"""Voice activity detection.

Why this exists: on 26 Oct 2026 we fed a near-silent 9-second room recording to
Sarvam and it returned "हो." — a confident, plausible, completely invented
Marathi word. ASR models are trained to always produce text, so silence is not
"no input" to them; it is a prompt to improvise. For a device that answers
"where is the passport", an invented answer is far worse than silence.

First attempt was a frame-energy threshold: anything 9 dB above the room's own
noise floor counts as voice. On the same recording it said "speech" for every
clip, because a fan or a distant TV is also 9 dB above the floor. Energy cannot
tell a human from a hairdryer, so we use WebRTC's VAD instead — the same
classifier that has been doing this job in phone calls for a decade. It weighs
spectral shape, not just loudness.

Silero VAD is the upgrade path if we later need better rejection of background
music; it is an ONNX model and would also run on the phone.
"""

from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import webrtcvad


@dataclass(frozen=True)
class Segment:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class VADResult:
    has_speech: bool
    segments: list[Segment]
    noise_db: float
    """Room noise floor in dBFS, for diagnostics."""
    peak_db: float
    """Peak frame energy in dBFS, for diagnostics."""


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError(f"expected 16-bit mono wav, got {path}")
        sr = w.getframerate()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    return data, sr


def _frame_db(signal: np.ndarray, sr: int, win_s: float = 0.03) -> np.ndarray:
    win = max(1, int(win_s * sr))
    n = max(1, len(signal) // win)
    frames = signal[: n * win].reshape(n, win).astype(np.float32) / 32768.0
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
    return 20.0 * np.log10(rms + 1e-12)


def detect(
    path: Path,
    *,
    mode: int = 2,
    frame_ms: int = 30,
    max_gap_s: float = 0.35,
    pad_s: float = 0.25,
    min_contiguous_s: float = 0.15,
    min_speech_s: float = 0.30,
) -> VADResult:
    """Return speech regions, or nothing if nobody was talking.

    `mode` is WebRTC's aggressiveness, 0-3. 2 is the balance point: 3 starts
    eating quiet speech from across a room, which is exactly our use case.
    """
    signal, sr = read_wav(path)
    db = _frame_db(signal, sr)
    floor = float(np.percentile(db, 20)) if len(db) else -120.0
    peak = float(np.max(db)) if len(db) else -120.0

    if len(signal) < sr // 10:  # under 100 ms
        return VADResult(False, [], floor, peak)

    vad = webrtcvad.Vad(mode)
    win = int(sr * frame_ms / 1000)
    hop_s = frame_ms / 1000.0

    # WebRTC wants exact 10/20/30 ms frames; pad the tail rather than drop it.
    pad = (-len(signal)) % win
    pcm = np.pad(signal, (0, pad)).astype("<i2").tobytes()

    voiced: list[Segment] = []
    run_start: float | None = None
    for i in range(len(pcm) // (win * 2)):
        chunk = pcm[i * win * 2 : (i + 1) * win * 2]
        is_speech = vad.is_speech(chunk, sr)
        t = i * hop_s
        if is_speech and run_start is None:
            run_start = t
        elif not is_speech and run_start is not None:
            voiced.append(Segment(run_start, t))
            run_start = None
    if run_start is not None:
        voiced.append(Segment(run_start, len(pcm) / 2 / sr))

    # Bridge short gaps so "nail ... cutter" stays one utterance.
    merged: list[Segment] = []
    for seg in voiced:
        if merged and (seg.start - merged[-1].end) <= max_gap_s:
            merged[-1] = Segment(merged[-1].start, seg.end)
        else:
            merged.append(seg)

    total = len(signal) / sr
    keep = [
        Segment(max(0.0, s.start - pad_s), min(total, s.end + pad_s))
        for s in merged
        if s.duration >= min_contiguous_s
    ]
    has_speech = sum(s.duration for s in keep) >= min_speech_s

    return VADResult(has_speech, keep if has_speech else [], floor, peak)
