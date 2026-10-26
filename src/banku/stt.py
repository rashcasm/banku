"""Marathi speech-to-text.

Sarvam is the primary engine. The interface is deliberately tiny (`transcribe`)
so that a local faster-whisper or IndicConformer backend can be dropped in
behind the same call later and compared on the same recordings.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import httpx

from . import config
from . import net
from .vad import detect


class STTError(RuntimeError):
    """Raised for anything that stops us producing a transcript."""


@dataclass(frozen=True)
class Transcript:
    text: str
    language: str | None = None
    language_probability: float | None = None
    request_id: str | None = None
    engine: str = ""
    seconds: float | None = None
    # True when the clip was too quiet to contain speech. Set without calling
    # the API, because silent audio makes ASR models hallucinate fluent text.
    no_speech: bool = False
    peak_db: float | None = None


def audio_level(path: Path) -> tuple[float, float]:
    """Return (mean_db, peak_db). Cheap loudness probe via ffmpeg."""
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", str(path), "-af", "volumedetect", "-f", "null", "-"],
        capture_output=True,
        text=True,
    )
    mean = peak = -100.0
    for line in proc.stderr.splitlines():
        if "mean_volume:" in line:
            mean = float(line.split("mean_volume:")[1].split("dB")[0])
        elif "max_volume:" in line:
            peak = float(line.split("max_volume:")[1].split("dB")[0])
    return mean, peak


def to_wav16k(src: Path, dst: Path | None = None, *, normalize: bool = True) -> Path:
    """Sarvam is most accurate at 16 kHz mono, and Termux records whatever it likes.

    Two things we learned from real recordings on the phone:

    1. Termux's MediaRecorder defaults to 8000 Hz, which throws away exactly the
       high-frequency detail that separates Marathi consonants. We resample.
    2. A phone on a shelf across the room records speech 30-40 dB quieter than a
       headset would. Left alone, that silence reads as noise to the ASR model.
       So we normalise loudness (with a peak limiter) and high-pass the rumble.
    """
    if shutil.which("ffmpeg") is None:
        raise STTError("ffmpeg not found; install it or send 16 kHz mono WAV directly")

    out = dst or Path(tempfile.mkstemp(suffix=".wav", prefix="banku-")[1])
    filters = ["highpass=f=80"]
    if normalize:
        # loudnorm gates silence, so it lifts speech without amplifying the room.
        filters.append("loudnorm=I=-18:TP=-2:LRA=11")

    proc = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(src),
            "-af", ",".join(filters),
            "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise STTError(f"ffmpeg could not decode {src.name}: {proc.stderr.strip()[:300]}")
    return out


def _cut(wav: Path, start: float, end: float) -> Path:
    """Slice a 16 kHz wav to [start, end) seconds."""
    out = Path(tempfile.mkstemp(suffix=".wav", prefix="banku-cut-")[1])
    proc = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
            "-i", str(wav), "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise STTError(f"ffmpeg could not cut audio: {proc.stderr.strip()[:300]}")
    return out


class SarvamSTT:
    def __init__(self, cfg: config.STTConfig | None = None) -> None:
        self.cfg = cfg or config.stt
        if not self.cfg.api_key:
            raise STTError(
                "SARVAM_API_KEY is not set. Put it in ~/banku/.env "
                "(get a key at https://dashboard.sarvam.ai)."
            )

    def transcribe(self, audio: Path, *, keyterms: list[str] | None = None) -> Transcript:
        if not audio.exists():
            raise STTError(f"no such file: {audio}")

        wav = to_wav16k(audio, normalize=self.cfg.normalize)
        try:
            vad = detect(wav)
            if not vad.has_speech:
                return Transcript(
                    text="",
                    engine=f"sarvam/{self.cfg.model}",
                    no_speech=True,
                    peak_db=vad.peak_db,
                )

            # Transcribe each speech segment separately. This is not just about
            # the 30 s REST limit: short clips around one utterance are also
            # more accurate than a long file with the washing machine in it.
            texts: list[str] = []
            langs: list[str] = []
            for seg in vad.segments:
                piece = _cut(wav, seg.start, seg.end)
                try:
                    got = self._post(piece, keyterms=keyterms)
                    texts.append(got.text)
                    if got.language:
                        langs.append(got.language)
                finally:
                    piece.unlink(missing_ok=True)

            text = " ".join(t for t in texts if t).strip()
            return Transcript(
                text=text,
                # Report what the API actually detected, not what we asked for.
                # We send mr-IN as a hint, but Sarvam is free to answer in Hindi
                # when the speaker switches, and that is information we want.
                language=langs[0] if langs else None,
                engine=f"sarvam/{self.cfg.model}",
                seconds=sum(s.duration for s in vad.segments),
                peak_db=vad.peak_db,
            )
        finally:
            wav.unlink(missing_ok=True)

    def _post(self, wav: Path, *, keyterms: list[str] | None) -> Transcript:
        fields: dict[str, str] = {
            "model": self.cfg.model,
            "language_code": self.cfg.language,
        }
        # `mode` only applies to saaras:v3 per the API reference; sending it to
        # v4 is harmless but we keep the logic honest.
        if self.cfg.model.startswith("saaras:v3"):
            fields["mode"] = self.cfg.mode
        if keyterms:
            # Up to 50 terms, 64 chars each. Only saaras:v4 honours these.
            import json

            fields["keyterms"] = json.dumps(keyterms[:50], ensure_ascii=False)

        try:
            with open(wav, "rb") as fh:
                resp = net.post(
                    self.cfg.endpoint,
                    headers={"api-subscription-key": self.cfg.api_key},
                    files={"file": (wav.name, fh, "audio/wav")},
                    data=fields,
                    timeout=self.cfg.timeout_s,
                )
        except httpx.HTTPError as exc:
            raise STTError(f"could not reach Sarvam: {exc}") from exc

        if resp.status_code != 200:
            raise STTError(f"Sarvam returned {resp.status_code}: {resp.text[:300]}")

        body = resp.json()
        return Transcript(
            text=(body.get("transcript") or "").strip(),
            language=body.get("language_code"),
            language_probability=body.get("language_probability"),
            request_id=body.get("request_id"),
            engine=f"sarvam/{self.cfg.model}",
        )


def get_stt() -> SarvamSTT:
    return SarvamSTT()
