"""Always-on ears: hear the phone's microphone over USB, answer through its speaker.

Why this replaced the chunked recorder (26 Oct 2026)
----------------------------------------------------
The phone used to record 2 s chunks with `termux-microphone-record` and POST each
one. It worked, but every restart of the Android recorder cost ~600 ms of
`am startservice` latency with the microphone switched *off*. Measured on the
device: 2 s of audio captured per 3 s wall clock, so roughly a third of everything
said fell into a hole between chunks. For a device whose whole job is understanding
a question, silently swallowing a third of the words is fatal.

scrcpy can stream the phone microphone over the existing USB connection as one
continuous WAV stream (`--audio-source=mic-voice-recognition`, designed for exactly
this). So the laptop now owns the microphone: it reads frames as they arrive, runs
WebRTC VAD locally, and decides where an utterance starts and ends. Nothing is
recorded and re-uploaded, no file containers get finalised mid-stream, and the
phone is left with a single job — speaking the answer.

The phone-side helper (`say.sh`) just watches a directory for text files; the
laptop writes one and waits for the matching `.done` marker, which tells us
precisely when the phone stopped talking. That is what keeps Banku from hearing
its own voice.
"""

from __future__ import annotations

import argparse
import os
import queue
import select
import struct
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import wave
from collections import deque
from pathlib import Path

import numpy as np
import webrtcvad

from . import config, wake
from .brain import Brain
from .stt import SarvamSTT, get_stt

FIFO = Path(os.getenv("BANKU_MIC_FIFO", "/tmp/opencode/banku-mic.wav"))

# VAD tuning. Frames are 30 ms, the only size WebRTC likes at every rate.
FRAME_MS = 30
PRE_ROLL_S = 0.30          # audio kept from before the first voiced frame
ONSET_FRAMES = 2           # 60 ms of speech before we commit to listening
END_SILENCE_S = 0.45       # this much quiet ends the utterance
TAIL_S = 0.18              # keep a little after the last voiced frame
MIN_UTTERANCE_S = 0.35
MAX_UTTERANCE_S = 18.0     # Sarvam refuses >30 s; nobody asks longer than this

WAKE_GRACE_S = float(os.getenv("BANKU_WAKE_GRACE", "12"))

SAY_DIR = os.getenv("BANKU_SAY_DIR", "/sdcard/Download/banku/say")


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# --------------------------------------------------------------------------- capture


class Capture:
    """scrcpy microphone -> FIFO -> fixed-size mono frames."""

    def __init__(self, fifo: Path = FIFO) -> None:
        self.fifo = fifo
        self.proc: subprocess.Popen[bytes] | None = None
        self.stream = None
        self.sr = 48000
        self.ch = 2
        self.eof = threading.Event()
        self.stderr_path = Path("/tmp/opencode/scrcpy-audio.log")
        self.stderr_file = None
        self._started = False

    # -- provenance

    def _device_label(self) -> str:
        """Which phone this audio is actually coming from."""
        try:
            out = subprocess.run(
                ["adb", "shell", "getprop", "ro.product.model"],
                capture_output=True,
                timeout=10,
            ).stdout.decode(errors="replace").strip()
            return out or "unknown phone"
        except Exception:  # noqa: BLE001 - purely informational
            return "unknown phone"

    def _transport(self) -> str:
        """USB or wireless, straight from adb rather than from memory."""
        try:
            out = subprocess.run(
                ["adb", "devices", "-l"], capture_output=True, timeout=10
            ).stdout.decode(errors="replace")
            for line in out.splitlines():
                if " device " in line:
                    return "USB" if "usb:" in line else "wireless"
        except Exception:  # noqa: BLE001 - purely informational
            pass
        return "unknown"

    # -- lifecycle

    def start(self) -> None:
        self.fifo.unlink(missing_ok=True)
        os.mkfifo(self.fifo)

        # Open the read end *before* starting scrcpy, and non-blocking. Opening a
        # fifo blocks until the other side opens it; if scrcpy tried to open for
        # writing in that window it could fail and then sit there doing nothing,
        # which is exactly what happened: scrcpy held the fifo open, never wrote a
        # byte, and the reader blocked forever. Holding a non-blocking reader from
        # the start removes the race and the deadlock.
        fd = os.open(self.fifo, os.O_RDONLY | os.O_NONBLOCK)
        self.stream = os.fdopen(fd, "rb", buffering=0)

        self.stderr_file = open(self.stderr_path, "ab", buffering=0)
        self.proc = subprocess.Popen(
            [
                "scrcpy",
                "--no-video",
                "--no-window",
                # Capture only. Without this scrcpy also *plays* the microphone
                # through the laptop speakers, which is a live feedback loop and
                # sounds exactly like something is wrong with the machine.
                "--no-audio-playback",
                "--audio-source=mic-voice-recognition",
                "--audio-codec=raw",
                "--audio-buffer=30",
                "--record", str(self.fifo),
                "--record-format=wav",
            ],
            stdout=self.stderr_file,
            stderr=self.stderr_file,
        )

        self._read_header()
        log(f"capture started: {self.sr} Hz, {self.ch} ch  <- microphone is the phone")
        log(f"  source: adb {self._device_label()} ({self._transport()})")

    def close(self) -> None:
        self.eof.set()
        if self.stream is not None:
            try:
                self.stream.close()
            except OSError:
                pass
            self.stream = None
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            self.proc = None
        self.fifo.unlink(missing_ok=True)
        if self.stderr_file is not None:
            self.stderr_file.close()
            self.stderr_file = None

    # -- wav header

    def _read_exact(self, n: int) -> bytes:
        data = b""
        fd = self.stream.fileno()
        deadline = time.monotonic() + 20
        while len(data) < n:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("no audio from the phone; scrcpy never sent a frame")
            ready, _, _ = select.select([fd], [], [], remaining)
            if not ready:
                continue
            try:
                chunk = os.read(fd, n - len(data))
            except BlockingIOError:
                continue
            if not chunk:
                # A fifo read returns nothing while there is no writer at all.
                # Only treat it as a close once audio has actually started.
                if self._started:
                    raise EOFError("microphone stream closed")
                time.sleep(0.02)
                continue
            self._started = True
            data += chunk
        return data

    def _read_header(self) -> None:
        head = self._read_exact(12)
        riff, _, wave_id = struct.unpack("<4sI4s", head)
        if riff != b"RIFF" or wave_id != b"WAVE":
            raise RuntimeError(f"unexpected stream header: {head!r}")

        while True:
            chunk_id, size = struct.unpack("<4sI", self._read_exact(8))
            if chunk_id == b"data":
                # The stream is live, so scrcpy leaves a placeholder length here
                # (0xFFFFFFFF). Reading `size` bytes would block forever; the
                # audio simply starts after this header.
                break
            if size > 4096:
                raise RuntimeError(f"unreasonable wav chunk size {size} for {chunk_id!r}")
            body = self._read_exact(size)
            if chunk_id == b"fmt ":
                fmt, ch, sr = struct.unpack("<HHI", body[:8])
                bits = struct.unpack("<H", body[14:16])[0]
                if fmt != 1 or bits != 16:
                    raise RuntimeError(f"need 16-bit PCM, got fmt={fmt} bits={bits}")
                self.sr, self.ch = sr, ch
                if sr not in (8000, 16000, 32000, 48000):
                    raise RuntimeError(f"unsupported sample rate {sr}")
            # any other chunk is skipped

    # -- frames

    @property
    def frame_samples(self) -> int:
        return self.sr * FRAME_MS // 1000

    @property
    def frame_bytes(self) -> int:
        return self.frame_samples * self.ch * 2

    def pump(self, out: queue.Queue[bytes]) -> None:
        """Read the stream and emit mono 16-bit frames. Runs in a thread."""
        nbytes = self.frame_bytes
        fd = self.stream.fileno()
        buf = b""
        try:
            while not self.eof.is_set():
                ready, _, _ = select.select([fd], [], [], 0.5)
                if not ready:
                    continue
                try:
                    data = os.read(fd, nbytes - len(buf))
                except BlockingIOError:
                    continue
                if not data:
                    # No writer yet, or scrcpy exited. If we were already getting
                    # audio, this is the end of the stream.
                    if self._started:
                        break
                    time.sleep(0.02)
                    continue
                buf += data
                while len(buf) >= nbytes:
                    raw, buf = buf[:nbytes], buf[nbytes:]
                    samples = np.frombuffer(raw, dtype="<i2")
                    if self.ch > 1:
                        samples = samples.reshape(-1, self.ch).mean(axis=1)
                    mono = samples.astype("<i2").tobytes()
                    try:
                        out.put_nowait(mono)
                    except queue.Full:
                        try:
                            out.get_nowait()
                        except queue.Empty:
                            pass
                        out.put_nowait(mono)
        except Exception:  # noqa: BLE001 - stream died; supervisor restarts
            traceback.print_exc()
        finally:
            self.eof.set()


# --------------------------------------------------------------------------- speaking


class PhoneSpeaker:
    """Ask the phone to say something, and know when it has finished."""

    def __init__(self) -> None:
        self.adb = os.getenv("BANKU_ADB", "adb")
        self.dir = SAY_DIR
        self._shell(["mkdir", "-p", self.dir])

    def _shell(self, args: list[str], timeout: float = 20) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(
            [self.adb, "shell", *args], capture_output=True, timeout=timeout
        )

    def say(self, text: str) -> bool:
        started = time.monotonic()
        stamp = f"{time.time():015.3f}".replace(".", "")
        base = f"{self.dir}/{stamp}"
        local = Path(tempfile.mkstemp(suffix=".txt")[1])
        try:
            local.write_text(text, encoding="utf-8")
            # Push under a name the helper ignores, then rename, so it never reads
            # a half-written file.
            subprocess.run([self.adb, "push", str(local), f"{base}.tmp"],
                           capture_output=True, timeout=30)
            self._shell(["mv", "-f", f"{base}.tmp", f"{base}.txt"])
        finally:
            local.unlink(missing_ok=True)

        deadline = time.monotonic() + 90
        confirmed = False
        while time.monotonic() < deadline:
            if self._shell(["test", "-f", f"{base}.done"]).returncode == 0:
                self._shell(["rm", "-f", f"{base}.done"])
                confirmed = True
                break
            time.sleep(0.15)
        if not confirmed:
            log("phone never confirmed it finished speaking")

        # `termux-tts-speak` reports completion when synthesis is done, not when
        # the last word has left the speaker, and the audio path adds a little
        # more delay. Measured on the device: it was still audibly talking two
        # seconds after `.done` appeared, and Banku then transcribed its own
        # answer as if it were a question. So keep the ears closed until the
        # sentence could plausibly have finished, based on its length.
        estimated = len(text) / 13.0 + 0.5
        remaining = estimated - (time.monotonic() - started)
        if remaining > 0:
            time.sleep(remaining)
        return confirmed


# --------------------------------------------------------------------------- listening


class Ears:
    def __init__(self, stt: SarvamSTT, brain: Brain, speaker: PhoneSpeaker) -> None:
        self.stt = stt
        self.brain = brain
        self.speaker = speaker
        self.vad = webrtcvad.Vad(2)  # 2: ignores fans and distant TV, keeps quiet speech
        self.busy = threading.Event()
        self._last_wake = 0.0
        self._keyterms: list[str] = []
        self._keyterms_at = 0.0
        self._frames: queue.Queue[bytes] | None = None

    def _drain(self) -> None:
        """Throw away audio that arrived while the reply was playing.

        The capture path lags real time by a moment, so frames captured during
        Banku's own sentence keep arriving for a short while after we think we
        have stopped listening. Left in the queue they would be heard as speech
        and answered — Banku talking to itself.
        """
        if self._frames is None:
            return
        while True:
            try:
                self._frames.get_nowait()
            except queue.Empty:
                return

    def _bias_terms(self) -> list[str]:
        """Household item names, to bias the transcriber toward them.

        Without this, "nail cutter" came back as "neelkata" and the lookup
        correctly said it didn't know. Every item Banku learns makes it better at
        hearing that item next time. Refreshed occasionally so new items count.
        """
        now = time.monotonic()
        if now - self._keyterms_at > 60:
            try:
                terms = self.brain.memory.keyterms()
                # Bias toward the wake word too. Without this the item keyterms
                # can win: "Banku, where are my spectacles?" came back as
                # "Nail cutter, where are my spectacles?" because "nail cutter"
                # is a much stronger keyterm than a name the engine has never
                # heard. Adding the name costs nothing and protects the one word
                # the whole conversation depends on.
                self._keyterms = ["Banku", *terms][:50]
            except Exception:  # noqa: BLE001 - never let this block listening
                traceback.print_exc()
            self._keyterms_at = now
        return self._keyterms

    # -- utterance handling

    def _write_wav(self, pcm: bytes, sr: int) -> Path:
        out = Path(tempfile.mkstemp(suffix=".wav", prefix="banku-utt-")[1])
        with wave.open(str(out), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(pcm)
        return out

    def _addressed(self, text: str) -> tuple[bool, str, bool]:
        addressed, command = wake.split(text)
        if addressed:
            self._last_wake = time.monotonic()
            return True, command, False
        if time.monotonic() - self._last_wake <= WAKE_GRACE_S and len(text.split()) >= 2:
            return True, text, True
        return False, "", False

    def handle(self, pcm: bytes, sr: int) -> None:
        t0 = time.monotonic()
        self.busy.set()
        wav = self._write_wav(pcm, sr)
        try:
            got = self.stt.transcribe(wav, keyterms=self._bias_terms())
            if got.no_speech or not got.text:
                return
            stt_ms = (time.monotonic() - t0) * 1000

            addressed, command, followup = self._addressed(got.text)
            if not addressed:
                log(f"heard: {got.text!r} -> not for me")
                return

            log(f"heard: {got.text!r}{' (follow-up)' if followup else ''} [stt {stt_ms:.0f} ms]")
            t1 = time.monotonic()
            query = command or got.text
            try:
                quick = self.brain.quick_answer(query)
                if quick is not None:
                    text = quick
                    log(f"answer [local lookup, no model]: {text}")
                else:
                    answer = self.brain.ask(query)
                    text = answer.text
                    log(f"reply [{(time.monotonic() - t1) * 1000:.0f} ms]: {text}")
            except Exception:  # noqa: BLE001 - network, rate limit, bad reply
                # Never answer a question with silence. Silence looks exactly
                # like a dead microphone, which once sent us hunting for a
                # hardware fault that did not exist. Say that something failed,
                # so the person knows to try again rather than shout louder.
                traceback.print_exc()
                text = "Sorry, my brain is not reachable right now. Please ask me again."
            if text:
                self.speaker.say(text)
                # A follow-up is usually asked straight after hearing the answer,
                # so the grace window has to start when Banku stops talking, not
                # when the question arrived. Before this, a nine-second answer
                # used up the whole eight-second window and the follow-up was
                # thrown away.
                self._last_wake = time.monotonic()
            log(f"total {(time.monotonic() - t0) * 1000:.0f} ms")
        except Exception:  # noqa: BLE001 - one bad utterance must not kill the ears
            traceback.print_exc()
        finally:
            wav.unlink(missing_ok=True)
            # Let the last bit of speaker sound clear before listening again.
            time.sleep(0.4)
            self._drain()
            self.busy.clear()

    # -- main loop

    def listen(self, capture: Capture, frames: queue.Queue[bytes]) -> None:
        self._frames = frames
        pre: deque[bytes] = deque(maxlen=max(1, int(PRE_ROLL_S * 1000 / FRAME_MS)))
        collected: list[bytes] = []
        voiced_run = 0
        silent_run = 0
        last_voice = 0
        collecting = False

        while True:
            try:
                frame = frames.get(timeout=0.4)
            except queue.Empty:
                if capture.eof.is_set():
                    return
                continue

            if self.busy.is_set():
                pre.clear()
                collecting, collected, voiced_run, silent_run = False, [], 0, 0
                continue

            voiced = self.vad.is_speech(frame, capture.sr)

            if not collecting:
                pre.append(frame)
                voiced_run = voiced_run + 1 if voiced else 0
                if voiced_run >= ONSET_FRAMES:
                    collecting = True
                    collected = list(pre)
                    pre.clear()
                    last_voice = len(collected) - 1
                    silent_run = 0
                continue

            collected.append(frame)
            if voiced:
                last_voice = len(collected) - 1
                silent_run = 0
            else:
                silent_run += 1

            seconds = len(collected) * FRAME_MS / 1000
            ended = silent_run * FRAME_MS / 1000 >= END_SILENCE_S
            if not ended and seconds < MAX_UTTERANCE_S:
                continue

            keep = min(len(collected), last_voice + 1 + int(TAIL_S * 1000 / FRAME_MS))
            utterance = b"".join(collected[:keep])
            collecting, collected, voiced_run, silent_run = False, [], 0, 0

            if len(utterance) / (2 * capture.sr) >= MIN_UTTERANCE_S:
                self.handle(utterance, capture.sr)


# --------------------------------------------------------------------------- entry


def serve() -> None:
    stt = get_stt()
    brain = Brain()
    speaker = PhoneSpeaker()
    ears = Ears(stt, brain, speaker)
    log(f"ears starting (lang={config.language.lang}, threshold={config.match.threshold})")
    log("  replies are spoken by the phone's own text-to-speech, not the laptop")

    while True:
        capture = Capture()
        try:
            capture.start()
            frames: queue.Queue[bytes] = queue.Queue(maxsize=300)
            reader = threading.Thread(target=capture.pump, args=(frames,), daemon=True)
            reader.start()
            ears.listen(capture, frames)
            log("microphone stream ended; restarting")
        except KeyboardInterrupt:
            log("stopping")
            return
        except Exception:  # noqa: BLE001 - scrcpy died, USB hiccup, etc.
            traceback.print_exc()
            log("capture failed; retrying")
        finally:
            capture.close()
        time.sleep(1.5)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Banku ears")
    parser.add_argument("--file", type=Path, help="run VAD+STT on a wav file, no capture")
    parser.add_argument("--no-speak", action="store_true", help="do not speak replies")
    args = parser.parse_args(argv)

    if args.file:
        stt = get_stt()
        brain = Brain()
        speaker = PhoneSpeaker()
        ears = Ears(stt, brain, speaker)
        if args.no_speak:
            ears.speaker = type("Silent", (), {"say": lambda self, _t: True})()  # type: ignore[assignment]
        with wave.open(str(args.file), "rb") as w:
            sr = w.getframerate()
            pcm = w.readframes(w.getnframes())
        ears.handle(pcm, sr)
        return 0

    serve()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
