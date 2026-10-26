"""HTTP endpoints the phone talks to.

Two ways in:

    POST /ask     one complete audio clip -> one reply. Simple, used by tests.
    POST /listen  one short chunk (about 1.5 s) -> accumulate until the speaker
                  stops, then one reply.

Why /listen exists (26 Oct 2026). The first design recorded fixed seven-second
windows. That is fine for catching speech and terrible for latency: if you finish
talking one second into a window, the phone still waits six more seconds doing
nothing, then thinks for two, then speaks. The user's verdict was "replies are
pretty late", and they were right — most of that delay was the phone waiting for
a clock, not for anything real.

So the phone now sends whatever it has every 1.5 s and the server decides when an
utterance has ended, using voice activity detection on each chunk. Speech chunks
are accumulated; the first chunk with no speech flushes the utterance for
transcription. That also fixes broken sentences at window boundaries, because a
sentence is no longer cut at an arbitrary time — the whole thing is stitched back
together before it is ever transcribed.

Side benefit: fewer, longer STT calls instead of one per window.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import traceback
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import config, wake
from .brain import Brain
from .stt import get_stt, to_wav16k
from .vad import detect

MAX_BYTES = 12 * 1024 * 1024

# How long after hearing "Banku" we keep treating speech as addressed to us.
#
# Without this, a capture boundary can cut the name off from the question, and
# the question on its own is (correctly) ignored. It also lets Aai ask a
# follow-up without repeating the name, which is how people actually talk.
WAKE_GRACE_S = float(os.getenv("BANKU_WAKE_GRACE", "8"))

# Hard ceiling on one utterance. Sarvam's REST API refuses audio over 30 s, and
# nobody asks a household question for longer than this.
MAX_UTTERANCE_S = float(os.getenv("BANKU_MAX_UTTERANCE", "18"))

# A session with no traffic for this long is forgotten.
SESSION_IDLE_S = 120.0

# How much silence at the end of a chunk means "they have finished talking".
#
# Without this we would wait for a chunk containing no speech at all, which costs
# a whole extra chunk of delay every time. Looking at where the last speech ends
# inside the chunk answers the question about 2 s sooner.
TRAILING_SILENCE_S = float(os.getenv("BANKU_TRAILING_SILENCE", "0.45"))

_wake_lock = threading.Lock()
_last_wake_at = 0.0


class Session:
    """Chunks of one in-progress utterance."""

    __slots__ = ("chunks", "seconds", "touched")

    def __init__(self) -> None:
        self.chunks: list[Path] = []
        self.seconds: float = 0.0
        self.touched: float = time.monotonic()

    def clear(self) -> None:
        for c in self.chunks:
            c.unlink(missing_ok=True)
        self.chunks = []
        self.seconds = 0.0
        self.touched = time.monotonic()


_sessions: dict[str, Session] = {}
_sessions_lock = threading.Lock()


def _merge_wavs(paths: list[Path], out: Path) -> None:
    """Concatenate 16 kHz mono 16-bit wavs.

    They all come from `to_wav16k`, so the frames are directly compatible and
    there is no need to shell out to ffmpeg for this.
    """
    with wave.open(str(out), "wb") as dst:
        dst.setnchannels(1)
        dst.setsampwidth(2)
        dst.setframerate(16000)
        for p in paths:
            with wave.open(str(p), "rb") as src:
                dst.writeframes(src.readframes(src.getnframes()))


def _seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def _sweep_sessions() -> None:
    now = time.monotonic()
    with _sessions_lock:
        for sid in [k for k, s in _sessions.items() if now - s.touched > SESSION_IDLE_S]:
            _sessions.pop(sid).clear()


class Handler(BaseHTTPRequestHandler):
    brain: Brain | None = None
    server_version = "Banku/0.2"

    def log_message(self, fmt: str, *args) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {fmt % args}", flush=True)

    # ------------------------------------------------------------------ plumbing

    def _json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> bytes | None:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            self._json(400, {"error": "empty body"})
            return None
        if length > MAX_BYTES:
            self._json(413, {"error": "audio too large"})
            return None
        return self.rfile.read(length)

    def do_GET(self) -> None:
        if self.path == "/health":
            _sweep_sessions()
            self._json(200, {"ok": True, "sessions": len(_sessions)})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path == "/ask":
            self._ask()
        elif self.path == "/listen":
            self._listen()
        else:
            self._json(404, {"error": "not found"})

    # ------------------------------------------------------------------ replies

    def _answer(self, transcript: str, t0: float) -> dict:
        """Decide whether this transcript is addressed to us, and answer it."""
        addressed, command = wake.split(transcript)
        followup = False
        global _last_wake_at
        if addressed:
            with _wake_lock:
                _last_wake_at = time.monotonic()
        else:
            with _wake_lock:
                since = time.monotonic() - _last_wake_at
            if since <= WAKE_GRACE_S and len(transcript.split()) >= 2:
                addressed, followup, command = True, True, transcript

        if not addressed:
            return {
                "transcript": transcript,
                "reply": "",
                "no_wake": True,
                "ms": int((time.monotonic() - t0) * 1000),
            }

        r = self.brain.ask(command or transcript)  # type: ignore[union-attr]
        return {
            "transcript": transcript,
            "command": command,
            "followup": followup,
            "reply": r.text,
            "tools": r.tool_calls,
            "tts": {
                "text": r.text,
                "lang": config.tts.lang,
                "region": config.tts.region,
                "rate": config.tts.rate,
                "engine": config.tts.engine,
            },
            "ms": int((time.monotonic() - t0) * 1000),
        }

    def _ask(self) -> None:
        audio = self._read_body()
        if audio is None:
            return
        ctype = (self.headers.get("Content-Type") or "audio/mp4").lower()
        suffix = ".wav" if "wav" in ctype else ".ogg" if "ogg" in ctype else ".m4a"
        tmp = Path(tempfile.mkstemp(suffix=suffix, prefix="banku-in-")[1])
        tmp.write_bytes(audio)
        t0 = time.monotonic()
        try:
            t = get_stt().transcribe(tmp)
            if t.no_speech or not t.text:
                self._json(200, {"transcript": "", "reply": "", "no_speech": True,
                                 "ms": int((time.monotonic() - t0) * 1000)})
                return
            self._json(200, self._answer(t.text, t0))
        except Exception as exc:  # noqa: BLE001 - never let the phone hang
            traceback.print_exc()
            self._json(500, {"error": str(exc)[:300]})
        finally:
            tmp.unlink(missing_ok=True)

    def _listen(self) -> None:
        audio = self._read_body()
        if audio is None:
            return
        t0 = time.monotonic()
        sid = self.headers.get("X-Banku-Session") or self.client_address[0]

        ctype = (self.headers.get("Content-Type") or "audio/mp4").lower()
        suffix = ".wav" if "wav" in ctype else ".ogg" if "ogg" in ctype else ".m4a"
        tmp = Path(tempfile.mkstemp(suffix=suffix, prefix="banku-chunk-")[1])
        tmp.write_bytes(audio)

        wav: Path | None = None
        try:
            # Decode without loudness normalisation: the merged utterance is
            # normalised once at the end, which keeps levels consistent across
            # chunks instead of pumping each 1.5 s piece separately.
            wav = to_wav16k(tmp, normalize=False)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            tmp.unlink(missing_ok=True)
            self._json(500, {"error": str(exc)[:300]})
            return
        finally:
            tmp.unlink(missing_ok=True)

        try:
            v = detect(wav)
            _sweep_sessions()

            with _sessions_lock:
                session = _sessions.get(sid)
                if session is None:
                    session = _sessions[sid] = Session()
                session.touched = time.monotonic()
                chunk_s = _seconds(wav)

                if v.has_speech:
                    session.chunks.append(wav)
                    session.seconds += sum(s.duration for s in v.segments)
                    wav = None  # ownership moved to the session

                utterance = None
                if v.has_speech:
                    # Speech ends inside this chunk when a stretch of silence
                    # follows the last voiced segment. Waiting for a chunk with
                    # no speech at all would add a whole chunk of delay to every
                    # single reply.
                    quiet_tail = chunk_s - v.segments[-1].end
                    ended = quiet_tail >= TRAILING_SILENCE_S
                else:
                    ended = True

                if ended and session.chunks:
                    utterance = session.chunks[:]
                    session.chunks = []
                    session.seconds = 0.0
                elif v.has_speech and session.seconds >= MAX_UTTERANCE_S:
                    utterance = session.chunks[:]
                    session.chunks = []
                    session.seconds = 0.0

            if utterance is None:
                self._json(200, {"action": "listening", "ms": int((time.monotonic() - t0) * 1000)})
                return

            merged = Path(tempfile.mkstemp(suffix=".wav", prefix="banku-utt-")[1])
            try:
                _merge_wavs(utterance, merged)
                t = get_stt().transcribe(merged)
            finally:
                for p in utterance:
                    p.unlink(missing_ok=True)
                merged.unlink(missing_ok=True)

            if t.no_speech or not t.text:
                self._json(200, {"action": "idle", "ms": int((time.monotonic() - t0) * 1000)})
                return

            result = self._answer(t.text, t0)
            result["action"] = "idle" if not result.get("reply") else "reply"
            self._json(200, result)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self._json(500, {"action": "error", "error": str(exc)[:300]})
        finally:
            if wav is not None:
                wav.unlink(missing_ok=True)


def main() -> None:
    Handler.brain = Brain()
    _sweep_sessions()
    srv = ThreadingHTTPServer((config.server.host, config.server.port), Handler)
    print(f"Banku listening on http://{config.server.host}:{config.server.port}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
