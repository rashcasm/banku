"""POST one audio chunk to the Banku server and print the reply as one line.

Uses only the standard library on purpose. The phone's `curl` broke once already
(libcurl linked against a newer openssl than was installed) and took the whole
assistant down with it. This talks plain HTTP over the LAN, so there is no TLS
stack involved and nothing to mismatch.

Output is nine fields separated by 0x1f (unit separator), on a single line, with
newlines and separators stripped out. The shell loop reads them with one `read`.
This replaced six `jq` invocations per chunk: each one cost the better part of a
second, and every millisecond spent parsing was a millisecond with the microphone
switched off, so words were falling into the gaps between chunks.

The session header keeps one phone's chunks together on the server, so a sentence
split across two chunks is stitched back into one utterance.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

SEP = "\x1f"
FIELDS = 9


def _clean(value: object) -> str:
    text = "" if value is None else str(value)
    for bad in ("\r", "\n", "\t", SEP):
        text = text.replace(bad, " ")
    return text.strip()


def _emit(values: list[object]) -> None:
    values = (values + [""] * FIELDS)[:FIELDS]
    sys.stdout.write(SEP.join(_clean(v) for v in values) + "\n")


def main() -> int:
    if len(sys.argv) < 3:
        _emit(["error", "usage: banku-ask.py <endpoint-url> <audiofile> [session]"])
        return 2

    endpoint, path = sys.argv[1], sys.argv[2]
    session = sys.argv[3] if len(sys.argv) > 3 else "phone"

    try:
        with open(path, "rb") as fh:
            payload = fh.read()
    except OSError as exc:
        _emit(["error", f"cannot read audio: {exc}"])
        return 1

    req = urllib.request.Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "audio/mp4", "X-Banku-Session": session},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            body = resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        detail = exc.read()[:200].decode("utf-8", "replace")
        _emit(["error", f"HTTP {exc.code}: {detail}"])
        return 1
    except Exception as exc:  # noqa: BLE001 - report, never crash the loop
        _emit(["error", f"{type(exc).__name__}: {exc}"])
        return 1

    try:
        data = json.loads(body)
    except ValueError:
        _emit(["error", f"bad response: {body[:180]}"])
        return 1

    tts = data.get("tts") or {}
    _emit([
        data.get("action", "idle"),
        data.get("transcript", ""),
        data.get("reply", ""),
        data.get("ms", 0),
        "true" if data.get("followup") else "false",
        tts.get("lang", "en"),
        tts.get("region", "IN"),
        tts.get("rate", 1.0),
        tts.get("engine", "com.google.android.tts"),
    ])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
