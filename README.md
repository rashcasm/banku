# Banku

A Marathi household-memory voice assistant. You ask where something is, and it
answers out loud.

It was built to run on a cheap Android phone lying around the house: the phone is
the microphone and the speaker, the laptop does the thinking. Audio is captured
over USB with `scrcpy`, transcribed by Sarvam, reasoned over by an LLM that can
call a small set of memory tools, and spoken back through the phone's own
text-to-speech via Termux.

```
phone mic ──adb/scrcpy──▶ laptop ──▶ STT (Sarvam) ──▶ wake check
                                                     │
                                            LLM + memory tools
                                                     │
phone speaker ◀──Termux TTS──── laptop ◀──────────────┘
```

## What is here

| Path | Purpose |
| --- | --- |
| `src/banku/ears.py` | Always-on listener: wakes on the phone's mic, decides when an utterance ends |
| `src/banku/wake.py` | Decides whether a transcript was addressed to Banku |
| `src/banku/vad.py` | Voice activity detection, so silence is never sent to the ASR |
| `src/banku/stt.py` | Speech-to-text (Sarvam, swappable backend) |
| `src/banku/brain.py` | Turns a Marathi sentence into tool calls plus a Marathi answer |
| `src/banku/memory.py` | Where things are kept: canonical names, aliases, fuzzy lookup |
| `src/banku/server.py` | Legacy HTTP endpoints used by the phone-side recorder |
| `src/banku/control.py` | Start/stop/status/restart, one PID file, no orphaned listeners |
| `src/banku/config.py` | All tunables, read from the environment |
| `phone/` | Termux scripts for the phone side (recorder loops, TTS watcher) |

## Setup

Requires Python 3.12+, `adb`, `scrcpy`, and Termux with Termux:API on the phone.

```bash
uv sync
cp .env.example .env      # then add your API keys
uv run banku status
```

`banku start` launches the listener in the background; `banku stop` stops the
listener and the `scrcpy` capture process together. `on` and `off` are aliases.

## Configuration

Everything is read from `.env`. The important ones:

- `SARVAM_API_KEY` — speech-to-text
- `GROQ_API_KEY` — LLM
- `BANKU_STT_PROVIDER`, `BANKU_LLM_PROVIDER`, `BANKU_TTS_ENGINE` — swap backends
- `BANKU_LANG`, `BANKU_STT_LANGUAGE`, `BANKU_TTS_LANG` — language (English or Marathi)

See `.env.example` for the full set.

## Notes

`.env`, `data/` and logs are git-ignored. `data/banku.db` holds real household
memory and is intentionally not part of this repository.
