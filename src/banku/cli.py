"""Small CLI so the whole thing is testable before any voice is involved."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import config
from .memory import Memory


def _memory() -> Memory:
    return Memory()


def cmd_stt(args: argparse.Namespace) -> int:
    from .stt import STTError, get_stt

    audio = Path(args.file)
    keyterms = None
    if args.bias:
        keyterms = _memory().keyterms()

    try:
        t = get_stt().transcribe(audio, keyterms=keyterms)
    except STTError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(t.text)
    if t.no_speech:
        print(f"[no speech detected — peak {t.peak_db:.1f} dB, API not called]", file=sys.stderr)
    if args.verbose:
        print(
            f"\n[{t.engine} lang={t.language} p={t.language_probability} id={t.request_id}]",
            file=sys.stderr,
        )
    return 0


def cmd_remember(args: argparse.Namespace) -> int:
    r = _memory().remember(args.item, args.place, args.note, aliases=args.alias or [])
    verb = "शिकलो" if r.created else "अपडेट केलं"
    print(f"{verb}: {r.item} → {r.place}")
    print(f"aliases: {', '.join(r.aliases)}")
    return 0


def cmd_find(args: argparse.Namespace) -> int:
    m = _memory()
    hits = m.find(args.query)
    if not hits:
        print("माहीत नाही (no match)")
        return 2
    if m.is_ambiguous(hits):
        print("अनेक शक्यता आहेत — खात्री करा:")
    for h in hits:
        print(f"  {h.score:>3}  {h.as_sentence()}")
    return 0


def cmd_forget(args: argparse.Namespace) -> int:
    ok = _memory().forget(args.item)
    print("विसरलो" if ok else "माहीत नाही (no match)")
    return 0 if ok else 2


def cmd_alias(args: argparse.Namespace) -> int:
    ok = _memory().add_alias(args.item, args.alias)
    print("जोडलं" if ok else "जमलं नाही (item not found, or alias already used)")
    return 0 if ok else 2


def cmd_list(_: argparse.Namespace) -> int:
    for name, place, note in _memory().all_items():
        print(f"{name:<24} {place}" + (f"  ({note})" if note else ""))
    return 0


def cmd_start(_: argparse.Namespace) -> int:
    from .control import start

    return start()


def cmd_stop(_: argparse.Namespace) -> int:
    from .control import stop

    return stop()


def cmd_status(_: argparse.Namespace) -> int:
    from .control import status

    return status()


def cmd_restart(_: argparse.Namespace) -> int:
    from .control import restart

    return restart()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="banku", description="Marathi household memory assistant")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("stt", help="transcribe an audio file")
    s.add_argument("file")
    s.add_argument("--bias", action="store_true", help="use remembered items as keyterms")
    s.add_argument("-v", "--verbose", action="store_true")
    s.set_defaults(fn=cmd_stt)

    r = sub.add_parser("remember", help="learn where something is")
    r.add_argument("item")
    r.add_argument("place")
    r.add_argument("--note")
    r.add_argument("--alias", action="append", help="another name for the same thing")
    r.set_defaults(fn=cmd_remember)

    f = sub.add_parser("find", help="ask where something is")
    f.add_argument("query")
    f.set_defaults(fn=cmd_find)

    g = sub.add_parser("forget", help="delete an item")
    g.add_argument("item")
    g.set_defaults(fn=cmd_forget)

    a = sub.add_parser("alias", help="attach another name to an item")
    a.add_argument("item")
    a.add_argument("alias")
    a.set_defaults(fn=cmd_alias)

    sub.add_parser("list", help="show everything Banku knows").set_defaults(fn=cmd_list)

    sub.add_parser(
        "start", aliases=["on"], help="start listening (no voice command needed)"
    ).set_defaults(fn=cmd_start)
    sub.add_parser(
        "stop", aliases=["off"], help="stop listening and clear queued speech"
    ).set_defaults(fn=cmd_stop)
    sub.add_parser("status", help="show whether Banku is running").set_defaults(fn=cmd_status)
    sub.add_parser("restart", help="stop, then start again").set_defaults(fn=cmd_restart)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
