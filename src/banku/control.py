"""Start/stop Banku without saying a word.

The ears process is deliberately detached from the terminal, but that used to
mean there was no reliable way to stop it: closing the terminal left it running,
and a second copy could be started by accident. This module owns one PID file,
starts exactly one listener, and stops the whole process group (ears + scrcpy).

Commands are wired into the CLI as `banku start|stop|status|restart` and are
also available as `on`/`off` aliases.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
PID_FILE = DATA / "banku.pid"
LOG_FILE = DATA / "banku.log"
FIFO = Path(os.getenv("BANKU_MIC_FIFO", "/tmp/opencode/banku-mic.wav"))
SAY_DIR = os.getenv("BANKU_SAY_DIR", "/sdcard/Download/banku/say")

READY_MARKER = "capture started:"
START_TIMEOUT_S = 30.0
STOP_TIMEOUT_S = 6.0


# --------------------------------------------------------------------------- process inspection


def _argv(pid: int) -> list[str]:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return []
    return [part.decode("utf-8", "replace") for part in raw.split(b"\0") if part]


def _exe(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/exe").resolve().name
    except OSError:
        return ""


def _is_running(pid: int) -> bool:
    """True unless the process is gone or already a zombie."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return False
    return stat.split()[2] != "Z"


def _find_ears() -> list[int]:
    """Find live `python -m banku.ears` processes, including manually started ones."""
    found: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        exe = _exe(pid)
        if not (exe.startswith("python") or exe == "uv"):
            continue
        if "banku.ears" in _argv(pid):
            found.append(pid)
    return sorted(found)


def _find_scrcpy() -> list[int]:
    """Find the scrcpy processes that belong to Banku, not an unrelated scrcpy."""
    found: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        argv = _argv(pid)
        if not argv or Path(argv[0]).name != "scrcpy":
            continue
        if any(a.startswith("--audio-source") for a in argv) and any(
            "banku-mic" in a for a in argv
        ):
            found.append(pid)
    return sorted(found)


def _read_pid() -> int | None:
    try:
        return int(PID_FILE.read_text().strip())
    except (OSError, ValueError):
        return None


def _is_managed(pid: int | None) -> bool:
    return bool(pid and _is_running(pid) and "banku.ears" in _argv(pid))


def _uptime(pid: int) -> str:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().split()
        start_ticks = int(fields[21])
        seconds = int(
            time.clock_gettime(time.CLOCK_BOOTTIME)
            - start_ticks / os.sysconf("SC_CLK_TCK")
        )
    except (OSError, ValueError, IndexError):
        return "unknown"
    minutes, seconds = divmod(max(0, seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


# --------------------------------------------------------------------------- phone helpers


def _adb(args: list[str], timeout: float = 10.0) -> subprocess.CompletedProcess[str] | None:
    if not shutil.which("adb"):
        return None
    try:
        return subprocess.run(
            ["adb", *args], capture_output=True, text=True, timeout=timeout
        )
    except (OSError, subprocess.TimeoutExpired):
        return None


def _phone() -> str | None:
    result = _adb(["devices"])
    if result is None:
        return None
    devices = [
        line.split()[0]
        for line in result.stdout.splitlines()
        if len(line.split()) >= 2 and line.split()[1] == "device"
    ]
    if not devices:
        return None
    serial = devices[0]
    model = _adb(["-s", serial, "shell", "getprop", "ro.product.model"])
    name = (model.stdout if model else "").strip() or serial
    suffix = f", +{len(devices) - 1} more" if len(devices) > 1 else ""
    return f"{name} ({serial}{suffix})"


def _queue_count() -> str:
    result = _adb(["shell", f"ls -1 {SAY_DIR}/*.txt 2>/dev/null | wc -l"])
    if result is None:
        return "unknown"
    return result.stdout.strip() or "0"


def _speaker_helper() -> bool:
    """Is the Termux-side say.sh watcher alive?"""
    result = _adb(
        [
            "shell",
            "ps -A -o ARGS 2>/dev/null | grep '/sdcard/Download/say.sh' | grep -v grep",
        ]
    )
    return bool(result and result.stdout.strip())


def _clear_phone_queue() -> bool:
    result = _adb(["shell", f"rm -f {SAY_DIR}/*.txt"])
    return bool(result is not None and result.returncode == 0)


def _log_tail(lines: int = 8) -> list[str]:
    try:
        return LOG_FILE.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
    except OSError:
        return []


def _new_log_text(offset: int) -> str:
    try:
        with LOG_FILE.open("rb") as fh:
            fh.seek(offset)
            return fh.read().decode("utf-8", "replace")
    except OSError:
        return ""


def _print_tail() -> None:
    tail = _log_tail()
    if tail:
        print("Last log lines:")
        for line in tail:
            print(f"  {line}")


# --------------------------------------------------------------------------- stop / start


def _signal(pid: int, sig: int, *, group: bool) -> bool:
    try:
        if group:
            os.killpg(pid, sig)
        else:
            os.kill(pid, sig)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return False


def _wait_gone(pids: list[int], timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not any(_is_running(pid) for pid in pids):
            return True
        time.sleep(0.1)
    return not any(_is_running(pid) for pid in pids)


def stop(*, clear_queue: bool = True) -> int:
    managed = _read_pid()
    managed = managed if _is_managed(managed) else None
    ears = _find_ears()
    scrcpy = _find_scrcpy()
    if managed and managed not in ears:
        ears.append(managed)
        ears.sort()

    if not ears and not scrcpy and managed is None:
        PID_FILE.unlink(missing_ok=True)
        FIFO.unlink(missing_ok=True)
        if clear_queue and _clear_phone_queue():
            print("Phone speech queue cleared.")
        print("Banku is already off.")
        return 0

    print("Stopping Banku...")
    for pid in ears:
        # Only the process we started is a session leader. A manually started
        # ears process may share a process group with the user's shell, so send
        # that one a signal directly instead of taking the shell down with it.
        _signal(pid, signal.SIGTERM, group=(pid == managed))
    for pid in scrcpy:
        _signal(pid, signal.SIGTERM, group=False)

    remaining = ears + scrcpy
    if not _wait_gone(remaining, STOP_TIMEOUT_S):
        for pid in ears:
            _signal(pid, signal.SIGKILL, group=(pid == managed))
        for pid in scrcpy:
            _signal(pid, signal.SIGKILL, group=False)
        _wait_gone(remaining, 2.0)

    FIFO.unlink(missing_ok=True)
    PID_FILE.unlink(missing_ok=True)

    if clear_queue:
        if _clear_phone_queue():
            print("Phone speech queue cleared.")
        else:
            print("Could not clear the phone speech queue (phone disconnected?).")

    print("Banku is off.")
    return 0


def start() -> int:
    existing = _find_ears()
    if existing:
        print(f"Banku is already running (pid {existing[0]}). Use `banku stop` first.")
        return 0

    if not shutil.which("adb"):
        print("Cannot start: `adb` is not installed or not on PATH.")
        return 1
    if not shutil.which("scrcpy"):
        print("Cannot start: `scrcpy` is not installed or not on PATH.")
        return 1

    phone = _phone()
    if phone is None:
        print("Cannot start: no Android phone is connected over adb.")
        print("Plug in the phone, enable USB debugging, then run `adb devices`.")
        return 1

    stale = _find_scrcpy()
    if stale:
        print("Cleaning up a leftover scrcpy...")
        for pid in stale:
            _signal(pid, signal.SIGTERM, group=False)
        if not _wait_gone(stale, 2.0):
            for pid in stale:
                _signal(pid, signal.SIGKILL, group=False)
            _wait_gone(stale, 1.0)

    DATA.mkdir(parents=True, exist_ok=True)
    offset = LOG_FILE.stat().st_size if LOG_FILE.exists() else 0
    with LOG_FILE.open("ab", buffering=0) as log:
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        proc = subprocess.Popen(
            [sys.executable, "-m", "banku.ears"],
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # own session: survives the terminal, killable as a group
            env=env,
        )
    PID_FILE.write_text(f"{proc.pid}\n")

    print(f"Starting Banku (pid {proc.pid})...")
    deadline = time.monotonic() + START_TIMEOUT_S
    while time.monotonic() < deadline:
        if not _is_running(proc.pid):
            print("Banku exited while starting.")
            _print_tail()
            PID_FILE.unlink(missing_ok=True)
            return 1
        if READY_MARKER in _new_log_text(offset):
            print(f"Banku is ready. Phone: {phone}")
            if _speaker_helper():
                print("Phone speaker: running.")
            else:
                print("Warning: the phone speaker helper is NOT running.")
                print("In Termux, start it with: bash /sdcard/Download/say.sh &")
            print("Say: “Banku, where is the nail cutter?”")
            print("Turn it off with: banku stop")
            return 0
        time.sleep(0.25)

    print(f"Banku did not report ready within {START_TIMEOUT_S:.0f}s.")
    _print_tail()
    print("Stopping the half-started process.")
    stop(clear_queue=False)
    return 1


def restart() -> int:
    stop(clear_queue=True)
    return start()


def status() -> int:
    ears = _find_ears()
    if not ears:
        PID_FILE.unlink(missing_ok=True)
        print("Banku is off.")
        return 1

    for pid in ears:
        print(f"Banku is running (pid {pid}, up {_uptime(pid)}).")
    print(f"  phone: {_phone() or 'not connected'}")
    print(f"  phone speaker: {'running' if _speaker_helper() else 'NOT running'}")
    print(f"  queued speech: {_queue_count()}")
    print(f"  log: {LOG_FILE}")
    _print_tail()
    return 0
