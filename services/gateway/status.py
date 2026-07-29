"""Gateway runtime status — PID file, lock management, and takeover markers.

All state files live under ``get_easybci_home()/``:
- ``gateway.pid`` — single-line PID of the running gateway
- ``gateway.status.json`` — runtime status dict (state + platforms)
- ``gateway.locks/`` — directory of scoped lock files
- ``gateway.takeover`` — marker written by --replace before SIGTERM
- ``gateway.planned_stop`` — marker written by `easybci gateway stop`
"""

from __future__ import annotations

import json
import os
import signal
import time
from pathlib import Path
from typing import Optional

from easybci_lib.constants import get_easybci_home


def _status_dir() -> Path:
    return get_easybci_home()


def _pid_file() -> Path:
    return _status_dir() / "gateway.pid"


def _status_file() -> Path:
    return _status_dir() / "gateway.status.json"


def _locks_dir() -> Path:
    d = _status_dir() / "gateway.locks"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _takeover_marker() -> Path:
    return _status_dir() / "gateway.takeover"


def _planned_stop_marker() -> Path:
    return _status_dir() / "gateway.planned_stop"


# ---------------------------------------------------------------------------
# PID management
# ---------------------------------------------------------------------------

def write_pid_file(pid: Optional[int] = None) -> None:
    pid = pid or os.getpid()
    _pid_file().write_text(str(pid), encoding="utf-8")


def remove_pid_file() -> None:
    try:
        _pid_file().unlink(missing_ok=True)
    except OSError:
        pass


def get_running_pid() -> Optional[int]:
    try:
        text = _pid_file().read_text(encoding="utf-8").strip()
        pid = int(text)
        if _pid_exists(pid):
            return pid
        return None
    except (FileNotFoundError, ValueError, OSError):
        return None


def _pid_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


def terminate_pid(pid: int, force: bool = False) -> None:
    sig = signal.SIGKILL if force else signal.SIGTERM
    os.kill(pid, sig)


def get_process_start_time(pid: int) -> Optional[float]:
    try:
        stat_path = Path(f"/proc/{pid}/stat")
        if stat_path.exists():
            fields = stat_path.read_text(encoding="utf-8").split()
            if len(fields) > 21:
                return float(fields[21])
    except (OSError, ValueError, IndexError):
        pass
    return None


# ---------------------------------------------------------------------------
# Runtime status
# ---------------------------------------------------------------------------

def read_runtime_status() -> dict:
    try:
        text = _status_file().read_text(encoding="utf-8")
        return json.loads(text)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {"gateway_state": "unknown", "platforms": {}}


def write_runtime_status(**kwargs) -> None:
    status = read_runtime_status()
    status.update(kwargs)
    try:
        _status_file().write_text(json.dumps(status), encoding="utf-8")
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Scoped locks
# ---------------------------------------------------------------------------

def acquire_scoped_lock(name: str) -> bool:
    lock_file = _locks_dir() / f"{name}.lock"
    try:
        lock_file.write_text(
            json.dumps({"pid": os.getpid(), "time": time.time()}),
            encoding="utf-8",
        )
        return True
    except OSError:
        return False


def release_scoped_lock(name: str) -> None:
    lock_file = _locks_dir() / f"{name}.lock"
    try:
        lock_file.unlink(missing_ok=True)
    except OSError:
        pass


def release_all_scoped_locks(owner_pid: Optional[int] = None, owner_start_time=None) -> int:
    released = 0
    try:
        for lock_file in _locks_dir().glob("*.lock"):
            try:
                data = json.loads(lock_file.read_text(encoding="utf-8"))
                if owner_pid is not None and data.get("pid") != owner_pid:
                    continue
            except (json.JSONDecodeError, OSError):
                pass
            lock_file.unlink(missing_ok=True)
            released += 1
    except OSError:
        pass
    return released


# ---------------------------------------------------------------------------
# Gateway runtime lock (single-instance guard)
# ---------------------------------------------------------------------------

_runtime_lock_name = "__gateway_runtime__"


def acquire_gateway_runtime_lock() -> bool:
    return acquire_scoped_lock(_runtime_lock_name)


def release_gateway_runtime_lock() -> None:
    release_scoped_lock(_runtime_lock_name)


# ---------------------------------------------------------------------------
# Takeover / planned-stop markers
# ---------------------------------------------------------------------------

def write_takeover_marker(target_pid: Optional[int] = None) -> None:
    data = {"target_pid": target_pid, "replacer_pid": os.getpid(), "time": time.time()}
    try:
        _takeover_marker().write_text(json.dumps(data), encoding="utf-8")
    except OSError:
        pass


def clear_takeover_marker() -> None:
    try:
        _takeover_marker().unlink(missing_ok=True)
    except OSError:
        pass


def consume_takeover_marker_for_self() -> bool:
    try:
        text = _takeover_marker().read_text(encoding="utf-8")
        data = json.loads(text)
        if data.get("target_pid") == os.getpid():
            _takeover_marker().unlink(missing_ok=True)
            return True
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass
    return False


def write_planned_stop_marker(target_pid: Optional[int] = None) -> None:
    data = {"target_pid": target_pid or get_running_pid(), "time": time.time()}
    try:
        _planned_stop_marker().write_text(json.dumps(data), encoding="utf-8")
    except OSError:
        pass


def consume_planned_stop_marker_for_self() -> bool:
    try:
        text = _planned_stop_marker().read_text(encoding="utf-8")
        data = json.loads(text)
        if data.get("target_pid") == os.getpid():
            _planned_stop_marker().unlink(missing_ok=True)
            return True
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass
    return False
