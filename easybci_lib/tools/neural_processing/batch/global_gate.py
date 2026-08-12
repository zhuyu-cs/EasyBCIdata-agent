"""Cross-process / cross-instance memory admission gate.

Root cause this closes: last night TWO independent easybci instances (separate
tmux sessions) each spawned a ~30 GB ``pipeline.py`` subprocess. Neither knew the
other existed — no global memory admission — so they summed to ~58 GB and the
kernel OOM-killer fired. The batch orchestrator only sees its own shell; it
cannot see a second instance in another terminal. So the gate has to live where
the memory is actually spent: in ``pipeline.py``, before it loads each file.

Design
------
A single machine-wide ledger ``<EASYBCI_HOME>/batch/mem_ledger.json`` records
every in-flight reservation ``{token: {pid, reserved_mb, ts, file_id}}``. A file
lock ``mem_ledger.json.lock`` (fcntl.flock, mirroring memory_tool.py:148)
serializes read-modify-write.

- ``acquire(peak_mb, file_id, timeout)`` blocks until ``sum(reserved) + peak_mb``
  fits the machine budget, then records a reservation and returns its token.
- ``release(token)`` drops the reservation.

Three hazards, three guards:

1. **Deadlock** — the flock is held ONLY for the millisecond-scale
   read-modify-write. The *wait* for capacity is lock-free polling with a
   timeout, so a blocked acquirer never pins the lock.
2. **Stale reservations** — every acquire first sweeps entries whose pid is dead
   (``os.kill(pid, 0)`` → ``ProcessLookupError``; catches an OOM-killed sibling)
   or older than ``EASYBCI_GATE_STALE_S``.
3. **Starvation** — a file whose own peak exceeds the whole budget (already
   vetted by Layer A) is admitted immediately when the ledger is otherwise
   empty, rather than blocking forever.

This module is used by the dispatch/tooling side. The generated ``pipeline.py``
carries a stdlib-only inline copy of the same logic (Rule 15: generated code is
self-contained) — keep the two in sync when editing.
"""
from __future__ import annotations

import fcntl
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from easybci_lib.constants import get_easybci_home

_MEMORY_BUDGET_RATIO = 0.7
_DEFAULT_STALE_S = 3600.0  # a reservation older than this is presumed dead
_POLL_INTERVAL_S = 0.5


def _ledger_dir() -> Path:
    d = get_easybci_home() / "batch"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _ledger_path() -> Path:
    return _ledger_dir() / "mem_ledger.json"


def _budget_mb() -> float:
    """Machine memory budget (MB). Honors EASYBCI_MEMORY_BUDGET_MB for tests."""
    env = os.environ.get("EASYBCI_MEMORY_BUDGET_MB")
    if env:
        try:
            v = float(env)
            if v > 0:
                return v * _MEMORY_BUDGET_RATIO
        except ValueError:
            pass
    total = None
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    total = int(line.split()[1]) / 1024  # kB -> MB
                    break
    except (OSError, ValueError, IndexError):
        pass
    if total is None:
        total = 8000.0
    return total * _MEMORY_BUDGET_RATIO


def _stale_s() -> float:
    env = os.environ.get("EASYBCI_GATE_STALE_S")
    if env:
        try:
            v = float(env)
            if v > 0:
                return v
        except ValueError:
            pass
    return _DEFAULT_STALE_S


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists but not ours
    except OSError:
        return False
    return True


@contextmanager
def _locked():
    """Hold the ledger lock for a read-modify-write ONLY (never while waiting)."""
    lock_path = _ledger_path().with_suffix(".json.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = open(lock_path, "a+", encoding="utf-8")
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        fd.close()


def _load() -> dict:
    p = _ledger_path()
    if not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _save(ledger: dict) -> None:
    p = _ledger_path()
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    tmp.replace(p)


def _sweep(ledger: dict, now: float) -> dict:
    """Drop reservations whose pid is dead or whose ts is older than stale_s."""
    stale = _stale_s()
    kept = {}
    for token, r in ledger.items():
        if not isinstance(r, dict):
            continue
        pid = int(r.get("pid", 0) or 0)
        ts = float(r.get("ts", 0.0) or 0.0)
        if pid and not _pid_alive(pid):
            continue
        if now - ts > stale:
            continue
        kept[token] = r
    return kept


def _reserved_total(ledger: dict) -> float:
    return sum(float(r.get("reserved_mb", 0.0) or 0.0)
               for r in ledger.values() if isinstance(r, dict))


def _try_reserve(peak_mb: float, file_id: str, token: str,
                 now: float) -> bool:
    """One atomic sweep+admit attempt. Returns True if the reservation landed."""
    with _locked():
        ledger = _sweep(_load(), now)
        budget = _budget_mb()
        reserved = _reserved_total(ledger)
        # Admit if it fits, OR if nothing else is reserved (anti-starvation: a
        # file bigger than the whole budget was already vetted by Layer A and
        # must not block forever on an empty machine).
        if reserved + peak_mb <= budget or not ledger:
            ledger[token] = {
                "pid": os.getpid(), "reserved_mb": round(float(peak_mb), 1),
                "ts": now, "file_id": file_id,
            }
            _save(ledger)
            return True
        _save(ledger)  # persist the sweep even when we can't admit yet
        return False


def acquire(peak_mb: float, *, file_id: str = "",
            timeout: Optional[float] = None,
            _now=time.time) -> str:
    """Block until ``peak_mb`` fits the machine budget, then reserve it.

    Returns an opaque token to pass to :func:`release`. Raises TimeoutError if
    capacity does not free up within ``timeout`` seconds (None = wait forever).
    The lock is held only during each brief sweep+admit attempt; the wait
    between attempts is lock-free polling.
    """
    token = f"{os.getpid()}-{int(_now() * 1000)}-{file_id}"
    start = _now()
    while True:
        if _try_reserve(peak_mb, file_id, token, _now()):
            return token
        if timeout is not None and (_now() - start) >= timeout:
            raise TimeoutError(
                f"memory gate: waited {timeout:.0f}s for ~{peak_mb:.0f} MB "
                f"(file_id={file_id!r}); other instances still hold reservations")
        time.sleep(_POLL_INTERVAL_S)


def release(token: str) -> None:
    """Drop the reservation identified by ``token``. Idempotent; never raises."""
    if not token:
        return
    try:
        with _locked():
            ledger = _load()
            if token in ledger:
                del ledger[token]
                _save(ledger)
    except OSError:
        pass


def peak_mb_for(*, n_channels: int, frequency: float, duration_s: float,
                has_ica: bool, target_hz: Optional[float] = None) -> float:
    """Convenience re-export of the authoritative estimator."""
    from easybci_lib.tools.neural_processing.preprocess.memory_strategy import (
        estimate_peak_mb,
    )
    return estimate_peak_mb(n_channels=n_channels, frequency=frequency,
                            duration_s=duration_s, has_ica=has_ica,
                            target_hz=target_hz)
