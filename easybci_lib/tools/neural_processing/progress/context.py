"""Module-level active-tracker registry.

The Phase-1 plan originally assumed `_handle_*` were AIAgent methods, but
they're module-level functions in `easybci_lib/tools/neural_tools.py`. This
context module bridges the gap: AIAgent registers its tracker once on init
and tools read it from here. Per-process singleton — safe for the single-
agent-per-session model.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

_ACTIVE_TRACKER = None
_ACTIVE_DAEMON = None


def set_active_tracker(tracker, daemon=None) -> None:
    global _ACTIVE_TRACKER, _ACTIVE_DAEMON
    _ACTIVE_TRACKER = tracker
    _ACTIVE_DAEMON = daemon


def clear_active_tracker() -> None:
    global _ACTIVE_TRACKER, _ACTIVE_DAEMON
    _ACTIVE_TRACKER = None
    _ACTIVE_DAEMON = None


def get_active_tracker():
    return _ACTIVE_TRACKER


def get_active_daemon():
    return _ACTIVE_DAEMON


def start_stage_if_active(
    stage: str,
    *,
    sub_total: Optional[int] = None,
    with_daemon: bool = False,
    operator: Optional[str] = None,
    fingerprint_hash: Optional[str] = None,
) -> None:
    """Guarded `start_stage`: silent no-op when no active tracker."""
    tracker = get_active_tracker()
    if tracker is None:
        return
    try:
        tracker.start_stage(
            stage,
            sub_total=sub_total,
            operator=operator,
            fingerprint_hash=fingerprint_hash,
        )
        if with_daemon:
            daemon = get_active_daemon()
            if daemon is not None:
                daemon.start()
    except Exception as exc:
        logger.debug("start_stage_if_active(%s) failed: %s", stage, exc)


def end_stage_if_active(*, with_daemon: bool = False) -> None:
    tracker = get_active_tracker()
    if tracker is None:
        return
    try:
        if with_daemon:
            daemon = get_active_daemon()
            if daemon is not None:
                daemon.stop()
        tracker.end_stage()
    except Exception as exc:
        logger.debug("end_stage_if_active failed: %s", exc)
