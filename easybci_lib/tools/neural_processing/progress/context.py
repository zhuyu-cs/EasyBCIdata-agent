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
_STAGE_ANCHOR_CB = None


def set_active_tracker(tracker, daemon=None, stage_anchor=None) -> None:
    global _ACTIVE_TRACKER, _ACTIVE_DAEMON, _STAGE_ANCHOR_CB
    _ACTIVE_TRACKER = tracker
    _ACTIVE_DAEMON = daemon
    _STAGE_ANCHOR_CB = stage_anchor


def clear_active_tracker() -> None:
    global _ACTIVE_TRACKER, _ACTIVE_DAEMON, _STAGE_ANCHOR_CB
    _ACTIVE_TRACKER = None
    _ACTIVE_DAEMON = None
    _STAGE_ANCHOR_CB = None


def get_active_tracker():
    return _ACTIVE_TRACKER


def get_active_daemon():
    return _ACTIVE_DAEMON


def get_stage_anchor_callback():
    """Return the registered per-stage turn-ETA anchor callback, or None.

    Called from :func:`start_stage_if_active` for stages in ``ANCHOR_STAGES``
    to re-anchor the turn-scope countdown (see AIAgent._emit_stage_anchor_eta).
    """
    return _STAGE_ANCHOR_CB


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
        # Re-anchor the turn-scope ETA countdown for phase-boundary stages
        # (plan / codegen). Best-effort: a raising callback must never break
        # the stage lifecycle.
        from .eta import ANCHOR_STAGES  # noqa: PLC0415 — avoid import cycle at module load
        if stage in ANCHOR_STAGES:
            cb = get_stage_anchor_callback()
            if cb is not None:
                try:
                    cb(stage)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("stage anchor callback(%s) failed: %s", stage, exc)
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
