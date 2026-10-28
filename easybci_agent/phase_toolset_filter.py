"""Pluggable phase-gated toolset filter.

Personas (algo agent, data agent, orchestrator) restrict the tool surface as a
run progresses through phases. Before a proposal is confirmed, the agent
should not be able to spin up write_file / patch / execute_code — that surface
is the one weak models used to bypass the standard tool chain by scratching
scripts under /tmp. After confirmation, more surface is unlocked so bug-fixing
generated code can proceed.

Mechanism is a single pluggable hook: `set_phase_filter_hook(hook)` registers
a callable that takes the list of tool schemas plus the currently active
work_dirs and returns a filtered list. `filter_tools(...)` dispatches to it;
if unset, all tools pass through. Policy lives in sync-point modules per
persona — that keeps the upstream general and the fork-specific whitelist
scoped to whichever persona is active.

Pattern mirrors set_read_allowlist_hook / set_write_allowlist_hook in
file_safety.py — same lifecycle (thread-safe global, exceptions swallowed).
"""
from __future__ import annotations

import threading
from typing import Callable, List, Optional, Sequence


PhaseFilterHook = Callable[[List[dict], Sequence[str]], List[dict]]


_phase_filter_hook_lock = threading.Lock()
_phase_filter_hook: Optional[PhaseFilterHook] = None


def set_phase_filter_hook(hook: Optional[PhaseFilterHook]) -> None:
    """Register a callable that filters the tool list per active phase.

    The hook receives ``(tools, work_dirs)`` where ``tools`` is the raw tool
    schema list about to be sent to the model transport, and ``work_dirs`` is
    the list of paths recorded via ``export.finalize.set_current_work_dir`` (or
    equivalent) that the current run has touched. The hook returns the
    filtered tool list.

    Only one hook is active at a time. Pass ``None`` to clear.
    """
    global _phase_filter_hook
    with _phase_filter_hook_lock:
        _phase_filter_hook = hook


def filter_tools(tools: Optional[List[dict]], work_dirs: Sequence[str]) -> Optional[List[dict]]:
    """Dispatch to the currently-registered phase filter hook.

    Returns the tools list unchanged when no hook is registered. Never raises —
    any exception in the hook is swallowed and the unfiltered list returned
    (a broken hook must never break the transport).
    """
    if tools is None:
        return None
    with _phase_filter_hook_lock:
        hook = _phase_filter_hook
    if hook is None:
        return tools
    try:
        result = hook(list(tools), tuple(work_dirs or ()))
    except Exception:
        return tools
    if not isinstance(result, list):
        return tools
    return result


def phase_by_markers(work_dir: str, markers: Sequence[str]) -> str:
    """Return the highest phase whose marker exists under work_dir.

    ``markers`` is an ordered list from lowest-to-highest phase, where each
    entry is a path relative to work_dir. Returns ``"P0"`` when none exist,
    otherwise ``"Pk"`` where k is the index of the highest matching marker,
    starting at 1. Non-existent work_dir → ``"P0"``.
    """
    from pathlib import Path
    try:
        base = Path(work_dir)
    except Exception:
        return "P0"
    if not base.is_dir():
        return "P0"
    phase_idx = 0
    for i, rel in enumerate(markers, start=1):
        if (base / rel).is_file():
            phase_idx = i
    return f"P{phase_idx}"
