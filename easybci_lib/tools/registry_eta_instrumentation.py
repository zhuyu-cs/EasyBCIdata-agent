"""Instrument every registered tool handler with the eta-record wrapper.

Idempotent (a wrapped handler carries the ``_eta_wrapped`` sentinel attribute so
re-application short-circuits). Call once after the model_tools import chain
has run all module-level ``registry.register()`` calls.
"""
from __future__ import annotations

import logging
from typing import Dict

logger = logging.getLogger(__name__)

# Map tool name → intent_kind. Mirrors services/gateway/turn_eta.py.
# Update both tables when adding a tool family that should appear in
# turn-scope ETA / get its elapsed samples written to history.
TOOL_NAME_TO_INTENT_KIND: Dict[str, str] = {
    # Web
    "web_search": "web_search",
    "web_extract": "web_search",
    "web_fetch": "web_search",
    # Shell
    "bash": "bash",
    "run_command": "bash",
    "shell": "bash",
    # File I/O
    "read_file": "read",
    "write_file": "write",
    # Stage-scope tools opt out — record_step_elapsed handles them via the
    # pipeline path (preprocess_neural/qc/etc emit StageProgress directly).
    "preprocess_neural": "unknown",
    "generate_code": "unknown",
    "plan_pipeline": "unknown",
    "quality_check": "unknown",
}


def instrument_registry_with_eta_recording(registry) -> int:
    """Walk the registry, wrap each known handler with wrap_handler_with_eta_record.

    Returns the number of handlers newly wrapped (idempotent — handlers carry
    a sentinel attribute so re-runs are no-ops).
    """
    from services.gateway.tool_eta_wrap import wrap_handler_with_eta_record  # noqa: PLC0415

    wrapped = 0
    # ToolRegistry exposes a private _tools dict + lock; both are stable
    # implementation contracts inside this repo.
    with registry._lock:  # noqa: SLF001
        for tool_name, entry in registry._tools.items():  # noqa: SLF001
            if getattr(entry.handler, "_eta_wrapped", False):
                continue
            kind = TOOL_NAME_TO_INTENT_KIND.get(tool_name)
            if kind is None:
                if tool_name.startswith("mcp__"):
                    kind = "mcp_tool"
                else:
                    continue
            if kind == "unknown":
                # Explicit opt-out — don't even touch the handler.
                continue
            new_handler = wrap_handler_with_eta_record(entry.handler, intent_kind=kind)
            entry.handler = new_handler
            wrapped += 1
    logger.debug("instrumented %d tool handlers with eta-record wrapper", wrapped)
    return wrapped
