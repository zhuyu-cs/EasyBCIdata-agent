"""Tool-name → EstimationIntent mapping + turn-scope progress payload builder.

The Gateway (or AIAgent) calls EtaEstimator.estimate(intent) before dispatching
each tool_use, then emits a `progress` SSE event with scope="turn".
"""
from __future__ import annotations

from typing import Any, Dict

from easybci_lib.tools.neural_processing.progress import (
    EstimationIntent,
    EtaEstimate,
)

# Static dispatch table. Keys are exact tool names registered with the tool
# registry; values are the IntentKind to bucket against.
# Update both this table and the HEURISTIC_TABLE in estimator.py if you add
# a new tool family that should appear in turn-scope ETA.
_TOOL_NAME_TO_INTENT_KIND: Dict[str, str] = {
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
    # Stage-scope tools: opt out of turn-scope ETA — the staged tracker
    # already drives `progress` events for these.
    "preprocess_neural": "unknown",
    "generate_code": "unknown",
    "plan_pipeline": "unknown",
    "quality_check": "unknown",
}


def classify_tool_use_to_intent(tool_name: str) -> EstimationIntent:
    """Return the EstimationIntent for one tool dispatch.

    Unknown tools fall back to kind="unknown" which makes EtaEstimator.estimate
    return None — caller should guard `if est is None: don't emit`.
    """
    kind = _TOOL_NAME_TO_INTENT_KIND.get(tool_name)
    if kind is not None:
        return EstimationIntent(kind=kind, tool_name=tool_name)  # type: ignore[arg-type]

    if tool_name.startswith("mcp__"):
        return EstimationIntent(kind="mcp_tool", tool_name=tool_name)

    return EstimationIntent(kind="unknown", tool_name=tool_name)


def build_turn_progress_payload(
    *,
    intent_kind: str,
    estimate: EtaEstimate,
    elapsed_seconds: int,
    heartbeat_ts: float,
) -> Dict[str, Any]:
    """Shape the SSE `progress` payload for a turn-scope ETA emission."""
    return {
        "scope": "turn",
        "next_intent_kind": intent_kind,
        "eta_seconds": estimate.seconds,
        "confidence": estimate.confidence,
        "source": estimate.source,
        "elapsed_seconds": elapsed_seconds,
        "heartbeat_ts": heartbeat_ts,
    }
