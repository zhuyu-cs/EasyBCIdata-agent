"""EtaEstimator — one front door for all ETA queries.

Three layers:
    1. history_store + fingerprint (preprocess_step) → mean + 2·stddev, conf=high if n>=5 else medium
    2. heuristic table (web_search / bash / llm_generation / mcp_tool / …) → static value, conf=low
    3. unknown / <5s → None (UI hides ETA entirely)

The estimator is stateless: it takes a ProgressHistoryStore by reference and
reads on every call. Wired into StagedProgressTracker (stage-scope ETA) and
into the Gateway turn-scope dispatch path (per-tool ETA).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

from .eta import ConfidenceLevel, lookup_elapsed_stats, lookup_intent_stats
from .history import ProgressHistoryStore

logger = logging.getLogger(__name__)

IntentKind = Literal[
    "preprocess_step",
    "web_search",
    "bash",
    "read",
    "write",
    "llm_generation",
    "mcp_tool",
    "unknown",
]

Source = Literal["history", "heuristic", "fallback"]


@dataclass(frozen=True)
class EstimationIntent:
    """Inputs to `EtaEstimator.estimate`. Frozen so it's hashable for caches."""
    kind: IntentKind
    operator: Optional[str] = None
    fingerprint: Optional[str] = None
    token_count: Optional[int] = None
    tool_name: Optional[str] = None


@dataclass(frozen=True)
class EtaEstimate:
    seconds: int
    confidence: ConfidenceLevel
    source: Source


HEURISTIC_TABLE: Dict[str, Dict[str, Any]] = {
    "web_search": {"base_s": 23, "stddev_s": 5},
    "bash": {"base_s": 9, "stddev_s": 3},
    "read": {"base_s": 2, "stddev_s": 0},
    "write": {"base_s": 2, "stddev_s": 0},
    "mcp_tool": {"base_s": 12, "stddev_s": 4},
    "preprocess_step": {"base_s": 45, "stddev_s": 15},
}

DISPLAY_THRESHOLD_S = 5


class EtaEstimator:
    """Stateless front door for all ETA queries."""

    def __init__(
        self,
        *,
        history_store: ProgressHistoryStore,
        table: Optional[Dict[str, Dict[str, Any]]] = None,
        display_threshold_s: int = DISPLAY_THRESHOLD_S,
    ) -> None:
        self._store = history_store
        self._table = table if table is not None else HEURISTIC_TABLE
        self._threshold = display_threshold_s

    def estimate(self, intent: EstimationIntent) -> Optional[EtaEstimate]:
        if intent.kind == "unknown":
            return None

        if intent.kind == "preprocess_step" and intent.operator and intent.fingerprint:
            stats = lookup_elapsed_stats(
                self._store,
                operator=intent.operator,
                fingerprint_hash=intent.fingerprint,
            )
            if stats.n_samples > 0 and stats.mean_s is not None:
                if stats.confidence == "high":
                    seconds = int(round(stats.mean_s + 2.0 * (stats.stddev_s or 0.0)))
                else:
                    seconds = int(round(stats.mean_s))
                if seconds < self._threshold:
                    return None
                return EtaEstimate(seconds=seconds, confidence=stats.confidence, source="history")

        if intent.kind in ("web_search", "bash", "mcp_tool"):
            intent_stats = lookup_intent_stats(self._store, kind=intent.kind)
            if intent_stats.n_samples > 0 and intent_stats.mean_s is not None:
                if intent_stats.confidence == "high":
                    seconds = int(round(intent_stats.mean_s + 2.0 * (intent_stats.stddev_s or 0.0)))
                else:
                    seconds = int(round(intent_stats.mean_s))
                if seconds < self._threshold:
                    return None
                return EtaEstimate(
                    seconds=seconds, confidence=intent_stats.confidence, source="history",
                )

        if intent.kind == "llm_generation":
            tokens = intent.token_count or 0
            seconds = max(5, int(tokens * 1.5 // 25))
            if seconds < self._threshold:
                return None
            return EtaEstimate(seconds=seconds, confidence="low", source="heuristic")

        entry = self._table.get(intent.kind)
        if entry is None:
            return None
        seconds = int(entry["base_s"])
        if seconds < self._threshold:
            return None
        return EtaEstimate(seconds=seconds, confidence="low", source="heuristic")
