"""ETA estimation from ProgressHistoryStore samples."""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Literal, Optional

from .history import ProgressHistoryStore

ConfidenceLevel = Literal["high", "medium", "low", "unknown"]


# ── Per-stage turn-ETA anchor floors ──────────────────────────────────────
#
# The user-visible turn-scope countdown (``est +Xm Ys`` in both CLI and WebUI)
# re-anchors when these stages start, so each "phase" shows a realistic ~10min
# expectation instead of the 60s cold-start fallback. This dict is the SINGLE
# source of truth for the floor — neither UI carries a hardcoded minimum (the
# WebUI's old ``Math.max(240, ...)`` was removed in favour of this).
#
# Phase 1 = "plan"  (inspect / research / propose).
# Phase 2 = the execution phase, anchored at "codegen" (the first execution
#           stage; preprocess + qc coast on this same anchor).
#
# ``max(floor, history)`` is applied at emit time, so accumulated history can
# only RAISE the estimate, never drop it below the floor.
STAGE_ETA_FLOOR_SECONDS: dict[str, int] = {
    "plan": 600,
    "codegen": 600,
}
ANCHOR_STAGES = frozenset(STAGE_ETA_FLOOR_SECONDS)


@dataclass(frozen=True)
class ElapsedStats:
    operator: str
    fingerprint_hash: str
    n_samples: int
    mean_s: Optional[float]
    stddev_s: Optional[float]
    confidence: ConfidenceLevel


def lookup_elapsed_stats(
    store: ProgressHistoryStore,
    *,
    operator: str,
    fingerprint_hash: str,
) -> ElapsedStats:
    """Return mean/stddev/confidence for one (operator, fingerprint) pair."""
    rows = list(store.query(operator=operator, fingerprint_hash=fingerprint_hash))
    n = len(rows)
    if n == 0:
        return ElapsedStats(
            operator=operator, fingerprint_hash=fingerprint_hash,
            n_samples=0, mean_s=None, stddev_s=None, confidence="unknown",
        )
    samples = [r.elapsed_s for r in rows]
    mean = statistics.fmean(samples)
    stddev = statistics.pstdev(samples) if n > 1 else 0.0
    confidence: ConfidenceLevel = "high" if n >= 5 else "medium"
    return ElapsedStats(
        operator=operator, fingerprint_hash=fingerprint_hash,
        n_samples=n, mean_s=mean, stddev_s=stddev, confidence=confidence,
    )


@dataclass(frozen=True)
class IntentStats:
    """Mean/stddev/confidence for one (intent_kind, optional fingerprint) tuple.

    Symmetric to ElapsedStats but keyed on intent_kind, used for turn-scope ETA.
    """
    intent_kind: str
    fingerprint_hash: Optional[str]
    n_samples: int
    mean_s: Optional[float]
    stddev_s: Optional[float]
    confidence: ConfidenceLevel


def lookup_intent_stats(
    store: ProgressHistoryStore,
    *,
    kind: str,
    fingerprint_hash: Optional[str] = None,
) -> IntentStats:
    """Return mean/stddev/confidence for elapsed samples of one intent_kind."""
    rows = list(store.query(intent_kind=kind, fingerprint_hash=fingerprint_hash))
    n = len(rows)
    if n == 0:
        return IntentStats(
            intent_kind=kind, fingerprint_hash=fingerprint_hash,
            n_samples=0, mean_s=None, stddev_s=None, confidence="unknown",
        )
    samples = [r.elapsed_s for r in rows]
    mean = statistics.fmean(samples)
    stddev = statistics.pstdev(samples) if n > 1 else 0.0
    confidence: ConfidenceLevel = "high" if n >= 5 else "medium"
    return IntentStats(
        intent_kind=kind, fingerprint_hash=fingerprint_hash,
        n_samples=n, mean_s=mean, stddev_s=stddev, confidence=confidence,
    )
