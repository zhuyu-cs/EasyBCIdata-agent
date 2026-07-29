"""Staged progress tracking — four-stage ETA + heartbeat.

Public exports:
- StageProgress: serialisable per-tick progress snapshot
- StagedProgressTracker: orchestrates the four stages of a session
- EtaEstimator / EstimationIntent / EtaEstimate: ETA query front door
- IntentStats / lookup_intent_stats: turn-scope history aggregation
"""
from .estimator import (
    EstimationIntent,
    EtaEstimate,
    EtaEstimator,
    HEURISTIC_TABLE,
    IntentKind,
)
from .eta import ElapsedStats, IntentStats, lookup_elapsed_stats, lookup_intent_stats
from .history import record_intent_elapsed
from .tracker import StageName, StageProgress, StagedProgressTracker

__all__ = [
    "ElapsedStats",
    "EstimationIntent",
    "EtaEstimate",
    "EtaEstimator",
    "HEURISTIC_TABLE",
    "IntentKind",
    "IntentStats",
    "StageName",
    "StageProgress",
    "StagedProgressTracker",
    "lookup_elapsed_stats",
    "lookup_intent_stats",
    "record_intent_elapsed",
]
