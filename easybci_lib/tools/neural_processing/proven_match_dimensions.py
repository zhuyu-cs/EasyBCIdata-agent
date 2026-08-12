"""Pluggable similarity dimensions.

Replaces the hard-coded 5-dim weighting in _compute_similarity. Modality stays
a hard filter; other dimensions contribute weighted partial scores.

Cohort is registered with weight 0 by default — enable via config:
  proven_pipelines:
    use_cohort: true
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from easybci_lib.tools.neural_processing.profile.cohort_taxonomy import cohort_similarity

logger = logging.getLogger(__name__)


@dataclass
class SimilarityDimension:
    name: str
    weight: float
    is_hard_filter: bool = False
    # compute_fn(entry, query_value) → score in [0, 1]
    compute_fn: Optional[Callable] = None


def weights_sum_to_one(dimensions: List[SimilarityDimension], *, tolerance: float = 1e-6) -> bool:
    return abs(sum(d.weight for d in dimensions) - 1.0) < tolerance


# ----------------------------------------------------------------- score helpers


def _score_modality(entry, query: str) -> float:
    if not entry.modality or not query:
        return 0.5  # neutral when info missing
    return 1.0 if entry.modality.lower() == query.lower() else 0.0


def _score_paradigm(entry, query: str) -> float:
    if not entry.paradigm or not query or query == "default":
        return 0.5
    a = entry.paradigm.lower().replace("-", "_").replace(" ", "_")
    b = query.lower().replace("-", "_").replace(" ", "_")
    if a == b:
        return 1.0
    if a in b or b in a:
        return 0.5
    return 0.0


def _score_n_channels(entry, query: int) -> float:
    if entry.n_channels <= 0 or query <= 0:
        return 0.5
    return min(entry.n_channels, query) / max(entry.n_channels, query)


def _score_frequency(entry, query: float) -> float:
    if entry.frequency_hz <= 0 or query <= 0:
        return 0.5
    return min(entry.frequency_hz, query) / max(entry.frequency_hz, query)


def _score_duration(entry, query: float) -> float:
    if entry.duration_s <= 0 or query <= 0:
        return 0.5
    log_ratio = abs(math.log2(entry.duration_s / query)) / 4.0
    return max(0.0, 1.0 - log_ratio)


def _score_cohort(entry, query: str) -> float:
    return cohort_similarity(entry.cohort_tag, query)


def _score_analysis_goal(entry, query: str) -> float:
    """analysis_goal hard-filter score.

    Mapping:
      * empty entry / empty query → 0.5 (neutral; legacy entries without the
        field stay visible in proven_matches, but the strict Reuse-gate check
        in neural_tools._handle_suggest_pipeline still requires exact equality
        before emitting a proven_recommendation).
      * query == "generic" → 0.5 (neutral; "generic" is the unspecified-goal
        sentinel the production caller substitutes when no goal is supplied —
        ``args.get("analysis_goal") or "generic"`` — so a generic *query* must
        not hard-filter out specific-goal entries like reference imports).
      * exact match → 1.0
      * mismatch    → 0.0 (hard filter short-circuits the entry).
    """
    a = (entry.analysis_goal or "").strip().lower()
    b = (query or "").strip().lower()
    if not a or not b or b == "generic":
        return 0.5
    return 1.0 if a == b else 0.0


DEFAULT_DIMENSIONS: List[SimilarityDimension] = [
    SimilarityDimension("modality",       0.30, is_hard_filter=True, compute_fn=_score_modality),
    SimilarityDimension("paradigm",       0.25, compute_fn=_score_paradigm),
    SimilarityDimension("n_channels",     0.15, compute_fn=_score_n_channels),
    SimilarityDimension("frequency_hz",   0.10, compute_fn=_score_frequency),
    SimilarityDimension("duration_s",     0.10, compute_fn=_score_duration),
    SimilarityDimension("cohort",         0.0,  compute_fn=_score_cohort),
    SimilarityDimension("analysis_goal",  0.10, is_hard_filter=True, compute_fn=_score_analysis_goal),
]


def _entry_value_for_log(entry, dim_name: str):
    if dim_name == "modality":
        return entry.modality
    if dim_name == "cohort":
        return entry.cohort_tag
    if dim_name == "analysis_goal":
        return entry.analysis_goal
    return getattr(entry, dim_name, None)


def compute_similarity_via_dimensions(
    entry,
    *,
    modality: str,
    paradigm: str,
    n_channels: int,
    frequency_hz: float,
    duration_s: float,
    cohort_tag: str = "",
    analysis_goal: str = "",
    dimensions: Optional[List[SimilarityDimension]] = None,
) -> Tuple[float, List[str]]:
    """Return (score, reasons). Modality mismatch short-circuits to (0.0, [])."""
    dims = dimensions if dimensions is not None else DEFAULT_DIMENSIONS

    query_values = {
        "modality": modality,
        "paradigm": paradigm,
        "n_channels": n_channels,
        "frequency_hz": frequency_hz,
        "duration_s": duration_s,
        "cohort": cohort_tag,
        "analysis_goal": analysis_goal,
    }

    total = 0.0
    reasons: List[str] = []
    for d in dims:
        if d.weight == 0.0 and not d.is_hard_filter:
            continue
        if d.compute_fn is None:
            continue
        q = query_values.get(d.name)
        if q is None and not d.is_hard_filter:
            continue
        partial = d.compute_fn(entry, q)
        if d.is_hard_filter and partial == 0.0:
            return 0.0, []
        total += partial * d.weight
        if partial > 0.7:
            reasons.append(f"{d.name} match ({_entry_value_for_log(entry, d.name)} vs {q})")
    return min(1.0, total), reasons
