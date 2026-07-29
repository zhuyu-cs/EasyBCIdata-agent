"""Diversity penalty for proven-pipeline match results.

Same-origin pipelines past ``max_per_origin`` are score-penalized
(multiplied by 0.7), not removed. Tiny libraries (≤ a few labs) skip the
penalty entirely to avoid degenerate top-N.
"""
from __future__ import annotations

from typing import List, Optional

_DEFAULT_PENALTY_MULT = 0.7


def apply_diversity_penalty(
    matches: List,  # List[MatchResult]
    *,
    max_per_origin: int = 2,
    origin_attr: str = "lab_id",
    penalty_mult: float = _DEFAULT_PENALTY_MULT,
    library_lab_count: Optional[int] = None,
) -> List:
    """Soft-cap same-origin matches in top-N. Returns the same list with
    similarity adjusted and `match_reasons` annotated.

    library_lab_count: total distinct labs in the library. When provided
    and ≤ max_per_origin, NO penalty is applied (tiny library mode).
    """
    if library_lab_count is not None and library_lab_count <= max_per_origin:
        return matches

    count_by_lab: dict[str, int] = {}
    for m in matches:
        lab = getattr(m.entry, origin_attr, "") or ""
        if not lab:
            continue
        prior = count_by_lab.get(lab, 0)
        if prior >= max_per_origin:
            m.similarity *= penalty_mult
            m.match_reasons.append(
                f"diversity-penalized: lab={lab} already has {prior} above"
            )
        count_by_lab[lab] = prior + 1
    return matches
