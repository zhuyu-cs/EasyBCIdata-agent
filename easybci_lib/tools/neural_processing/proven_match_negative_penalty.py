"""Apply negative-example penalty to MatchResult list.

A match is penalized if its pipeline (via .entry.steps) contains a step that
appears in ANY relevant NegativeExample.failed_step. The penalty is multiplied
ONCE — having more negatives for the same step doesn't compound.

Both sides are reduced to canonical operator names via ``canonical_step`` —
``entry.steps`` is stored as full ``operator[:params]`` strings (e.g.
``"ica:eog"``) while ``NegativeExample.failed_step`` is canonical-only
(``"ica"``). Without the normalisation step the set intersection was almost
always empty and the penalty never fired.
"""
from __future__ import annotations

from typing import List, Optional

from .experience import ExperienceStore
from .experience._step_key import canonical_op

try:
    from easybci_lib.constants import get_easybci_home as _get_easybci_home
except Exception:  # noqa: BLE001
    _get_easybci_home = None  # type: ignore[assignment]

_HARD_MULT = 0.3
_SOFT_MULT = 0.7


def apply_negative_penalty(
    matches: List,  # List[MatchResult]
    *,
    modality: str,
    paradigm: str,
    cohort_tag: str,
    analysis_goal: str,
    store: Optional[ExperienceStore] = None,
) -> List:
    """In-place penalty on ``matches``. ``store`` is injectable for tests; defaults
    to a fresh ``ExperienceStore`` rooted at EASYBCI_HOME (when available)."""
    if store is None:
        if _get_easybci_home is not None:
            try:
                store = ExperienceStore(store_dir=str(_get_easybci_home() / "experience"))
            except Exception:  # noqa: BLE001
                store = ExperienceStore()
        else:
            store = ExperienceStore()
    negatives = store.find_relevant_negatives(
        modality=modality, paradigm=paradigm,
        cohort_tag=cohort_tag, analysis_goal=analysis_goal,
    )
    if not negatives:
        return matches

    # Canonicalise both sides into operator-name sets. Empty operator strings
    # (legacy / unparseable failed_step entries) are dropped — they can't
    # match anything meaningful and shouldn't sweep in arbitrary matches.
    hard_steps = {n.failed_step for n in negatives if n.severity == "hard" and n.failed_step}
    soft_steps = {n.failed_step for n in negatives if n.severity == "soft" and n.failed_step}

    for m in matches:
        raw_steps = getattr(m.entry, "steps", []) or []
        ops = {op for op in (canonical_op(s) for s in raw_steps) if op}
        if ops & hard_steps:
            m.similarity *= _HARD_MULT
            hits = sorted(ops & hard_steps)[:3]
            m.match_reasons.append(
                f"penalized: known failure mode (hard) — {', '.join(hits)}"
            )
        elif ops & soft_steps:
            m.similarity *= _SOFT_MULT
            hits = sorted(ops & soft_steps)[:3]
            m.match_reasons.append(
                f"penalized: known failure mode (soft) — {', '.join(hits)}"
            )
    return matches
