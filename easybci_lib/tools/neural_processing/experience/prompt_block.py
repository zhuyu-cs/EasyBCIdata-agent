"""Render NegativeExample list into a system-prompt block.

Tone is intentionally **hint, not directive** to avoid LLM overgeneralisation.
The list is capped at 5 — too many makes the LLM treat them as rules.
"""
from __future__ import annotations

from typing import Iterable, List

from . import NegativeExample


_MAX_NEGATIVES_IN_PROMPT = 5


def build_negatives_prompt_block(negatives: Iterable[NegativeExample]) -> str:
    """Return a markdown block to prepend to the system prompt's 'Constraints'
    section. Empty string if no negatives apply."""
    negs = list(negatives)
    if not negs:
        return ""

    hard = sorted([n for n in negs if n.severity == "hard"], key=lambda n: n.recorded_at, reverse=True)
    soft = sorted([n for n in negs if n.severity == "soft"], key=lambda n: n.recorded_at, reverse=True)
    selected: List[NegativeExample] = (hard + soft)[:_MAX_NEGATIVES_IN_PROMPT]

    lines = [
        "### Known failure modes for similar data",
        "",
        "*These are observed past failures on similar recordings. Treat as hints,",
        "not hard rules — exceptional cases may still warrant the failed step.*",
        "",
    ]
    for n in selected:
        sev_tag = f"[{n.severity}]"
        bullet = (
            f"- {sev_tag} **{n.failed_step}** failed on "
            f"{n.modality}/{n.paradigm} ({n.failure_mode}): "
            f"{n.failure_evidence[:200]}"
        )
        lines.append(bullet)
    lines.append("")
    return "\n".join(lines)
