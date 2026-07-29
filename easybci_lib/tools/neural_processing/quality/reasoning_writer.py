"""Reasoning markdown renderer.

A pure function from (step_index, operator, params, rationale, evidence_dict)
to a markdown section. Used by:
  - tools/neural_tools.py:_handle_propose_pipeline_evidence (renders the
    reasoning.md text into the staged envelope; the text materializes to
    plan/reasoning.md when mark_proposal_confirmed accepts the proposal)
  - easybci_cli/cli_output.py rendering of the CONFIRM display
  - mini-repo README assembly in tools/neural_processing/export/

Each step renders as:

    ## Step N — <operator>

    **Parameters**: k=v, ...

    **Rationale**: <synthesized from param_evidence rationales>

    **Effect**: <from operator_effects.OPERATOR_EFFECTS>

    ### Parameter evidence
    - **k = v** — *source*
      > <rationale snippet>

The "Rationale" line is synthesized by concatenating each parameter's
``rationale`` from ``param_evidence``. The "Effect" line comes from the
operator effects registry. Step-level rationale supplied by the agent
(when non-empty) takes precedence over the auto-synthesis.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable

from easybci_lib.tools.neural_processing.preprocess.operator_effects import (
    get_effect as _get_operator_effect,
)
from easybci_lib.tools.neural_processing.research.parameter_evidence import ParameterEvidence


def _synthesize_step_rationale(evidence: Dict[str, ParameterEvidence]) -> str:
    """Build a "Rationale" sentence from per-parameter rationales.

    Picks unique non-empty rationales in declaration order and joins them
    with whitespace. Empty result means no parameter carried a rationale —
    caller decides whether to omit the line entirely.
    """
    seen: set[str] = set()
    parts: list[str] = []
    for ev in evidence.values():
        text = (getattr(ev, "rationale", "") or "").strip()
        if not text:
            continue
        # Dedup repeated rationale strings (some operators share the same
        # justification across multiple params).
        if text in seen:
            continue
        seen.add(text)
        parts.append(text)
    return " ".join(parts)


def render_step(
    *,
    index: int,
    operator: str,
    params: Dict[str, Any],
    rationale: str,
    evidence: Dict[str, ParameterEvidence],
) -> str:
    lines: list[str] = []
    lines.append(f"## Step {index} — {operator}")
    lines.append("")
    if params:
        param_str = ", ".join(f"{k}={v}" for k, v in params.items())
        lines.append(f"**Parameters**: {param_str}")
    else:
        lines.append("**Parameters**: (none)")
    lines.append("")

    # Rationale — synthesize from evidence when the step-level rationale is empty.
    reason_text = (rationale or "").strip() or _synthesize_step_rationale(evidence)
    if reason_text:
        lines.append(f"**Rationale**: {reason_text}")
        lines.append("")

    # Effect — operator-level effect string from the registry.
    effect = _get_operator_effect(operator).strip()
    if effect:
        lines.append(f"**Effect**: {effect}")
        lines.append("")

    if evidence:
        lines.append("### Parameter evidence")
        for pname, ev in evidence.items():
            lines.append(_render_evidence(pname, ev))
        lines.append("")
    return "\n".join(lines)


def _render_evidence(pname: str, ev: ParameterEvidence) -> str:
    """Render one parameter-evidence bullet, surfacing the rationale snippet."""
    rationale_snippet = (getattr(ev, "rationale", "") or "").strip()
    if ev.source == "web":
        cites = ", ".join(
            f"[{c.title or c.url}]({c.url})" for c in ev.citations if c.url
        )
        out = f"- **{pname} = {ev.value}** — *web research, confidence {ev.confidence:.2f}*"
        if rationale_snippet:
            out += f"\n  > {rationale_snippet[:400]}"
        elif ev.summary:
            out += f"\n  > {ev.summary[:300]}"
        if cites:
            out += f"\n  Sources: {cites}"
        return out
    if ev.source == "empirical_default":
        suffix = ""
        if ev.fallback_reason:
            suffix = f" — fallback ({ev.fallback_reason})"
        head = (
            f"- **{pname} = {ev.value}** — *empirical default* "
            f"({ev.default_origin}){suffix}"
        )
        if rationale_snippet:
            head += f"\n  > {rationale_snippet[:400]}"
        return head
    if ev.source == "registry_miss":
        return f"- **{pname}** — *registry miss; needs user input*"
    if ev.source == "user_provided":
        prev = ev.previous_evidence
        prev_note = f" (previously: {prev.value})" if prev else ""
        head = f"- **{pname} = {ev.value}** — *user override*{prev_note}"
        if rationale_snippet:
            head += f"\n  > {rationale_snippet[:400]}"
        elif ev.summary:
            head += f"\n  > {ev.summary}"
        return head
    # Generic / inspection_report / domain_skill / ... all share the same
    # one-liner shape; surface the rationale snippet underneath when present.
    head = f"- **{pname} = {ev.value}** — *{ev.source}*"
    if rationale_snippet:
        head += f"\n  > {rationale_snippet[:400]}"
    return head


def render_full_reasoning(
    *,
    title: str,
    steps: list,
    rationales: list,
    evidence_per_step: list,
) -> str:
    """Compose plan/reasoning.md from a full pipeline.

    ``steps`` is a list of {operator, params}; ``rationales`` is the same length;
    ``evidence_per_step[i]`` is a dict[str, ParameterEvidence] for step i.

    Either the per-step rationale or the per-parameter evidence rationales
    must supply text for the "Rationale" line; the "Effect" line is always
    rendered from the operator_effects registry.
    """
    parts = [f"# {title}", ""]
    for i, step in enumerate(steps):
        parts.append(render_step(
            index=i + 1,
            operator=step.get("operator", ""),
            params=step.get("params", {}) or {},
            rationale=rationales[i] if i < len(rationales) else "",
            evidence=evidence_per_step[i] if i < len(evidence_per_step) else {},
        ))
    return "\n".join(parts).rstrip() + "\n"
