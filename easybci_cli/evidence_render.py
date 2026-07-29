"""Render ParameterEvidence dicts into ANSI-colorized terminal output."""

from __future__ import annotations

from typing import Dict

try:
    from easybci_cli.colors import Colors, color as _raw_color
except ImportError:
    class Colors:  # type: ignore[no-redef]
        RESET = BOLD = DIM = RED = GREEN = YELLOW = BLUE = MAGENTA = CYAN = ""

    def _raw_color(text: str, *_codes) -> str:  # type: ignore[no-redef]
        return text

from easybci_lib.tools.neural_processing.research.parameter_evidence import ParameterEvidence


_NAME_TO_CODE = {
    "green": Colors.GREEN,
    "yellow": Colors.YELLOW,
    "red": Colors.RED,
    "blue": Colors.BLUE,
    "dim": Colors.DIM,
}


def color(text: str, name: str = "") -> str:
    """Color helper that accepts a friendly name (green/yellow/red/blue/dim)."""
    code = _NAME_TO_CODE.get(name, "")
    if not code:
        return text
    return _raw_color(text, code)


def _label(ev: ParameterEvidence) -> str:
    if ev.source == "web":
        if ev.confidence >= 0.7:
            return color(f"[research • {ev.confidence:.2f}]", "green")
        if ev.confidence >= 0.4:
            return color(f"[research • {ev.confidence:.2f}]", "yellow")
        return color(f"[research • {ev.confidence:.2f}]", "red")
    if ev.source == "empirical_default":
        if ev.fallback_reason:
            return color("[default • !]", "yellow")
        return color("[default]", "dim")
    if ev.source == "registry_miss":
        return color("[registry_miss]", "red")
    if ev.source == "user_provided":
        return color("[user]", "blue")
    return f"[{ev.source}]"


def _origin(ev: ParameterEvidence) -> str:
    if ev.source == "web":
        if ev.citations:
            return ev.citations[0].title or ev.citations[0].url
        return ev.summary[:60]
    if ev.source == "empirical_default":
        suffix = f"; {ev.fallback_reason}" if ev.fallback_reason else ""
        return f"{ev.default_origin}{suffix}"
    if ev.source == "user_provided":
        prev = ev.previous_evidence
        if prev:
            return f"override (was {prev.value})"
        return "user override"
    return ""


def render_step_for_terminal(
    *,
    index: int,
    operator: str,
    evidence: Dict[str, ParameterEvidence],
) -> str:
    if not evidence:
        return f"Step {index}: {operator}  (no params)"
    lines = [f"Step {index}: {operator}"]
    items = list(evidence.items())
    for i, (pname, ev) in enumerate(items):
        connector = "└─" if i == len(items) - 1 else "├─"
        eq = f"{pname}={ev.value}"
        line = f"  {connector} {eq:<20} {_label(ev)}  {_origin(ev)}"
        lines.append(line.rstrip())
    return "\n".join(lines)


def render_proposal(steps: list, evidence_per_step: list) -> str:
    """Render a full proposal block (called by CLI confirm)."""
    out: list[str] = []
    for i, step in enumerate(steps):
        out.append(render_step_for_terminal(
            index=i + 1,
            operator=step.get("operator", ""),
            evidence=evidence_per_step[i] if i < len(evidence_per_step) else {},
        ))
    return "\n".join(out)


def apply_user_override(step: dict, param_name: str, new_value, reason: str = "") -> None:
    """Mutate a step in place: replace ``params[param_name]`` and rewrite
    ``param_evidence[param_name]`` as a ``user_provided`` ParameterEvidence
    with ``previous_evidence`` retained.

    Used by the CLI CONFIRM step's "edit l_freq=4.0" branch and by the
    WebUI's override POST handler so both surfaces share the same semantics.
    """
    old_block = step.setdefault("param_evidence", {})
    old_ev = old_block.get(param_name)
    previous = ParameterEvidence.from_dict(old_ev) if old_ev else None
    new_ev = ParameterEvidence(
        operator=step.get("operator", ""),
        parameter=param_name,
        value=new_value,
        source="user_provided",
        confidence=1.0,
        summary=reason or "user override during CONFIRM",
        previous_evidence=previous,
    )
    old_block[param_name] = new_ev.to_dict()
    step.setdefault("params", {})[param_name] = new_value
