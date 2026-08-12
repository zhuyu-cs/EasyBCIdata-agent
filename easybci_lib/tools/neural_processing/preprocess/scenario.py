"""Single-source-of-truth registry for the `scenario` enum.

`scenario` is the delivery-context axis (who the output is for), ORTHOGONAL to
`analysis_goal` (the methodology axis, see analysis_goals.py). It does NOT
force any code-level pipeline branching — it only biases the parameters the
LLM recommends (via `param_bias_notes`) and the default deliverable set. Every
step still lands in proposal.json for the user to review and change.

Consumers:
- neural_tools.py — PLAN_PIPELINE_SCHEMA enum + propose handlers (writes scenario
  into proposal.json / goal.json / staged envelope)
- easybci_cli/web_server.py /api/schema/scenario-enum — dynamic enum response
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ScenarioSpec:
    name: str
    display_name: Dict[str, str]
    description: str
    default_deliverables: List[str] = field(default_factory=lambda: ["preprocessed"])
    param_bias_notes: str = ""


DEFAULT_SCENARIO = "research"

SCENARIO_REGISTRY: Dict[str, ScenarioSpec] = {
    "research": ScenarioSpec(
        name="research",
        display_name={"en": "Research", "zh": "科研"},
        description="Offline research analysis — reproducibility and full information retention prioritised.",
        default_deliverables=["preprocessed"],
        param_bias_notes=(
            "Favour reproducible, well-documented defaults; retain information "
            "(avoid over-aggressive cleaning); prefer standard published parameters."
        ),
    ),
    "clinical": ScenarioSpec(
        name="clinical",
        display_name={"en": "Clinical", "zh": "临床"},
        description="Clinical review/screening — conservative, interpretable, artefact-cautious processing.",
        default_deliverables=["preprocessed"],
        param_bias_notes=(
            "Favour conservative, interpretable steps; avoid aggressive filtering "
            "that could mask clinically-relevant morphology; prefer transparent, "
            "auditable parameters over maximal automation."
        ),
    ),
    "deployment": ScenarioSpec(
        name="deployment",
        display_name={"en": "Deployment", "zh": "部署"},
        description="Real-time / online deployment — low latency, no offline-only steps.",
        default_deliverables=["preprocessed"],
        param_bias_notes=(
            "Favour low-latency, causal steps; avoid offline-only operations "
            "(e.g. non-causal filtering, full-recording ICA); keep the pipeline "
            "streaming-compatible."
        ),
    ),
}


def is_valid_scenario(scenario: str) -> bool:
    return scenario in SCENARIO_REGISTRY


def get_scenario(scenario: str) -> ScenarioSpec:
    """Return spec or raise KeyError."""
    return SCENARIO_REGISTRY[scenario]
