"""Single source of truth for the mini-repo layout contract.

Every checker, fixer, tool schema, and skill.md renderer that touches the
work_dir layout MUST read from ``CANONICAL`` (or ``resolve_for_goal(...)``
when it needs goal-conditional flags). Do not re-declare these constants
elsewhere — grep for ``layout_spec.CANONICAL`` when adding a new consumer.

See ``improved_docs/plans/strict-layout-enforcement/00-overview.md`` for
the design.

Consumers (grow as later phases land):

- ``contract_check.py``      Phase 0 (this)
- ``layout_repair.py``       Phase 1 — detect + fix
- ``neural_tools.py``        Phase 2 — verify_and_repair hook
- ``services/gateway/run.py``Phase 2 — verify_and_repair hook
- ``export/finalize.py``     Phase 2 — verify_and_repair on finalize
- ``repair_layout`` tool     Phase 3 — schema derives required paths
- ``pipeline`` SKILL.md      Phase 4 — Step 12 rewrite references CANONICAL

Any new consumer must be added here so we can grep the surface later.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

try:
    from easybci_lib.tools.neural_processing.preprocess.analysis_goals import (
        REGISTRY as _GOAL_REGISTRY,
    )
except Exception:  # noqa: BLE001 - registry import must never break layout_spec
    _GOAL_REGISTRY = None  # type: ignore[assignment]


@dataclass(frozen=True)
class LayoutSpec:
    """Immutable structural contract for a preprocessed mini-repo."""

    required_dirs: tuple[str, ...] = ("plan", "code", "preprocessed_output")
    required_files: tuple[str, ...] = ("README.md",)
    plan_files_required: tuple[str, ...] = (
        "proposal.json",
        "reasoning.md",
        "pipeline_record.json",
    )
    plan_files_optional: tuple[str, ...] = (
        "goal.json",
        "config.yaml",
        "web_evidence.json",
        "input_ref.json",
        "repair_report.json",
    )
    code_files_always: tuple[str, ...] = (
        "pipeline.py",
        "qc.py",
        "run.py",
        "requirements.txt",
    )
    code_files_conditional: Mapping[str, str] = field(default_factory=lambda: {
        "vis.py": "produces_figures",
        "build_ai_ready.py": "produces_ai_ready",
    })
    preproc_halves: tuple[str, ...] = (
        "preprocessed_output/preprocessed",
        "preprocessed_output/AI_ready",
    )
    figures_root: str = "preprocessed_output/figures"
    qc_root: str = "preprocessed_output/QC_out"
    forbidden_paths: tuple[str, ...] = (
        "code/middle_process",
    )
    filename_disallowed_chars: tuple[str, ...] = (" ",)
    preprocessed_ext_allowed: tuple[str, ...] = (".nwb",)
    ai_ready_ext_allowed: tuple[str, ...] = (".pkl",)
    figures_ext_allowed: tuple[str, ...] = (".png",)
    transient_top_level: tuple[str, ...] = ("middle_process",)


@dataclass(frozen=True)
class ResolvedLayoutSpec:
    """LayoutSpec with goal-conditional fields flattened to concrete tuples."""

    base: LayoutSpec
    analysis_goal: str
    code_files_required: tuple[str, ...]
    produces_figures: bool
    produces_ai_ready: bool


CANONICAL = LayoutSpec()


def resolve_for_goal(
    analysis_goal: str | None,
    deliverables: "list[str] | None" = None,
) -> ResolvedLayoutSpec:
    """Return a ResolvedLayoutSpec with conditional files/flags flattened.

    Unknown or missing goals fall back to the safe default (no figures /
    no AI_ready), matching ``generic`` semantics. This is the same fallback
    ``check_contract`` already uses.

    ``deliverables`` (when provided) is the SOURCE OF TRUTH for the AI-ready
    expectation — it OVERRIDES the goal's legacy ``produces_ai_ready`` hint.
    This is what decouples "does the method usually want epochs" (goal) from
    "did the user actually ask for AI-ready this run" (deliverables). When
    ``deliverables`` is None (legacy callers), we fall back to the goal hint so
    behaviour is unchanged.
    """
    goal = (analysis_goal or "").strip() or "generic"

    produces_figures = False
    produces_ai_ready = False
    if _GOAL_REGISTRY is not None:
        spec = _GOAL_REGISTRY.get(goal)
        if spec is not None:
            produces_figures = bool(getattr(spec, "produces_figures", False))
            produces_ai_ready = bool(getattr(spec, "produces_ai_ready", False))

    # deliverables overrides the goal's AI-ready hint (the whole point of the
    # goal × deliverables decoupling): NWB is always produced; ai_ready is only
    # expected when explicitly in the confirmed deliverables.
    if deliverables is not None:
        produces_ai_ready = "ai_ready" in deliverables

    code_required = list(CANONICAL.code_files_always)
    for name, flag_attr in CANONICAL.code_files_conditional.items():
        if flag_attr == "produces_figures" and produces_figures:
            code_required.append(name)
        elif flag_attr == "produces_ai_ready" and produces_ai_ready:
            code_required.append(name)

    return ResolvedLayoutSpec(
        base=CANONICAL,
        analysis_goal=goal,
        code_files_required=tuple(code_required),
        produces_figures=produces_figures,
        produces_ai_ready=produces_ai_ready,
    )
