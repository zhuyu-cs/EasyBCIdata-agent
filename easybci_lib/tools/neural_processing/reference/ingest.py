"""Reference ingest orchestrator: gold-standard project → enhanced proven skill.

Pipeline: parse_recipe → build_skeleton → build_qc_baselines → assemble an
enhanced-skill `profile` dict → render via _render_skill_md → write via
skill_manager_tool._create_skill(category="proven-pipelines").
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from easybci_lib.tools.neural_processing.reference.recipe_parser import parse_recipe
from easybci_lib.tools.neural_processing.reference.skeleton import build_skeleton
from easybci_lib.tools.neural_processing.reference.qc_baseline import build_qc_baselines

logger = logging.getLogger(__name__)

_DEFAULT_GOAL = "clinical_screening"


def _skill_name(rp) -> str:
    nch = rp.n_signal_channels or 0
    freq = int(rp.source_sfreq or 0)
    stem = re.sub(r"[^a-z0-9]+", "-", rp.stem.lower()).strip("-") or "case"
    return f"seeg-ref-{stem}-{nch}ch-{freq}hz"


def _step_params(step: str) -> dict:
    """Extract inline params from a 'op:arg,arg' step into a dict for the table."""
    if ":" not in step:
        return {}
    op, arg = step.split(":", 1)
    if op == "notch":
        return {"freq_hz": arg}
    if op == "bandpass":
        parts = arg.split(",")
        return {"low_hz": parts[0], "high_hz": parts[1] if len(parts) > 1 else ""}
    if op == "resample":
        return {"target_hz": arg}
    if op == "drop_bads":
        return {"mode": arg}
    return {"value": arg}


def ingest_reference(
    reference_dir: str,
    analysis_goal: str = _DEFAULT_GOAL,
    modality: str = "seeg",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Ingest a gold-standard project into an enhanced proven-pipeline skill.

    Returns {success, profile, unmapped, skill_name, skill_path?} or
    {success: False, error}.
    """
    try:
        rp = parse_recipe(reference_dir)
    except (FileNotFoundError, ValueError) as exc:
        return {"success": False, "error": str(exc)}

    skel = build_skeleton(rp)
    baselines = build_qc_baselines(rp)
    name = _skill_name(rp)

    profile: dict[str, Any] = {
        "steps": skel.steps,
        "step_string": skel.step_string,
        "modality": modality or rp.modality,
        "paradigm": "resting_state",
        "n_channels": rp.n_signal_channels,
        # YAML-safety: never emit a bare '?' — use int 0 when sfreq is missing.
        "sampling_rate": int(rp.source_sfreq) if rp.source_sfreq else 0,
        "duration_s": rp.duration_sec,
        "source_format": "nihon_kohden",
        "source_file": rp.stem,
        "analysis_goal": analysis_goal or _DEFAULT_GOAL,
        "cohort_tag": "(none)",
        "step_params": [
            {"name": s.split(":")[0], "params": _step_params(s)} for s in skel.steps
        ],
        "step_rationale": [f"From gold-standard recipe: {s}" for s in skel.steps],
        "qc_metrics": {"bad_channels_count": rp.n_bad_channels},
        "source_kind": "reference_import",
        "reference_origin": f"{Path(reference_dir).name}/{rp.stem}",
        # Persist gold reject keywords so reject_by_labels can excise labelled
        # windows per recording (component-3 adaptive reuse). Previously parsed
        # then dropped — the reason whole recordings were discarded instead.
        "reject_keywords": list(rp.reject_keywords or []),
        "adaptation_slots": skel.adaptation_slots,
        "qc_baselines": baselines,
    }
    if skel.unmapped:
        profile["unmapped_steps"] = skel.unmapped

    if dry_run:
        return {"success": True, "profile": profile, "unmapped": skel.unmapped,
                "skill_name": name}

    from easybci_lib.tools.neural_processing.export.contract_check import _render_skill_md
    from easybci_lib.tools import skill_manager_tool as smt
    from easybci_lib import time_utils

    date_str = time_utils.now().strftime("%Y-%m-%d")

    content = _render_skill_md(name, profile, "B", date_str)
    create_res = smt._create_skill(name, content, category="proven-pipelines")
    if not create_res.get("success"):
        return {
            "success": False,
            "error": create_res.get("error", "skill creation failed"),
            "profile": profile,
            "unmapped": skel.unmapped,
            "skill_name": name,
            "create_result": create_res,
        }
    # _create_skill returns absolute SKILL.md path under "skill_md";
    # "path" is only relative to SKILLS_DIR.
    skill_path = create_res.get("skill_md") or ""
    return {
        "success": True,
        "profile": profile,
        "unmapped": skel.unmapped,
        "skill_name": name,
        "skill_path": str(skill_path),
        "create_result": create_res,
    }
