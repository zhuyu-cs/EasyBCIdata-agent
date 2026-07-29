"""Programmatic safety net for the standardized preprocessing mini-repo.

Two responsibilities, both invoked AFTER an agent run completes (typically
from the Gateway/WebUI path, where the agent may have skipped the orchestrator
flow):

1. ``validate_mini_repo`` — does the produced work_dir conform to the contract
   (plan/, code/, preprocessed_output/, README.md)? Used to *flag* drift, not
   to rewrite the user's outputs (source-data immutability principle).
2. ``maybe_crystallize_proven`` — if a valid, QC-passing run was NOT
   crystallized into a proven-pipeline skill, do it programmatically by calling
   the same backend the ``skill_manage`` tool uses. This closes the
   proven-pipeline flywheel even when the LLM forgets / lacks the tool.

Everything is best-effort and never raises into the caller.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from easybci_lib.tools.neural_processing.preprocess.analysis_goals import (
        REGISTRY as _GOAL_REGISTRY,
    )
except Exception:  # noqa: BLE001
    _GOAL_REGISTRY = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

from easybci_lib.tools.neural_processing.export.layout_spec import CANONICAL as _LAYOUT

# Preserved public names for legacy readers; both consumers already grep
# by these identifiers. New code must import from layout_spec directly.
_REQUIRED_DIRS = _LAYOUT.required_dirs
_REQUIRED_FILES = _LAYOUT.required_files
# Mini-repo layout: at least one of these subtrees must contain a
# subject/session directory for preprocessed_output/ to be considered
# non-empty. Both halves are not required — segmenting may not have run.
_PREPROC_OUT_HALVES = _LAYOUT.preproc_halves


def _is_preprocessed_output_populated(root: Path) -> bool:
    """True if either ``preprocessed/sub-*`` or ``AI_ready/*`` has subject content.

    Tolerates the legacy flat layout (``preprocessed_output/{subject}/``) so
    pre-rename runs still validate.
    """
    for sub in _PREPROC_OUT_HALVES:
        half = root / sub
        if half.is_dir():
            for child in half.iterdir():
                if child.is_dir():
                    return True
    # Back-compat: legacy flat layout.
    preproc = root / "preprocessed_output"
    if preproc.is_dir():
        legacy_skip = {"figures", "QC_out", "preprocessed", "AI_ready"}
        for child in preproc.iterdir():
            if child.is_dir() and child.name not in legacy_skip:
                return True
    return False


def check_contract(work_dir, *, analysis_goal: str = "generic") -> List[Dict[str, Any]]:
    """Return a list of contract issues for the given mini-repo.

    Each issue is a dict with keys ``kind`` (machine-readable category),
    ``severity`` (``"error"``/``"warning"``), ``detail`` (human-readable
    message) and optionally ``path``. Empty list means the repo conforms.

    Mini-repo contract: when ``analysis_goal`` opts out of figures (e.g.
    ``online_inference``), the figures_missing check is skipped. The
    ``REGISTRY[goal].produces_figures`` flag is the single source of truth.
    """
    out_dir = Path(work_dir)
    issues: List[Dict[str, Any]] = []

    base = validate_mini_repo(str(out_dir))
    for missing in base.get("missing", []):
        issues.append({
            "kind": "missing_path",
            "severity": "error",
            "detail": f"Required path missing: {missing}",
            "path": missing,
        })

    # figures requirement is goal-conditional (REGISTRY[goal].produces_figures).
    figures_required = True
    if _GOAL_REGISTRY is not None:
        spec = _GOAL_REGISTRY.get(analysis_goal)
        if spec is not None:
            figures_required = spec.produces_figures

    if not figures_required:
        # Even when figures aren't required, still lint the pipeline.
        issues.extend(_check_pipeline_code_standard(out_dir))
        return issues

    figures_root = out_dir / "preprocessed_output" / "figures"
    figures_found = False
    if figures_root.is_dir():
        # qc.py writes to figures/sub-{id}/{ses}/*.png. The legacy layout
        # was figures/{ses}/*.png (one level shallower); rglob covers both.
        figures_found = any(figures_root.rglob("*.png"))
    if not figures_found:
        issues.append({
            "kind": "figures_missing",
            "severity": "error",
            "detail": (
                "preprocessed_output/figures/sub-<id>/<session>/ contains no PNG files. "
                "code/vis.py should produce at least one figure per session — "
                "Step 8 may have been skipped or failed."
            ),
            "path": str(figures_root),
        })

    # T7 Sub-phase P-D — code standard check on the emitted pipeline.py.
    issues.extend(_check_pipeline_code_standard(out_dir))

    return issues


def verify_layout_strict(
    work_dir,
    *,
    auto_repair: bool = False,
    allow_subprocess: bool = False,
    analysis_goal: Optional[str] = None,
) -> None:
    """Hard-constraint version of check_contract for gateway / CLI exit.

    When ``auto_repair=True``, first runs ``layout_repair.verify_and_repair``
    with the supplied ``analysis_goal`` and ``allow_subprocess`` flags. Only
    residual violations after the repair loop become ``LayoutContractError``.
    ``allow_subprocess=False`` is the safe default because most callers run
    on hot tool-return paths where a script re-run would starve the request.

    Raises:
        LayoutContractError: if any of the required top-level dirs is
            missing OR if plan/proposal.json contains the string
            ``"unknown"`` in modality / paradigm / analysis_goal OR
            if ``<work_dir>/code/middle_process/`` exists (forbidden — see
            plans/multi-session-routing/00-overview.md).

    Returns:
        None when layout is valid. The function does NOT swallow errors —
        callers MUST handle LayoutContractError explicitly (typically by
        emitting an SSE error event in the gateway path or printing a
        diagnostic and exiting non-zero in the CLI path).
    """
    import json as _json

    from easybci_lib.tools.neural_processing.export.errors import (
        LayoutContractError,
    )

    wd = Path(work_dir)

    if auto_repair:
        try:
            from easybci_lib.tools.neural_processing.export.layout_repair import (
                verify_and_repair,
            )
            verify_and_repair(
                wd, analysis_goal=analysis_goal,
                allow_subprocess=allow_subprocess,
                dry_run=False, write_report=True,
            )
        except Exception as exc:  # noqa: BLE001 - never let repair kill verify
            logger.warning("auto_repair on %s failed: %s", wd, exc)

    missing: List[str] = []
    for d in _REQUIRED_DIRS:
        if not (wd / d).is_dir():
            missing.append(d)

    husk_fields: List[str] = []
    proposal_path = wd / "plan" / "proposal.json"
    if proposal_path.is_file():
        try:
            data = _json.loads(proposal_path.read_text(encoding="utf-8"))
        except (OSError, _json.JSONDecodeError) as exc:
            raise LayoutContractError(
                f"plan/proposal.json unreadable: {exc!r}",
                missing=missing,
                husk_fields=["proposal_unreadable"],
            )
        for field in ("modality", "paradigm", "analysis_goal"):
            val = data.get(field) if isinstance(data, dict) else None
            if isinstance(val, str) and val.lower() == "unknown":
                husk_fields.append(field)

    # Layout invariant: middle_process/ lives at <wd>/middle_process/, NEVER
    # under <wd>/code/. The forbidden path here is a hard violation; surfacing
    # it via missing[] keeps the existing LayoutContractError shape.
    if (wd / "code" / "middle_process").exists():
        missing.append("FORBIDDEN: code/middle_process/ (move to middle_process/code/)")

    if not missing and not husk_fields:
        return

    msg_parts = []
    if missing:
        msg_parts.append(f"missing dirs: {missing}")
    if husk_fields:
        msg_parts.append(f"husk fields: {husk_fields}")
    raise LayoutContractError(
        "; ".join(msg_parts) or "layout invalid",
        missing=missing,
        husk_fields=husk_fields,
    )


def enumerate_pending(
    work_dir,
    routing_data=None,
    analysis_goal: str = "generic",
) -> dict:
    """Return a per-entry completeness snapshot for a multi-input work_dir.

    Reads ``middle_process/inputs_routing.json`` when ``routing_data`` is
    None. For each routing entry, checks whether the four artefact groups
    exist under ``work_dir``:

      - preprocessed:  preprocessed/sub-<sub>/ses-<ses>/<stem>_preprocessed.nwb
      - figures:       figures/sub-<sub>/ses-<ses>/<stem>_*.png (any one)
      - qc_report:     QC_out/sub-<sub>/ses-<ses>/qc_report.json
      - ai_ready:      AI_ready/<sub>/ses-<ses>/<stem>_epochs.pkl
                       (only when entry has ``events_path`` AND the goal
                        produces AI_ready per _GOAL_REGISTRY)

    Returns::

        {
          "work_dir": "...",
          "total": N,
          "done": D,
          "pending": N - D,
          "pending_file_ids": [...],
          "missing_by_entry": {
            "<file_id>": {
              "subject_id": ..., "session_id": ..., "stem_safe": ...,
              "missing": ["preprocessed", "figures", ...],  # empty when done
            },
            ...
          },
        }

    Does NOT raise on missing artefacts — that's the caller's decision.
    Raises FileNotFoundError only when ``routing_data`` is None and the
    routing table does not exist.
    """
    import json as _json

    wd = Path(work_dir)
    if routing_data is None:
        routing_path = wd / "middle_process" / "inputs_routing.json"
        if not routing_path.is_file():
            raise FileNotFoundError(
                f"routing table not found: {routing_path}"
            )
        try:
            routing_data = _json.loads(routing_path.read_text(encoding="utf-8"))
        except (OSError, _json.JSONDecodeError) as exc:
            raise ValueError(
                f"inputs_routing.json unreadable: {exc!r}"
            ) from exc

    inputs = (routing_data or {}).get("inputs") or []

    goal_produces_ai_ready = True
    if _GOAL_REGISTRY is not None:
        spec = _GOAL_REGISTRY.get(analysis_goal)
        if spec is not None:
            goal_produces_ai_ready = spec.produces_ai_ready

    missing_by_entry: dict = {}
    pending_ids: list = []
    for inp in inputs:
        sub = inp.get("subject_id")
        ses = inp.get("session_id")
        stem = inp.get("stem_safe")
        fid = inp.get("file_id") or "<no file_id>"

        missing: list = []
        if not (sub and ses and stem):
            missing.append("routing_entry_incomplete")
        else:
            nwb = (
                wd / "preprocessed_output" / "preprocessed"
                / f"sub-{sub}" / f"ses-{ses}" / f"{stem}_preprocessed.nwb"
            )
            if not nwb.is_file():
                missing.append("preprocessed")

            fig_dir = wd / "preprocessed_output" / "figures" / f"sub-{sub}" / f"ses-{ses}"
            if not fig_dir.is_dir() or not any(fig_dir.glob(f"{stem}_*.png")):
                missing.append("figures")

            qc = (
                wd / "preprocessed_output" / "QC_out"
                / f"sub-{sub}" / f"ses-{ses}" / "qc_report.json"
            )
            if not qc.is_file():
                missing.append("qc_report")

            if inp.get("events_path") and goal_produces_ai_ready:
                epochs = (
                    wd / "preprocessed_output" / "AI_ready"
                    / sub / f"ses-{ses}" / f"{stem}_epochs.pkl"
                )
                if not epochs.is_file():
                    missing.append("ai_ready")

        missing_by_entry[fid] = {
            "subject_id": sub, "session_id": ses, "stem_safe": stem,
            "missing": missing,
        }
        if missing:
            pending_ids.append(fid)

    total = len(inputs)
    pending = len(pending_ids)
    return {
        "work_dir": str(wd),
        "total": total,
        "done": total - pending,
        "pending": pending,
        "pending_file_ids": pending_ids,
        "missing_by_entry": missing_by_entry,
    }


def verify_layout_strict_multi(
    work_dir,
    *,
    auto_repair: bool = False,
    allow_subprocess: bool = False,
    analysis_goal: Optional[str] = None,
) -> None:
    """Multi-input contract check — uses inputs_routing.json as the truth.

    For each routing entry, verifies:
      - preprocessed/sub-<sub>/ses-<ses>/<stem_safe>_preprocessed.nwb exists
      - figures/sub-<sub>/ses-<ses>/ contains at least one PNG for <stem_safe>
      - QC_out/sub-<sub>/ses-<ses>/qc_report.json exists
      - AI_ready/<sub>/ses-<ses>/<stem_safe>_epochs.pkl exists IF the entry
        has an events_path OR the run's analysis_goal needs AI_ready

    Reverse scan: every preprocessed .nwb in the tree must correspond to a
    routing entry's (sub, ses, stem_safe) triple — orphaned files are
    reported. Stray .pkl in the preprocessed/ bucket is now a contract
    violation (the preprocessed layer is NWB-only since the format
    unification) and is reported as such.

    Also enforces:
      - No filename in any bucket contains a space (un-normalized stem leak).
      - <wd>/code/middle_process/ does not exist (location invariant).

    Falls through to ``verify_layout_strict`` when no routing table is present
    (single-file work_dirs). Raises ``LayoutContractError`` on any violation.

    When ``auto_repair=True``, runs ``layout_repair.verify_and_repair`` first;
    only residual violations surface.
    """
    import json as _json

    from easybci_lib.tools.neural_processing.export.errors import (
        LayoutContractError,
    )

    wd = Path(work_dir)

    if auto_repair:
        try:
            from easybci_lib.tools.neural_processing.export.layout_repair import (
                verify_and_repair,
            )
            verify_and_repair(
                wd, analysis_goal=analysis_goal,
                allow_subprocess=allow_subprocess,
                dry_run=False, write_report=True,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("auto_repair on %s failed: %s", wd, exc)

    routing_path = wd / "middle_process" / "inputs_routing.json"
    if not routing_path.is_file():
        return verify_layout_strict(wd)

    try:
        table = _json.loads(routing_path.read_text(encoding="utf-8"))
    except (OSError, _json.JSONDecodeError) as exc:
        raise LayoutContractError(
            f"inputs_routing.json unreadable: {exc!r}",
            missing=["inputs_routing.json unparseable"],
            husk_fields=[],
        )

    inputs = table.get("inputs") or []
    if not inputs:
        raise LayoutContractError(
            "inputs_routing.json has zero entries — multi-input contract violated",
            missing=["inputs_routing.json empty"],
            husk_fields=[],
        )

    errors: List[str] = []

    # Read analysis_goal from plan/proposal.json once — same source as
    # check_contract(). Used to gate the AI_ready expectation.
    goal = "generic"
    try:
        proposal_path = wd / "plan" / "proposal.json"
        if proposal_path.is_file():
            proposal = _json.loads(proposal_path.read_text(encoding="utf-8"))
            if isinstance(proposal, dict):
                goal = proposal.get("analysis_goal") or "generic"
    except Exception:  # noqa: BLE001
        pass

    # Forward pass: delegate to enumerate_pending. Convert per-entry
    # `missing` lists back into the flat error-string form this function
    # has always emitted so downstream error surfacing is unchanged.
    snapshot = enumerate_pending(wd, routing_data=table, analysis_goal=goal)
    for fid, entry in snapshot["missing_by_entry"].items():
        sub, ses, stem = entry["subject_id"], entry["session_id"], entry["stem_safe"]
        for kind in entry["missing"]:
            if kind == "routing_entry_incomplete":
                errors.append(
                    f"routing entry incomplete (file_id={fid}): missing sub/ses/stem"
                )
            elif kind == "preprocessed":
                expect_nwb = (
                    wd / "preprocessed_output" / "preprocessed"
                    / f"sub-{sub}" / f"ses-{ses}" / f"{stem}_preprocessed.nwb"
                )
                errors.append(
                    f"missing preprocessed for file_id={fid}: {expect_nwb}"
                )
            elif kind == "figures":
                fig_dir = wd / "preprocessed_output" / "figures" / f"sub-{sub}" / f"ses-{ses}"
                errors.append(
                    f"missing figures for file_id={fid} in {fig_dir} "
                    f"(expected {stem}_*.png)"
                )
            elif kind == "qc_report":
                qc_dir = wd / "preprocessed_output" / "QC_out" / f"sub-{sub}" / f"ses-{ses}"
                errors.append(
                    f"missing qc_report.json for file_id={fid}: {qc_dir / 'qc_report.json'}"
                )
            elif kind == "ai_ready":
                expect_epochs = (
                    wd / "preprocessed_output" / "AI_ready"
                    / sub / f"ses-{ses}" / f"{stem}_epochs.pkl"
                )
                errors.append(
                    f"missing AI_ready epochs for file_id={fid}: {expect_epochs}"
                )

    # Build the (sub, ses, stem) set for reverse-lookup.
    valid_triples = {
        (inp.get("subject_id"), inp.get("session_id"), inp.get("stem_safe"))
        for inp in inputs
    }

    # Reverse pass: every file in the four buckets must trace back to a triple.
    pre_base = wd / "preprocessed_output" / "preprocessed"
    if pre_base.is_dir():
        for sub_dir in pre_base.glob("sub-*"):
            sub_name = sub_dir.name.removeprefix("sub-")
            for sess_dir in sub_dir.iterdir():
                if not sess_dir.is_dir():
                    continue
                ses_name = sess_dir.name.removeprefix("ses-")
                for entry in sess_dir.iterdir():
                    if not entry.is_file():
                        continue
                    # NOTE: Extension allowlist (preprocessed=.nwb / AI_ready=.pkl)
                    # is enforced by _fix_disallowed_ext_in_{preprocessed,ai_ready}
                    # in verify_and_repair, which runs at every neural-tool
                    # boundary and at export_repo exit — disallowed-ext files are
                    # swept to middle_process/sweep_<ts>/ before we get here.
                    # contract_check no longer double-reports illegal extensions.
                    if entry.suffix != ".nwb":
                        continue
                    if " " in entry.name:
                        errors.append(
                            f"filename contains space (un-normalized stem): {entry}"
                        )
                    stem = entry.stem
                    if stem.endswith("_preprocessed"):
                        stem = stem[: -len("_preprocessed")]
                    if (sub_name, ses_name, stem) not in valid_triples:
                        errors.append(
                            f"orphaned preprocessed file: {entry} "
                            f"(no routing entry with sub={sub_name}, "
                            f"ses={ses_name}, stem={stem})"
                        )

    fig_base = wd / "preprocessed_output" / "figures"
    if fig_base.is_dir():
        for png in fig_base.rglob("*.png"):
            if " " in png.name:
                errors.append(f"figure filename contains space: {png}")
            # parts: .../figures/sub-X/ses-Y/<stem>_*.png
            parts = png.relative_to(fig_base).parts
            if len(parts) < 3:
                continue
            sub_name = parts[0].removeprefix("sub-")
            ses_name = parts[1].removeprefix("ses-")
            # Figure name is "<stem>_<rest>.png"; match prefix against any triple's stem.
            matched = any(
                s == sub_name and se == ses_name and png.name.startswith(stem + "_")
                for (s, se, stem) in valid_triples
            )
            if not matched:
                errors.append(
                    f"orphaned figure: {png} (no routing entry covers it)"
                )

    # Forbidden directory invariant.
    if (wd / "code" / "middle_process").exists():
        errors.append(
            "forbidden directory: <wd>/code/middle_process/ — middle_process/ "
            "MUST live at <wd>/middle_process/, never under code/"
        )

    if errors:
        raise LayoutContractError(
            "multi-session layout contract violated",
            missing=errors,
            husk_fields=[],
        )


def _check_pipeline_code_standard(out_dir: Path) -> List[Dict[str, Any]]:
    """T7 P-D — surface code-standard violations as contract issues so
    ``validate_mini_repo`` / ``easybci finalize`` flag pipelines that fail
    the lint.  Best-effort: if the checker import fails (unexpected) we
    log and return zero issues rather than blocking finalize."""
    pipeline_py = out_dir / "code" / "pipeline.py"
    if not pipeline_py.exists():
        return []
    try:
        from easybci_lib.tools.neural_processing.codegen.code_standard_check import (
            check_pipeline_code_standard,
        )
    except Exception:  # noqa: BLE001
        return []
    out: List[Dict[str, Any]] = []
    for v in check_pipeline_code_standard(pipeline_py):
        out.append({
            "kind": "code_standard_violation",
            "severity": "error" if v.get("blocking", True) else "warning",
            "detail": f"[{v['rule']}] {v['message']}",
            "path": str(pipeline_py.relative_to(out_dir)),
            "line": v.get("line", 0),
        })
    return out


def validate_mini_repo(work_dir: str) -> Dict[str, Any]:
    """Check whether *work_dir* conforms to the standard mini-repo contract.

    Returns ``{"ok": bool, "missing": [str, ...]}``. Never raises.
    """
    missing: List[str] = []
    try:
        root = Path(work_dir)
        if not root.is_dir():
            return {"ok": False, "missing": ["<work_dir does not exist>"]}
        for d in _REQUIRED_DIRS:
            if not (root / d).is_dir():
                missing.append(f"{d}/")
        for f in _REQUIRED_FILES:
            if not (root / f).is_file():
                missing.append(f)
        # A pipeline_record under plan/ is the explainability anchor.
        if (root / "plan").is_dir() and not (root / "plan" / "pipeline_record.json").is_file():
            missing.append("plan/pipeline_record.json")
        # preprocessed_output must contain at least one subject (continuous or
        # epoched). Empty preprocessed_output/ means the run dropped its
        # outputs into middle_process/ or never produced any.
        if (root / "preprocessed_output").is_dir() and not _is_preprocessed_output_populated(root):
            missing.append(
                "preprocessed_output is empty (need preprocessed/sub-* or AI_ready/*)"
            )
    except Exception as exc:
        logger.debug("validate_mini_repo failed for %s: %s", work_dir, exc)
        return {"ok": False, "missing": ["<validation error>"]}
    return {"ok": not missing, "missing": missing}


def _load_yaml(path: Path) -> Optional[dict]:
    try:
        import yaml  # PyYAML is a base dep
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return data if isinstance(data, dict) else None
    except Exception as exc:
        logger.debug("Could not read %s: %s", path, exc)
        return None


def _load_json(path: Path) -> Optional[dict]:
    try:
        import json
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else None
    except Exception as exc:
        logger.debug("Could not read %s: %s", path, exc)
        return None


def _sanitize(token: str) -> str:
    """Make a token safe for a skill name segment (lowercase a-z0-9._-)."""
    out = "".join(c if (c.isalnum() or c in "._-") else "-" for c in str(token).lower())
    out = out.strip("-._")
    return out or "x"


def _find_qc_grade(work_dir: Path) -> Optional[str]:
    """Best-effort scan of QC reports for a pass/fail-ish grade."""
    qc_root = work_dir / "preprocessed_output" / "QC_out"
    if not qc_root.is_dir():
        return None
    for jf in sorted(qc_root.rglob("qc_report_*.json")):
        data = _load_json(jf)
        if not data:
            continue
        fb = data.get("qc_feedback")
        if isinstance(fb, dict) and fb.get("grade"):
            return str(fb.get("grade"))
    return None


def _extract_step_rationale(reasoning_md: Path, step_names: list[str]) -> list[str]:
    """Parse plan/reasoning.md and pull rationale paragraph per step.

    Convention: each step appears as ``## Step N: <name>`` or ``### <name>``
    or ``## <name>``. Returns list aligned with step_names; missing step → "".
    """
    if not reasoning_md.exists():
        return ["" for _ in step_names]
    try:
        text = reasoning_md.read_text(encoding="utf-8")
    except Exception:
        return ["" for _ in step_names]
    out = []
    for name in step_names:
        pattern = re.compile(
            r"(?:^|\n)#{2,3}\s+(?:Step\s+\d+:\s+)?"
            + re.escape(str(name))
            + r"\b(.*?)(?=\n#{1,3}\s|\Z)",
            re.DOTALL | re.IGNORECASE,
        )
        m = pattern.search(text)
        out.append(m.group(1).strip() if m else "")
    return out


def _extract_step_params(proposal_json: Path, step_names: list[str]) -> list[dict]:
    """Pull (name, params) entries from plan/proposal.json, aligned with step_names."""
    if not proposal_json.exists():
        return [{"name": n, "params": {}} for n in step_names]
    try:
        data = json.loads(proposal_json.read_text(encoding="utf-8"))
    except Exception:
        return [{"name": n, "params": {}} for n in step_names]
    steps = data.get("steps") or []
    by_name: dict[str, dict] = {}
    for s in steps:
        if isinstance(s, dict):
            key = s.get("name") or s.get("op") or ""
            by_name[str(key)] = s
    return [
        {"name": n, "params": (by_name.get(str(n)) or {}).get("params") or {}}
        for n in step_names
    ]


def _gather_profile(work_dir: Path) -> Dict[str, Any]:
    """Pull modality/paradigm/steps/freq/channels from the contract files.

    config.yaml (plan/) is the reliable structured source; QC report fills in
    sampling rate and channel count.
    """
    profile: Dict[str, Any] = {
        "modality": None, "paradigm": None, "steps": [],
        "sampling_rate": None, "n_channels": None,
        "source_file": None, "source_format": None, "subject_id": None,
        "analysis_goal": None,
        "duration_s": None,
        "cohort_tag": None,
        "qc_metrics": {},
        "step_params": [],
        "step_rationale": [],
        "web_evidence_used": False,
        "web_evidence_refs": [],
        "source_run_path": str(work_dir),
    }
    cfg = _load_yaml(work_dir / "plan" / "config.yaml")
    if cfg:
        profile["modality"] = cfg.get("modality")
        profile["paradigm"] = cfg.get("paradigm")
        profile["subject_id"] = cfg.get("subject_id")
        steps = cfg.get("preprocessing")
        if isinstance(steps, list):
            profile["steps"] = [str(s) for s in steps]
        _goal_from_cfg = cfg.get("analysis_goal")
        if _goal_from_cfg:
            profile["analysis_goal"] = str(_goal_from_cfg).strip()

    # source_file / source_format come from the routing table (multi-input,
    # modern path) first, then pipeline_record.json input_path (legacy
    # single-input). config.yaml no longer carries input/output paths — see
    # generator.generate_config_yaml docstring.
    _src_path: Optional[str] = None
    rt_data = _load_json(work_dir / "middle_process" / "inputs_routing.json")
    if rt_data:
        _inputs = rt_data.get("inputs") or []
        if _inputs and isinstance(_inputs[0], dict):
            _src_path = (_inputs[0].get("data_path") or "").strip() or None

    # Priority: pipeline_record.analysis_goal > plan/goal.json > config.yaml
    pr = _load_json(work_dir / "plan" / "pipeline_record.json")
    if pr and pr.get("analysis_goal"):
        profile["analysis_goal"] = str(pr.get("analysis_goal")).strip()
    if pr:
        dp = pr.get("data_profile") or {}
        if isinstance(dp, dict):
            profile["cohort_tag"] = dp.get("cohort_tag")
        profile["web_evidence_used"] = bool(pr.get("web_evidence_used"))
        refs = pr.get("web_evidence_refs") or pr.get("web_evidence", {}).get("refs")
        if isinstance(refs, list):
            profile["web_evidence_refs"] = refs
        if not _src_path:
            _src_path = (
                (pr.get("input_path") or "").strip()
                or ((pr.get("data_info") or {}).get("file") or "").strip()
                or None
            )
    if _src_path:
        profile["source_file"] = Path(str(_src_path)).name
        profile["source_format"] = Path(str(_src_path)).suffix.lstrip(".") or None
    if not profile.get("analysis_goal"):
        gj = _load_json(work_dir / "plan" / "goal.json")
        if gj and gj.get("analysis_goal"):
            profile["analysis_goal"] = str(gj.get("analysis_goal")).strip()

    # Sampling rate + channel count + duration + qc metrics from any QC report.
    qc_root = work_dir / "preprocessed_output" / "QC_out"
    if qc_root.is_dir():
        for jf in sorted(qc_root.rglob("qc_report_*.json")):
            data = _load_json(jf)
            if not data:
                continue
            sr = data.get("sampling_rate")
            if isinstance(sr, dict):
                profile["sampling_rate"] = sr.get("after") or sr.get("before")
            elif sr:
                profile["sampling_rate"] = sr
            chans = data.get("channels")
            if isinstance(chans, list) and chans:
                profile["n_channels"] = len(chans)
            ds = data.get("data_shape")
            if isinstance(ds, dict):
                if not profile["n_channels"]:
                    profile["n_channels"] = ds.get("n_channels")
                if ds.get("duration_s") is not None:
                    profile["duration_s"] = ds.get("duration_s")
            bad = data.get("bad_channels")
            profile["qc_metrics"] = {
                "bad_channels_count": len(bad) if isinstance(bad, list) else 0,
                "qc_grade": data.get("qc_grade"),
                "artifact_rate": data.get("artifact_rate"),
                "ica_components_rejected": data.get("ica_components_rejected"),
            }
            break

    if profile["steps"]:
        profile["step_rationale"] = _extract_step_rationale(
            work_dir / "plan" / "reasoning.md", profile["steps"]
        )
        profile["step_params"] = _extract_step_params(
            work_dir / "plan" / "proposal.json", profile["steps"]
        )
    return profile


def _build_skill_name(profile: Dict[str, Any], date_str: str) -> Optional[str]:
    mod = _sanitize(profile.get("modality") or "")
    par = _sanitize(profile.get("paradigm") or "")
    # Without real modality+paradigm the name isn't meaningful enough to
    # crystallize. ``unknown`` / ``x`` / empty all signal that no real data
    # signature was captured — refusing the crystallization keeps the
    # proven-pipeline library clean (2026-06-17 bug #5: an auto-finalized
    # husk produced a polluting ``unknown-unknown-YYYYMMDD`` skill).
    _REJECT = {"", "x", "unknown"}
    if mod in _REJECT or par in _REJECT:
        return None
    parts = [mod, par]
    nch = profile.get("n_channels")
    if nch:
        parts.append(f"{int(nch)}ch")
    freq = profile.get("sampling_rate")
    if freq:
        try:
            parts.append(f"{int(round(float(freq)))}hz")
        except Exception:
            pass
    parts.append(_sanitize(date_str))
    name = "-".join(parts)
    return name[:64]


def _render_skill_md(name: str, profile: Dict[str, Any], grade: str, date_str: str) -> str:
    steps = profile.get("steps") or []
    chain = " → ".join(steps) if steps else "(steps unavailable)"
    mod = profile.get("modality") or "unknown"
    par = profile.get("paradigm") or "unknown"
    nch = profile.get("n_channels") or "?"
    freq = profile.get("sampling_rate") or "?"
    dur = profile.get("duration_s")
    dur_str = f"{dur:.1f}" if isinstance(dur, (int, float)) else "?"
    fmt = profile.get("source_format") or "unknown"
    src = profile.get("source_file") or "unknown"
    goal = profile.get("analysis_goal") or "generic"
    cohort = profile.get("cohort_tag") or "(none)"
    qcm = profile.get("qc_metrics") or {}
    bad_n = qcm.get("bad_channels_count", 0)
    art = qcm.get("artifact_rate")
    art_str = f"{art:.3f}" if isinstance(art, (int, float)) else "n/a"
    ica_n = qcm.get("ica_components_rejected")
    ica_str = str(ica_n) if ica_n is not None else "n/a"
    step_params = profile.get("step_params") or []
    step_rats = profile.get("step_rationale") or []
    web_used = bool(profile.get("web_evidence_used"))
    web_refs = profile.get("web_evidence_refs") or []
    src_run = profile.get("source_run_path") or "(unknown)"

    rat_lines = []
    for i, sname in enumerate(steps, 1):
        params = ({} if i - 1 >= len(step_params)
                  else (step_params[i - 1].get("params") or {}))
        params_str = (", ".join(f"{k}={v}" for k, v in params.items())
                      if params else "(no params)")
        rationale = (step_rats[i - 1]
                     if i - 1 < len(step_rats) and step_rats[i - 1] else "")
        rat_lines.append(
            f"{i}. **{sname}** — *params: {params_str}*\n\n   {rationale}"
        )
    rationale_block = "\n".join(rat_lines) if rat_lines else "(none)"

    param_rows = []
    for sp in step_params:
        sname = sp.get("name", "")
        params = sp.get("params") or {}
        if not params:
            param_rows.append(f"| {sname} | — | — | |")
            continue
        for k, v in params.items():
            param_rows.append(f"| {sname} | {k} | {v} | |")
    params_table = "\n".join(param_rows) if param_rows else "| — | — | — | — |"

    ref_lines = [f"- Source run: `{src_run}`",
                 f"- Reasoning detail: `{src_run}/plan/reasoning.md`"]
    if web_used and web_refs:
        ref_lines.append("- Web evidence:")
        for r in web_refs:
            if isinstance(r, dict):
                title = r.get("title") or r.get("url") or "(untitled)"
                url = r.get("url", "")
                ref_lines.append(f"  - {title} — <{url}>")
            else:
                ref_lines.append(f"  - {r}")
    elif web_used:
        ref_lines.append("- Web evidence: used (no structured refs captured)")
    refs_block = "\n".join(ref_lines)

    step_string = "→".join(steps) if steps else ""

    return f"""---
name: {name}
description: "Proven pipeline for {mod} {par}, {nch}ch @ {freq}Hz, QC grade {grade}, goal={goal}"
layer: L1
group: proven-pipelines
metadata:
  analysis_goal: {goal}
  analysis_goal_allowed: [{goal}]
  modalities: [{mod}]
  paradigm: {par}
  tags: [proven, auto-crystallized, {mod}, {par}, {goal}]
  step_string: "{step_string}"
  data_profile:
    channels: {nch}
    sfreq_hz: {freq}
    duration_s: {dur_str}
    cohort_tag: {cohort}
  qc_grade: {grade}
  qc_metrics:
    bad_channels_count: {bad_n}
    artifact_rate: {art_str}
    ica_components_rejected: {ica_str}
  source_run: "{src_run}"
  web_evidence_used: {str(web_used).lower()}
  version: 1
  auto_crystallized: true
  proven_date: "{date_str}"
---

# Proven Pipeline: {mod} {par} ({nch}ch @ {freq}Hz)

> Auto-crystallized from a successful run on {date_str}. Goal: {goal}.

## When to Reuse
- Modality: **{mod}** (hard match required)
- Paradigm: **{par}**
- Channels: recommend within ±25% of {nch}
- Sampling rate: recommend within ±20% of {freq} Hz
- Cohort: {cohort}
- QC grade observed: {grade}

## Data Profile
- Channels: **{nch}**
- Sampling rate: **{freq} Hz**
- Duration: {dur_str} s
- Source format: {fmt}
- Source file: {src}
- Cohort tag: {cohort}

## Pipeline Steps
```
{chain}
```

### Per-Step Rationale
{rationale_block}

## Parameters Used
| Step | Parameter | Value | Notes |
|------|-----------|-------|-------|
{params_table}

## QC Result
- **Grade**: {grade}
- **Bad channels**: {bad_n}
- **Artifact rate**: {art_str}
- **ICA components rejected**: {ica_str}
- See the original run: `{src_run}/preprocessed_output/QC_out/`

## When NOT to Reuse
- Modality / paradigm mismatch
- Channel count deviates from profile by > 30%
- Sampling rate deviates from profile by > 30%
- Different analysis_goal (this skill is only safe for `{goal}`)

## References
{refs_block}
"""


# Grades we treat as "passing enough" to crystallize. D/F are not stored.
_PASSING_GRADES = {"A", "B", "C"}


def maybe_crystallize_proven(work_dir: str, date_str: str) -> Dict[str, Any]:
    """If *work_dir* is a valid, QC-passing run not yet crystallized, save it
    as a proven-pipeline skill via the same backend ``skill_manage`` uses.

    *date_str* must be supplied by the caller (``YYYYMMDD``) — this module
    avoids wall-clock calls so it stays deterministic/testable.

    Returns a small status dict; never raises.
    """
    try:
        root = Path(work_dir)
        contract = validate_mini_repo(work_dir)
        if not contract["ok"]:
            return {"crystallized": False, "reason": "contract_incomplete",
                    "missing": contract["missing"]}

        grade = _find_qc_grade(root)
        if grade is not None and grade.upper() not in _PASSING_GRADES:
            return {"crystallized": False, "reason": f"qc_grade_{grade}"}

        profile = _gather_profile(root)

        # Gate 1: goal must be crystallize-eligible
        goal_name = (profile.get("analysis_goal") or "").strip()
        if _GOAL_REGISTRY is not None:
            goal_spec = _GOAL_REGISTRY.get(goal_name)
            if goal_spec is None or not getattr(
                goal_spec, "crystallize_eligible", True
            ):
                logger.info(
                    "Skip crystallize: analysis_goal=%r not crystallize-eligible",
                    goal_name,
                )
                return {"crystallized": False,
                        "reason": "goal_not_crystallize_eligible",
                        "goal": goal_name}

        # Gate 2: data profile must be complete (channels + sampling rate)
        if not profile.get("n_channels") or not profile.get("sampling_rate"):
            return {"crystallized": False,
                    "reason": "incomplete_data_profile",
                    "n_channels": profile.get("n_channels"),
                    "sampling_rate": profile.get("sampling_rate")}

        # Gate 3: at least one non-empty per-step rationale required
        rats = profile.get("step_rationale") or []
        if not rats or not any(r and r.strip() for r in rats):
            return {"crystallized": False, "reason": "missing_step_rationale"}

        name = _build_skill_name(profile, date_str)
        if not name:
            return {"crystallized": False, "reason": "insufficient_profile"}

        from easybci_lib.tools import skill_manager_tool as smt

        # Dedup: skip if a skill with this name already exists anywhere.
        try:
            if smt._find_skill(name):
                return {"crystallized": False, "reason": "already_exists", "name": name}
        except Exception:
            pass

        content = _render_skill_md(name, profile, grade or "C", date_str)
        result = smt._create_skill(name, content, category="proven-pipelines")
        if result.get("success"):
            logger.info("Crystallized proven pipeline skill: %s", name)
            return {"crystallized": True, "name": name, "path": result.get("path")}
        return {"crystallized": False, "reason": "create_failed",
                "error": result.get("error"), "name": name}
    except Exception as exc:
        logger.debug("maybe_crystallize_proven failed for %s: %s", work_dir, exc)
        return {"crystallized": False, "reason": "exception"}
