"""Mini-repo builder — assembles a self-contained, reproducible preprocessing directory.

Reuses generators from codegen/ for pipeline.py, run.py, config.yaml, requirements.txt.
Adds: reasoning.md, input reference, QC figures, pipeline logs.

Output structure (after finalization):
    {work_dir}/
    ├── README.md                    # Comprehensive explainability document
    ├── preprocessed_output/         # Run artefacts (split by readiness)
    │   ├── preprocessed/            # Continuous post-pipeline signal (BIDS-friendly, NWB)
    │   │   └── sub-{subject_id}/{session_id}/
    │   │       └── {stem}_preprocessed.nwb
    │   ├── AI_ready/                # Epoched / segmented data — feed straight to model
    │   │   └── {subject_id}/{session_id}/
    │   │       └── *_epochs.pkl
    │   ├── figures/
    │   │   └── sub-{subject_id}/{session_id}/
    │   │       ├── {stem}_timeseries_before_after.png
    │   │       ├── {stem}_psd.png
    │   │       ├── {stem}_variance.png
    │   │       ├── {stem}_amplitude.png
    │   │       └── {stem}_timeseries.png
    │   └── QC_out/
    │       └── sub-{subject_id}/{session_id}/
    │           ├── qc_report_{session_id}.md
    │           └── qc_report_{session_id}.json
    ├── code/                        # Reproducible pipeline scripts
    │   ├── pipeline.py
    │   ├── run.py
    │   ├── visualize.py
    │   ├── qc.py
    │   └── requirements.txt
    └── plan/                        # Configuration + explainability
        ├── config.yaml
        ├── reasoning.md
        ├── pipeline_record.json
        └── input_ref.json

Note: middle_process/ is used during build for intermediate artifacts but is
removed by finalize_mini_repo() once the pipeline is confirmed stable.
"""

import hashlib
import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from easybci_lib.tools.neural_processing._seed import EASYBCI_SEED
from easybci_lib.tools.neural_processing.export._lint import ruff_fix
from easybci_lib.tools.neural_processing.export._size_utils import (
    dir_size_bytes, format_size, paths_size_bytes,
)

logger = logging.getLogger(__name__)

# Directories that are part of the canonical output structure
_CANONICAL_DIRS = frozenset({"preprocessed_output", "code", "plan", "middle_process"})


def _compute_storage_footprint(out: Path, input_path: str) -> Dict[str, Any]:
    """Raw-input vs preprocessed-output byte totals for the run summary.

    Raw size priority: routing table (batch, sum every input's ``data_path``) >
    ``plan/input_ref.json['size_bytes']`` (single-file) > ``input_path`` itself.
    Output size = recursive size of ``preprocessed_output/``. Missing/unreadable
    sources degrade to 0 (never raises). Returns {} when nothing is measurable.
    """
    raw_bytes = 0
    # Batch: sum the actual routed source files.
    try:
        from easybci_lib.tools.neural_processing.io.routing_table import load_routing_table
        table = load_routing_table(out)
        if table and getattr(table, "inputs", None):
            raw_bytes = paths_size_bytes(e.data_path for e in table.inputs)
    except Exception:
        raw_bytes = 0
    # Single-file fallbacks.
    if not raw_bytes:
        ref = out / "plan" / "input_ref.json"
        if ref.exists():
            try:
                raw_bytes = int(json.loads(ref.read_text(encoding="utf-8")).get("size_bytes") or 0)
            except (OSError, json.JSONDecodeError, ValueError, TypeError):
                raw_bytes = 0
    if not raw_bytes and input_path:
        p = Path(input_path)
        raw_bytes = dir_size_bytes(p) if p.is_dir() else paths_size_bytes([input_path])

    output_bytes = dir_size_bytes(out / "preprocessed_output")
    if not raw_bytes and not output_bytes:
        return {}
    fp: Dict[str, Any] = {
        "raw_size_bytes": raw_bytes,
        "output_size_bytes": output_bytes,
        "raw_size_human": format_size(raw_bytes),
        "output_size_human": format_size(output_bytes),
    }
    if raw_bytes > 0:
        fp["reduction_pct"] = round((1 - output_bytes / raw_bytes) * 100, 1)
    return fp

# --- Output layout: continuous (BIDS sub-) + epoched (no prefix) ---
# Continuous post-pipeline data lives under preprocessed_output/preprocessed/
# with a BIDS-style ``sub-{id}`` prefix so MNE-BIDS / nilearn / similar can
# index it. Epoched / model-input data lives under preprocessed_output/AI_ready/
# without the prefix (ML / training-loop convention).
_PREPROC_SUBDIR = "preprocessed"
_AIREADY_SUBDIR = "AI_ready"
# Top-level entries inside preprocessed_output/ that are NOT subject containers
# (used by hierarchy walkers and migration helpers to skip them).
_NON_SUBJECT_TOPLEVEL = frozenset({"figures", "QC_out", _PREPROC_SUBDIR, _AIREADY_SUBDIR})


def _preprocessed_path(out: Path, subject_id: str, session_id: str) -> Path:
    """Return ``{out}/preprocessed_output/preprocessed/sub-{subject_id}/{session_id}/``."""
    return out / "preprocessed_output" / _PREPROC_SUBDIR / f"sub-{subject_id}" / session_id


def _load_identity_from_workdir(out: Path) -> tuple[Optional[str], Optional[str]]:
    """Read ``(subject_id, session_id_with_ses_prefix)`` from inspection_report.

    Single source of truth for who/which-session this recording is. When
    deep_inspect recorded an identity (sibling BIDS scan, parent
    dir heuristic, …) we MUST defer to it instead of re-inferring from
    the data_path stem, which is usually a timestamp rather than a
    subject identifier.

    Returns ``(None, None)`` when the report is missing or has no identity
    field — callers should fall back to legacy stem-based inference.
    """
    try:
        report_path = out / "middle_process" / "inspection_report.json"
        if not report_path.is_file():
            return (None, None)
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return (None, None)
    identity = report.get("identity") or {}
    sub = identity.get("subject_id")
    ses = identity.get("session_id")
    if not sub or not ses:
        return (None, None)
    ses_prefixed = ses if str(ses).startswith("ses-") else f"ses-{ses}"
    return (str(sub), ses_prefixed)


def _ai_ready_path(out: Path, subject_id: str, session_id: str) -> Path:
    """Return ``{out}/preprocessed_output/AI_ready/{subject_id}/{session_id}/``."""
    return out / "preprocessed_output" / _AIREADY_SUBDIR / subject_id / session_id
_CANONICAL_FILES = frozenset({"README.md"})

# File extensions that look like pipeline outputs — never bury these in
# middle_process, always promote to preprocessed_output. Used by both
# _consolidate_intermediates (forward path) and _salvage_layout (recovery path).
_OUTPUT_DATA_SUFFIXES = frozenset({".pkl", ".npy", ".npz", ".h5", ".hdf5", ".nwb"})


def _salvage_layout(out: Path) -> List[str]:
    """Best-effort repair pass that runs at the start of every build.

    Even after a noisy multi-step run (failed steps, hand-rolled scripts,
    interrupted exports, agent goofs that wrote files at the wrong level), the
    final repo must conform to the canonical layout. This helper rescues two
    classes of damage that no other path covers:

    1. **Sibling pollution.** A buggy work_dir derivation in a tool handler can
       drop ``pipeline.yaml`` / ``plan/`` next to ``work_dir`` instead of
       inside it. We pull those siblings in (only when the canonical slot is
       empty — never overwrite a real artifact).
    2. **Buried outputs.** A previous consolidation pass running before this
       fix may have swept ``session*_preprocessed.nwb`` into
       ``middle_process/``, leaving ``preprocessed_output/`` empty. When the
       canonical output dir is empty and middle_process clearly holds the real
       outputs, we lift them back out.

    Idempotent: calling this on an already-clean work_dir is a no-op. Returns
    a list of human-readable salvage notes for the repair report.
    """
    notes: List[str] = []

    # ---- (1) Sibling pollution rescue ------------------------------------
    parent = out.parent
    if parent.exists() and parent != out:  # don't try to climb out of /
        # pipeline.yaml at sibling level → pull in if work_dir's own slot is empty
        sibling_yaml = parent / "pipeline.yaml"
        own_yaml = out / "pipeline.yaml"
        if sibling_yaml.is_file() and not own_yaml.exists():
            try:
                shutil.move(str(sibling_yaml), str(own_yaml))
                notes.append(f"moved sibling pipeline.yaml → {own_yaml.name}")
            except OSError as exc:
                logger.debug("salvage: pipeline.yaml move failed: %s", exc)

        # plan/ at sibling level → merge files into our plan/ that don't exist yet.
        # We deliberately limit to known plan side-files (goal.json /
        # web_evidence.json / proposal.json / config.yaml) to avoid swallowing
        # an unrelated parent-of-many-experiments "plan" folder.
        sibling_plan = parent / "plan"
        if sibling_plan.is_dir():
            own_plan = out / "plan"
            own_plan.mkdir(parents=True, exist_ok=True)
            _SALVAGE_PLAN_FILES = {"goal.json", "web_evidence.json", "proposal.json", "config.yaml", "reasoning.md", "pipeline_record.json"}
            for child in sorted(sibling_plan.iterdir()):
                if child.is_file() and child.name in _SALVAGE_PLAN_FILES:
                    dest = own_plan / child.name
                    if not dest.exists():
                        try:
                            shutil.move(str(child), str(dest))
                            notes.append(f"moved sibling plan/{child.name} → plan/{child.name}")
                        except OSError as exc:
                            logger.debug("salvage: plan/%s move failed: %s", child.name, exc)
            # Drop the sibling plan/ if we emptied it
            try:
                if not any(sibling_plan.iterdir()):
                    sibling_plan.rmdir()
                    notes.append("removed empty sibling plan/")
            except OSError:
                pass

    # ---- (2) Output rescue from middle_process ---------------------------
    # Only fires when preprocessed_output is empty / has no data files AND
    # middle_process holds output-looking pkls. This catches damage from a
    # previous run that swept .pkl into middle_process before the
    # _consolidate_intermediates fix shipped.
    preproc = out / "preprocessed_output"
    middle = out / "middle_process"
    has_preproc_data = (
        preproc.is_dir()
        and any(p.suffix.lower() in _OUTPUT_DATA_SUFFIXES for p in preproc.rglob("*"))
    )
    if middle.is_dir() and not has_preproc_data:
        try:
            from easybci_lib.tools.neural_processing.batch.processor import _infer_session_id
        except Exception:
            _infer_session_id = lambda _: "ses-001"  # noqa: E731

        for item in sorted(middle.iterdir()):
            if not (item.is_file() and item.suffix.lower() in _OUTPUT_DATA_SUFFIXES):
                continue
            stem = item.stem
            for marker in ("_preprocessed", "_epochs", "_clean", "_processed"):
                if stem.endswith(marker):
                    stem = stem[: -len(marker)]
                    break
            try:
                session_id = _infer_session_id(item.name) or "ses-001"
            except Exception:
                session_id = "ses-001"
            dest = preproc / (stem or item.stem) / session_id / item.name
            if dest.exists():
                continue
            try:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(item), str(dest))
                rel = dest.relative_to(out)
                notes.append(f"rescued middle_process/{item.name} → {rel}")
            except OSError as exc:
                logger.debug("salvage: %s rescue failed: %s", item.name, exc)

    # ---- (3) Drop redundant code/middle_process archives ----------------
    # When the codegen archive holds only byte-identical copies of the
    # current code/pipeline.py, the directory is pure noise — kill the
    # duplicates and remove the archive dir if it ends up empty. (The
    # codegen-side fix already prevents NEW identical archives; this
    # rescues directories that were polluted by older builds.)
    code_archive = out / "code" / "middle_process"
    pipeline_py = out / "code" / "pipeline.py"
    if code_archive.is_dir() and pipeline_py.is_file():
        try:
            current_bytes = pipeline_py.read_bytes()
        except OSError:
            current_bytes = None
        if current_bytes is not None:
            for archived in sorted(code_archive.glob("pipeline_*.py")):
                try:
                    if archived.read_bytes() == current_bytes:
                        archived.unlink()
                        notes.append(f"removed identical archive code/middle_process/{archived.name}")
                except OSError as exc:
                    logger.debug("salvage: %s unlink failed: %s", archived, exc)
            try:
                if not any(code_archive.iterdir()):
                    code_archive.rmdir()
                    notes.append("removed empty code/middle_process/")
            except OSError:
                pass

    if notes:
        logger.info("salvage_layout: %d action(s) on %s", len(notes), out)
    return notes


def build_mini_repo(
    output_dir: str,
    steps: List[str],
    data_info: Dict[str, Any],
    pipeline_record: Dict[str, Any],
    input_path: str = "",
    modality: str = "eeg",
    segment_duration: float = 2.0,
    stride: float = 1.0,
    subject_id: str = "",
    paradigm: str = "",
    pkl_path: str = "",
    force: bool = False,
    qc_figures_dir: Optional[str] = None,
    label_config: Optional[Dict[str, Any]] = None,
    split_config: Optional[Dict[str, Any]] = None,
    status: Literal["ok", "partial", "migrated"] = "ok",
    partial_reason: Optional[str] = None,
    analysis_goal: Optional[str] = None,
    scenario: Optional[str] = None,
    deliverables: Optional[list] = None,
) -> Dict[str, Any]:
    """Build a reproducible mini-repo from pipeline execution results.

    Parameters
    ----------
    label_config : dict, optional
        L3 continuous label configuration.
    split_config : dict, optional
        Data splitting configuration.
    status : {"ok", "partial", "migrated"}, default "ok"
        Lifecycle status for the produced mini-repo.

        - ``ok``: full, contract-compliant build (default; used by the LLM's
          explicit ``export_repo`` tool call and by the auto-finalize hook
          when the run completed cleanly).
        - ``partial``: best-effort build invoked from a finally hook after the
          run was interrupted or failed. Skips ``_consolidate_intermediates``
          so debugging artifacts stay where the user (or LLM) left them; the
          README gets a banner and the record gains ``pending_consolidations``.
        - ``migrated``: produced by ``easybci migrate-work-dir`` from an old
          layout; behaves like ``partial`` for consolidation purposes.
    partial_reason : str, optional
        Human-readable reason recorded alongside ``status`` when not ``ok``
        (e.g. ``"keyboard_interrupt"``, ``"failed: TimeoutError: ..."``).
    """
    from easybci_lib.tools.neural_processing.codegen.generator import (
        _enforce_clean_output,
        generate_config_yaml,
        generate_pipeline_script,
        generate_requirements,
        generate_run_script_v2,
        generate_split_code,
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Resolve analysis_goal with priority:
    #   explicit param > pipeline_record > plan/goal.json side file > "generic"
    # The side file is written by _handle_propose_pipeline, so by the time
    # build_mini_repo runs (either through export_repo or auto-finalize) the
    # goal is already on disk. "generic" is the safe fallback.
    _goal = analysis_goal or (pipeline_record or {}).get("analysis_goal")
    if not _goal:
        _goal_path = out / "plan" / "goal.json"
        if _goal_path.exists():
            try:
                _goal = json.loads(_goal_path.read_text(encoding="utf-8")).get("analysis_goal")
            except (OSError, json.JSONDecodeError):
                _goal = None
    analysis_goal = (str(_goal).strip() if _goal else "") or "generic"

    # Resolve scenario/deliverables with the same priority as analysis_goal:
    #   explicit param > pipeline_record > plan/goal.json side file > default.
    # propose_pipeline already wrote both into plan/goal.json, so an
    # export/auto-finalize call that omits them still recovers the confirmed
    # values instead of silently defaulting.
    from easybci_lib.tools.neural_processing.preprocess.deliverables import (
        normalize_deliverables as _normalize_deliverables,
    )
    _scenario = scenario or (pipeline_record or {}).get("scenario")
    _deliv = deliverables
    if _deliv is None:
        _pr_deliv = (pipeline_record or {}).get("deliverables")
        if isinstance(_pr_deliv, list):
            _deliv = _pr_deliv
    if _scenario is None or _deliv is None:
        _goal_path = out / "plan" / "goal.json"
        if _goal_path.exists():
            try:
                _gj = json.loads(_goal_path.read_text(encoding="utf-8"))
                if _scenario is None:
                    _scenario = _gj.get("scenario")
                if _deliv is None and isinstance(_gj.get("deliverables"), list):
                    _deliv = _gj["deliverables"]
            except (OSError, json.JSONDecodeError):
                pass
    _scenario = (str(_scenario).strip() if _scenario else "") or "research"
    try:
        _deliverables = _normalize_deliverables(_deliv)
    except ValueError:
        _deliverables = ["preprocessed"]

    # Resolve web_evidence with the same priority pattern as
    # analysis_goal (pipeline_record > plan/web_evidence.json > "unavailable").
    # The result is rendered as a banner in reasoning.md and stamped into
    # pipeline_record.json (web_evidence_used / web_evidence_provider).
    web_evidence: Dict[str, Any] = {}
    if isinstance((pipeline_record or {}).get("web_evidence"), dict):
        web_evidence = dict(pipeline_record["web_evidence"])
    if not web_evidence:
        _ev_path = out / "plan" / "web_evidence.json"
        if _ev_path.exists():
            try:
                web_evidence = json.loads(_ev_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                web_evidence = {"status": "unavailable", "reason": "side-file unreadable"}
    if not web_evidence:
        web_evidence = {"status": "unavailable", "reason": "no web_evidence recorded"}

    # Enforce data-only cleanup on the canonical step list
    # at the build_mini_repo boundary so reasoning.md, config.yaml, and
    # pipeline.py all see the same enforced steps. Idempotent — calling
    # _enforce_clean_output a second time on already-enforced steps is a
    # no-op (rule 1). Goal-conditional gate decides whether to inject at all.
    steps = _enforce_clean_output(steps, analysis_goal=analysis_goal)

    # --- Layout salvage (runs first, idempotent) -------------------------
    # Robustness ramp: rescue stray sibling files and unbury misplaced .pkl
    # outputs BEFORE the idempotency short-circuit. Past runs that crashed
    # mid-way may have left damage that an "ok-on-record" finalize would
    # otherwise refuse to touch — so we scrub first, then check the cache.
    salvage_notes: List[str] = []
    try:
        salvage_notes = _salvage_layout(out)
    except Exception as exc:
        # Salvage must never block the build — log and continue.
        logger.warning("salvage_layout failed for %s: %s", out, exc)

    # --- Idempotency: pipeline_record.json drives status-based short-circuit ---
    # This runs BEFORE the manifest check so a finalize hook firing on an
    # already-finalized work_dir is a true no-op (no re-walk, no re-write).
    record_path = out / "plan" / "pipeline_record.json"
    existing_record: Optional[Dict[str, Any]] = None
    if record_path.exists():
        try:
            existing_record = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing_record = None

    if existing_record is not None:
        existing_status = str(existing_record.get("status") or "ok")
        # The auto-finalize safety net stamps ``auto_synthesized: true`` on
        # records it synthesizes from scratch (turn-1 early-stop case).
        # Such "husk" records carry placeholder modality/paradigm/steps and
        # MUST be overwritten by the next real ok finalize — otherwise the
        # user sees a 'modality=unknown' README forever (Bug #1/#2 from
        # the 2026-06-17 WebUI test).
        existing_is_husk = bool(existing_record.get("auto_synthesized"))
        # ok + ok → no-op return existing record. Don't re-walk the tree.
        # Exception 1: if salvage rescued anything, we MUST re-run the full
        # build so README / manifest / consolidation reflect the rescued
        # files. A no-op short-circuit on a damaged tree is exactly what we
        # were trying to avoid.
        # Exception 2: husk records (see above) — never short-circuit.
        # Exception 3: ``force=True`` callers (finalize asking for a
        # proposal-driven rebuild) — always proceed past the no-op.
        if (
            existing_status == "ok"
            and status == "ok"
            and not salvage_notes
            and not existing_is_husk
            and not force
        ):
            logger.info(
                "build_mini_repo: %s already finalized (status=ok) — no-op", output_dir
            )
            return {
                "success": True,
                "output_dir": str(out),
                "files": [],
                "n_files": 0,
                "cached": True,
                "status": "ok",
                "pipeline_record": existing_record,
            }
        if existing_is_husk and status == "ok":
            logger.info(
                "build_mini_repo: %s previously finalized as husk "
                "(auto_synthesized=True) — overwriting with real record",
                output_dir,
            )
        # partial + partial → overwrite with newer partial (latest reason wins)
        # partial/migrated + ok → upgrade to full build (proceed)
        # (no early return; fall through to full build)

    # --- Idempotency check: skip if manifest indicates no changes ---
    # Only honored when caller did not request a status downgrade/upgrade above,
    # AND the prior export wasn't a husk (auto_synthesized=True) — a husk
    # manifest would short-circuit the very rebuild it's supposed to be replaced by.
    if (
        not force
        and status == "ok"
        and not (existing_record and existing_record.get("auto_synthesized"))
    ):
        cached = _check_export_manifest(out, pkl_path, input_path)
        if cached:
            logger.info("Export manifest valid — skipping rebuild of %s", output_dir)
            cached.setdefault("status", "ok")
            return cached

    created_files: List[str] = []
    reasoning = pipeline_record.get("reasoning") or {}

    # --- Consolidate intermediate artifacts into middle_process/ ---
    # SKIP in partial mode so debug artifacts stay in place for the user to
    # inspect; record what *would* have been moved so a later `easybci
    # finalize --status ok` (or a clean re-run) can pick up where this left off.
    pending_consolidations: List[str] = []
    if status == "ok":
        _consolidate_intermediates(out)
    else:
        pending_consolidations = _list_pending_consolidations(out)

    # --- plan/ ---
    plan_dir = out / "plan"
    plan_dir.mkdir(parents=True, exist_ok=True)

    # Salvage audit — when _salvage_layout actually moved files, write a small
    # report into plan/ so the user can see exactly what auto-recovery did.
    # Idempotent: a clean run leaves no report behind.
    if salvage_notes:
        _write(
            plan_dir / "repair_report.json",
            json.dumps(
                {"actions": salvage_notes, "n_actions": len(salvage_notes)},
                indent=2, ensure_ascii=False,
            ),
        )
        created_files.append("plan/repair_report.json")

    _input_stem = Path(input_path or data_info.get("file", "data")).stem

    # Identity is authoritative when present. Falls back to legacy stem-based
    # inference for older sessions where the inspection_report predates the
    # identity field.
    _ident_sub, _ident_ses = _load_identity_from_workdir(out)
    if _ident_sub and not subject_id:
        subject_id = _ident_sub

    from easybci_lib.tools.neural_processing.batch.processor import _infer_session_id
    if _ident_ses:
        _session_id = _ident_ses
    else:
        _session_id = _infer_session_id(input_path or data_info.get("file", "data"))

    # The preprocessed/ layer is NWB-only — the chosen_format field is
    # kept on the pipeline_record purely for audit (so consumers can see
    # what the LLM was instructed to write).
    _chosen_fmt = "nwb"

    config_yaml = generate_config_yaml(
        steps=steps,
        modality=modality,
        segment_duration=segment_duration,
        stride=stride,
        subject_id=subject_id,
        paradigm=paradigm,
        output_format=_chosen_fmt,
        split_config=split_config,
    )
    _write(plan_dir / "config.yaml", config_yaml)
    created_files.append("plan/config.yaml")

    existing_reasoning = plan_dir / "reasoning.md"
    if _is_propose_authored_reasoning(existing_reasoning):
        # propose_pipeline's evidence-driven render is the single source of
        # truth for reasoning.md. When its fingerprint is present we
        # must NOT overwrite it with the finalize-time fallback, which has empty
        # steps/data_info and would clobber it with a 16-line husk.
        created_files.append("plan/reasoning.md")
    else:
        # Fallback path: propose didn't write (or wrote a degraded stub). Recover
        # steps / data_info from proposal.json so even the fallback has content
        # instead of an empty husk.
        fb_steps, fb_data_info = _recover_reasoning_inputs(
            plan_dir, steps, data_info
        )
        _write_reasoning_md(
            plan_dir, fb_steps, fb_data_info, pipeline_record, paradigm, modality,
            analysis_goal=analysis_goal,
            web_evidence=web_evidence,
        )
        created_files.append("plan/reasoning.md")

    if pipeline_record:
        # Stamp reproducibility info so consumers (CLI/WebUI) can render a badge.
        pipeline_record.setdefault("reproducibility", {
            "seed": EASYBCI_SEED,
            "locked": True,
        })
        # Keep the canonical "steps" field aligned with the
        # cleanup-enforced step list that codegen / config.yaml / reasoning.md
        # all see. Historical "steps_applied" in middle_process records is
        # left untouched — it documents what *ran*; "steps" is the canonical
        # documented pipeline going forward.
        pipeline_record["steps"] = list(steps)
        # analysis_goal is one of the four single-source-of-truth
        # fields (proposal/reasoning/pipeline_record/schema). Stamp it here so
        # downstream consumers and the finalize-recovery path see it.
        pipeline_record["analysis_goal"] = analysis_goal
        pipeline_record["scenario"] = _scenario
        pipeline_record["deliverables"] = list(_deliverables)
        # Raw-vs-preprocessed storage footprint (e.g. raw 5.6 TB → 98 GB) so the
        # summary/README and machine consumers can report the reduction.
        _footprint = _compute_storage_footprint(out, input_path)
        if _footprint:
            pipeline_record["storage_footprint"] = _footprint
        # Record the user's output_format choice + the resolved
        # chosen_format. .get() fallback elsewhere keeps consumers tolerant of
        # legacy records that lack these keys.
        # Record the resolved output format for audit. The preprocessed/
        # layer is NWB-only, so chosen_format is fixed; output_format
        # echoes whatever override the proposal carried (informational).
        pipeline_record["chosen_format"] = "nwb"
        _proposal_path = plan_dir / "proposal.json"
        if _proposal_path.is_file() and "output_format" not in pipeline_record:
            try:
                import json as _json_or
                _prop = _json_or.loads(_proposal_path.read_text(encoding="utf-8"))
                if "output_format" in _prop:
                    pipeline_record["output_format"] = _prop["output_format"]
            except Exception:
                pass
        # Surface the dispatch-time web_evidence on the record
        # so consumers (CLI summary, WebUI, finalize fallback) can render
        # the "used / unavailable" badge without re-reading the side file.
        pipeline_record["web_evidence"] = web_evidence
        pipeline_record["web_evidence_used"] = (
            isinstance(web_evidence, dict)
            and web_evidence.get("status") == "ok"
            and bool(web_evidence.get("recommendations"))
        )
        pipeline_record["web_evidence_provider"] = (
            web_evidence.get("provider") if isinstance(web_evidence, dict) else None
        )
        # Stamp lifecycle status so downstream tooling (CLI listings, validators,
        # the proven-pipeline flywheel) can distinguish partial vs. ok builds.
        pipeline_record["status"] = status
        if partial_reason is not None:
            pipeline_record["partial_reason"] = partial_reason
        elif "partial_reason" in pipeline_record and status == "ok":
            # Clean up stale partial_reason when upgrading to ok.
            pipeline_record.pop("partial_reason", None)
        if pending_consolidations:
            pipeline_record["pending_consolidations"] = pending_consolidations
        elif status == "ok":
            pipeline_record.pop("pending_consolidations", None)
        _write(out / "plan" / "pipeline_record.json", json.dumps(
            pipeline_record, indent=2, default=str, ensure_ascii=False))
        created_files.append("plan/pipeline_record.json")
    elif status != "ok":
        # Auto-finalize path may have no pipeline_record at all (LLM stopped
        # before calling propose_pipeline). Still write a minimal record so
        # the work_dir is recognizably partial and `easybci finalize` can see it.
        minimal_record = {
            "status": status,
            "partial_reason": partial_reason,
            "reproducibility": {"seed": EASYBCI_SEED, "locked": True},
            "steps": steps,
            "modality": modality,
            "paradigm": paradigm,
            "analysis_goal": analysis_goal,
            "scenario": _scenario,
            "deliverables": list(_deliverables),
            "web_evidence": web_evidence,
            "web_evidence_used": (
                isinstance(web_evidence, dict)
                and web_evidence.get("status") == "ok"
                and bool(web_evidence.get("recommendations"))
            ),
            "web_evidence_provider": (
                web_evidence.get("provider") if isinstance(web_evidence, dict) else None
            ),
            "auto_finalized": True,
        }
        if pending_consolidations:
            minimal_record["pending_consolidations"] = pending_consolidations
        _write(out / "plan" / "pipeline_record.json", json.dumps(
            minimal_record, indent=2, default=str, ensure_ascii=False))
        created_files.append("plan/pipeline_record.json")
    elif existing_record is not None and not existing_record.get("auto_synthesized"):
        # Upgrading a partial/migrated record to ok but caller didn't supply a
        # fresh pipeline_record — promote the existing one in place so
        # downstream idempotency checks see status=ok.
        # NOTE: husk records (auto_synthesized=True from a turn-1 premature
        # finalize) are intentionally NOT promoted here — they would re-inject
        # 'unknown' modality/paradigm. They fall through to the minimal-record
        # branch below so caller-supplied modality/paradigm/steps win.
        promoted = dict(existing_record)
        promoted["status"] = "ok"
        promoted.pop("partial_reason", None)
        promoted.pop("pending_consolidations", None)
        promoted.setdefault("reproducibility", {"seed": EASYBCI_SEED, "locked": True})
        _write(out / "plan" / "pipeline_record.json", json.dumps(
            promoted, indent=2, default=str, ensure_ascii=False))
        created_files.append("plan/pipeline_record.json")
    else:
        # Fix B1 — caller passed empty pipeline_record, status="ok", and no
        # existing record on disk. Without this branch plan/pipeline_record.json
        # never lands and the README falls back to "?" / "unknown" because
        # downstream finalize/recovery has nothing to read. Synthesize a minimal
        # record from whatever args we DO have (steps/modality/paradigm/
        # analysis_goal/data_info) so README sampling-rate, channels, and the
        # source-file row are populated.
        # `auto_synthesized` marks records the safety net pulled out of thin
        # air with no real preprocessing info; downstream code uses it to
        # decide whether to allow re-finalize overwrite. A caller-provided
        # real modality + paradigm + steps means this is NOT thin-air —
        # it's a real run whose pipeline_record arg happened to be empty.
        _real_signal = (
            modality not in ("", "unknown")
            and paradigm not in ("", "unknown")
            and bool(steps)
        )
        minimal_record = {
            "status": status,
            "steps": list(steps),
            "modality": modality,
            "paradigm": paradigm,
            "analysis_goal": analysis_goal,
            "scenario": _scenario,
            "deliverables": list(_deliverables),
            "data_info": dict(data_info) if data_info else {},
            "web_evidence": web_evidence,
            "web_evidence_used": (
                isinstance(web_evidence, dict)
                and web_evidence.get("status") == "ok"
                and bool(web_evidence.get("recommendations"))
            ),
            "web_evidence_provider": (
                web_evidence.get("provider") if isinstance(web_evidence, dict) else None
            ),
            "reproducibility": {"seed": EASYBCI_SEED, "locked": True},
            "auto_synthesized": not _real_signal,
        }
        _write(out / "plan" / "pipeline_record.json", json.dumps(
            minimal_record, indent=2, default=str, ensure_ascii=False))
        created_files.append("plan/pipeline_record.json")

    if input_path:
        # Prefer the full-content hasher; fall back to the local manifest
        # hasher if that module isn't importable. If both raise (disk
        # error, permissions, etc.), skip input_ref.json with a warning
        # rather than blowing up the whole export.
        try:
            from easybci_lib.tools.neural_processing.export.reproducibility import write_input_hash
            write_input_hash(str(plan_dir), input_path)
            created_files.append("plan/input_ref.json")
        except Exception as _hash_err:
            try:
                _write_input_ref(plan_dir, input_path)
                created_files.append("plan/input_ref.json")
            except Exception as _fallback_err:
                logger.warning(
                    "input_ref.json skipped — write_input_hash failed (%s); "
                    "fallback _write_input_ref also failed (%s)",
                    _hash_err, _fallback_err,
                )

    alignment_info = pipeline_record.get("alignment_report") or pipeline_record.get("alignment_step")
    if alignment_info:
        _write(out / "plan" / "stream_correspondence.json", json.dumps(
            alignment_info, indent=2, default=str, ensure_ascii=False))
        created_files.append("plan/stream_correspondence.json")

    # --- code/ ---
    code_dir = out / "code"
    code_dir.mkdir(parents=True, exist_ok=True)

    code = generate_pipeline_script(
        steps=steps,
        data_info=data_info,
        modality=modality,
        analysis_goal=analysis_goal,
    )
    _write(code_dir / "pipeline.py", code)
    created_files.append("code/pipeline.py")

    # Bundle the standalone Nihon Kohden io_loader plugin into the repo so the
    # generated pipeline reads NK (.EEG) correctly on ANY machine — not as
    # 1-channel BrainVision garbage. matches() only claims NK sets (sibling
    # .21E), so it is inert in non-NK repos → provision unconditionally. Also
    # drop a machine-global copy for interactive/QC reuse. Best-effort.
    try:
        from easybci_lib.tools.neural_processing.io.nk_loader_plugin import (
            ensure_global_plugin,
            ensure_repo_plugin,
        )
        ensure_repo_plugin(code_dir)
        created_files.append("code/io_loaders/nihon_kohden.py")
        ensure_global_plugin()
    except Exception as _nk_err:  # noqa: BLE001
        logger.warning("NK io_loader provisioning failed: %s", _nk_err)

    # Best-effort lint pass on the generated pipeline (auto-fix import order
    # and explicit text-mode encoding). Never blocks export — see
    # tools/neural_processing/export/_lint.py for the rationale and the
    # ruff-absent / failure paths.
    _lint_remaining = ruff_fix(code_dir / "pipeline.py")
    if _lint_remaining:
        _write(code_dir / "pipeline.lint.txt", _lint_remaining + "\n")
        created_files.append("code/pipeline.lint.txt")

    # run.py: multi-input-aware wrapper. has_build_ai_ready is True when the
    # caller already wrote build_ai_ready.py (or expects to in this batch).
    # has_vis follows the same pattern — picks up code/vis.py if it's on disk.
    _has_bar = (code_dir / "build_ai_ready.py").is_file()
    _has_vis = (code_dir / "vis.py").is_file()
    run_code = generate_run_script_v2(has_build_ai_ready=_has_bar, has_vis=_has_vis)
    _write(code_dir / "run.py", run_code)
    created_files.append("code/run.py")

    reqs = generate_requirements()
    _write(code_dir / "requirements.txt", reqs)
    created_files.append("code/requirements.txt")

    # NOTE: qc.py / build_ai_ready.py are written by `generate_code` at Step 5
    # of the pipeline skill (the new contract). repo_builder still
    # emits run.py + requirements.txt for backward compatibility with mini-repos
    # that pre-date the codegen bundle, but no longer duplicates the qc /
    # visualize emission.

    if split_config:
        try:
            n_segments_est = 0
            if segment_duration > 0 and stride > 0:
                dur_s = data_info.get("duration_seconds", data_info.get("duration", 0))
                if dur_s:
                    n_segments_est = max(1, int((dur_s - segment_duration) / stride) + 1)
            split_code = generate_split_code(split_config, n_segments_est)
            _write(code_dir / "split.py", split_code)
            created_files.append("code/split.py")
        except Exception as exc:
            logger.debug("Split code generation failed: %s", exc)

    # --- preprocessed_output/ ---
    # Migrate per-subject output+figures into preprocessed_output/{subject}/{session}/
    _migrate_subject_outputs(out)

    # For single-file mode with explicit pkl_path
    if pkl_path and Path(pkl_path).exists():
        sub_id = subject_id or _input_stem
        sub_out = _preprocessed_path(out, sub_id, _session_id)
        sub_out.mkdir(parents=True, exist_ok=True)
        _copy_output(sub_out, pkl_path, output_stem=f"{_input_stem}_preprocessed")
        src_suffix = Path(pkl_path).suffix or ".pkl"
        rel = sub_out.relative_to(out) / f"{_input_stem}_preprocessed{src_suffix}"
        created_files.append(str(rel))

    # Clean up duplicate processed/continuous files
    _cleanup_duplicate_outputs(out)

    # Save L3 continuous labels
    if label_config:
        try:
            import numpy as np
            sub_id = subject_id or _input_stem
            label_out = _preprocessed_path(out, sub_id, _session_id)
            label_out.mkdir(parents=True, exist_ok=True)
            aligned_labels = label_config.get("aligned_labels")
            if aligned_labels is not None and hasattr(aligned_labels, "shape"):
                np.save(str(label_out / "labels.npy"), aligned_labels)
                created_files.append(str(label_out.relative_to(out) / "labels.npy"))
            elif label_config.get("label_path"):
                src_label = Path(label_config["label_path"])
                if src_label.exists():
                    dest_label = label_out / f"labels_source{src_label.suffix}"
                    shutil.copy2(str(src_label), str(dest_label))
                    created_files.append(str(label_out.relative_to(out) / f"labels_source{src_label.suffix}"))
        except Exception as exc:
            logger.debug("Label export failed: %s", exc)

    # QC report → preprocessed_output/QC_out/sub-{id}/{session}/
    qc_data = pipeline_record.get("qc_result") or pipeline_record.get("qc")
    if qc_data:
        sub_id = subject_id or _input_stem
        qc_out = out / "preprocessed_output" / "QC_out" / f"sub-{sub_id}" / _session_id
        qc_out.mkdir(parents=True, exist_ok=True)
        _write(qc_out / "qc_metrics.json", json.dumps(
            qc_data, indent=2, default=str, ensure_ascii=False))
        created_files.append(
            f"preprocessed_output/QC_out/sub-{sub_id}/{_session_id}/qc_metrics.json"
        )

    # Record which subjects/sessions are in preprocessed_output/
    output_hierarchy = _list_output_hierarchy(out)
    subjects_in_output = list(output_hierarchy.keys())
    for sub, sessions in output_hierarchy.items():
        for ses in sessions:
            for ses_dir in (
                _preprocessed_path(out, sub, ses),
                _ai_ready_path(out, sub, ses),
            ):
                if not ses_dir.exists():
                    continue
                for f in sorted(ses_dir.rglob("*")):
                    if f.is_file():
                        rel = f.relative_to(out)
                        if str(rel) not in created_files:
                            created_files.append(str(rel))

    # --- README.md ---
    _write_readme(out, steps, data_info, pipeline_record, modality, paradigm,
                  input_path, split_config, output_hierarchy,
                  status=status, partial_reason=partial_reason,
                  pending_consolidations=pending_consolidations)
    created_files.append("README.md")

    logger.info("Mini-repo built at %s with %d files", output_dir, len(created_files))

    result = {
        "success": True,
        "output_dir": str(out),
        "files": created_files,
        "n_files": len(created_files),
        "reproducibility": {"seed": EASYBCI_SEED, "locked": True},
        "status": status,
    }
    if partial_reason is not None:
        result["partial_reason"] = partial_reason
    if pending_consolidations:
        result["pending_consolidations"] = pending_consolidations

    # Write export manifest into middle_process/ (only meaningful for ok builds;
    # partial builds intentionally skip so a later upgrade re-runs the full path).
    if status == "ok":
        _write_export_manifest(out, pkl_path, input_path, created_files)

    return result


def finalize_mini_repo(output_dir: str) -> Dict[str, Any]:
    """Remove middle_process/ from a completed, stable mini-repo.

    Call this ONLY after the pipeline is fully complete and confirmed stable.
    Removes intermediate artifacts that are not part of the final deliverable.
    """
    out = Path(output_dir)
    middle = out / "middle_process"
    removed = False
    if middle.exists() and middle.is_dir():
        shutil.rmtree(str(middle))
        removed = True
        logger.info("Finalized mini-repo: removed middle_process/ from %s", output_dir)
    return {
        "success": True,
        "output_dir": str(out),
        "middle_process_removed": removed,
        "reproducibility": {"seed": EASYBCI_SEED, "locked": True},
    }


# ---------------------------------------------------------------------------
# Directory structure management
# ---------------------------------------------------------------------------

def _list_pending_consolidations(out: Path) -> List[str]:
    """Enumerate files/dirs that ``_consolidate_intermediates`` *would* move.

    Mirrors the predicate logic in ``_consolidate_intermediates`` but performs
    no filesystem changes. Used in partial-mode builds so the user can see what
    a future ``--status ok`` upgrade will clean up, without losing visibility
    into the debug artifacts now.
    """
    pending: List[str] = []
    if not out.is_dir():
        return pending
    for item in sorted(out.iterdir()):
        name = item.name
        if name in _CANONICAL_DIRS or name in _CANONICAL_FILES:
            continue
        if name == "middle_process":
            continue
        if name == ".export_manifest.json":
            pending.append(name)
            continue
        if item.is_file():
            pending.append(name)
        elif item.is_dir():
            # Subject-like dirs (with output/ or figures/) are handled by
            # _migrate_subject_outputs, not consolidation — skip them.
            if (item / "output").exists() or (item / "figures").exists():
                continue
            pending.append(name + "/")
    return pending


def _consolidate_intermediates(out: Path) -> None:
    """Move all intermediate/exploration artifacts into middle_process/.

    Anything not part of the canonical structure (README, preprocessed_output,
    code, plan) is considered intermediate and belongs in middle_process/.

    Exception: root-level data outputs (``*.pkl`` / ``*.npy`` / ``*.npz`` /
    ``*.h5`` / ``*.hdf5``) are migrated to
    ``preprocessed_output/{stem}/{session}/`` instead. These are typically
    written by an agent's hand-rolled preprocess script that didn't know about
    the canonical layout — burying them under middle_process leaves
    preprocessed_output empty and forces the user to dig 591 MB of pkls out of
    the "history archive" folder.
    """
    middle = out / "middle_process"
    middle.mkdir(parents=True, exist_ok=True)

    # Files/dirs at root that are intermediate artifacts
    for item in sorted(out.iterdir()):
        name = item.name
        # Skip canonical dirs and README
        if name in _CANONICAL_DIRS or name in _CANONICAL_FILES:
            continue
        # Skip hidden manifest (will be placed in middle_process explicitly)
        if name == ".export_manifest.json":
            _move_safe(item, middle / name)
            continue
        # Skip items already in the correct location
        if name == "middle_process":
            continue
        # Per-subject dirs (Acq..., sub-...) will be handled by _migrate_subject_outputs
        # But other files (debug scripts, batch reports, pipeline.yaml, etc.) → middle
        if item.is_file():
            # Route data-output extensions to preprocessed_output instead of
            # burying them in middle_process. Strip a "_preprocessed"/"_epochs"
            # suffix from the stem to recover the input/session identifier so
            # related files cluster under the same subject dir.
            if item.suffix.lower() in _OUTPUT_DATA_SUFFIXES:
                stem = item.stem
                is_epoched = False
                for marker in ("_preprocessed", "_epochs", "_clean", "_processed"):
                    if stem.endswith(marker):
                        if marker == "_epochs":
                            is_epoched = True
                        stem = stem[: -len(marker)]
                        break
                try:
                    from easybci_lib.tools.neural_processing.batch.processor import _infer_session_id
                    session_id = _infer_session_id(item.name) or "ses-001"
                except Exception:
                    session_id = "ses-001"
                # Route by readiness: *_epochs.* → AI_ready; everything else → preprocessed/sub-.
                sub_id = stem or item.stem
                if is_epoched:
                    dest = _ai_ready_path(out, sub_id, session_id) / item.name
                else:
                    dest = _preprocessed_path(out, sub_id, session_id) / item.name
                _move_safe(item, dest)
            else:
                _move_safe(item, middle / name)
        elif item.is_dir():
            # Directories that look like subjects (have output/ or figures/) are handled later
            if (item / "output").exists() or (item / "figures").exists():
                continue
            # Pipeline cache → middle
            if name.startswith("."):
                _move_safe(item, middle / name)
            else:
                _move_safe(item, middle / name)


def _migrate_subject_outputs(out: Path) -> None:
    """Migrate per-subject output+figures into the new split layout.

    Handles the old layout where subjects live at ``{work_dir}/{subject_id}/output/``
    and moves them to ``{work_dir}/preprocessed_output/preprocessed/sub-{subject_id}/``
    (continuous-data side, BIDS-friendly). Figures are migrated to
    ``preprocessed_output/figures/sub-{id}/{session_id}/``. Also fixes double-nesting
    (``preprocessed_output/preprocessed_output/``) from batch runs.
    """
    preproc_out = out / "preprocessed_output"
    preproc_out.mkdir(parents=True, exist_ok=True)

    # Fix double-nesting: flatten preprocessed_output/preprocessed_output/ → preprocessed_output/
    nested = preproc_out / "preprocessed_output"
    if nested.is_dir():
        for item in sorted(nested.iterdir()):
            dest = preproc_out / item.name
            if item.is_dir() and not dest.exists():
                item.rename(dest)
            elif item.is_dir():
                for child in item.rglob("*"):
                    if child.is_file():
                        rel = child.relative_to(item)
                        target = dest / rel
                        target.parent.mkdir(parents=True, exist_ok=True)
                        _move_safe(child, target)
        try:
            shutil.rmtree(str(nested))
        except OSError:
            pass

    for item in sorted(out.iterdir()):
        if not item.is_dir():
            continue
        if item.name in _CANONICAL_DIRS or item.name in ("middle_process",):
            continue

        # Check if this looks like a subject directory
        has_output = (item / "output").exists()
        has_figures = (item / "figures").exists()
        if not (has_output or has_figures):
            continue

        # Sanitize subject id: strip BIDS sub- prefix if the source dir already
        # has it; the helpers will re-prepend it under preprocessed/.
        raw_name = item.name
        sub_id = raw_name[len("sub-"):] if raw_name.startswith("sub-") else raw_name

        if has_output:
            # Move output/ files into preprocessed_output/preprocessed/sub-{id}/.
            # Per-session structure is best-effort: legacy `output/` is flat —
            # land everything in a synthetic ``ses-001`` until the codegen
            # writer fans it out properly on the next real run.
            dest_root = preproc_out / _PREPROC_SUBDIR / f"sub-{sub_id}"
            for f in (item / "output").iterdir():
                if f.is_file():
                    target = dest_root / "ses-001" / f.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    _move_safe(f, target)
                elif f.is_dir() and f.name.startswith("ses-"):
                    # Real session dir — preserve.
                    target_ses = dest_root / f.name
                    target_ses.mkdir(parents=True, exist_ok=True)
                    for child in f.iterdir():
                        _move_safe(child, target_ses / child.name)
                    try:
                        f.rmdir()
                    except OSError:
                        pass
                else:
                    target = dest_root / "ses-001" / f.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    _move_safe(f, target)
            try:
                (item / "output").rmdir()
            except OSError:
                pass

        if has_figures:
            _migrate_figures_to_toplevel(
                item / "figures",
                preproc_out / "figures" / f"sub-{sub_id}",
            )

        # Remove empty source dir
        try:
            shutil.rmtree(str(item))
        except OSError:
            pass

    # Also migrate any figures already inside subject/session dirs to the
    # canonical preprocessed_output/figures/sub-{id}/{ses}/ root. Walk all
    # three subject-container shapes:
    #   - preprocessed_output/preprocessed/sub-{id}/{ses}/figures/
    #   - preprocessed_output/AI_ready/{id}/{ses}/figures/
    #   - preprocessed_output/{legacy_id}/{ses}/figures/   (pre-rename runs)
    def _walk_subject_roots():
        cont_root = preproc_out / _PREPROC_SUBDIR
        if cont_root.is_dir():
            for d in cont_root.iterdir():
                if d.is_dir():
                    yield d
        ai_root = preproc_out / _AIREADY_SUBDIR
        if ai_root.is_dir():
            for d in ai_root.iterdir():
                if d.is_dir():
                    yield d
        for d in preproc_out.iterdir():
            if d.is_dir() and d.name not in _NON_SUBJECT_TOPLEVEL:
                yield d

    for sub_dir in _walk_subject_roots():
        # Resolve sub-{id} display name regardless of which half this dir came
        # from (preprocessed/ stores `sub-S01`, AI_ready/ stores `S01`).
        _sub_name = sub_dir.name
        sub_label = _sub_name if _sub_name.startswith("sub-") else f"sub-{_sub_name}"
        for ses_dir in sorted(sub_dir.iterdir()):
            if not ses_dir.is_dir():
                continue
            fig_dir = ses_dir / "figures"
            if fig_dir.exists():
                dest_figs = preproc_out / "figures" / sub_label / ses_dir.name
                dest_figs.mkdir(parents=True, exist_ok=True)
                for f in fig_dir.rglob("*"):
                    if f.is_file():
                        _move_safe(f, dest_figs / f.name)
                try:
                    shutil.rmtree(str(fig_dir))
                except OSError:
                    pass

    # Legacy salvage: pre-fix runs wrote figures/{session}/ and QC_out/{session}/
    # without the sub-{id} layer. Migrate them up under sub-{id}/ when we can
    # unambiguously resolve the subject from preprocessed/sub-*/{session}/. If
    # multiple subjects share the same session id, leave the legacy dir in
    # place rather than guess.
    _legacy_layout_migrate(preproc_out)


def _legacy_layout_migrate(preproc_out: Path) -> None:
    """Promote legacy `figures/{session}/` and `QC_out/{session}/` dirs into
    the new `figures/sub-{id}/{session}/` layout when subject is resolvable.

    A session is resolvable when exactly one `preprocessed/sub-*/{session}/`
    matches it. Idempotent: dirs already under a `sub-*/` prefix are skipped.
    """
    cont_root = preproc_out / _PREPROC_SUBDIR
    if not cont_root.is_dir():
        return
    session_to_sub: Dict[str, List[str]] = {}
    for sub_dir in cont_root.iterdir():
        if not sub_dir.is_dir() or not sub_dir.name.startswith("sub-"):
            continue
        for ses_dir in sub_dir.iterdir():
            if ses_dir.is_dir() and ses_dir.name.startswith("ses-"):
                session_to_sub.setdefault(ses_dir.name, []).append(sub_dir.name)

    for half in ("figures", "QC_out"):
        half_root = preproc_out / half
        if not half_root.is_dir():
            continue
        for child in list(half_root.iterdir()):
            if not child.is_dir():
                continue
            # Already correctly nested as sub-XXX/ — skip.
            if child.name.startswith("sub-"):
                continue
            # Legacy direct ses-XXX/ at half root — promote when unambiguous.
            if child.name.startswith("ses-"):
                subs = session_to_sub.get(child.name, [])
                if len(subs) != 1:
                    continue
                dest = half_root / subs[0] / child.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                if dest.exists():
                    for f in child.rglob("*"):
                        if f.is_file():
                            _move_safe(f, dest / f.relative_to(child))
                    try:
                        shutil.rmtree(str(child))
                    except OSError:
                        pass
                else:
                    try:
                        child.rename(dest)
                    except OSError:
                        # Fall back to copy+remove on cross-device.
                        for f in child.rglob("*"):
                            if f.is_file():
                                _move_safe(f, dest / f.relative_to(child))
                        try:
                            shutil.rmtree(str(child))
                        except OSError:
                            pass


def _migrate_figures_to_toplevel(src_figures: Path, dest_figures_root: Path) -> None:
    """Move figures from a subject's figures/ dir to preprocessed_output/figures/sub-{id}/{session}/."""
    dest_figures_root.mkdir(parents=True, exist_ok=True)
    # If src has session subdirs (ses-*), preserve them
    has_session_subdirs = any(
        d.is_dir() and d.name.startswith("ses-") for d in src_figures.iterdir()
    ) if src_figures.exists() else False

    if has_session_subdirs:
        for ses_dir in src_figures.iterdir():
            if ses_dir.is_dir() and ses_dir.name.startswith("ses-"):
                dest = dest_figures_root / ses_dir.name
                dest.mkdir(parents=True, exist_ok=True)
                for f in ses_dir.rglob("*"):
                    if f.is_file():
                        _move_safe(f, dest / f.name)
    else:
        # Flat figures dir — move all files into a generic location
        for f in src_figures.rglob("*"):
            if f.is_file():
                _move_safe(f, dest_figures_root / f.name)


def _list_output_hierarchy(out: Path) -> Dict[str, List[str]]:
    """List subject/session hierarchy across both preprocessed_output/ halves.

    Returns dict mapping subject_id → list of session_id directories. The
    continuous side lives under ``preprocessed_output/preprocessed/sub-{id}/``
    (BIDS-style); the epoched side lives under
    ``preprocessed_output/AI_ready/{id}/``. Subjects appearing in only one
    half still surface; sessions are unioned. The legacy flat layout
    (``preprocessed_output/{id}/``) is also picked up for back-compat with
    pre-rename runs.
    """
    preproc = out / "preprocessed_output"
    if not preproc.exists():
        return {}

    found: Dict[str, set] = {}

    cont_root = preproc / _PREPROC_SUBDIR
    if cont_root.is_dir():
        for sub_dir in cont_root.iterdir():
            if not sub_dir.is_dir() or not sub_dir.name.startswith("sub-"):
                continue
            sid = sub_dir.name[len("sub-"):]
            ses_set = found.setdefault(sid, set())
            ses_set.update(
                d.name for d in sub_dir.iterdir()
                if d.is_dir() and d.name.startswith("ses-")
            )

    ai_root = preproc / _AIREADY_SUBDIR
    if ai_root.is_dir():
        for sub_dir in ai_root.iterdir():
            if not sub_dir.is_dir():
                continue
            ses_set = found.setdefault(sub_dir.name, set())
            ses_set.update(
                d.name for d in sub_dir.iterdir()
                if d.is_dir() and d.name.startswith("ses-")
            )

    # Legacy flat layout — pre-rename runs may still drop {subject}/{session}/
    # at the top of preprocessed_output/. Surface them so back-compat tooling
    # can still find old runs.
    for sub_dir in preproc.iterdir():
        if not sub_dir.is_dir() or sub_dir.name in _NON_SUBJECT_TOPLEVEL:
            continue
        sid = sub_dir.name[len("sub-"):] if sub_dir.name.startswith("sub-") else sub_dir.name
        ses_dirs = sorted(
            d.name for d in sub_dir.iterdir()
            if d.is_dir() and d.name.startswith("ses-")
        )
        ses_set = found.setdefault(sid, set())
        if ses_dirs:
            ses_set.update(ses_dirs)
        elif not ses_set:
            ses_set.add("ses-001")  # legacy flat — synthetic id

    return {sid: sorted(seslist) for sid, seslist in found.items() if seslist}


def _move_safe(src: Path, dest: Path) -> None:
    """Move file/dir safely, skipping if source doesn't exist."""
    if not src.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if src.is_file() and dest.is_file():
            return  # skip duplicate
        elif src.is_dir() and dest.is_dir():
            return  # skip duplicate dir
    try:
        shutil.move(str(src), str(dest))
    except (OSError, shutil.Error):
        pass


# ---------------------------------------------------------------------------
# Cleanup duplicates
# ---------------------------------------------------------------------------

def _cleanup_duplicate_outputs(out: Path) -> None:
    """Remove duplicate processed/continuous files when {stem}_preprocessed exists,
    and remove unlabeled `*_epochs.{pkl,h5,npz,mat}` when a `*_labeled.{ext}`
    sibling lands in the same directory.

    The labeled-supersedes-unlabeled rule covers the common LLM workflow where
    an exploratory `save_processed` call dumps a sliding-window slice as
    ``{stem}_action_epochs.pkl`` (often hundreds of MB) and a later script
    overrides it with the trigger-aligned ``action_epochs_labeled.pkl``. The
    unlabeled file is then redundant — we drop it (plus any ``.meta.json``
    sidecar) so AI_ready/ contains only the model-ready labeled epochs.
    Conservative: only fires inside dirs that already host a labeled artifact.
    """
    _stale_prefixes = ("processed", "continuous")
    _data_suffixes = (".pkl", ".h5", ".hdf5", ".npz", ".mat", ".nwb")

    def _clean_dir(directory: Path) -> None:
        if not directory.is_dir():
            return
        has_preprocessed = any(
            f.name.endswith("_preprocessed.pkl") or
            f.name.endswith("_preprocessed.h5") or
            f.name.endswith("_preprocessed.npz") or
            f.name.endswith("_preprocessed.mat") or
            f.name.endswith("_preprocessed.nwb")
            for f in directory.iterdir() if f.is_file()
        )
        if has_preprocessed:
            for f in list(directory.iterdir()):
                if not f.is_file():
                    continue
                if f.stem in _stale_prefixes:
                    try:
                        f.unlink()
                    except OSError:
                        pass
                if f.suffix == ".json" and any(
                    f.stem == f"{p}.meta" for p in _stale_prefixes
                ):
                    try:
                        f.unlink()
                    except OSError:
                        pass

        # Labeled epochs supersede plain `_epochs` siblings in the same dir.
        labeled_present = any(
            f.is_file()
            and f.suffix.lower() in _data_suffixes
            and f.stem.endswith("_labeled")
            for f in directory.iterdir()
        )
        if not labeled_present:
            return
        for f in list(directory.iterdir()):
            if not f.is_file():
                continue
            suffix = f.suffix.lower()
            if suffix in _data_suffixes and f.stem.endswith("_epochs") \
                    and not f.stem.endswith("_labeled"):
                try:
                    f.unlink()
                except OSError:
                    continue
                # Drop the JSON sidecar alongside the data file.
                sidecar = f.with_suffix(".meta.json")
                if sidecar.exists():
                    try:
                        sidecar.unlink()
                    except OSError:
                        pass

    # Walk every subject/session under both halves of preprocessed_output/.
    preproc = out / "preprocessed_output"
    if preproc.exists():
        for half in (_PREPROC_SUBDIR, _AIREADY_SUBDIR):
            half_root = preproc / half
            if not half_root.is_dir():
                continue
            for sub_dir in half_root.iterdir():
                if not sub_dir.is_dir():
                    continue
                for ses_dir in sub_dir.iterdir():
                    if ses_dir.is_dir():
                        _clean_dir(ses_dir)
                _clean_dir(sub_dir)
        # Back-compat: legacy flat layout (pre-rename runs).
        for sub_dir in preproc.iterdir():
            if not sub_dir.is_dir() or sub_dir.name in _NON_SUBJECT_TOPLEVEL:
                continue
            for ses_dir in sub_dir.iterdir():
                if ses_dir.is_dir():
                    _clean_dir(ses_dir)
            _clean_dir(sub_dir)

    # Legacy: also check {out}/output/ (single-file mode)
    _clean_dir(out / "output")


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------

def _write(path: Path, content: str) -> None:
    """Write text content to a file, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _copy_output(dest_dir: Path, pkl_path: str, output_stem: str = "processed") -> None:
    """Move processed output into destination directory.

    Skips if the file already exists at the destination (idempotent).
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    src = Path(pkl_path)
    dest = dest_dir / f"{output_stem}{src.suffix}"
    if src.resolve() == dest.resolve():
        return
    if dest.exists():
        return
    try:
        shutil.move(str(src), str(dest))
    except OSError:
        shutil.copy2(str(src), str(dest))
        try:
            src.unlink()
        except OSError:
            pass
    # Also move the .meta.json sidecar if present
    meta_src = src.with_suffix(".meta.json")
    if meta_src.exists():
        meta_dest = dest.with_suffix(".meta.json")
        if not meta_dest.exists():
            try:
                shutil.move(str(meta_src), str(meta_dest))
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Export manifest — idempotency mechanism
# ---------------------------------------------------------------------------

_MANIFEST_NAME = ".export_manifest.json"


def _collect_source_mtimes(pkl_path: str, input_path: str) -> Dict[str, float]:
    """Collect mtime of relevant source files for manifest comparison."""
    mtimes: Dict[str, float] = {}
    for p in (pkl_path, input_path):
        if p:
            path = Path(p)
            if path.exists():
                try:
                    mtimes[str(path)] = path.stat().st_mtime
                except OSError:
                    pass
    return mtimes


def _write_export_manifest(
    out: Path, pkl_path: str, input_path: str, created_files: List[str]
) -> None:
    """Write manifest into middle_process/ for idempotent re-export detection."""
    manifest = {
        "exported_at": time.time(),
        "source_mtimes": _collect_source_mtimes(pkl_path, input_path),
        "n_files": len(created_files),
        "files": created_files,
    }
    middle = out / "middle_process"
    middle.mkdir(parents=True, exist_ok=True)
    manifest_path = middle / _MANIFEST_NAME
    try:
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except OSError:
        logger.debug("Failed to write export manifest")


def _check_export_manifest(
    out: Path, pkl_path: str, input_path: str
) -> Optional[Dict[str, Any]]:
    """Check if a valid export manifest exists and sources are unchanged."""
    # Check both locations (new: middle_process/, legacy: root)
    for manifest_path in (out / "middle_process" / _MANIFEST_NAME, out / _MANIFEST_NAME):
        if not manifest_path.exists():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        saved_mtimes = manifest.get("source_mtimes", {})
        current_mtimes = _collect_source_mtimes(pkl_path, input_path)

        if not saved_mtimes and not current_mtimes:
            continue

        valid = True
        for path_str, saved_mtime in saved_mtimes.items():
            current = current_mtimes.get(path_str)
            if current is None or abs(current - saved_mtime) > 0.01:
                valid = False
                break

        if valid:
            return {
                "success": True,
                "output_dir": str(out),
                "files": manifest.get("files", []),
                "n_files": manifest.get("n_files", 0),
                "cached": True,
            }

    return None


# ---------------------------------------------------------------------------
# README generation — comprehensive explainability
# ---------------------------------------------------------------------------

def _removed_channels_for_step(step_states: List[Dict[str, Any]], idx: int) -> List[str]:
    """Channel names present before step ``idx`` but absent after it.

    Returns [] when the snapshot lacks channel names or no channels were dropped.
    Used to annotate the README pipeline table and keep it in sync with the
    name-aligned before/after figure.
    """
    if idx < 0 or idx >= len(step_states):
        return []
    entry = step_states[idx] or {}
    before = (entry.get("before") or {}).get("channels")
    after = (entry.get("after") or {}).get("channels")
    if not before or not after:
        return []
    after_set = set(after)
    return [ch for ch in before if ch not in after_set]


def _write_readme(
    out: Path,
    steps: List[str],
    data_info: Dict[str, Any],
    pipeline_record: Dict[str, Any],
    modality: str,
    paradigm: str,
    input_path: str,
    split_config: Optional[Dict[str, Any]] = None,
    output_hierarchy: Optional[Dict[str, List[str]]] = None,
    status: str = "ok",
    partial_reason: Optional[str] = None,
    pending_consolidations: Optional[List[str]] = None,
) -> None:
    """Generate a comprehensive README.md — the human entry point for the mini-repo.

    This README serves as the primary explainability artifact. It must be detailed
    enough that a researcher opening this directory for the first time can understand:
    1. What data was processed and why
    2. What each processing step did to the signal
    3. Where to find each artifact and what it means
    4. How to reproduce or modify the pipeline
    """
    filename = Path(input_path).name if input_path else "unknown"
    n_ch = data_info.get("n_channels", "?")
    freq = data_info.get("frequency_hz", data_info.get("frequency", "?"))
    dur = data_info.get("duration_seconds", data_info.get("duration", "?"))
    output_hierarchy = output_hierarchy or {}

    qc = pipeline_record.get("qc_result") or pipeline_record.get("qc")
    qc_passed = qc and qc.get("passed", qc.get("status") == "pass") if qc else False

    # preprocessed/ is NWB-only.
    _preproc_ext = ".nwb"

    lines = [
        "# Neural Data Preprocessing Results",
        "",
        f"> **Modality:** {modality.upper()} | **Paradigm:** {paradigm or 'unspecified'} | "
        f"**Generated:** {time.strftime('%Y-%m-%d %H:%M')}",
        "",
    ]

    # When modality wasn't recorded (auto-finalize on a run that never
    # called propose_pipeline / export_repo), surface a clear manual-review
    # banner instead of letting downstream readers assume EEG.
    if modality == "unknown":
        lines.extend([
            "> **Modality not recorded — manual review needed.** This mini-repo was "
            "produced by the auto-finalize safety net before the agent could record "
            "the recording modality (EEG / MEG / sEEG / spike / ...). Treat the "
            "paradigm-specific commentary below as placeholder; consult `plan/` "
            "and the source data before reusing this pipeline.",
            "",
        ])

    # Partial-build banner: prepended so it's the first thing a reader sees.
    if status != "ok":
        banner_status = status
        reason_line = f" — reason: `{partial_reason}`" if partial_reason else ""
        lines.extend([
            f"> ⚠️ **Status: {banner_status}**{reason_line}",
            ">",
            "> This mini-repo was produced by the auto-finalize safety net "
            "after the run was interrupted, failed, or migrated from an old layout. "
            "Intermediate debug artifacts have **not** been consolidated into "
            "`middle_process/` — review them, then run `easybci finalize "
            f"{out} --status ok` to upgrade this build.",
            "",
        ])
        if pending_consolidations:
            lines.append("> Files that *would* have been moved to `middle_process/` on a clean build:")
            lines.append(">")
            for item in pending_consolidations[:20]:
                lines.append(f"> - `{item}`")
            if len(pending_consolidations) > 20:
                lines.append(f"> - ... and {len(pending_consolidations) - 20} more")
            lines.append("")

    # --- 1. Overview ---
    lines.extend([
        "## 1. Overview",
        "",
        "This directory contains the complete output of an automated neural data "
        "preprocessing pipeline. The pipeline was designed, configured, and executed "
        "by EasyBCI-Data Agent with full traceability from raw input to AI-ready output.",
        "",
    ])

    # --- 2. Data Summary ---
    lines.extend([
        "## 2. Input Data",
        "",
        "| Property | Value |",
        "|----------|-------|",
        f"| Source file | `{filename}` |",
        f"| Channels | {n_ch} |",
        f"| Sampling rate | {freq} Hz |",
        f"| Duration | {dur} seconds |",
        f"| Modality | {modality} |",
        f"| Paradigm | {paradigm or 'unspecified'} |",
    ])
    n_subjects = len(output_hierarchy)
    n_sessions = sum(len(ses) for ses in output_hierarchy.values())
    if n_subjects > 0:
        lines.append(f"| Subjects | {n_subjects} |")
        lines.append(f"| Sessions | {n_sessions} |")
    lines.append("")

    # --- 3. Pipeline Steps (detailed) ---
    lines.extend([
        "## 3. Processing Pipeline",
        "",
        "The following steps were applied in sequence. Each step transforms the signal "
        "in a specific way — see `plan/reasoning.md` for the full scientific rationale "
        "behind each decision.",
        "",
        "| # | Step | Effect |",
        "|---|------|--------|",
    ])

    step_effects = {
        "notch": "Suppresses power line interference ({param} Hz) and harmonics",
        "bandpass": "Retains only the {param} Hz frequency band, removing drift and HF noise",
        "resample": "Downsamples to {param} Hz (Nyquist-safe), reducing data volume",
        "car": "Subtracts common average reference, removing spatially global noise",
        "ica": "Removes artifact components (eye blinks, cardiac) via blind source separation",
        "scale": "Normalizes amplitude across channels ({param} scaling)",
        "clip": "Clamps extreme values beyond ±{param} to prevent outlier influence",
        "fill_nan": "Replaces non-finite values to ensure numerical stability",
        "pick_channels": "Selects relevant channel subset, discards non-neural sensors",
        "drop_bads": "Removes channels flagged as bad (flat/noisy/disconnected)",
        "interpolate_bads": "Reconstructs bad channels from spatial neighbors (spline)",
        "bipolar_ref": "Computes bipolar derivations between adjacent depth contacts",
        "hilbert": "Extracts instantaneous envelope (analytic signal amplitude)",
    }

    step_states = pipeline_record.get("step_states") or []

    for i, step in enumerate(steps, 1):
        step_name = step.split(":")[0]
        param = step.split(":", 1)[1] if ":" in step else ""
        effect_template = step_effects.get(step_name, "Applies {step_name} processing")
        effect = effect_template.format(param=param, step_name=step_name)
        # Annotate the step that actually removed channels with their names so
        # the table reflects the drop visible in timeseries_before_after.png.
        removed_here = _removed_channels_for_step(step_states, i - 1)
        if removed_here:
            preview = ", ".join(removed_here[:6])
            if len(removed_here) > 6:
                preview += f", … (+{len(removed_here) - 6})"
            effect += f" — removed {len(removed_here)} channel(s): {preview}"
        lines.append(f"| {i} | `{step}` | {effect} |")
    lines.append("")

    # --- 4. Before/After summary ---
    if step_states:
        first_before = step_states[0].get("before", {}) if step_states else {}
        last_after = step_states[-1].get("after", {}) if step_states else {}
        lines.extend([
            "## 4. Signal Transformation Summary",
            "",
            "| Metric | Before (raw) | After (processed) |",
            "|--------|-------------|-------------------|",
        ])
        if first_before.get("n_channels") or last_after.get("n_channels"):
            lines.append(f"| Channels | {first_before.get('n_channels', '?')} | {last_after.get('n_channels', '?')} |")
        if first_before.get("frequency") or last_after.get("frequency"):
            lines.append(f"| Sampling rate | {first_before.get('frequency', '?')} Hz | {last_after.get('frequency', '?')} Hz |")
        if first_before.get("n_samples") or last_after.get("n_samples"):
            lines.append(f"| Samples | {first_before.get('n_samples', '?')} | {last_after.get('n_samples', '?')} |")
        b_range = first_before.get("value_range")
        a_range = last_after.get("value_range")
        if b_range:
            lines.append(f"| Amplitude range | [{b_range[0]:.4f}, {b_range[1]:.4f}] | [{a_range[0]:.4f}, {a_range[1]:.4f}] |" if a_range else f"| Amplitude range | [{b_range[0]:.4f}, {b_range[1]:.4f}] | ? |")
        b_std = first_before.get("mean_std", [None, None])
        a_std = last_after.get("mean_std", [None, None])
        if b_std[1] and a_std[1]:
            reduction = (1 - a_std[1] / b_std[1]) * 100 if b_std[1] > 0 else 0
            lines.append(f"| Std deviation | {b_std[1]:.4f} | {a_std[1]:.4f} ({reduction:+.1f}%) |")
        lines.append("")

    # --- 4b. Storage Footprint (raw inputs vs preprocessed output) ---
    _fp = (pipeline_record or {}).get("storage_footprint") or {}
    if _fp.get("raw_size_bytes") or _fp.get("output_size_bytes"):
        lines.extend([
            "## 4b. Storage Footprint",
            "",
            "| | Size |",
            "|--|--|",
            f"| Raw input | {_fp.get('raw_size_human', '?')} |",
            f"| Preprocessed output | {_fp.get('output_size_human', '?')} |",
        ])
        if _fp.get("reduction_pct") is not None:
            lines.append(f"| Reduction | {_fp['reduction_pct']:.1f}% |")
        lines.append("")

    # --- 5. QC Status ---
    lines.extend([
        "## 5. Quality Check",
        "",
        f"**Status: {'PASS ✓' if qc_passed else 'PENDING / WARNING'}**",
        "",
    ])
    if qc:
        issues = qc.get("issues") or []
        if issues:
            lines.append("Issues detected:")
            for issue in issues:
                lines.append(f"- {issue}")
            lines.append("")
        metrics = qc.get("metrics") or {}
        if metrics:
            lines.append("Key metrics:")
            for k, v in metrics.items():
                lines.append(f"- {k}: {v}")
            lines.append("")

    # --- 6. Directory Structure ---
    lines.extend([
        "## 6. Directory Structure",
        "",
        "```",
        f"{out.name}/",
        "├── README.md                          ← You are here",
        "├── preprocessed_output/               ← Run artefacts (split by readiness)",
        "│   ├── preprocessed/                  ⏵ Continuous post-pipeline signal (BIDS-friendly)",
    ])
    if output_hierarchy:
        for sub, sessions in list(output_hierarchy.items())[:3]:
            lines.append(f"│   │   └── sub-{sub}/")
            for ses in sessions[:3]:
                lines.append(f"│   │       └── {ses}/")
                lines.append(f"│   │           └── {{stem}}_preprocessed{_preproc_ext}")
            if len(sessions) > 3:
                lines.append(f"│   │       └── ... ({len(sessions) - 3} more sessions)")
        if len(output_hierarchy) > 3:
            lines.append(f"│   │   └── ... ({len(output_hierarchy) - 3} more subjects)")
        lines.append("│   ├── AI_ready/                      ⏵ Epoched / segmented data — feed straight to model")
        for sub, sessions in list(output_hierarchy.items())[:3]:
            lines.append(f"│   │   └── {sub}/")
            for ses in sessions[:3]:
                lines.append(f"│   │       └── {ses}/")
                lines.append(f"│   │           └── *_epochs.pkl")
            if len(sessions) > 3:
                lines.append(f"│   │       └── ... ({len(sessions) - 3} more sessions)")
        lines.append("│   ├── figures/                       ⏵ QC visualisations")
        for sub, sessions in list(output_hierarchy.items())[:1]:
            sub_label = sub if sub.startswith("sub-") else f"sub-{sub}"
            lines.append(f"│   │   └── {sub_label}/")
            for ses in sessions[:2]:
                lines.append(f"│   │       └── {ses}/")
                lines.append(f"│   │           ├── {{stem}}_timeseries_before_after.png")
                lines.append(f"│   │           ├── {{stem}}_psd.png")
                lines.append(f"│   │           └── {{stem}}_variance.png")
        lines.append("│   └── QC_out/                        ⏵ QC report (.json + .md)")
        for sub, sessions in list(output_hierarchy.items())[:1]:
            sub_label = sub if sub.startswith("sub-") else f"sub-{sub}"
            lines.append(f"│       └── {sub_label}/")
            for ses in sessions[:2]:
                lines.append(f"│           └── {ses}/")
                lines.append(f"│               ├── qc_report_{ses}.md")
                lines.append(f"│               └── qc_report_{ses}.json")
    else:
        lines.append("│   │   └── sub-{subject_id}/{session_id}/")
        lines.append(f"│   │       └── {{stem}}_preprocessed{_preproc_ext}")
        lines.append("│   ├── AI_ready/                      ⏵ Epoched / segmented data — feed straight to model")
        lines.append("│   │   └── {subject_id}/{session_id}/")
        lines.append("│   │       └── *_epochs.pkl")
        lines.append("│   ├── figures/sub-{subject_id}/{session_id}/  ⏵ QC visualisations")
        lines.append("│   └── QC_out/sub-{subject_id}/{session_id}/   ⏵ QC report (.json + .md)")
    lines.extend([
        "├── code/                              ← Reproducible pipeline scripts",
        "│   ├── pipeline.py                    Complete preprocessing code",
        "│   ├── run.py                         One-click execution wrapper",
        "│   └── requirements.txt              Python dependencies",
        "└── plan/                              ← Configuration + explainability",
        "    ├── config.yaml                    Pipeline parameters",
        "    ├── reasoning.md                   Per-step scientific rationale",
        "    ├── pipeline_record.json           Execution log + timing",
        "    └── input_ref.json                 Source data fingerprint",
        "```",
        "",
    ])

    # --- 7. Plan explanation (key explainability section) ---
    lines.extend([
        "## 7. Plan & Explainability (plan/)",
        "",
        "The `plan/` directory is the **explainability core** of this pipeline. "
        "Each file serves a specific documentation purpose:",
        "",
        "### plan/config.yaml",
        "",
        "The complete pipeline configuration in YAML format. Specifies:",
        "- Input file path and detected modality",
        "- Ordered list of processing steps with parameters",
        "- Segmentation settings (window size, stride)",
        "- Output path and format",
        "",
        "This file can be passed directly to `code/run.py --config plan/config.yaml` "
        "to reproduce the exact pipeline.",
        "",
        "### plan/reasoning.md",
        "",
        "**The scientific justification for every processing decision.** For each step:",
        "",
        "- **Input:** What the data looked like before this step (channels, samples, amplitude stats)",
        "- **Why:** Three-part rationale: (1) what was observed in the data, "
        "(2) why this method addresses it, (3) what the implementation specifically does",
        "- **Output:** Measurable changes after the step (amplitude reduction, channel count change, etc.)",
        "",
        "This is where reviewers should look to understand whether the pipeline "
        "decisions are scientifically sound for the given paradigm and data characteristics.",
        "",
        "### plan/pipeline_record.json",
        "",
        "Machine-readable execution metadata:",
        "- Per-step timing information",
        "- Before/after data state snapshots (shapes, statistics)",
        "- QC results and metrics",
        "- Error recovery attempts (if any)",
        "",
        "### plan/input_ref.json",
        "",
        "Source data traceability:",
        "- Original file path and filename",
        "- File size and modification timestamp",
        "- SHA-256 hash (first 1MB) for integrity verification",
        "",
    ])

    # --- 8. Code explanation ---
    lines.extend([
        "## 8. Reproducible Code (code/)",
        "",
        "The `code/` directory contains standalone Python scripts that reproduce "
        "this exact preprocessing pipeline without requiring EasyBCI:",
        "",
        "```bash",
        "cd code/",
        "pip install -r requirements.txt",
        "python run.py --config ../plan/config.yaml",
        "```",
        "",
        "- `pipeline.py` — The complete preprocessing logic: data loading, "
        "filtering, artifact removal, and output saving",
        "- `run.py` — CLI wrapper that reads config.yaml and executes pipeline.py",
        "- `requirements.txt` — Exact Python package versions needed",
    ])
    if split_config:
        split_method = split_config.get("method", "random")
        split_ratios = split_config.get("ratios", {})
        ratios_str = ", ".join(f"{k}={v}" for k, v in split_ratios.items())
        lines.extend([
            f"- `split.py` — Data splitting script (method: {split_method}, ratios: {ratios_str})",
        ])
        warnings = split_config.get("warnings", [])
        if warnings:
            lines.append("")
            lines.append("**Split warnings:**")
            for w in warnings:
                lines.append(f"- {w}")
    lines.append("")

    # --- 9. Output explanation ---
    lines.extend([
        "## 9. Processed Output (preprocessed_output/)",
        "",
        "Organized into three directories:",
        "",
        "### {subject_id}/{session_id}/ — AI-Ready Data",
        "",
        f"- **`{{stem}}_preprocessed{_preproc_ext}`** — The AI-ready preprocessed data in "
        + ("NWB (Neurodata Without Borders) format. Self-describing: channel "
           "names/types, sampling rate and shape live in the electrode table + "
           "ElectricalSeries; the full processing provenance (analysis_goal, "
           "modality, ordered steps, dropped_channels) is embedded as a JSON "
           "`easybci_provenance` scratch entry inside the file — no external "
           "sidecar needed."
           if _preproc_ext == ".nwb"
           else "pickle format. Structure: `{\"data\": {\"neural\": ndarray}, \"labels\": {...}, \"meta\": {...}}`"),
        "",
        "### figures/sub-{subject_id}/{session_id}/ — Visual Quality Evidence",
        "",
        "- **`{stem}_timeseries_before_after.png`** — Before/after time-domain comparison",
        "- **`{stem}_psd.png`** — Power spectral density of processed signal",
        "- **`{stem}_variance.png`** — Per-channel variance (outlier detection)",
        "- **`{stem}_amplitude.png`** — Amplitude distribution histogram",
        "- **`{stem}_timeseries.png`** — Processed signal preview",
        "",
        "### QC_out/sub-{subject_id}/{session_id}/ — Quality Control Reports",
        "",
        "- **`qc_report_{session}.md`** — Human-readable quality report with metrics, "
        "grades, per-step transitions, and remediation recommendations",
        "- **`qc_report_{session}.json`** — Machine-readable QC data for automated workflows",
    ])
    if qc:
        lines.append("- **`QC_out/{session}/`** — Automated quality check results with detailed reports")
    lines.append("")

    # --- 10. Figures explanation ---
    lines.extend([
        "## 10. Visualization",
        "",
        "Each subject's `figures/comparison/` directory contains before/after plots:",
        "",
        "| Figure | What it proves |",
        "|--------|----------------|",
        "| `timeseries_before_after.png` | Raw vs. processed time-domain signal (amplitude, noise level) |",
        "",
        "These figures provide immediate visual confirmation that:",
        "1. Noise was reduced (smaller amplitude variance in processed signal)",
        "2. Signal morphology was preserved (waveform shape intact)",
        "3. No over-filtering occurred (signal not flattened)",
        "",
    ])

    _write(out / "README.md", "\n".join(lines))


# ---------------------------------------------------------------------------
# reasoning.md generation
# ---------------------------------------------------------------------------

def _is_propose_authored_reasoning(path: Path) -> bool:
    """True when reasoning.md was written by propose_pipeline's evidence renderer.

    propose's ``render_full_reasoning`` emits ``## Step N — <operator>`` headers
    plus ``**Rationale**:`` / ``### Parameter evidence`` blocks. The finalize-time
    fallback (``_write_reasoning_md``) uses ``### Step N:`` / ``**Why:**`` instead.
    We fingerprint on the propose-only markers so we never overwrite the real,
    user-facing reasoning with the husk (replaces the byte-size heuristic).
    """
    if not path.exists():
        return False
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return False
    # A degraded propose stub ("# Reasoning (degraded; renderer failed)") is NOT
    # authoritative — let the fallback replace it with real content.
    if "degraded; renderer failed" in text:
        return False
    return "**Rationale**:" in text or "### Parameter evidence" in text


def _recover_reasoning_inputs(
    plan_dir: Path,
    steps: List[str],
    data_info: Dict[str, Any],
) -> tuple:
    """Backfill empty steps / data_info from plan/proposal.json for the fallback.

    The finalize path often calls build_mini_repo with empty ``steps`` and
    ``data_info`` (the husk root cause). When that happens we read the steps the
    proposal recorded so even the fallback reasoning.md lists real operators
    rather than an empty "Pipeline Steps" heading.
    """
    if steps and data_info:
        return steps, data_info

    recovered_steps = list(steps) if steps else []
    recovered_info = dict(data_info) if data_info else {}

    proposal_path = plan_dir / "proposal.json"
    if not proposal_path.exists():
        return recovered_steps, recovered_info
    try:
        with open(proposal_path, "r", encoding="utf-8") as f:
            proposal = json.load(f)
    except (OSError, json.JSONDecodeError):
        return recovered_steps, recovered_info

    if not recovered_steps:
        for s in proposal.get("steps") or []:
            operator = s.get("operator", "")
            params = s.get("params") or {}
            if not operator:
                continue
            # Re-encode as "operator:k=v;k=v" to match the string-step shape
            # _write_reasoning_md expects (it splits on ":" for the name).
            if params:
                pstr = ";".join(f"{k}={v}" for k, v in params.items())
                recovered_steps.append(f"{operator}:{pstr}")
            else:
                recovered_steps.append(operator)

    return recovered_steps, recovered_info


def _write_reasoning_md(
    out: Path,
    steps: List[str],
    data_info: Dict[str, Any],
    pipeline_record: Dict[str, Any],
    paradigm: str,
    modality: str,
    analysis_goal: str = "generic",
    web_evidence: Optional[Dict[str, Any]] = None,
) -> None:
    """Generate reasoning.md — natural-language explainability for each pipeline step."""
    from easybci_lib.tools.neural_processing.preprocess.pipeline import STEP_FULL_NAMES

    # analysis_goal banner is the first thing the user sees.
    # Two flavours: known goal vs. generic fallback (with retry hint).
    goal_key = (analysis_goal or "generic").strip() or "generic"
    if goal_key == "generic":
        goal_banner = (
            "**Analysis goal:** generic (no explicit target — using safe defaults). "
            "If you later need classification or source localization, re-run "
            "with the specific goal to get better-tuned parameters."
        )
    else:
        goal_banner = f"**Analysis goal:** {goal_key} (inferred by user/LLM)"
    try:
        from easybci_agent.i18n import t as _t
        i18n_key = (
            "preprocessing.banner.analysis_goal.generic"
            if goal_key == "generic"
            else "preprocessing.banner.analysis_goal.known"
        )
        translated = _t(i18n_key, goal=goal_key)
        if translated and translated != i18n_key:
            goal_banner = f"**{translated}**"
    except Exception:
        pass

    # Data-only-cleanup banner: when the pipeline contains a data_only cleanup
    # (typically appended by codegen._enforce_clean_output after ICA), call
    # it out so users understand why VEOG/HEOG/Trigger don't appear in the
    # final output. This banner is goal-conditional and sits alongside the
    # analysis_goal banner above.
    _has_data_only = any(
        isinstance(s, str) and s.split(":", 1)[0].strip() == "drop_nondata_channels"
        and (s.split(":", 1)[1].strip() if ":" in s else "") == "data_only"
        for s in steps
    )
    _has_ica = any(
        isinstance(s, str) and s.split(":", 1)[0].strip() == "ica" for s in steps
    )

    lines = [
        "# Pipeline Reasoning",
        "",
        f"Generated: {time.strftime('%Y-%m-%dT%H:%M:%S')}",
        "",
        f"> {goal_banner}",
        "",
    ]

    # web_evidence banner immediately after the goal banner.
    if isinstance(web_evidence, dict) and web_evidence:
        if web_evidence.get("status") == "ok" and web_evidence.get("recommendations"):
            provider = web_evidence.get("provider") or "unknown provider"
            confidence = web_evidence.get("confidence")
            applied = web_evidence.get("applied_to_steps") or []
            applied_txt = (
                f" applied to {', '.join(applied)}" if applied else ""
            )
            conf_txt = (
                f" · confidence {float(confidence):.2f}"
                if isinstance(confidence, (int, float))
                else ""
            )
            lines.append(
                f"> **Web evidence:** queried {provider} for SOTA preprocessing"
                f"{conf_txt}{applied_txt}. See `plan/proposal.json:web_evidence`."
            )
        else:
            reason = web_evidence.get("reason", "unknown")
            lines.append(
                f"> **Web evidence:** unavailable ({reason}) — using "
                "domain-skill defaults for this run. Install/configure a "
                "web search provider to get SOTA parameter recommendations."
            )
        lines.append("")

    if _has_data_only:
        try:
            from easybci_agent.i18n import t as _t
            note = _t("preprocessing.output_cleanup_appended")
            if note == "preprocessing.output_cleanup_appended":
                # Catalog miss — fall back to English so the user doesn't
                # see a bare key in their reasoning.md.
                raise KeyError
        except Exception:
            if _has_ica:
                note = (
                    "Output cleanup: appended `drop_nondata_channels:data_only` "
                    "after ICA to ensure final channels exclude EOG/marker/physio."
                )
            else:
                note = (
                    "Output cleanup: `drop_nondata_channels:data_only` is "
                    "applied to ensure final channels exclude EOG/marker/physio."
                )
        lines.append(f"> **{note}**")
        lines.append("")
    elif goal_key in ("source_localization", "exploratory"):
        lines.append(
            f"> **Output cleanup skipped (analysis_goal={goal_key}):** "
            f"physiological channels (EOG/ECG) are intentionally retained "
            f"so downstream source modelling / exploratory feature work has "
            f"the full montage available."
        )
        lines.append("")
    lines.extend([
        "## Data Fingerprint",
        "",
        f"- **File**: {data_info.get('file', 'unknown')}",
        f"- **Modality**: {modality}",
        f"- **Paradigm**: {paradigm or 'unspecified'}",
        f"- **Channels**: {data_info.get('n_channels', '?')}",
        f"- **Sampling Rate**: {data_info.get('frequency_hz', data_info.get('frequency', '?'))} Hz",
        f"- **Duration**: {data_info.get('duration_seconds', data_info.get('duration', '?'))} s",
        "",
        "---",
        "",
        "## Pipeline Steps",
        "",
    ])

    reasoning = pipeline_record.get("reasoning") or {}
    step_states = pipeline_record.get("step_states") or []

    for i, step in enumerate(steps, 1):
        step_name = step.split(":")[0]
        full_name = STEP_FULL_NAMES.get(step_name, step_name)
        reason = reasoning.get(step) or reasoning.get(step_name, "")
        state_entry = step_states[i - 1] if i - 1 < len(step_states) else None

        lines.append(f"### Step {i}: {full_name} — `{step}`")
        lines.append("")
        lines.append(f"**Input:** {_describe_state(state_entry, 'before', i, steps)}")
        lines.append("")
        why_text = reason if reason and len(reason) >= 80 else _default_rationale(step_name, step, paradigm, modality)
        lines.append(f"**Why:** {why_text}")
        lines.append("")
        lines.append(f"**Output:** {_describe_output(state_entry, step_name)}")
        lines.append("")
        lines.append("---")
        lines.append("")

    # QC summary
    qc = pipeline_record.get("qc_result") or pipeline_record.get("qc")
    if qc:
        lines.append("## Quality Check Summary")
        lines.append("")
        passed = qc.get("passed", qc.get("status") == "pass")
        lines.append(f"- **Status**: {'PASS' if passed else 'FAIL/WARNING'}")
        issues = qc.get("issues") or []
        if issues:
            for issue in issues:
                lines.append(f"- {issue}")
        lines.append("")

    timing = pipeline_record.get("timing")
    if timing:
        lines.append("## Execution Timing")
        lines.append("")
        if isinstance(timing, dict):
            for key, val in timing.items():
                lines.append(f"- **{key}**: {val}")
        else:
            lines.append(f"- Total: {timing}")
        lines.append("")

    _write(out / "reasoning.md", "\n".join(lines))


# ---------------------------------------------------------------------------
# State description helpers
# ---------------------------------------------------------------------------

def _describe_state(state_entry: Optional[Dict], phase: str, step_idx: int, steps: List[str]) -> str:
    """Describe data state in natural language."""
    if not state_entry:
        if phase == "before" and step_idx == 1:
            return "Raw data as loaded from file, no processing applied yet."
        elif phase == "before":
            return f"Data after completing step {step_idx - 1} (`{steps[step_idx - 2]}`)."
        return "State information not available."

    state = state_entry.get(phase, {})
    n_ch = state.get("n_channels", 0)
    n_samp = state.get("n_samples", 0)
    freq = state.get("frequency", 0)
    dur = state.get("duration_s", 0)
    val_range = state.get("value_range")
    mean_std = state.get("mean_std")

    parts = []
    if n_ch and freq:
        parts.append(f"{n_ch} channels at {freq} Hz")
    if n_samp and dur:
        parts.append(f"{n_samp} samples ({dur}s)")
    if val_range:
        parts.append(f"amplitude range [{val_range[0]:.4f}, {val_range[1]:.4f}]")
    if mean_std:
        parts.append(f"mean={mean_std[0]:.4f}, std={mean_std[1]:.4f}")

    if not parts:
        if phase == "before" and step_idx == 1:
            return "Raw data as loaded from file."
        return "Data from previous step."

    return "; ".join(parts) + "."


def _describe_output(state_entry: Optional[Dict], step_name: str) -> str:
    """Describe the output state and what changed, in natural language."""
    if not state_entry:
        return _output_fallback(step_name)

    before = state_entry.get("before", {})
    after = state_entry.get("after", {})
    changes = []

    if before.get("n_channels") and after.get("n_channels"):
        if before["n_channels"] != after["n_channels"]:
            changes.append(f"channels changed from {before['n_channels']} to {after['n_channels']}")

    if before.get("frequency") and after.get("frequency"):
        if before["frequency"] != after["frequency"]:
            changes.append(f"sampling rate changed from {before['frequency']} Hz to {after['frequency']} Hz")

    if before.get("n_samples") and after.get("n_samples"):
        if before["n_samples"] != after["n_samples"]:
            changes.append(f"sample count changed from {before['n_samples']} to {after['n_samples']}")

    before_std = before.get("mean_std", [None, None])[1]
    after_std = after.get("mean_std", [None, None])[1]
    if before_std and after_std and before_std > 0:
        ratio = after_std / before_std
        if ratio < 0.85:
            changes.append(f"signal amplitude reduced by {(1 - ratio) * 100:.0f}% (noise removal)")
        elif ratio > 1.15:
            changes.append(f"signal amplitude increased by {(ratio - 1) * 100:.0f}%")

    before_range = before.get("value_range")
    after_range = after.get("value_range")
    if before_range and after_range:
        before_span = before_range[1] - before_range[0]
        after_span = after_range[1] - after_range[0]
        if before_span > 0 and after_span / before_span < 0.7:
            changes.append(f"value range narrowed from [{before_range[0]:.3f}, {before_range[1]:.3f}] to [{after_range[0]:.3f}, {after_range[1]:.3f}]")

    if changes:
        return "; ".join(changes) + "."

    n_ch = after.get("n_channels", 0)
    freq = after.get("frequency", 0)
    if n_ch and freq:
        return f"Data shape preserved ({n_ch} channels, {freq} Hz). Spectral content modified in-place."
    return _output_fallback(step_name)


def _output_fallback(step_name: str) -> str:
    """Fallback output description when no state data available."""
    fallbacks = {
        "notch": "Power line frequency and harmonics suppressed; neural signal preserved.",
        "bandpass": "Signal now contains only the target frequency band; drift and high-frequency noise removed.",
        "resample": "Data resampled to target frequency; anti-aliasing filter applied.",
        "car": "Common average subtracted; shared noise removed, channel-specific activity enhanced.",
        "ica": "Artifact components identified and removed; neural signal reconstructed.",
        "scale": "Amplitude normalized across channels to uniform scale.",
        "clip": "Extreme outlier values clamped; data distribution tightened.",
        "fill_nan": "Non-finite values replaced; data is now numerically clean.",
        "pick_channels": "Channel subset selected; irrelevant channels discarded.",
        "drop_bads": "Bad channels removed from the data.",
        "interpolate_bads": "Bad channels reconstructed from neighbors; full montage restored.",
        "bipolar_ref": "Bipolar derivations computed; local activity maximized, volume conduction rejected.",
        "hilbert": "Analytic signal envelope extracted; instantaneous power now available.",
    }
    return fallbacks.get(step_name, "Processing step completed.")


def _default_rationale(step_name: str, step_str: str, paradigm: str = "", modality: str = "") -> str:
    """Provide a context-aware rationale when the agent didn't supply one."""
    parts = step_str.split(":", 1)
    param = parts[1] if len(parts) > 1 else ""
    mod = modality.upper() or "neural"
    para = paradigm or "general"

    bp_parts = param.split(",") if param else ["", ""]
    bp_low = bp_parts[0] if bp_parts[0] else "?"
    bp_high = bp_parts[1] if len(bp_parts) > 1 and bp_parts[1] else "?"

    rationales = {
        "notch": (
            f"Based on spectral inspection, the {mod} data shows a sharp peak at "
            f"{param or '50/60'} Hz, which is the characteristic signature of power line "
            f"interference rather than neural activity. "
            f"The strategy is to apply a notch filter that precisely suppresses the "
            f"fundamental frequency and its harmonics. "
            f"The implementation uses a zero-phase notch filter removing "
            f"{param or '50'} Hz and all integer multiples up to the Nyquist frequency, "
            f"affecting only an extremely narrow band while leaving adjacent neural "
            f"content completely intact."
        ),
        "bandpass": (
            f"Based on inspection, the data contains slow drift below {bp_low} Hz "
            f"(electrode impedance changes, sweat artifacts) and high-frequency "
            f"contamination above {bp_high} Hz (muscle EMG, electronic noise) that carry "
            f"no useful neural information for the {para} paradigm. "
            f"The strategy is to retain only the physiologically relevant frequency band. "
            f"The implementation applies a {bp_low} to {bp_high} Hz bandpass filter "
            f"that removes DC drift via high-pass and rejects muscle noise via low-pass, "
            f"preserving the target neural oscillations in between."
        ),
        "resample": (
            f"Based on inspection, the current sampling rate exceeds what is needed "
            f"since the bandpass filter has already removed content above the relevant "
            f"frequency range. "
            f"The strategy is to downsample to reduce data volume and accelerate "
            f"downstream computation without losing information. "
            f"The implementation resamples to {param or '?'} Hz, which satisfies the "
            f"Nyquist criterion (greater than twice the bandpass upper cutoff) and "
            f"halves the data size while preserving all effective signal content."
        ),
        "car": (
            f"Based on inspection, all channels share spatially global noise components "
            f"such as residual line noise and movement artifacts. "
            f"The strategy is to subtract the common average to remove noise that is "
            f"uniform across the scalp while preserving local neural differences. "
            f"The implementation computes the mean of all channels at each time point "
            f"and subtracts it from every channel, enhancing channel-specific neural "
            f"activity relative to shared interference."
        ),
        "ica": (
            f"Based on inspection, the data contains mixed artifact sources including "
            f"eye blinks (high-amplitude frontal slow waves) and cardiac contamination "
            f"(periodic sharp peaks). "
            f"The strategy is to decompose the signal into statistically independent "
            f"components and remove those matching artifact patterns. "
            f"The implementation uses FastICA decomposition, automatically identifies "
            f"artifact components via correlation with EOG/ECG reference signals, "
            f"excludes them, and reconstructs the cleaned neural signal."
        ),
        "scale": (
            f"Based on inspection, channel amplitudes vary significantly with some "
            f"channels (possibly near muscles or with poor contact impedance) showing "
            f"much higher variance than others. "
            f"The strategy is to normalize all channels to a common amplitude scale "
            f"so that high-variance channels do not dominate downstream analysis. "
            f"The implementation applies "
            f"{'robust scaling based on median and interquartile range, which is less sensitive to residual outliers than standard z-score' if 'robust' in param.lower() else 'standard z-score normalization, centering each channel at zero mean with unit standard deviation'}."
        ),
        "clip": (
            f"Based on inspection, the filtered data still contains occasional transient "
            f"extreme values from residual artifacts that survived earlier steps. "
            f"The strategy is to cap extreme values to prevent them from distorting "
            f"variance estimates and downstream model training. "
            f"The implementation clamps all values exceeding plus or minus {param or '?'} "
            f"to the boundary, preserving the normal data distribution while eliminating "
            f"rare outlier spikes."
        ),
        "fill_nan": (
            f"Based on inspection, the data may contain non-finite values arising from "
            f"numerical instability in prior filtering steps or acquisition dropouts. "
            f"The strategy is to replace all non-finite values to ensure numerical "
            f"stability in subsequent computations. "
            f"The implementation substitutes NaN and Inf values with {param or '0'}, "
            f"guaranteeing that all matrix operations and statistical calculations "
            f"execute without error."
        ),
        "pick_channels": (
            f"Based on inspection, the data includes channels irrelevant to {para} "
            f"analysis (such as EMG, EOG, or stimulus trigger channels). "
            f"The strategy is to select only the channel subset carrying target "
            f"neural signals. "
            f"The implementation retains {param or 'the specified'} channels and "
            f"discards irrelevant sensors, reducing computational load and preventing "
            f"non-neural signals from interfering with spatial filtering."
        ),
        "drop_bads": (
            f"Based on inspection, certain channels are marked as bad (flat signal, "
            f"excessive noise, or disconnected during recording). "
            f"The strategy is to remove these channels to prevent their noise from "
            f"propagating through spatial filters and corrupting averages. "
            f"The implementation deletes all channels flagged as bad from the data "
            f"matrix, ensuring only reliable channels contribute to subsequent "
            f"processing steps."
        ),
        "interpolate_bads": (
            f"Based on inspection, certain channels are marked as bad but the full "
            f"electrode montage is needed for topographic analysis or spatial filters. "
            f"The strategy is to reconstruct bad channel data from surrounding good "
            f"channels rather than discarding them. "
            f"The implementation uses spherical spline interpolation based on electrode "
            f"spatial positions, estimating the bad channel signal from its neighbors "
            f"and restoring the complete montage without propagating the original noise."
        ),
        "bipolar_ref": (
            f"Based on inspection, the data comes from depth electrodes (sEEG) where "
            f"local field potential sensitivity needs to be maximized. "
            f"The strategy is to compute bipolar derivations between adjacent contacts "
            f"to cancel far-field volume-conducted activity. "
            f"The implementation subtracts consecutive contact pairs on each probe, "
            f"maximizing the signal from nearby neural sources and rejecting distant "
            f"common-mode activity."
        ),
        "hilbert": (
            f"Based on inspection, the {para} paradigm requires extraction of "
            f"instantaneous power (such as ERD/ERS in motor imagery). "
            f"The strategy is to compute the signal envelope via the analytic signal "
            f"to obtain time-varying amplitude in the filtered band. "
            f"The implementation applies the Hilbert transform and takes the modulus "
            f"of the resulting analytic signal, yielding the instantaneous envelope "
            f"that reflects moment-to-moment neural oscillation power."
        ),
    }
    return rationales.get(step_name, f"Applied {step_name} ({param}) as a standard preprocessing step for {mod} data." if param else f"Applied {step_name} as a standard preprocessing step for {mod} data.")


def _write_input_ref(meta_dir: Path, input_path: str) -> None:
    """Write input_ref.json to plan/ directory.

    Accepts a file or directory (BIDS / multi-run input). For a directory,
    skip the 1MB partial hash (not meaningful) and use a manifest hash of
    the tree instead. The fuller version with full sha256 lives in
    ``reproducibility.write_input_hash``; this is the local fallback when
    that module isn't importable.
    """
    path = Path(input_path)
    ref: Dict[str, Any] = {
        "path": str(path.resolve()) if path.exists() else input_path,
        "filename": path.name,
    }

    if path.is_dir():
        ref["is_directory"] = True
        files = [f for f in path.rglob("*") if f.is_file()]
        ref["size_bytes"] = sum(f.stat().st_size for f in files)
        ref["file_count"] = len(files)
        # Manifest of (relpath, size, mtime) keeps the helper cheap; full
        # content hash is the responsibility of write_input_hash.
        manifest = sorted(
            (str(f.relative_to(path)), f.stat().st_size, int(f.stat().st_mtime))
            for f in files
        )
        sha = hashlib.sha256()
        sha.update(json.dumps(manifest, separators=(",", ":")).encode("utf-8"))
        ref["sha256_1mb"] = sha.hexdigest()
    elif path.exists():
        stat = path.stat()
        ref["size_bytes"] = stat.st_size
        ref["modified"] = time.strftime(
            "%Y-%m-%dT%H:%M:%S", time.localtime(stat.st_mtime))
        sha = hashlib.sha256()
        with open(path, "rb") as f:
            sha.update(f.read(1024 * 1024))
        ref["sha256_1mb"] = sha.hexdigest()

    _write(meta_dir / "input_ref.json", json.dumps(ref, indent=2, ensure_ascii=False))
