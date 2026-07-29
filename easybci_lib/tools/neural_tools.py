"""BCI/Neural data processing tools — self-registering into the easybci tool registry.

Provides 11 tools for inspecting, preprocessing, segmenting, and exporting
neural data (EEG, sEEG, ECoG, MEG, Spike). All handlers lazy-import from
``tools.neural_processing`` so this module is importable even without MNE
installed (the check_fn gates availability).

Primary tools:
    inspect_data, preprocess_neural, quality_check, segment_data,
    save_processed, plan_pipeline, list_data, export_repo, bin_spikes,
    batch_process, research_preprocessing

Backward-compatible aliases:
    inspect_neural → inspect_data
    suggest_pipeline → plan_pipeline
    propose_pipeline → plan_pipeline
    confirm_output_format → save_processed
    generate_code → export_repo
"""

import json
import logging
import os
from pathlib import Path

from easybci_agent.source_data_guard import register_source_path, check_output_path
from easybci_lib.tools.neural_processing.progress.context import (
    end_stage_if_active,
    start_stage_if_active,
)
from easybci_lib.tools.registry import registry

try:
    from easybci_lib.constants import get_easybci_home as _get_easybci_home  # type: ignore
    from easybci_lib.tools.neural_processing.research.citation_banner import (
        build_banner as _build_citation_banner,
        latest_flagged_citation_ids as _latest_flagged_citation_ids,
    )
except Exception:  # noqa: BLE001
    _get_easybci_home = None  # type: ignore[assignment]
    _build_citation_banner = None  # type: ignore[assignment]
    _latest_flagged_citation_ids = None  # type: ignore[assignment]

# Top-level binding so tests can mock.patch via
# ``easybci_lib.tools.neural_tools.diagnose_active_provider``. Falls back to
# None if the agent package isn't importable in light-install environments;
# call sites guard with an explicit None check.
try:
    from easybci_agent.web_search_registry import (
        diagnose_active_provider,  # noqa: F401
    )
except Exception:  # noqa: BLE001
    diagnose_active_provider = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Session-level data cache — avoids redundant load_neural() calls within a
# single preprocessing session. Each tool that loads data checks here first.
# Bounded to 1 file (raw + processed) to prevent memory accumulation.
# When a new file enters, the old file's data is evicted.
# ---------------------------------------------------------------------------

_data_cache: dict = {}
_cache_current_file: str = ""

# ---------------------------------------------------------------------------
# Pending QC payload cache — preprocess_neural runs the pipeline in-memory
# but defers writing figures + QC report to disk until quality_check is
# called. This dict holds (data_path, modality) -> payload so quality_check
# can pop the latest run and finalize ``preprocessed_output/{figures, QC_out}``
# from it. Process-local; lost on Gateway restart, which matches existing
# session semantics.
# ---------------------------------------------------------------------------

_PENDING_QC_PAYLOADS: dict = {}


# Session-level fallback for `_require_inspection_report`: when `suggest_pipeline`
# / `propose_pipeline` / `generate_code` are called WITHOUT a work_dir
# argument, fall back to whatever work_dir the last `deep_inspect` was given.
# The agent occasionally drops work_dir from the args between turns; this
# keeps the gate from failing with a confusing "auto-discovery failed" error
# when middle_process/inspection_report.json is sitting right there on disk.
_LAST_DEEP_INSPECT_WORK_DIR: str | None = None


def _register_last_work_dir(work_dir: str | None) -> None:
    """Stash work_dir from deep_inspect for later auto-discovery fallback."""
    global _LAST_DEEP_INSPECT_WORK_DIR
    if work_dir:
        _LAST_DEEP_INSPECT_WORK_DIR = str(work_dir)


def _get_last_work_dir() -> str | None:
    return _LAST_DEEP_INSPECT_WORK_DIR


def _stash_qc_payload(data_path: str, modality: str, payload: dict) -> None:
    """Stash a QC payload for later quality_check() to finalize."""
    _PENDING_QC_PAYLOADS[(os.path.realpath(data_path), modality)] = payload


def _pop_qc_payload(data_path: str, modality: str):
    """Pop the latest QC payload for (data_path, modality), or None."""
    return _PENDING_QC_PAYLOADS.pop(
        (os.path.realpath(data_path), modality), None
    )


def _drain_qc_payloads_for(work_dir: str) -> list:
    """Pop and return every pending QC payload whose work_dir matches.

    Safety net for the case where the agent jumps from preprocess_neural
    straight to export_repo without calling quality_check — without this
    drain, figures + per-session QC reports would stay in memory until
    process exit and never land on disk.
    """
    if not work_dir:
        return []
    try:
        target = os.path.realpath(work_dir)
    except Exception:
        return []
    matches = []
    for key in list(_PENDING_QC_PAYLOADS.keys()):
        payload = _PENDING_QC_PAYLOADS.get(key)
        if not isinstance(payload, dict):
            continue
        try:
            payload_wd = os.path.realpath(payload.get("work_dir", ""))
        except Exception:
            continue
        if payload_wd == target:
            matches.append(_PENDING_QC_PAYLOADS.pop(key))
    return matches


def _peek_qc_payload_for(work_dir: str):
    """Return the first pending QC payload whose work_dir matches, without popping.

    Used by _handle_export_repo to backfill input_path / data_info before calling
    build_mini_repo. The actual pop happens after the build via _drain_qc_payloads_for.
    """
    if not work_dir:
        return None
    try:
        target = os.path.realpath(work_dir)
    except Exception:
        return None
    for payload in _PENDING_QC_PAYLOADS.values():
        if not isinstance(payload, dict):
            continue
        try:
            payload_wd = os.path.realpath(payload.get("work_dir", ""))
        except Exception:
            continue
        if payload_wd == target:
            return payload
    return None


# ---------------------------------------------------------------------------
# Two-phase pipeline helpers — shared across Phase 1 gating
# (_require_inspection_report) and Phase 2 AutoFixer attempt-cap state
# (_bump_autofix_attempts / _clear_autofix_stage). See
# docs/superpowers/specs/2026-06-16-data-preprocessing-two-phase-design.md
# ---------------------------------------------------------------------------

MAX_AUTOFIX_ATTEMPTS = 3


def _resolve_work_dir_from_args(args: dict) -> Path | None:
    """Best-effort recovery of <work_dir> from any of the path-like args a
    propose / plan / generate call may carry. Returns None when nothing is
    usable. Mirrors the work_dir resolution logic in ``_handle_propose_pipeline``.
    """
    for key in ("work_dir", "output_path", "output_dir"):
        raw = args.get(key)
        if not raw:
            continue
        p = Path(raw)
        # output_path can be a file inside work_dir/results/, a file inside
        # work_dir/, or work_dir itself. Replicate the propose-side heuristic.
        try:
            is_dir = p.is_dir()
        except OSError:
            is_dir = False
        treat_as_dir = (
            is_dir
            or not p.suffix
            or p.name.endswith("_preprocess_work_dir")
        )
        if treat_as_dir:
            return p
        if p.parent.name == "results":
            return p.parent.parent
        return p.parent
    return None


def _maybe_archive_prior_run(args: dict, kw: dict, *, phase: str) -> None:
    """Best-effort: if work_dir resolved from args already contains a finalized
    (``plan/pipeline_record.json status=="ok"``) run, rename it to ``_runN`` so
    a fresh preprocessing request starts in a clean directory.

    Idempotent within a session — only the first handler entry in a given turn
    actually archives; subsequent calls on the same (session_id, work_dir) are
    no-ops. See ``maybe_archive_completed_work_dir`` in finalize.py.
    """
    try:
        from easybci_lib.tools.neural_processing.export.finalize import (
            maybe_archive_completed_work_dir,
        )
        wd = _resolve_work_dir_from_args(args)
        if wd is None:
            return
        sid = (kw.get("session_id") or args.get("_session_id") or "").strip() or None
        archived = maybe_archive_completed_work_dir(wd, sid)
        if archived is not None:
            logger.info(
                "%s: archived previous finalized run %s → %s", phase, wd, archived
            )
    except Exception as exc:  # noqa: BLE001
        logger.debug("%s: archive check failed (continuing): %s", phase, exc)


def _require_inspection_report(args: dict) -> dict | None:
    """Phase 1 gate: returns None on success or an error payload on failure.

    Resolution order, since proposal.confirmed / inspection_report.json on
    disk are the ground truth (the LLM-supplied path arg is just a hint):

      1. ``args['inspection_report_path']`` set + file exists → use it.
      2. Fall back to ``<work_dir>/middle_process/inspection_report.json``
         derived from work_dir/output_path/output_dir. Inject the discovered
         path back into ``args`` so downstream code sees it.
      3. Last-resort fallback to the work_dir registered by the most recent
         ``deep_inspect`` call this session — covers the case where the agent
         forgot to forward work_dir between turns but inspection_report
         clearly exists on disk.
      4. Otherwise return the error payload.

    Mutating ``args`` (step 2/3) is intentional — the caller passes its own
    kwargs dict, and the discovered path needs to flow into subsequent gates
    (e.g. ``_do_handle_generate_code`` re-reads ``args['inspection_report_path']``
    later in the call). Without injection the gate would pass here but the
    second use would still see the empty arg.
    """
    p = args.get("inspection_report_path")
    if p and Path(p).is_file():
        return None

    wd = _resolve_work_dir_from_args(args)
    if wd is not None:
        candidate = wd / "middle_process" / "inspection_report.json"
        if candidate.is_file():
            args["inspection_report_path"] = str(candidate)
            # Backfill work_dir so downstream tools that read it directly
            # (e.g. _handle_propose_pipeline) get the recovered path too.
            args.setdefault("work_dir", str(wd))
            return None

    # Session-level fallback — agent forgot to pass any path arg, but
    # deep_inspect already established a work_dir earlier this session.
    last_wd = _get_last_work_dir()
    if last_wd:
        candidate = Path(last_wd) / "middle_process" / "inspection_report.json"
        if candidate.is_file():
            args["inspection_report_path"] = str(candidate)
            args.setdefault("work_dir", str(last_wd))
            logger.info(
                "inspection_report auto-discovered via session-level "
                "last_work_dir=%s",
                last_wd,
            )
            return None

    if not p:
        return {
            "success": False,
            "error": (
                "inspection_report_path is required, and "
                "<work_dir>/middle_process/inspection_report.json was not "
                "found via work_dir / output_path / output_dir args either. "
                "Run deep_inspect first (SKILL.md Phase 1 Step 3) and pass "
                "the returned report_path, OR pass work_dir so it can be "
                "auto-discovered."
            ),
            "fix_hint": (
                "deep_inspect(data_path=<...>, work_dir=<work_dir>) → use "
                "the returned 'report_path' as inspection_report_path."
            ),
        }
    return {
        "success": False,
        "error": f"inspection_report_path does not exist: {p}",
        "fix_hint": (
            "Re-run deep_inspect; the path is created at "
            "<work_dir>/middle_process/inspection_report.json."
            ),
        }
    return None


def _autofix_state_path(work_dir: str) -> Path:
    return Path(work_dir) / "middle_process" / "autofix_state.json"


def _read_autofix_state(work_dir: str) -> dict:
    p = _autofix_state_path(work_dir)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _bump_autofix_attempts(*, work_dir: str, stage: str) -> dict:
    """Increment failure counter for a Phase 2 stage; returns stage record."""
    state = _read_autofix_state(work_dir)
    rec = state.get(stage) or {}
    rec["attempts"] = int(rec.get("attempts", 0)) + 1
    state[stage] = rec
    p = _autofix_state_path(work_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return rec


def _clear_autofix_stage(work_dir: str, stage: str) -> None:
    """Clear a stage's slot on success. Other stages preserved."""
    state = _read_autofix_state(work_dir)
    if state.pop(stage, None) is not None:
        _autofix_state_path(work_dir).write_text(
            json.dumps(state, indent=2), encoding="utf-8",
        )


def _recovery_exhausted_payload(*, stage: str, attempts: int, last: dict) -> dict:
    return {
        "success": False,
        "fatal": True,
        "recovery_exhausted": True,
        "stage": stage,
        "attempts": attempts,
        "last_traceback": last.get("traceback"),
        "stdout_tail": last.get("stdout_tail", ""),
        "stderr_tail": last.get("stderr_tail", ""),
        "fix_hint": (
            "AutoFixer exhausted after 3 attempts. Do NOT call write_file again. "
            "Return to Phase 1 Step 6 PROPOSE: surface this failure to the user "
            "along with inspection_report summary + the last traceback, then ask "
            "them to adjust steps/params and re-confirm (mark_proposal_confirmed). "
            "The counter will reset on the next confirm."
        ),
    }


def _write_qc_artifacts(payload: dict) -> dict:
    """Materialize figures + QC report from a stashed preprocess payload.

    Extracted from _handle_quality_check so _handle_export_repo can call the
    same path as a safety net. Never raises — failures are logged.
    Returns a summary dict (figures_dir, figure list, qc_report_dir, ...).
    """
    summary: dict = {}
    if not isinstance(payload, dict):
        return summary
    try:
        from easybci_lib.tools.neural_processing.quality.compare_viz import (
            generate_comparison_figures,
        )
        from easybci_lib.tools.neural_processing.quality.visualize import (
            generate_qc_figures as _gen_qc_figs,
        )
        from easybci_lib.tools.neural_processing.quality.qc_report_writer import (
            write_session_qc_report,
        )
        import base64 as _b64

        work_dir = payload["work_dir"]
        session_id = payload["session_id"]
        subject_id = payload["subject_id"]
        # Identity from inspection_report is authoritative when present —
        # the stale payload subject_id from a previous file would otherwise
        # land figures in the wrong sub-X/ses-Y/ directory.
        try:
            _ir_path = Path(work_dir) / "middle_process" / "inspection_report.json"
            if _ir_path.is_file():
                _ir = json.loads(_ir_path.read_text(encoding="utf-8"))
                _ident = (_ir or {}).get("identity") or {}
                _ir_sub = _ident.get("subject_id")
                _ir_ses = _ident.get("session_id")
                if _ir_sub:
                    subject_id = _ir_sub
                if _ir_ses:
                    session_id = (
                        _ir_ses if str(_ir_ses).startswith("ses-")
                        else f"ses-{_ir_ses}"
                    )
        except (OSError, json.JSONDecodeError) as _ident_exc:
            logger.debug("identity read failed in _write_qc_artifacts: %s", _ident_exc)
        input_stem = Path(payload["data_path"]).stem
        before_snippet = payload["before_snippet"]
        after_data = payload["after_data"]
        freq_before = payload["freq"]
        freq_after = payload["out_freq"]
        channels_before = payload["channels"]
        channels_after = payload["out_channels"]
        steps = payload["steps"]

        fig_dir = (
            Path(work_dir) / "preprocessed_output" / "figures"
            / f"sub-{subject_id}" / session_id
        )
        fig_dir.mkdir(parents=True, exist_ok=True)

        # Single contract for every figure helper: build one FinalDataView
        # snapshot with classifier-based safety net (multi-modal). After this
        # point, no figure code can mistakenly use raw arrays / channels.
        from easybci_lib.tools.neural_processing.quality.final_view import FinalDataView

        modality = payload.get("modality", "unknown")
        after_2d = after_data if after_data.ndim >= 2 else after_data.reshape(1, -1)
        try:
            view = FinalDataView.from_pipeline_result(
                after_data=after_2d, channels=channels_after,
                frequency=freq_after, modality=modality,
                enforce_data_only=True,
            )
        except ValueError as exc:
            logger.warning(
                "FinalDataView construction failed (%s); falling back to "
                "unfiltered view", exc,
            )
            view = FinalDataView(
                data=after_2d, channels=tuple(channels_after),
                frequency=freq_after, modality=modality,
            )

        comp_figs = generate_comparison_figures(
            before_data=before_snippet, before_freq=freq_before,
            channels_before=channels_before, after_view=view,
            steps=steps, subject_id=input_stem,
        )
        for name, png_bytes in comp_figs.items():
            (fig_dir / name).write_bytes(png_bytes)

        try:
            qc_fig_b64 = _gen_qc_figs(
                view, max_channels_display=min(view.data.shape[0], 32),
            )
            for fig_type, b64_str in qc_fig_b64.items():
                png_bytes = _b64.b64decode(b64_str)
                (fig_dir / f"{input_stem}_{fig_type}.png").write_bytes(png_bytes)
        except Exception as e_figs:
            logger.debug("Additional QC figures failed: %s", e_figs)

        try:
            from easybci_lib.tools.neural_processing.quality.paradigm_viz import (
                generate_paradigm_figures,
            )
            para_written = generate_paradigm_figures(
                view, fig_dir=str(fig_dir), stem=input_stem,
            )
            if para_written:
                summary["paradigm_figures"] = para_written
        except Exception as e_para:
            logger.debug("Paradigm figures failed: %s", e_para)

        summary["comparison_figures_dir"] = str(fig_dir)
        summary["comparison_figures"] = list(comp_figs.keys())

        try:
            qc_out_dir = (
                Path(work_dir) / "preprocessed_output" / "QC_out"
                / f"sub-{subject_id}" / session_id
            )
            qc_out_dir.mkdir(parents=True, exist_ok=True)
            write_session_qc_report(
                output_dir=str(qc_out_dir),
                session_id=session_id,
                subject_id=subject_id,
                data_path=payload["data_path"],
                steps=steps,
                frequency_before=freq_before,
                frequency_after=freq_after,
                channels_before=channels_before,
                channels_after=channels_after,
                qc_metrics=payload.get("qc_metrics"),
                qc_feedback=payload.get("qc_feedback"),
                step_states=payload.get("step_states"),
                data_shape_before=(
                    list(before_snippet.shape)
                    if hasattr(before_snippet, "shape") else None
                ),
                data_shape_after=list(after_data.shape),
            )
            summary["qc_report_dir"] = str(qc_out_dir)
        except Exception as e_qc:
            logger.warning("QC report write failed: %s", e_qc)
    except Exception as exc:
        logger.warning("Could not finalize QC artifacts: %s", exc)
    return summary


def _cache_key(filepath: str, modality: str = "auto", processed: bool = False) -> str:
    """Build a cache key from filepath + modality + processing state."""
    tag = "processed" if processed else "raw"
    return f"{os.path.realpath(filepath)}::{modality}::{tag}"


def _cache_evict_if_new_file(filepath: str) -> None:
    """Evict cache if a different file is being accessed (LRU-1 policy)."""
    global _cache_current_file
    real = os.path.realpath(filepath)
    if _cache_current_file and _cache_current_file != real:
        _data_cache.clear()
    _cache_current_file = real


def _cache_get(filepath: str, modality: str = "auto", processed: bool = False):
    """Retrieve cached data dict, or None if not cached."""
    key = _cache_key(filepath, modality, processed)
    return _data_cache.get(key)


def _cache_put(filepath: str, data_dict: dict, modality: str = "auto", processed: bool = False):
    """Store a data dict in the session cache."""
    _cache_evict_if_new_file(filepath)
    key = _cache_key(filepath, modality, processed)
    _data_cache[key] = data_dict


def _cache_get_processed(filepath: str, modality: str = "auto"):
    """Get processed data for a source file, regardless of exact modality."""
    result = _cache_get(filepath, modality, processed=True)
    if result:
        return result
    real = os.path.realpath(filepath)
    for key, val in _data_cache.items():
        if key.startswith(real + "::") and key.endswith("::processed"):
            return val
    return None


def _load_cached(filepath: str, modality: str = "auto", **kwargs):
    """Load neural data with caching — avoids re-reading the same file."""
    from easybci_lib.tools.neural_processing.io.loader import load_neural

    cached = _cache_get(filepath, modality)
    if cached is not None:
        logger.debug("Cache hit for %s (modality=%s)", filepath, modality)
        return cached

    data_dict = load_neural(filepath, modality=modality, **kwargs)
    # The loader returns a (1, 0) zeros-sentinel with meta.load_error when the
    # file is missing or unreadable. Don't cache that — a corrected path on the
    # next call must hit the loader, not stale empty data, and downstream tools
    # check meta.load_error to short-circuit cleanly instead of crashing on
    # np.min(empty_array).
    if not (isinstance(data_dict.get("meta"), dict) and data_dict["meta"].get("load_error")):
        _cache_put(filepath, data_dict, modality)
    return data_dict


def _check_neural_requirements() -> bool:
    try:
        import numpy  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

INSPECT_DATA_SCHEMA = {
    "name": "inspect_data",
    "description": (
        "Load a neural data file and return a non-destructive summary: "
        "channels, sampling frequency, duration, modality, and basic stats. "
        "Only call when the user provides a specific file path. "
        "Use this FIRST before proposing a pipeline."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "data_path": {"type": "string", "description": "Path to neural data file (EDF, FIF, SET, NWB, MAT, CSV, etc.)"},
            "modality": {"type": "string", "enum": ["auto", "eeg", "seeg", "ecog", "meg", "spike"], "description": "Data modality (auto-detected if omitted)"},
        },
        "required": ["data_path"],
    },
}

DEEP_INSPECT_SCHEMA = {
    "name": "deep_inspect",
    "description": (
        "Full-data scan: per-channel variance/NaN/flat/spike stats, welch PSD "
        "(50/60 Hz peak detection), artifact rate, bad-channel candidates. "
        "Writes <work_dir>/middle_process/inspection_report.json (schema v1). "
        "Phase 1 of the two-phase pipeline flow — call this AFTER "
        "inspect_data + before plan_pipeline."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "data_path": {"type": "string", "description": "Path to neural data file"},
            "work_dir": {"type": "string", "description": "Preprocessing work directory"},
            "sample_pct": {"type": "number", "description": "Fraction of data sampled for artifact rate (default 0.10)"},
            "psd_resolution_hz": {"type": "number", "description": "Welch PSD frequency resolution in Hz (default 1.0)"},
            "max_preload_mb": {"type": "integer", "description": "Cap on full-preload size in MB; degrades if exceeded (default 4096)"},
            "timeout_s": {"type": "integer", "description": "Wall-clock cap in seconds; degrades if exceeded (default 300)"},
        },
        "required": ["data_path", "work_dir"],
    },
}

MARK_PROPOSAL_CONFIRMED_SCHEMA = {
    "name": "mark_proposal_confirmed",
    "description": (
        "Record the user's decision on the Phase 1 proposal AND materialize "
        "the post-confirmation deliverable on 'confirm'. "
        "propose_pipeline stages the proposal into "
        "<work_dir>/middle_process/proposal.staged.json but writes nothing to "
        "plan/ — this tool reads the staged envelope and materializes "
        "plan/proposal.json + plan/goal.json + plan/web_evidence.json "
        "(+ plan/reasoning.md when the evidence-driven step form was used) "
        "+ pipeline.yaml (legacy form), then writes the "
        "<work_dir>/middle_process/proposal.confirmed marker that "
        "generate_code requires. 'modify' clears the marker but keeps the "
        "staged envelope so the next propose_pipeline overwrites it; "
        "'abort' clears both. Call AFTER the user explicitly accepts the "
        "proposal — never self-confirm."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "work_dir": {"type": "string", "description": "Preprocessing work directory"},
            "user_decision": {
                "type": "string", "enum": ["confirm", "modify", "abort"],
                "description": "Verbatim user choice from Phase 1 Step 7",
            },
            "proposal_summary": {
                "type": "string",
                "description": "One-line summary of the proposal (audit trail)",
            },
        },
        "required": ["work_dir", "user_decision"],
    },
}

PREPROCESS_NEURAL_SCHEMA = {
    "name": "preprocess_neural",
    "description": (
        "Execute the preprocessing pipeline by running code/pipeline.py in a "
        "subprocess. If code/pipeline.py is missing, the tool generates the "
        "full bundle from `steps` first; if an existing EASYBCI-versioned "
        "script's header doesn't match, the tool archives it and regenerates. "
        "User-provided scripts (no version marker) are preserved as-is. On "
        "non-zero exit, returns a structured traceback "
        "(file/line/error_type/suggestion_kind) without raising — repair the "
        "script via write_file and re-invoke."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "data_path": {"type": "string", "description": "Path to raw neural data file"},
            "steps": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Ordered processing steps, e.g. ['notch:50', 'bandpass:1,40', 'resample:256', 'scale:robust']",
            },
            "modality": {"type": "string", "enum": ["auto", "eeg", "seeg", "ecog", "meg", "spike"]},
            "analysis_goal": {
                "type": "string",
                "enum": [
                    "classification", "source_localization", "feature_extraction",
                    "clinical_screening", "exploratory", "generic",
                ],
                "description": (
                    "Same enum as plan_pipeline. Drives drop_bads:auto + "
                    "drop_nondata_channels:data_only injection per the §1.3 "
                    "decision table. Falls back to plan/goal.json side file or 'generic' when omitted."
                ),
            },
            "output_path": {
                "type": "string",
                "description": (
                    "Directory to write the mini-repo into. MUST end with "
                    "'_preprocess_work_dir' — the layout contract "
                    "(LayoutSpec.CANONICAL) requires that shape. Preprocessed "
                    "NWBs land at output_path/preprocessed_output/preprocessed/"
                    "sub-<id>/ses-<ses>/, figures under preprocessed_output/"
                    "figures/... and QC under preprocessed_output/QC_out/... "
                    "Any layout drift at tool return will be auto-repaired by "
                    "verify_and_repair; if the repair loop still reports "
                    "residual violations, call the repair_layout tool with the "
                    "returned work_dir before proceeding — do NOT issue manual "
                    "mv/rm/mkdir commands. Default: derived from data_path."
                ),
            },
            "data_info": {"type": "object", "description": "Fingerprint from inspect_data — used when script must be generated"},
            "reasoning": {"type": "object", "description": "Step → rationale string"},
            "label_config": {"type": "object"},
            "segment_duration": {"type": "number"},
            "stride": {"type": "number"},
            "timeout": {"type": "integer", "description": "Subprocess wall-clock limit (seconds, default 900)"},
        },
        "required": ["data_path", "steps"],
    },
}

QUALITY_CHECK_SCHEMA = {
    "name": "quality_check",
    "description": (
        "Run code/qc.py in a subprocess to produce figures "
        "(preprocessed_output/figures/sub-{id}/{ses}/) and the QC report "
        "(preprocessed_output/QC_out/sub-{id}/{ses}/). The script must already be "
        "written by generate_code in Step 5 — quality_check returns a clear "
        "error if code/qc.py is missing. On non-zero exit, returns a "
        "structured traceback so the agent can repair via write_file and re-invoke."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "data_path": {"type": "string", "description": "Path to RAW neural data file (qc.py reads raw + preprocessed)"},
            "output_path": {
                "type": "string",
                "description": (
                    "Root of the mini-repo. The layout contract requires "
                    "preprocessed_output/preprocessed/sub-<id>/ses-<ses>/ to "
                    "hold NWB inputs and preprocessed_output/QC_out/sub-<id>/"
                    "ses-<ses>/ to receive qc_report.{json,md}. Missing or "
                    "mislocated artefacts trigger auto-repair; residual drift "
                    "becomes a tool error. Default: derived from data_path."
                ),
            },
            "modality": {"type": "string", "enum": ["auto", "eeg", "seeg", "ecog", "meg", "spike"]},
            "timeout": {"type": "integer", "description": "Subprocess wall-clock limit (seconds, default 450)"},
        },
        "required": ["data_path"],
    },
}

SEGMENT_DATA_SCHEMA = {
    "name": "segment_data",
    "description": (
        "Segment continuous neural data into fixed-size epochs. "
        "Supports sliding window or event-triggered segmentation. "
        "Optionally applies preprocessing before segmentation."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "data_path": {"type": "string", "description": "Path to neural data file"},
            "method": {"type": "string", "enum": ["sliding", "event"], "description": "Segmentation method"},
            "duration": {"type": "number", "description": "Segment duration in seconds (default: 2.0)"},
            "stride": {"type": "number", "description": "Stride in seconds for sliding window (default: 1.0)"},
            "events": {"type": "array", "description": "Event list for event-triggered (each with 'start' key)"},
            "offset": {"type": "number", "description": "Offset from event onset in seconds"},
            "modality": {"type": "string", "enum": ["auto", "eeg", "seeg", "ecog", "meg", "spike"]},
            "preprocess_steps": {"type": "array", "items": {"type": "string"}, "description": "Optional preprocessing before segmentation"},
        },
        "required": ["data_path"],
    },
}

RESUME_PREPROCESSING_SCHEMA = {
    "name": "resume_preprocessing",
    "description": (
        "Resume an incomplete preprocessing run in an existing work_dir. "
        "Reads middle_process/inputs_routing.json and reports which routing "
        "entries still need work (missing preprocessed.nwb / figures / "
        "qc_report.json / AI_ready epochs). When check_only=True, only "
        "returns the completeness snapshot. Otherwise runs pipeline → qc → "
        "vis → build_ai_ready via the existing generated scripts (which are "
        "per-file idempotent — already-done entries are skipped automatically) "
        "and returns before/after snapshots. Set force=True to disable the "
        "per-file skip (injects EASYBCI_FORCE_REPROCESS=1 for the subprocess)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "work_dir": {
                "type": "string",
                "description": (
                    "Absolute or relative path to the work_dir containing "
                    "code/pipeline.py + middle_process/inputs_routing.json. "
                    "The layout contract requires "
                    "preprocessed_output/preprocessed/sub-<id>/ses-<ses>/ to "
                    "hold NWB inputs and preprocessed_output/QC_out/sub-<id>/"
                    "ses-<ses>/ to receive qc_report.{json,md}. Missing or "
                    "mislocated artefacts trigger auto-repair; residual drift "
                    "becomes a tool error."
                ),
            },
            "check_only": {
                "type": "boolean",
                "description": (
                    "When true, do not run any stage; only enumerate pending "
                    "entries and return the snapshot. Default false."
                ),
                "default": False,
            },
            "force": {
                "type": "boolean",
                "description": (
                    "When true, disables per-file idempotent skip by setting "
                    "EASYBCI_FORCE_REPROCESS=1 in the subprocess env. "
                    "Default false."
                ),
                "default": False,
            },
            "timeout": {
                "type": "integer",
                "description": (
                    "Wall-clock timeout per stage in seconds (default 900)."
                ),
            },
        },
        "required": ["work_dir"],
    },
}

SAVE_PROCESSED_SCHEMA = {
    "name": "save_processed",
    "description": (
        "Run code/build_ai_ready.py in a subprocess to write "
        "AI_ready/{id}/{ses}/*_epochs.pkl. When code/build_ai_ready.py is "
        "absent and no events / label_config are available, the tool reports "
        "skipped=true (AI_ready is conditional on labels). Set confirm=true "
        "to delegate to the legacy output-format selector."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "data_path": {"type": "string", "description": "Raw input or *_preprocessed.nwb path"},
            "output_path": {
                "type": "string",
                "description": (
                    "Root of the mini-repo (must end with '_preprocess_work_dir'). "
                    "Continuous outputs go under preprocessed_output/preprocessed/"
                    "sub-<id>/ses-<ses>/ as NWB — the layout contract enforces "
                    "preprocessed_output/preprocessed/ is NWB-only since the "
                    "format unification. If a drift is detected, verify_and_repair "
                    "moves stray files to middle_process/sweep_<ts>/; do not "
                    "attempt manual fixes. Default: derived from data_path."
                ),
            },
            "output_format": {
                "type": "string",
                "enum": ["nwb", "auto"],
                "description": "Output format for the preprocessed/ layer. NWB-only since the format unification — 'auto' and 'nwb' both resolve to NWB; the legacy 'pkl' override has been removed. AI_ready/*_epochs.pkl is unaffected and remains pkl.",
            },
            "modality": {"type": "string", "enum": ["auto", "eeg", "seeg", "ecog", "meg", "spike"]},
            "analysis_goal": {
                "type": "string",
                "enum": [
                    "classification", "source_localization", "feature_extraction",
                    "clinical_screening", "exploratory", "generic",
                ],
            },
            "data_info": {"type": "object", "description": "events / fingerprint — used when build_ai_ready.py must be generated"},
            "label_config": {"type": "object"},
            "preprocess_steps": {"type": "array", "items": {"type": "string"}},
            "segment_method": {"type": "string", "enum": ["sliding", "event"]},
            "segment_duration": {"type": "number"},
            "stride": {"type": "number"},
            "subject_id": {"type": "string"},
            "paradigm": {"type": "string"},
            "timeout": {"type": "integer", "description": "Subprocess wall-clock limit (seconds, default 450)"},
            "confirm": {"type": "boolean", "description": "If true, return format options for user to choose instead of saving immediately"},
            "n_segments": {"type": "integer", "description": "Number of segments (used with confirm=true)"},
            "data_shape": {"type": "object", "description": "Dict of modality -> shape (used with confirm=true)"},
        },
        "required": ["data_path", "output_path"],
    },
}

PLAN_PIPELINE_SCHEMA = {
    "name": "plan_pipeline",
    "description": (
        "Plan a preprocessing pipeline. mode='suggest': return best-practice steps for a "
        "modality+paradigm (no data_path needed). mode='propose': build a full YAML config "
        "with per-step rationale for user review before execution."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "mode": {"type": "string", "enum": ["suggest", "propose"], "description": "Planning mode (default: suggest)"},
            "modality": {"type": "string", "enum": ["eeg", "seeg", "ecog", "meg", "spike"], "description": "Data modality"},
            "paradigm": {"type": "string", "description": "BCI paradigm (motor_imagery, erp, ssvep, or default)"},
            "analysis_goal": {
                "type": "string",
                "enum": [
                    "classification",
                    "source_localization",
                    "feature_extraction",
                    "clinical_screening",
                    "exploratory",
                    "generic",
                    "connectivity",
                    "phase_amplitude_coupling",
                    "online_inference",
                ],
                "description": (
                    "Required. The downstream analysis the user actually wants — drives "
                    "channel-cleanup mode, bandpass, ICA strategy, segmentation, output "
                    "format. Infer from the user's natural-language intent: "
                    "'classify / decoder / 解码 / 分类' → classification; "
                    "'source / dipole / 源定位' → source_localization; "
                    "'feature / 特征 / spectrogram' → feature_extraction; "
                    "'clinical / 临床 / screening / 筛查' → clinical_screening; "
                    "'explore / 探索 / 看一下' → exploratory; "
                    "'connectivity / coherence / PLV / 功能连接 / 网络' → connectivity; "
                    "'PAC / cross-frequency / theta-gamma coupling / 跨频耦合' → phase_amplitude_coupling; "
                    "'online / realtime / 实时 / 在线 / streaming inference' → online_inference. "
                    "If you cannot infer (user said 'just process this' / no signal), use "
                    "'generic' — never leave empty, never invent a goal. The decision table "
                    "lives in improved_docs/plans/goal-driven-preprocessing/02-phase1-goal-first.md §1.3."
                ),
            },
            "data_path": {"type": "string", "description": "Path to input neural data file (required for propose mode unless steps are passed in object form)"},
            "steps": {
                "description": (
                    "Ordered preprocessing steps. **Preferred — evidence-driven form**: "
                    "array of objects [{operator, method?, params, param_evidence?}, ...] where "
                    "`method` is an optional one-sentence human description of the "
                    "concrete technique applied at this step (e.g. "
                    "'Remove MEG line noise using ZapLine/DSS line-noise removal'). "
                    "It complements the abstract `operator` token by naming the "
                    "specific algorithm / variant used, and is written verbatim to "
                    "plan/proposal.json. param_evidence is keyed by param name and carries "
                    "{source, value, confidence, rationale?, default_origin?}. "
                    "This form writes the full post-confirmation deliverable: "
                    "plan/proposal.json (with per-parameter evidence AND embedded "
                    "web_evidence), plan/goal.json, plan/web_evidence.json (raw web "
                    "search payload when research_preprocessing ran), and "
                    "plan/reasoning.md (rendered Parameter evidence tables + "
                    "web-evidence banner). "
                    "**Fallback only — legacy string form**: array of strings like "
                    "'bandpass:1,40'. Produces a husk proposal.json "
                    "(params:{raw:\"…\"}, param_evidence:{} empty) and SKIPS "
                    "plan/reasoning.md entirely. Use only when no parameter rationale "
                    "is available; you must accept that the post-confirm deliverable "
                    "will be missing explainability content. Pair with the top-level "
                    "`methods` array to attach a per-step one-sentence method description."
                ),
            },
            "rationale": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Per-step reasoning (same length as steps). Each entry MUST be 3+ sentences (80+ words) following observation/strategy/implementation structure. Short entries will be replaced by system defaults.",
            },
            "methods": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Optional per-step one-sentence method description (same length and order as "
                    "`steps`), e.g. 'Remove MEG line noise using ZapLine/DSS line-noise removal'. "
                    "Unlike `rationale` (long observation/strategy/implementation prose), each "
                    "entry should be a single sentence naming the concrete technique used. "
                    "Written verbatim to the per-step `method` field of plan/proposal.json. "
                    "Used by the legacy string-step form; when steps are passed as objects, "
                    "prefer embedding `method` directly inside each step object instead."
                ),
            },
            "output_path": {"type": "string", "description": "Output file path or work directory (auto-generated if omitted). Object-form steps require output_path."},
            "subject_id": {"type": "string"},
            "segment_method": {"type": "string", "enum": ["sliding", "event", "none"]},
            "segment_duration": {"type": "number"},
            "stride": {"type": "number"},
            "output_format": {"type": "string", "enum": ["nwb", "auto"]},
            "reuse_source": {
                "type": "string",
                "description": (
                    "In Reuse Mode ONLY: the proven-pipeline skill name being reused "
                    "(from suggest_pipeline's proven_recommendation.name). Persisted into "
                    "plan/proposal.json so downstream tooling knows this run reuses a "
                    "proven pipeline. Omit for New-Plan Mode."
                ),
            },
        },
        "required": ["modality", "analysis_goal"],
    },
}

LIST_DATA_SCHEMA = {
    "name": "list_data",
    "description": "List available neural data files in a directory. Scans for known neural data extensions.",
    "parameters": {
        "type": "object",
        "properties": {
            "directory": {"type": "string", "description": "Directory to scan (default: current working directory)"},
            "pattern": {"type": "string", "description": "Glob pattern filter (default: *)"},
        },
    },
}

EXPORT_REPO_SCHEMA = {
    "name": "export_repo",
    "description": (
        "Export a reproducible mini-repo with executable code (pipeline.py + run.py), "
        "config.yaml, reasoning.md, and results. Set code_only=true to return generated "
        "pipeline code without building the full repo. "
        "The mini-repo is standalone: scripts inline mne/scipy/numpy/sklearn logic and "
        "MUST NOT import from easybci_lib / easybci_agent / easybci_cli / services.* / "
        "run_agent — anyone with `pip install mne numpy scipy scikit-learn matplotlib` "
        "must be able to run it without easybci installed (CODE_STANDARD.md Rule 15)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "output_dir": {
                "type": "string",
                "description": (
                    "Output directory for the mini-repo — MUST match the "
                    "'_preprocess_work_dir' shape defined by the layout "
                    "contract. verify_and_repair runs after build_mini_repo; "
                    "if it reports residual issues, invoke the repair_layout "
                    "tool to converge before returning."
                ),
            },
            "steps": {"type": "array", "items": {"type": "string"}},
            "data_info": {"type": "object", "description": "Data inspection result"},
            "pipeline_record": {"type": "object", "description": "Pipeline record dict with timing and QC"},
            "input_path": {"type": "string"},
            "modality": {"type": "string"},
            "segment_duration": {"type": "number"},
            "stride": {"type": "number"},
            "subject_id": {"type": "string"},
            "paradigm": {"type": "string"},
            "pkl_path": {"type": "string", "description": "Path to pkl output to include"},
            "code_only": {"type": "boolean", "description": "If true, return generated code only without building repo"},
            "work_dir": {"type": "string", "description": "Optional. When provided in code_only mode, the generated pipeline is written directly to {work_dir}/code/pipeline.py (any pre-existing pipeline.py is archived to middle_process/code/<timestamp>). Without work_dir, code is returned as a string only and a warning is emitted."},
            "reasoning": {"type": "object", "description": "Step -> reason mapping for code comments (code_only mode)"},
            "step_states": {"type": "array", "description": "Per-step state transitions from preprocess_neural (before/after for each step)"},
        },
        "required": ["steps", "data_info"],
    },
}

BIN_SPIKES_SCHEMA = {
    "name": "bin_spikes",
    "description": "Load spike data (NWB/HDF5) and bin spike trains into a dense array at target frequency.",
    "parameters": {
        "type": "object",
        "properties": {
            "data_path": {"type": "string", "description": "Path to spike data file (NWB or HDF5)"},
            "bin_frequency": {"type": "number", "description": "Target bin frequency in Hz (default: 100)"},
        },
        "required": ["data_path"],
    },
}

REPAIR_LAYOUT_SCHEMA = {
    "name": "repair_layout",
    "description": (
        "Bring a preprocess work_dir back into contract via code-driven "
        "repair. Detects layout drift (missing required dirs/files, "
        "forbidden paths, orphan files, husk proposal fields) and applies "
        "matched Python fix primitives; residual violations are returned "
        "unchanged. This tool REPLACES manual `terminal(command='mv ...')` "
        "moves in Step 12 pre-export self-check — never issue mv/rm/mkdir "
        "for layout drift yourself; call this tool instead so the whole "
        "session leaves a consistent audit trail in plan/repair_report.json."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "work_dir": {
                "type": "string",
                "description": "Absolute path to the {subject}_preprocess_work_dir/",
            },
            "dry_run": {
                "type": "boolean",
                "description": (
                    "When true, report what would be done without touching "
                    "the filesystem (default false)."
                ),
            },
            "allow_subprocess": {
                "type": "boolean",
                "description": (
                    "When true, allow re-running code/<stage>.py subprocesses "
                    "to regenerate missing artefacts (default false — hot "
                    "tool path). Set true only when calling from a slow "
                    "finalize context."
                ),
            },
            "only_kinds": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Optional whitelist of violation kinds to attempt. Kinds "
                    "match the prefix (e.g. 'missing_dir:', 'forbidden:'); "
                    "use the full kind for exact match. Everything else is "
                    "left alone."
                ),
            },
            "analysis_goal": {
                "type": "string",
                "description": (
                    "Optional. When omitted, the tool reads it from "
                    "plan/proposal.json. Passing it explicitly overrides — "
                    "useful when proposal.json is a husk."
                ),
            },
        },
        "required": ["work_dir"],
    },
}

BATCH_PROCESS_SCHEMA = {
    "name": "batch_process",
    "description": (
        "Batch process multiple neural data files in parallel. "
        "Accepts a glob pattern and applies the same pipeline to all matching files. "
        "Concurrency is memory-aware: actual workers may be fewer than max_workers "
        "if file sizes would exceed available RAM. Hard limit: 8 workers."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Glob pattern for input files, e.g. '/data/sub-*/eeg.fif'"},
            "steps": {"type": "array", "items": {"type": "string"}, "description": "Preprocessing steps"},
            "output_dir": {"type": "string", "description": "Output directory for results"},
            "modality": {"type": "string", "enum": ["auto", "eeg", "seeg", "ecog", "meg", "spike"]},
            "segment_duration": {"type": "number"},
            "stride": {"type": "number"},
            "max_workers": {"type": "integer", "description": "Max parallel workers (default: 4, hard limit: 8). Actual workers may be lower based on available memory."},
        },
        "required": ["pattern", "steps", "output_dir"],
    },
}


# ---------------------------------------------------------------------------
# Visualization helpers — produce lightweight JSON for WebUI rendering
# ---------------------------------------------------------------------------

def _build_signal_compare_viz(before, after, before_freq, after_freq, channels, out_channels):
    """Build a signal_compare viz payload (before/after snippets, max 5s, 8 channels)."""
    import numpy as np

    try:
        max_ch = 8
        duration_s = 5.0

        b_ch = min(before.shape[0], max_ch) if before.ndim >= 2 else 1
        b_samples = min(before.shape[-1], int(before_freq * duration_s))
        b_snippet = before[:b_ch, :b_samples] if before.ndim >= 2 else before[:b_samples].reshape(1, -1)

        a_ch = min(after.shape[0], max_ch) if after.ndim >= 2 else 1
        a_samples = min(after.shape[-1], int(after_freq * duration_s))
        a_snippet = after[:a_ch, :a_samples] if after.ndim >= 2 else after[:a_samples].reshape(1, -1)

        return {
            "type": "signal_compare",
            "before": {
                "data": np.round(b_snippet, 4).tolist(),
                "channels": (channels[:b_ch] if channels else [f"Ch{i}" for i in range(b_ch)]),
                "frequency": before_freq,
            },
            "after": {
                "data": np.round(a_snippet, 4).tolist(),
                "channels": (out_channels[:a_ch] if out_channels else [f"Ch{i}" for i in range(a_ch)]),
                "frequency": after_freq,
            },
            "duration_shown": duration_s,
        }
    except Exception:
        return None


def _build_qc_viz(data, frequency, channels):
    """Build a qc_dashboard viz payload (PSD + channel variance)."""
    import numpy as np

    try:
        max_ch = 8
        n_ch = min(data.shape[0], max_ch) if data.ndim >= 2 else 1

        variance = np.nanvar(data, axis=-1) if data.ndim >= 2 else [float(np.nanvar(data))]
        ch_names = channels if channels else [f"Ch{i}" for i in range(data.shape[0] if data.ndim >= 2 else 1)]
        var_data = [{"ch": ch_names[i], "var": round(float(variance[i]), 4)} for i in range(min(len(variance), 32))]

        psd_data = []
        try:
            from scipy.signal import welch
            subset = data[:n_ch] if data.ndim >= 2 else data.reshape(1, -1)
            nperseg = min(1024, subset.shape[-1])
            if nperseg >= 8:
                freqs, psd = welch(subset[0], fs=frequency, nperseg=nperseg)
                max_freq_idx = min(len(freqs), 128)
                psd_data = [[round(float(freqs[i]), 2), round(float(psd[i]), 6)] for i in range(max_freq_idx)]
        except (ImportError, Exception):
            pass

        return {
            "type": "qc_dashboard",
            "psd": psd_data,
            "variance": var_data,
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

# Signal file extensions recognized by the loader
_SIGNAL_EXTENSIONS = {
    ".edf", ".bdf", ".gdf", ".fif", ".set", ".cnt", ".vhdr",
    ".xdf", ".mat", ".npy", ".npz", ".csv", ".hdf5", ".h5",
    ".mef", ".nwb", ".plx", ".nex", ".nex5", ".smr",
}


def _scan_directory_for_signals(directory: str) -> list:
    """Scan a directory for neural signal files, returning sorted paths."""
    dir_path = Path(directory)
    if not dir_path.is_dir():
        return []
    signal_files = []
    try:
        for entry in sorted(dir_path.iterdir()):
            if entry.is_file() and entry.suffix.lower() in _SIGNAL_EXTENSIONS:
                signal_files.append(str(entry))
    except (PermissionError, OSError):
        pass
    return signal_files


def _build_channel_summary(channels, meta, modality):
    """Compact channel-type summary for inspect_data fingerprint.

    Pure: derives from already-loaded meta (ch_types/bad_channels). Never
    re-reads the file. Returns a JSON-friendly dict the LLM can act on.
    """
    from easybci_lib.tools.neural_processing.io.channel_classifier import classify_channels

    cls = classify_channels(
        channels,
        ch_types=(meta or {}).get("ch_types"),
        modality=modality,
        bad_channels=(meta or {}).get("bad_channels"),
    )
    if not cls["applicable"]:
        return {"applicable": False}
    return {
        "applicable": True,
        "used_fallback": cls["used_fallback"],
        "counts": cls["summary"],
        "must_drop": cls["must_drop"],
        "suggest_drop": cls["suggest_drop"],
    }


def _handle_inspect_data(args, **kw):
    from easybci_lib.tools.neural_processing.io.loader import load_neural
    import numpy as np

    data_path = args["data_path"]

    # --- Directory input: sampling mode ---
    if Path(data_path).is_dir():
        return _handle_inspect_directory(data_path, args, kw)

    register_source_path(data_path)
    modality = args.get("modality", "auto")

    data_dict = load_neural(data_path, modality=modality, inspect_only=True)
    data = data_dict["data"]
    freq = data_dict.get("frequency", 0)
    channels = data_dict.get("channels", [])
    meta = data_dict.get("meta", {})

    # The loader returns a zero-size sentinel + meta.load_error when it cannot
    # find / read the file. Surface that as a clean tool error instead of
    # marching downstream into np.nanmin(empty) — that crash forced the agent
    # to retry the same wrong path 3× before the same_tool_failure_warning
    # finally kicked in (see gateway.stderr.log root-cause trace).
    if isinstance(meta, dict) and meta.get("load_error"):
        return json.dumps({
            "success": False,
            "error": meta["load_error"],
            "file": data_path,
            "hint": (
                "Verify the path exists and is readable. Use list_data on the "
                "parent directory to enumerate available files before retrying."
            ),
        })

    # Handle spike data (list of arrays) vs continuous data (ndarray)
    if isinstance(data, list):
        n_channels = len(data)
        n_samples = meta.get("n_samples_total", sum(len(t) for t in data))
        duration_s = data_dict.get("duration", 0)
        stats = {
            "n_units": n_channels,
            "total_spikes": sum(len(t) for t in data),
            "sample_only": False,
        }
    else:
        n_channels = meta.get("n_channels") or (data.shape[0] if data.ndim >= 2 else 1)
        n_samples = meta.get("n_samples_total") or meta.get("n_samples") or data.shape[-1]
        duration_s = data_dict.get("duration") or (n_samples / freq if freq > 0 else 0)
        # Defense in depth: even though load_error short-circuits above, any
        # other path that produces an empty sample (truncated file, all-NaN
        # channels) must not crash inspect_data with the cryptic
        # "zero-size array to reduction operation fmin" — emit zeros and let
        # the caller see an "empty" flag.
        if getattr(data, "size", 0) > 0:
            stats = {
                "mean": float(np.nanmean(data)),
                "std": float(np.nanstd(data)),
                "min": float(np.nanmin(data)),
                "max": float(np.nanmax(data)),
                "nan_count": int(np.isnan(data).sum()),
                "sample_only": True,
            }
        else:
            stats = {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "nan_count": 0,
                "sample_only": True,
                "empty": True,
            }

    result = {
        "success": True,
        "file": data_path,
        "modality": data_dict.get("modality", modality),
        "n_channels": int(n_channels),
        "n_samples": int(n_samples),
        "frequency_hz": float(freq),
        "duration_seconds": round(float(duration_s), 2),
        "channels": channels[:32],
        "dtype": str(data.dtype) if hasattr(data, "dtype") else "object",
        "stats": stats,
        "data_nbytes_estimate": int(meta.get("data_nbytes_estimate", n_channels * n_samples * 4)),
    }

    # Channel-type summary — surfaces marker/physio/misc channels so the LLM
    # can drop non-data channels in PLAN. Best-effort; never fails inspect.
    try:
        result["channel_summary"] = _build_channel_summary(
            channels, meta, result.get("modality", modality),
        )
    except Exception as exc:
        logger.debug("channel_summary skipped: %s", exc)

    if meta.get("annotations"):
        annotations = meta["annotations"]
        n_events = len(annotations.get("onset", []))
        unique_types = len(set(annotations.get("description", [])))
        result["events"] = {"n_events": n_events, "n_types": unique_types}

    # Sidecar file detection — discover companion event/behavior/aux files
    try:
        from easybci_lib.tools.neural_processing.io.sidecar_detector import (
            detect_sidecar_files, build_event_source_report,
        )
        sidecar_result = detect_sidecar_files(data_path)
        if sidecar_result["sidecar_files"]:
            result["sidecar_files"] = sidecar_result["sidecar_files"]
            result["data_type"] = sidecar_result["data_type"]
            result["relationships"] = sidecar_result["relationships"]
        else:
            result["data_type"] = "signal-only"

        # Unified event source report (embedded + sidecar)
        event_report = build_event_source_report(meta, sidecar_result)
        result["event_sources"] = event_report

        # Cross-validate multiple event sources if more than one exists
        try:
            from easybci_lib.tools.neural_processing.io.event_cross_validator import cross_validate_events
            embedded_evts = None
            if meta.get("annotations"):
                ann = meta["annotations"]
                embedded_evts = [
                    {"onset": o, "duration": d, "type": t}
                    for o, d, t in zip(
                        ann.get("onset", []),
                        ann.get("duration", []),
                        ann.get("description", []),
                    )
                ]
            elif meta.get("embedded_events"):
                embedded_evts = meta["embedded_events"]

            external_event_paths = [
                sf["path"] for sf in sidecar_result.get("sidecar_files", [])
                if sf.get("role") in ("event", "marker", "trigger")
                and sf.get("path")
            ]

            if embedded_evts and external_event_paths:
                xval = cross_validate_events(
                    embedded_events=embedded_evts,
                    external_paths=external_event_paths,
                    frequency=freq,
                )
                result["event_cross_validation"] = xval
        except Exception as exc:
            logger.debug("Event cross-validation failed: %s", exc)
    except Exception as exc:
        logger.debug("Sidecar detection failed: %s", exc)

    # BIDS directory structure recognition
    try:
        from easybci_lib.tools.neural_processing.io.bids_detector import detect_bids_structure
        bids_info = detect_bids_structure(data_path)
        if bids_info and bids_info.get("is_bids"):
            result["bids"] = bids_info
            if bids_info.get("associated_files"):
                result.setdefault("sidecar_files", [])
                for key, info in bids_info["associated_files"].items():
                    if isinstance(info, dict) and "path" in info:
                        result["sidecar_files"].append({
                            "path": info["path"],
                            "role": key,
                            "source": "bids",
                        })
                    elif isinstance(info, list):
                        for item in info:
                            if isinstance(item, dict) and "path" in item:
                                result["sidecar_files"].append({
                                    "path": item["path"],
                                    "role": key,
                                    "source": "bids",
                                })
    except Exception as exc:
        logger.debug("BIDS detection failed: %s", exc)

    # XDF multi-stream info (from loader meta)
    if meta.get("stream_index"):
        result["streams"] = meta["stream_index"]
        result["data_type"] = "multi-stream"

        # Auto-trigger alignment step configuration (Step 1.5 ALIGN)
        try:
            from easybci_lib.tools.neural_processing.preprocess.alignment import build_alignment_step_config
            align_config = build_alignment_step_config(result)
            if align_config:
                result["alignment_step"] = align_config
        except Exception as exc:
            logger.debug("Alignment step config failed: %s", exc)

    # Compute data profile for adaptive routing (continuous data only)
    if not isinstance(data, list) and data.ndim >= 2 and freq > 0:
        try:
            from easybci_lib.tools.neural_processing.profile.data_profile import compute_profile
            # T1.5 — pass data_path so compute_profile auto-resolves cohort_tag
            # via BIDS participants.tsv. The CLI override (set by `easybci
            # profile set-cohort`) lives in the per-session data_profile.json
            # written by this very handler one turn ago; read any prior tag
            # so a re-inspect on the same session honors it.
            _prior_cohort: Optional[str] = None
            try:
                _sess_id = (kw.get("session_id") or args.get("_session_id") or "").strip()
                if _sess_id:
                    from easybci_lib.constants import get_easybci_home as _gh
                    _prof_path = _gh() / "sessions" / _sess_id / "data_profile.json"
                    if _prof_path.exists():
                        _prior = json.loads(_prof_path.read_text(encoding="utf-8"))
                        _prior_cohort = (_prior.get("cohort_tag") or "").strip() or None
            except Exception:
                _prior_cohort = None
            profile = compute_profile(
                data, freq,
                channels=channels,
                data_path=str(data_path),
                cli_cohort_override=_prior_cohort,
            )
            result["data_profile"] = profile.to_dict()
        except Exception as exc:
            logger.debug("Data profile computation failed: %s", exc)

    # Memory estimation for large file handling
    try:
        from easybci_lib.tools.neural_processing.preprocess.chunked import estimate_memory_requirements
        n_ch = data.shape[0] if not isinstance(data, list) and data.ndim >= 2 else 0
        dur = data.shape[-1] / freq if not isinstance(data, list) and data.ndim >= 2 and freq > 0 else 0
        mem_est = estimate_memory_requirements(
            data_path, n_channels=n_ch, frequency=freq, duration_s=dur,
        )
        if mem_est.get("needs_chunking"):
            result["memory_warning"] = mem_est
    except Exception as exc:
        logger.debug("Memory estimation failed: %s", exc)

    return json.dumps(result, default=str)


def _handle_deep_inspect(args, **kw):
    """Phase 1: full-data scan producing inspection_report.json.

    Validates args, calls deep_inspect, returns the result as a JSON string.
    Never raises into the agent loop — degraded paths still return success.
    """
    from easybci_lib.tools.neural_processing.io.deep_inspect import deep_inspect

    data_path = args.get("data_path")
    work_dir = args.get("work_dir")
    if not data_path:
        return json.dumps({"success": False, "error": "data_path is required"})
    if not work_dir:
        return json.dumps({
            "success": False,
            "error": (
                "work_dir is required — pass <your_preprocess_work_dir> so the "
                "inspection_report.json lands in <work_dir>/middle_process/."
            ),
        })
    register_source_path(data_path)
    _register_last_work_dir(work_dir)
    result = deep_inspect(
        data_path=data_path,
        work_dir=work_dir,
        sample_pct=float(args.get("sample_pct", 0.10)),
        psd_resolution_hz=float(args.get("psd_resolution_hz", 1.0)),
        max_preload_mb=int(args.get("max_preload_mb", 4096)),
        timeout_s=int(args.get("timeout_s", 300)),
        cli_subject_id=args.get("subject_id") or None,
        cli_session_id=args.get("session_id") or None,
    )
    return json.dumps(result, ensure_ascii=False)


def _handle_mark_proposal_confirmed(args, **kw):
    """Phase 1 → Phase 2 hand-off: confirms a staged proposal and materializes
    the post-confirmation deliverable.

    On ``confirm``, reads ``<work_dir>/middle_process/proposal.staged.json``
    (written by propose_pipeline), materializes the files it carries — plan/
    side files (proposal.json, goal.json, web_evidence.json, optionally
    reasoning.md) and root-level files (pipeline.yaml for the legacy step
    form) — then writes the proposal.confirmed marker and clears
    autofix_state.json so Phase 2 starts with a clean 3-attempt budget per
    stage. plan/ does not exist on disk until this moment.

    On ``abort``, cleans both the marker and the staged envelope so a
    future run starts from scratch. On ``modify``, cleans the marker only;
    the staged envelope persists so the next propose_pipeline call can
    overwrite it with the revised proposal.
    """
    from datetime import datetime as _dt
    work_dir = args.get("work_dir")
    user_decision = args.get("user_decision")
    if not work_dir:
        return json.dumps({"success": False, "error": "work_dir is required"})
    if user_decision not in ("confirm", "modify", "abort"):
        return json.dumps({
            "success": False,
            "error": "user_decision must be one of: confirm, modify, abort",
        })
    work_dir_path = Path(work_dir)
    middle = work_dir_path / "middle_process"
    middle.mkdir(parents=True, exist_ok=True)
    marker = middle / "proposal.confirmed"
    autofix_state = middle / "autofix_state.json"
    staged = middle / "proposal.staged.json"

    if user_decision == "confirm":
        # Read the staged envelope — without it there is nothing to confirm.
        if not staged.is_file():
            return json.dumps({
                "success": False,
                "error": (
                    f"no staged proposal at {staged} — call propose_pipeline "
                    "before mark_proposal_confirmed(user_decision='confirm')."
                ),
            })
        try:
            envelope = json.loads(staged.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return json.dumps({
                "success": False,
                "error": f"failed to read staged proposal: {exc!r}",
            })
        if not isinstance(envelope, dict):
            return json.dumps({
                "success": False,
                "error": "staged proposal envelope is not a JSON object",
            })

        materialized: list = []

        # Root-level files (pipeline.yaml for the legacy form).
        root_files = envelope.get("root_files") or {}
        if isinstance(root_files, dict):
            for relname, content in root_files.items():
                if not isinstance(relname, str) or not isinstance(content, str):
                    continue
                target = work_dir_path / relname
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
                materialized.append(relname)

        # plan/* files — this is the moment plan/ first appears on disk.
        plan_dir = work_dir_path / "plan"
        plan_files = envelope.get("plan_files") or {}
        if isinstance(plan_files, dict) and plan_files:
            plan_dir.mkdir(parents=True, exist_ok=True)
            for relname, content in plan_files.items():
                if not isinstance(relname, str) or not isinstance(content, str):
                    continue
                target = plan_dir / relname
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
                materialized.append(f"plan/{relname}")

        marker.write_text(
            json.dumps({
                "user_decision": "confirm",
                "proposal_summary": args.get("proposal_summary", ""),
                "confirmed_at": _dt.utcnow().isoformat(timespec="seconds"),
                "envelope_kind": envelope.get("kind", "unknown"),
                "materialized": materialized,
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if autofix_state.exists():
            autofix_state.unlink()
        return json.dumps({
            "success": True,
            "marker_written": True,
            "materialized": materialized,
            "plan_dir": str(plan_dir),
        })

    if user_decision == "abort":
        if marker.exists():
            marker.unlink()
        # On abort the staged envelope is no longer relevant — drop it so a
        # later run is not surprised by a stale proposal from this session.
        if staged.exists():
            staged.unlink()
        return json.dumps({
            "success": True, "marker_written": False, "aborted": True,
        })

    # modify → keep staged.json so the next propose_pipeline call overwrites
    # it with the revised proposal. Clear any stale confirmation marker so
    # the agent cannot accidentally trip the Phase 2 gate with a previous
    # confirmation.
    if marker.exists():
        marker.unlink()
    return json.dumps({"success": True, "marker_written": False})


def _handle_inspect_directory(directory: str, args: dict, kw: dict) -> str:
    """Handle inspect when input is a directory — uses sampling for large datasets.

    Strategy:
    - If <= 10 signal files: inspect all individually
    - If > 10: inspect one representative sample, generate Dataset Fingerprint
    """
    from easybci_lib.tools.neural_processing.io.loader import load_neural
    from easybci_lib.tools.neural_processing.preprocess.memory_strategy import should_use_sampling
    import numpy as np

    signal_files = _scan_directory_for_signals(directory)
    if not signal_files:
        return json.dumps({
            "success": False,
            "error": f"No recognized neural signal files found in {directory}",
            "directory": directory,
        })

    modality = args.get("modality", "auto")
    n_files = len(signal_files)

    # Classify all files in directory by role and build relationship matrix
    file_classification = None
    try:
        from easybci_lib.tools.neural_processing.io.file_classifier import classify_directory_files
        file_classification = classify_directory_files(directory)
    except Exception as exc:
        logger.debug("File classification failed: %s", exc)

    try:
        from easybci_lib.tools.neural_processing.io.sidecar_detector import detect_sidecar_files
        # Use first file to get sidecar context for the whole directory
        sidecar_result = detect_sidecar_files(signal_files[0])
    except Exception:
        sidecar_result = {"sidecar_files": [], "data_type": "signal-only", "relationships": {}}

    # Decide: full inspect vs sampling
    use_sampling = should_use_sampling(signal_files)

    if not use_sampling:
        # Small directory: inspect all files individually
        inspections = []
        for fpath in signal_files:
            register_source_path(fpath)
            try:
                data_dict = load_neural(fpath, modality=modality, inspect_only=True)
                data = data_dict["data"]
                freq = data_dict.get("frequency", 0)
                channels = data_dict.get("channels", [])
                n_channels = data_dict.get("meta", {}).get("n_channels") or (
                    data.shape[0] if hasattr(data, "ndim") and data.ndim >= 2 else 1
                )
                inspections.append({
                    "file": fpath,
                    "n_channels": int(n_channels),
                    "frequency_hz": float(freq),
                    "channels_preview": channels[:8],
                })
            except Exception as exc:
                inspections.append({"file": fpath, "error": str(exc)})

        return json.dumps({
            "success": True,
            "mode": "full_directory",
            "directory": directory,
            "n_signal_files": n_files,
            "inspections": inspections,
            "sidecar_files": sidecar_result.get("sidecar_files", []),
            "data_type": sidecar_result.get("data_type", "signal-only"),
            "file_classification": file_classification,
        }, default=str)

    # Large directory: sampling mode — inspect one representative
    sample_path = signal_files[0]
    register_source_path(sample_path)

    try:
        data_dict = load_neural(sample_path, modality=modality, inspect_only=True)
    except Exception as exc:
        return json.dumps({
            "success": False,
            "error": f"Failed to inspect sample file {sample_path}: {exc}",
            "directory": directory,
            "n_signal_files": n_files,
        })

    data = data_dict["data"]
    freq = data_dict.get("frequency", 0)
    channels = data_dict.get("channels", [])
    meta = data_dict.get("meta", {})

    if isinstance(data, list):
        n_channels = len(data)
        n_samples = sum(len(t) for t in data)
        duration_s = data_dict.get("duration", 0)
    else:
        n_channels = meta.get("n_channels") or (data.shape[0] if data.ndim >= 2 else 1)
        n_samples = meta.get("n_samples_total") or meta.get("n_samples") or data.shape[-1]
        duration_s = data_dict.get("duration") or (n_samples / freq if freq > 0 else 0)

    # Estimate total size across all files
    try:
        from easybci_lib.tools.neural_processing.preprocess.memory_strategy import compute_execution_strategy
        strategy = compute_execution_strategy(
            signal_files, n_channels=int(n_channels), frequency=float(freq), duration_s=float(duration_s)
        )
        batch_strategy = {
            "mode": strategy.mode,
            "max_workers": strategy.max_workers,
            "estimated_per_file_mb": round(strategy.estimated_per_file_mb, 1),
            "total_estimated_mb": round(strategy.total_estimated_mb, 1),
            "reason": strategy.reason,
        }
    except Exception:
        batch_strategy = None

    # Dataset fingerprint — derived from the sample
    dataset_fingerprint = {
        "sample_file": sample_path,
        "format": Path(sample_path).suffix.lower(),
        "n_channels": int(n_channels),
        "frequency_hz": float(freq),
        "duration_seconds": round(float(duration_s), 2),
        "modality": data_dict.get("modality", modality),
        "channels_preview": channels[:16],
    }

    # Label type detection on sample
    try:
        from easybci_lib.tools.neural_processing.io.label_classifier import classify_label_type
        label_info = classify_label_type(meta, sidecar_result)
        dataset_fingerprint["label_type"] = label_info.get("type", "none")
        dataset_fingerprint["label_source"] = label_info.get("source", "")
    except Exception:
        dataset_fingerprint["label_type"] = "none"

    return json.dumps({
        "success": True,
        "mode": "sampling",
        "directory": directory,
        "n_signal_files": n_files,
        "sampled_from": sample_path,
        "dataset_fingerprint": dataset_fingerprint,
        "sidecar_files": sidecar_result.get("sidecar_files", []),
        "data_type": sidecar_result.get("data_type", "signal-only"),
        "batch_strategy": batch_strategy,
        "file_classification": file_classification,
        "all_signal_files": signal_files[:50],  # cap listing at 50
        "note": (
            f"Sampled 1 of {n_files} files. Dataset appears homogeneous "
            f"({dataset_fingerprint['format']}, {n_channels}ch, {freq}Hz). "
            f"Confirm this fingerprint applies to all files before proceeding with batch processing."
        ),
    }, default=str)


def _resolve_analysis_goal_for_run(args: dict, work_dir: str) -> str:
    """Resolve `analysis_goal` for the actual-execution path.

    Priority: explicit args > plan/goal.json side file (written by
    propose_pipeline) > "generic". The fallback matches the schema default so
    `_enforce_clean_output` can validate without raising and the run-time
    enforcement always agrees with what the plan/pipeline_record path applied.
    """
    from pathlib import Path as _Path

    goal = args.get("analysis_goal")
    if isinstance(goal, str) and goal.strip():
        return goal.strip()
    if work_dir:
        side = _Path(work_dir) / "plan" / "goal.json"
        if side.exists():
            try:
                payload = json.loads(side.read_text(encoding="utf-8"))
                cand = payload.get("analysis_goal")
                if isinstance(cand, str) and cand.strip():
                    return cand.strip()
            except (OSError, json.JSONDecodeError):
                pass
    return "generic"


def _script_header_matches(script_path: Path, *, steps: list, analysis_goal: str) -> bool:
    """Return True iff the existing script's EASYBCI_STEPS / GOAL header matches.

    Compared by the canonical representation of the (post-enforce) step list
    serialized as ``repr(list)``. Mismatch means the steps changed since the
    script was last generated — the handler will archive + regenerate. Match
    means the script's logic still corresponds to the current call (so any
    agent-applied repair edits are preserved across retries).
    """
    try:
        head = script_path.read_text(encoding="utf-8")[:1024]
    except OSError:
        return False
    from easybci_lib.tools.neural_processing.codegen.generator import _enforce_clean_output
    try:
        enforced = _enforce_clean_output(list(steps), analysis_goal=analysis_goal)
    except Exception:
        return False
    want_steps = f"EASYBCI_STEPS: {repr(enforced)}"
    want_goal = f"EASYBCI_GOAL: {analysis_goal}"
    return want_steps in head and want_goal in head


def _script_has_version_marker(script_path: Path) -> bool:
    """True iff the script carries an EASYBCI_VERSION header — ie. was emitted
    by codegen.generator. Hand-written / user-provided scripts (no marker) are
    preserved as-is by the handlers, never regenerated."""
    try:
        head = script_path.read_text(encoding="utf-8")[:1024]
    except OSError:
        return False
    return "EASYBCI_VERSION" in head


def _inject_nwb_banners(work_dir: Path) -> None:
    """Prepend NWB-related banners to plan/reasoning.md when applicable.

    Banner A (downgrade, Spec § 6.2): ``middle_process/format_downgrade.json``
        exists — the pipeline.py runtime fell back to .pkl because pynwb
        wasn't available.

    (Banner B — "non-invasive saved as NWB per user override" — was removed
    when NWB became the universal default for preprocessed/ output; that
    case is no longer noteworthy.)
    """
    reasoning_md = work_dir / "plan" / "reasoning.md"
    if not reasoning_md.is_file():
        return

    banner_a = ""

    # Banner A — runtime downgrade.
    downgrade_path = work_dir / "middle_process" / "format_downgrade.json"
    if downgrade_path.is_file():
        try:
            info = json.loads(downgrade_path.read_text(encoding="utf-8"))
        except Exception:
            info = {}
        requested = info.get("requested", "?")
        actual = info.get("actual", "?")
        banner_a = (
            f"⚠️ **Output format downgraded**: requested `{requested}` but "
            f"`pynwb` is not available on the execution environment. "
            f"Saved as `.{actual}` instead.\n\n"
            f"To enable NWB output, install `pynwb==3.1.3` and `hdmf==4.3.1` "
            f"in the active venv (or set `security.allow_lazy_installs: true`).\n\n"
        )

    if not banner_a:
        return

    existing = reasoning_md.read_text(encoding="utf-8")
    if "Output format downgraded" not in existing:
        reasoning_md.write_text(banner_a + existing, encoding="utf-8")


def _handle_preprocess_neural(args, **kw):
    """Adapted: ensure code/pipeline.py exists matching the requested steps,
    then run it as subprocess. Returns success+status sidecar or structured
    error report; never raises into the agent loop.

    Normalizes ``work_dir`` to an absolute path so CLI/gateway/sandbox
    produce identical artifacts regardless of the caller's cwd. After the
    inner handler returns, runs ``verify_layout_strict`` on the resulting
    work_dir — if the layout violates the mini-repo contract (missing
    required dirs or 'unknown' husk fields in plan/proposal.json), the
    success envelope is replaced with an error envelope so the agent /
    SSE relay shows the failure instead of writing a half-broken repo.
    """
    if isinstance(args, dict):
        wd = args.get("work_dir")
        if wd:
            try:
                args = {**args, "work_dir": str(Path(wd).expanduser().resolve())}
            except (OSError, RuntimeError) as exc:
                logger.warning("work_dir resolve failed for %r: %s", wd, exc)

    # Retry hygiene: if a previous attempt on this stage failed, sweep its partial
    # outputs so the new attempt starts from a clean state. First-attempt is a no-op.
    try:
        _wd = _resolve_work_dir_from_args(args)
        if _wd is not None:
            # _read_autofix_state takes a str work_dir and returns the whole state;
            # index the per-stage record to read this stage's attempts.
            _state = _read_autofix_state(str(_wd))
            _rec = (_state or {}).get("preprocess_neural") or {}
            if int(_rec.get("attempts", 0)) > 0:
                from easybci_lib.tools.neural_processing.export.layout_repair import (
                    sweep_failed_partials,
                )
                _sweep_result = sweep_failed_partials(_wd, "preprocess_neural")
                if _sweep_result.get("moved_files"):
                    logger.info(
                        "swept %d partial file(s) from previous preprocess_neural retry to %s",
                        len(_sweep_result["moved_files"]),
                        _sweep_result.get("target_dir"),
                    )
    except Exception:  # never let hygiene break the tool
        logger.exception("sweep_failed_partials failed for preprocess_neural — ignoring")

    _steps_for_progress = (args or {}).get("steps", []) if isinstance(args, dict) else []

    # Compute a coarse fingerprint from args so ETA can hit history.
    _operator_repr = None
    _fp_hash = None
    if isinstance(args, dict) and _steps_for_progress:
        try:
            from easybci_lib.tools.neural_processing.progress.fingerprint import coarse_fingerprint
            _first = _steps_for_progress[0]
            _operator_repr = (_first.get("operator") if isinstance(_first, dict) else None) or "pipeline"
            _di = args.get("data_info") or {}
            _modality = (args.get("modality") or _di.get("modality") or "unknown")
            _n_ch = int(_di.get("n_channels", 0) or 0)
            _sfreq = float(_di.get("sampling_rate", 0) or _di.get("sfreq", 0) or 0.0)
            _dur = float(_di.get("duration_seconds", 0) or _di.get("duration_s", 0) or 0.0)
            _fp_hash = coarse_fingerprint(
                modality=str(_modality),
                n_channels=_n_ch,
                frequency_hz=_sfreq,
                duration_s=_dur,
            )
        except Exception as _fp_exc:
            logger.debug("coarse_fingerprint failed: %s", _fp_exc)

    start_stage_if_active(
        "preprocess",
        sub_total=len(_steps_for_progress) or None,
        with_daemon=True,
        operator=_operator_repr,
        fingerprint_hash=_fp_hash,
    )
    try:
        result = _do_handle_preprocess_neural(args, **kw)
    finally:
        end_stage_if_active(with_daemon=True)

    # Hard-constraint verify on success envelopes — see Task 3 plan.
    try:
        return _verify_preprocess_result_envelope(result)
    except Exception as _v_err:
        logger.debug("verify_layout_strict wrapper failed: %s", _v_err)
        return result


def _verify_preprocess_result_envelope(raw_result):
    """Run verify_layout_strict on a preprocess tool result.

    Accepts either a JSON string (the canonical handler return format) or a
    dict envelope. Returns the same shape as the input: when verification
    fails on a success envelope, returns an error envelope; otherwise
    returns the input unchanged.
    """
    import json as _json

    if isinstance(raw_result, str):
        try:
            envelope = _json.loads(raw_result)
        except _json.JSONDecodeError:
            return raw_result
        repaired = _check_envelope_against_layout(envelope)
        if repaired is envelope:
            return raw_result
        return _json.dumps(repaired)
    if isinstance(raw_result, dict):
        return _check_envelope_against_layout(raw_result)
    return raw_result


def _check_envelope_against_layout(envelope):
    """Return the same envelope when no violation; else an error envelope."""
    if not isinstance(envelope, dict):
        return envelope
    if not envelope.get("success"):
        return envelope
    wd = envelope.get("work_dir") or envelope.get("output_path")
    if not wd:
        return envelope

    from easybci_lib.tools.neural_processing.export.contract_check import (
        verify_layout_strict,
        verify_layout_strict_multi,
    )
    from easybci_lib.tools.neural_processing.export.errors import (
        LayoutContractError,
    )

    try:
        # Multi-input runs use the routing-table-driven checker; single-file
        # runs fall through to the legacy verifier (preserves existing behaviour).
        # auto_repair=True lets verify_and_repair heal fixable drift before we
        # decide whether to reject; allow_subprocess=False keeps the hot
        # tool-return path fast (subprocess re-run happens only at finalize).
        routing = Path(wd) / "middle_process" / "inputs_routing.json"
        checker = verify_layout_strict_multi if routing.is_file() else verify_layout_strict
        checker(wd, auto_repair=True, allow_subprocess=False)
    except LayoutContractError as exc:
        return {
            "success": False,
            "error": f"layout contract violation: {exc}",
            "missing": exc.missing,
            "husk_fields": exc.husk_fields,
            "work_dir": str(wd),
        }
    return envelope


def _handle_repair_layout(args, **_kw):
    """Tool handler for repair_layout — always returns JSON string envelope.

    Public LLM entry-point that wraps ``layout_repair.verify_and_repair``.
    Supports the ``only_kinds`` filter to let the LLM converge one class of
    violation at a time (useful in Step 12 self-check).
    """
    import json as _json

    if not isinstance(args, dict):
        return _json.dumps({"success": False, "error": "args must be a dict"})
    wd_raw = args.get("work_dir")
    if not wd_raw:
        return _json.dumps({"success": False, "error": "work_dir is required"})
    wd = Path(str(wd_raw)).expanduser()
    try:
        wd = wd.resolve()
    except OSError as exc:
        return _json.dumps({"success": False, "error": f"work_dir resolve failed: {exc}"})
    if not wd.is_dir():
        return _json.dumps({"success": False, "error": f"work_dir not a directory: {wd}"})

    dry_run = bool(args.get("dry_run", False))
    allow_subprocess = bool(args.get("allow_subprocess", False))
    only_kinds = args.get("only_kinds")
    goal = args.get("analysis_goal") or None

    from easybci_lib.tools.neural_processing.export.layout_repair import (
        FixResult,
        RepairReport,
        _default_codegen_generator,
        _default_script_runner,
        _dispatch_violation,
        detect_violations,
        resolve_for_goal,
        verify_and_repair,
    )

    # Fast path: no filter → the whole loop.
    if not only_kinds:
        report = verify_and_repair(
            wd, analysis_goal=goal, dry_run=dry_run,
            allow_subprocess=allow_subprocess, write_report=not dry_run,
        )
        payload = report.to_dict()
        payload["success"] = True
        return _json.dumps(payload)

    # Filtered path: only touch violations whose kind matches any of only_kinds.
    resolved = resolve_for_goal(goal)
    generator = None if dry_run else _default_codegen_generator(wd, resolved)
    runner = None if (dry_run or not allow_subprocess) else _default_script_runner()

    kinds = tuple(only_kinds)

    def _matches(kind: str) -> bool:
        return any(kind == k or kind.startswith(k) for k in kinds)

    initial_matches = [v for v in detect_violations(wd, resolved=resolved) if _matches(v.kind)]
    fixes = []
    for v in initial_matches:
        try:
            r = _dispatch_violation(
                wd, v, dry_run=dry_run, allow_subprocess=allow_subprocess,
                generator=generator, runner=runner,
            )
        except Exception as exc:  # noqa: BLE001
            r = FixResult(kind=v.kind, applied=False, notes=[str(exc)], residual=True)
        fixes.append(r)
    remaining = len(
        [v for v in detect_violations(wd, resolved=resolved) if _matches(v.kind)]
    )
    filtered_report = RepairReport(
        work_dir=str(wd),
        initial_violations=len(initial_matches),
        remaining_violations=remaining,
        rounds=1,
        fixes=fixes,
        wall_clock_s=0.0,
        unrepairable=[],
    )
    payload = filtered_report.to_dict()
    payload["success"] = True
    payload["filtered"] = list(kinds)
    return _json.dumps(payload)


def _do_handle_preprocess_neural(args, **kw):
    from easybci_lib.tools.neural_processing.codegen.script_runner import run_script

    # Archive any previously-finalized run on this work_dir (same session).
    _maybe_archive_prior_run(args, kw, phase="preprocess_neural")

    data_path = args["data_path"]
    register_source_path(data_path)
    steps = args["steps"]
    modality = args.get("modality", "auto")
    output_path = args.get("output_path") or _derive_work_dir(data_path)
    analysis_goal = (
        args.get("analysis_goal")
        or _resolve_analysis_goal_for_run(args, output_path)
        or "generic"
    )

    work_dir = Path(output_path) if output_path else Path.cwd()
    work_dir.mkdir(parents=True, exist_ok=True)
    script_path = work_dir / "code" / "pipeline.py"

    # 1. Ensure the script exists with a matching header (only regenerate
    #    when missing OR when the existing script is EasyBCI-versioned but
    #    stale; user-supplied scripts without a marker are preserved as-is).
    needs_regen = (
        not script_path.exists()
        or (
            _script_has_version_marker(script_path)
            and not _script_header_matches(
                script_path, steps=steps, analysis_goal=analysis_goal
            )
        )
    )
    if needs_regen:
        data_info = args.get("data_info") or {}
        # Internal regen path: Phase 2 already past the human gate. Synthesize
        # proposal_confirmed=True + inspection_report_path so the gate doesn't
        # block code regeneration during AutoFixer retries.
        insp_path = args.get("inspection_report_path") or str(
            work_dir / "middle_process" / "inspection_report.json"
        )
        gen_args = {
            "steps": steps,
            "data_info": data_info,
            "modality": modality if modality != "auto" else "eeg",
            "analysis_goal": analysis_goal,
            "work_dir": str(work_dir),
            "reasoning": args.get("reasoning"),
            "label_config": args.get("label_config"),
            "segment_duration": args.get("segment_duration", 2.0),
            "stride": args.get("stride", 1.0),
            "inspection_report_path": insp_path,
            "proposal_confirmed": True,
        }
        _handle_generate_code(gen_args)

    # 2. Run via subprocess. Multi-input mode (routing table present) →
    #    pipeline.py loops internally and we pass input_path=None. Legacy
    #    single-file mode keeps the original argv shape.
    routing_path = work_dir / "middle_process" / "inputs_routing.json"
    multi_input = routing_path.is_file()
    if multi_input:
        # Register every input for source-path tracking so source_data_guard
        # protects them all, not just the one carried in args["data_path"].
        try:
            _table = json.loads(routing_path.read_text(encoding="utf-8"))
            for _inp in (_table.get("inputs") or []):
                _ip = _inp.get("data_path")
                if _ip:
                    register_source_path(_ip)
        except Exception as exc:
            logger.warning("routing table read failed (falling back to single-file): %s", exc)
            multi_input = False

    result = run_script(
        work_dir=str(work_dir),
        stage="pipeline",
        input_path=None if multi_input else str(data_path),
        timeout=int(args.get("timeout", 900)),
    )

    if result["ok"]:
        # Stage succeeded — clear its AutoFixer counter (other stages preserved).
        _clear_autofix_stage(str(work_dir), "preprocess_neural")
        status = result.get("status") or {}
        # NWB-output banners — see Spec § 5.3 / § 6.2.
        # Inspect the mini-repo for two conditions and prepend banners to
        # plan/reasoning.md if either is present. Idempotent: banners are
        # only inserted on the first run that produces them.
        try:
            _inject_nwb_banners(Path(work_dir))
        except Exception as _ban_err:
            logger.debug("NWB banner injection skipped: %s", _ban_err)
        return json.dumps({
            "success": True,
            "stage": "pipeline",
            "status": status,
            "output_file": status.get("output_file"),
            "stdout_tail": result["stdout_tail"],
            "analysis_goal": analysis_goal,
        })

    # Failure path: bump counter and check cap.
    rec = _bump_autofix_attempts(work_dir=str(work_dir), stage="preprocess_neural")
    if rec["attempts"] >= MAX_AUTOFIX_ATTEMPTS:
        return json.dumps(_recovery_exhausted_payload(
            stage="preprocess_neural", attempts=rec["attempts"], last=result,
        ))

    return json.dumps({
        "success": False,
        "stage": "pipeline",
        "attempts": rec["attempts"],
        "attempts_remaining": MAX_AUTOFIX_ATTEMPTS - rec["attempts"],
        "retcode": result["retcode"],
        "stdout_tail": result["stdout_tail"],
        "stderr_tail": result["stderr_tail"],
        "traceback": result["traceback"],
        "archived_to": result["archived_to"],
        "hint": (
            "Read traceback.error_type + traceback.error_message. Edit "
            f"{script_path} via write_file to fix the error, then re-invoke "
            "preprocess_neural with the SAME args. "
            f"Attempts remaining: {MAX_AUTOFIX_ATTEMPTS - rec['attempts']}."
        ),
    })


def _derive_work_dir(data_path: str) -> str:
    """Derive the preprocessing work directory from data file path.

    Convention: {data_parent_parent}/{data_parent_name}_preprocess_work_dir/
    """
    from pathlib import Path
    p = Path(data_path)
    if not p.exists():
        return ""
    parent = p.parent
    work_dir = parent.parent / f"{parent.name}_preprocess_work_dir"
    return str(work_dir)


def _handle_resume_preprocessing(args, **kw):
    """Enumerate and (optionally) resume an incomplete preprocessing run.

    Non-mutating when check_only=True; otherwise runs each stage script in
    order and relies on their per-file idempotent skip (Phase 2) to avoid
    reprocessing already-done inputs.
    """
    from easybci_lib.tools.neural_processing.codegen.script_runner import run_script
    from easybci_lib.tools.neural_processing.export.contract_check import (
        enumerate_pending,
    )

    if not isinstance(args, dict):
        return json.dumps({"success": False, "error": "invalid args"})

    wd_raw = args.get("work_dir")
    if not wd_raw:
        return json.dumps({
            "success": False,
            "error": "work_dir is required",
        })
    try:
        wd = Path(str(wd_raw)).expanduser().resolve()
    except (OSError, RuntimeError) as exc:
        return json.dumps({
            "success": False,
            "error": f"work_dir resolve failed: {exc!r}",
        })

    if not wd.is_dir():
        return json.dumps({
            "success": False,
            "error": f"work_dir does not exist: {wd}",
        })

    routing_path = wd / "middle_process" / "inputs_routing.json"
    if not routing_path.is_file():
        return json.dumps({
            "success": False,
            "error": (
                "no inputs_routing.json — nothing to resume. Multi-input "
                "resume requires the routing table produced by deep_inspect."
            ),
            "work_dir": str(wd),
        })

    # Discover analysis_goal (same source as verify_layout_strict_multi).
    goal = "generic"
    proposal_path = wd / "plan" / "proposal.json"
    if proposal_path.is_file():
        try:
            proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
            if isinstance(proposal, dict):
                goal = proposal.get("analysis_goal") or "generic"
        except (OSError, json.JSONDecodeError) as exc:
            logger.debug("proposal.json unreadable, using goal=generic: %s", exc)

    # Track work_dir on the thread-local so downstream tools (WebUI /artifacts)
    # can find it after a resume triggered outside the normal handler chain.
    try:
        from easybci_lib.tools.neural_processing.export.finalize import (
            set_current_work_dir,
        )
        set_current_work_dir(str(wd))
    except Exception as exc:
        logger.debug("set_current_work_dir failed (non-fatal): %s", exc)

    try:
        before = enumerate_pending(wd, analysis_goal=goal)
    except (FileNotFoundError, ValueError) as exc:
        return json.dumps({
            "success": False,
            "error": f"enumerate_pending failed: {exc!r}",
            "work_dir": str(wd),
        })

    check_only = bool(args.get("check_only"))
    if check_only or before["pending"] == 0:
        return json.dumps({
            "success": True,
            "resumed": False,
            "reason": "check_only" if check_only else "already_complete",
            "work_dir": str(wd),
            "analysis_goal": goal,
            "before": before,
        })

    force = bool(args.get("force"))
    timeout = int(args.get("timeout") or 900)

    # Register every input for source-path tracking (mirrors _do_handle_preprocess_neural).
    try:
        table = json.loads(routing_path.read_text(encoding="utf-8"))
        for _inp in (table.get("inputs") or []):
            _ip = _inp.get("data_path")
            if _ip:
                register_source_path(_ip)
    except Exception as exc:
        logger.debug("source-path registration during resume failed: %s", exc)

    # force → EASYBCI_FORCE_REPROCESS=1 for the duration of this call.
    _prev_force = os.environ.get("EASYBCI_FORCE_REPROCESS")
    if force:
        os.environ["EASYBCI_FORCE_REPROCESS"] = "1"

    stages_run: list = []
    stage_results: dict = {}
    try:
        for stage in ("pipeline", "qc", "vis", "build_ai_ready"):
            script = wd / "code" / f"{stage}.py"
            if not script.is_file():
                stage_results[stage] = {"ok": False, "skipped_stage": True,
                                        "reason": f"{stage}.py missing"}
                continue
            result = run_script(
                work_dir=str(wd),
                stage=stage,
                input_path=None,   # multi-input; script reads routing table
                timeout=timeout,
            )
            stages_run.append(stage)
            stage_results[stage] = {
                "ok": bool(result.get("ok")),
                "retcode": result.get("retcode"),
                "stdout_tail": result.get("stdout_tail"),
                "stderr_tail": result.get("stderr_tail"),
            }
            if not result.get("ok"):
                break
    finally:
        if force:
            if _prev_force is None:
                os.environ.pop("EASYBCI_FORCE_REPROCESS", None)
            else:
                os.environ["EASYBCI_FORCE_REPROCESS"] = _prev_force

    try:
        after = enumerate_pending(wd, analysis_goal=goal)
    except (FileNotFoundError, ValueError) as exc:
        after = {"error": f"enumerate_pending failed: {exc!r}"}

    return json.dumps({
        "success": True,
        "resumed": True,
        "work_dir": str(wd),
        "analysis_goal": goal,
        "force": force,
        "stages_run": stages_run,
        "stage_results": stage_results,
        "before": before,
        "after": after,
    })


def _handle_quality_check(args, **kw):
    """Adapted: run code/qc.py as a subprocess to write figures + QC report.

    The script must already exist at ``<work_dir>/code/qc.py`` (generate_code in
    Step 5). On non-zero exit, returns a structured traceback so the agent can
    repair via write_file and re-invoke.
    """
    start_stage_if_active("qc")

    # Retry hygiene: sweep partial QC/figure outputs from a previous failed
    # attempt of this stage before re-running. First-attempt is a no-op.
    try:
        _wd = _resolve_work_dir_from_args(args)
        if _wd is not None:
            _state = _read_autofix_state(str(_wd))
            _rec = (_state or {}).get("quality_check") or {}
            if int(_rec.get("attempts", 0)) > 0:
                from easybci_lib.tools.neural_processing.export.layout_repair import (
                    sweep_failed_partials,
                )
                _sweep_result = sweep_failed_partials(_wd, "quality_check")
                if _sweep_result.get("moved_files"):
                    logger.info(
                        "swept %d partial file(s) from previous quality_check retry to %s",
                        len(_sweep_result["moved_files"]),
                        _sweep_result.get("target_dir"),
                    )
    except Exception:
        logger.exception("sweep_failed_partials failed for quality_check — ignoring")

    try:
        return _do_handle_quality_check(args, **kw)
    finally:
        end_stage_if_active()


def _do_handle_quality_check(args, **kw):
    from easybci_lib.tools.neural_processing.codegen.script_runner import run_script

    data_path = args["data_path"]
    register_source_path(data_path)
    output_path = args.get("output_path") or _derive_work_dir(data_path)
    work_dir = Path(output_path) if output_path else Path.cwd()
    work_dir.mkdir(parents=True, exist_ok=True)

    script_path = work_dir / "code" / "qc.py"
    if not script_path.exists():
        return json.dumps({
            "success": False,
            "error": (
                "code/qc.py is missing — call generate_code (Step 5) before "
                "quality_check, or pass enough data_info for the handler to "
                "generate the bundle now."
            ),
            "missing_script": str(script_path),
        })

    result = run_script(
        work_dir=str(work_dir),
        stage="qc",
        input_path=None if (work_dir / "middle_process" / "inputs_routing.json").is_file() else str(data_path),
        timeout=int(args.get("timeout", 450)),
    )
    if result["ok"]:
        _clear_autofix_stage(str(work_dir), "quality_check")
        status = result.get("status") or {}

        # ── Chain vis.py: figures live in code/vis.py since the qc.py
        # refactor. When vis.py is absent, the analysis_goal opted out of
        # figures (REGISTRY[goal].produces_figures=False, e.g.
        # online_inference) — return the qc-only success payload.
        vis_script_path = work_dir / "code" / "vis.py"
        if not vis_script_path.exists():
            return json.dumps({
                "success": True,
                "stage": "qc",
                "status": status,
                "grade": status.get("grade"),
                "figures": status.get("figures", []),
                "stdout_tail": result["stdout_tail"],
                "vis": {"skipped": True, "reason": "vis.py not generated for this goal"},
            })

        vis_result = run_script(
            work_dir=str(work_dir),
            stage="vis",
            input_path=None if (work_dir / "middle_process" / "inputs_routing.json").is_file() else str(data_path),
            timeout=int(args.get("vis_timeout", args.get("timeout", 900))),
        )
        if vis_result["ok"]:
            _clear_autofix_stage(str(work_dir), "quality_check_vis")
            vis_status = vis_result.get("status") or {}
            # Prefer figures reported in vis_status.json; fall back to a glob.
            figures = vis_status.get("figures") or status.get("figures") or []
            if not figures:
                try:
                    figures = [
                        str(p.relative_to(work_dir))
                        for p in sorted(
                            (work_dir / "preprocessed_output" / "figures").rglob("*.png")
                        )
                    ]
                except Exception:
                    figures = []
            return json.dumps({
                "success": True,
                "stage": "qc",
                "status": status,
                "grade": status.get("grade"),
                "figures": figures,
                "stdout_tail": result["stdout_tail"],
                "vis": {
                    "ok": True,
                    "stdout_tail": vis_result["stdout_tail"],
                    "status": vis_status,
                },
            })

        # vis.py failed — separate autofix counter so qc and vis budgets
        # don't interfere. The agent must repair code/vis.py via write_file
        # and re-invoke quality_check (qc.py re-runs idempotently).
        rec = _bump_autofix_attempts(work_dir=str(work_dir), stage="quality_check_vis")
        if rec["attempts"] >= MAX_AUTOFIX_ATTEMPTS:
            return json.dumps(_recovery_exhausted_payload(
                stage="quality_check_vis", attempts=rec["attempts"], last=vis_result,
            ))
        return json.dumps({
            "success": False,
            "stage": "vis",
            "qc_ok": True,
            "qc_grade": status.get("grade"),
            "attempts": rec["attempts"],
            "attempts_remaining": MAX_AUTOFIX_ATTEMPTS - rec["attempts"],
            "retcode": vis_result["retcode"],
            "stdout_tail": vis_result["stdout_tail"],
            "stderr_tail": vis_result["stderr_tail"],
            "traceback": vis_result["traceback"],
            "archived_to": vis_result["archived_to"],
            "hint": (
                "qc.py succeeded but vis.py failed — edit code/vis.py via write_file "
                "to fix the error, then re-invoke quality_check (qc.py will re-run "
                f"idempotently). Attempts remaining: {MAX_AUTOFIX_ATTEMPTS - rec['attempts']}."
            ),
        })
    rec = _bump_autofix_attempts(work_dir=str(work_dir), stage="quality_check")
    if rec["attempts"] >= MAX_AUTOFIX_ATTEMPTS:
        return json.dumps(_recovery_exhausted_payload(
            stage="quality_check", attempts=rec["attempts"], last=result,
        ))
    return json.dumps({
        "success": False,
        "stage": "qc",
        "attempts": rec["attempts"],
        "attempts_remaining": MAX_AUTOFIX_ATTEMPTS - rec["attempts"],
        "retcode": result["retcode"],
        "stdout_tail": result["stdout_tail"],
        "stderr_tail": result["stderr_tail"],
        "traceback": result["traceback"],
        "archived_to": result["archived_to"],
        "hint": (
            "Edit code/qc.py via write_file to fix the error, then re-invoke quality_check. "
            f"Attempts remaining: {MAX_AUTOFIX_ATTEMPTS - rec['attempts']}."
        ),
    })


def _handle_segment_data(args, **kw):
    from easybci_lib.tools.neural_processing.preprocess.pipeline import preprocess
    from easybci_lib.tools.neural_processing.segment.segment import segment_data, sliding_windows

    data_path = args["data_path"]
    register_source_path(data_path)
    method = args.get("method", "sliding")
    duration = args.get("duration", 2.0)
    stride = args.get("stride", 1.0)
    events = args.get("events")
    offset = args.get("offset", 0.0)
    modality = args.get("modality", "auto")
    preprocess_steps = args.get("preprocess_steps")

    # Use processed cache if available and no custom preprocess_steps given
    if not preprocess_steps:
        data_dict = _cache_get_processed(data_path, modality)
        if data_dict is None:
            data_dict = _load_cached(data_path, modality=modality)
    else:
        data_dict = _load_cached(data_path, modality=modality)
        import copy
        data_dict = copy.deepcopy(data_dict)
        data_dict = preprocess(data_dict, steps=preprocess_steps)

    data = data_dict["data"]
    freq = data_dict["frequency"]

    if method == "event" and events:
        result = segment_data(data, freq, events, duration, offset=offset)
    else:
        result = sliding_windows(data, freq, duration, stride)

    return json.dumps({
        "success": True,
        "n_segments": result["segments"].shape[0],
        "segment_shape": list(result["segments"].shape),
        "frequency": freq,
    })


def _handle_confirm_output_format(args, **kw):
    """Internal helper: legacy output-format selector.

    Since the preprocessed/ layer is NWB-only, there is nothing to ask the
    user — we return the resolved format immediately, mirroring the JSON
    shape callers expect but with ``awaiting_user_input=False``.
    """
    default_path = args.get("output_path") or args.get("default_path", "")
    return json.dumps({
        "success": True,
        "awaiting_user_input": False,
        "chosen_format": "nwb",
        "default_format": "nwb",
        "supported_formats": ["nwb"],
        "default_path": default_path,
        "note": (
            "preprocessed/ is NWB-only since the format unification; the "
            "legacy interactive format prompt is now a no-op."
        ),
    })


def _handle_save_processed(args, **kw):
    """Adapted: run code/build_ai_ready.py as a subprocess to write
    AI_ready/{id}/{ses}/*_epochs.pkl.

    ``data_path`` may point to either the raw input (the script then
    auto-discovers the corresponding preprocessed nwb under work_dir) or
    directly to a ``*_preprocessed.nwb`` file. When neither events nor a
    label_config are available, AI_ready generation is intentionally skipped.
    """
    # Legacy interactive flow (output-format selector) — keep behavior intact.
    if args.get("confirm"):
        return _handle_confirm_output_format(args, **kw)

    from easybci_lib.tools.neural_processing.codegen.script_runner import run_script

    data_path = args["data_path"]
    register_source_path(data_path)
    output_path = args.get("output_path") or _derive_work_dir(data_path)
    work_dir = Path(output_path) if output_path else Path.cwd()
    work_dir.mkdir(parents=True, exist_ok=True)

    # Retry hygiene: sweep partial AI_ready outputs from a previous failed
    # attempt before re-running. First-attempt (and the skip path below) both
    # short-circuit to a no-op when attempts==0. Placed before the skip-check so
    # a retry cleans residual regardless of which path it later takes.
    try:
        _state = _read_autofix_state(str(work_dir))
        _rec = (_state or {}).get("save_processed") or {}
        if int(_rec.get("attempts", 0)) > 0:
            from easybci_lib.tools.neural_processing.export.layout_repair import (
                sweep_failed_partials,
            )
            _sweep_result = sweep_failed_partials(work_dir, "save_processed")
            if _sweep_result.get("moved_files"):
                logger.info(
                    "swept %d partial file(s) from previous save_processed retry to %s",
                    len(_sweep_result["moved_files"]),
                    _sweep_result.get("target_dir"),
                )
    except Exception:
        logger.exception("sweep_failed_partials failed for save_processed — ignoring")

    analysis_goal = (
        args.get("analysis_goal")
        or _resolve_analysis_goal_for_run(args, str(work_dir))
        or "generic"
    )
    modality = args.get("modality", "eeg")

    script_path = work_dir / "code" / "build_ai_ready.py"
    if not script_path.exists():
        data_info = args.get("data_info") or {}
        events = data_info.get("events") or []
        label_config = args.get("label_config")
        if not events and not label_config:
            return json.dumps({
                "success": False,
                "skipped": True,
                "reason": (
                    "No events and no label_config — AI_ready generation skipped per "
                    "the design contract (events_present or label_config required)."
                ),
            })
        # Generate the bundle (writes build_ai_ready.py among others).
        # Internal regen path: Phase 2 already past the human gate, so synth
        # proposal_confirmed=True + inspection_report_path to satisfy the gate.
        insp_path = args.get("inspection_report_path") or str(
            work_dir / "middle_process" / "inspection_report.json"
        )
        _handle_generate_code({
            "steps": args.get("preprocess_steps") or args.get("steps") or [],
            "data_info": data_info,
            "modality": modality if modality != "auto" else "eeg",
            "analysis_goal": analysis_goal,
            "work_dir": str(work_dir),
            "label_config": label_config,
            "segment_duration": args.get("segment_duration", 2.0),
            "stride": args.get("stride", 1.0),
            "inspection_report_path": insp_path,
            "proposal_confirmed": True,
        })

    if not script_path.exists():
        return json.dumps({
            "success": False,
            "skipped": True,
            "reason": "build_ai_ready.py was not generated (no events and no label_config).",
        })

    result = run_script(
        work_dir=str(work_dir),
        stage="build_ai_ready",
        input_path=None if (work_dir / "middle_process" / "inputs_routing.json").is_file() else str(data_path),
        timeout=int(args.get("timeout", 450)),
    )
    if result["ok"]:
        _clear_autofix_stage(str(work_dir), "save_processed")
        status = result.get("status") or {}
        return json.dumps({
            "success": True,
            "stage": "ai_ready",
            "status": status,
            "output_file": status.get("output_file"),
            "stdout_tail": result["stdout_tail"],
        })
    rec = _bump_autofix_attempts(work_dir=str(work_dir), stage="save_processed")
    if rec["attempts"] >= MAX_AUTOFIX_ATTEMPTS:
        return json.dumps(_recovery_exhausted_payload(
            stage="save_processed", attempts=rec["attempts"], last=result,
        ))
    return json.dumps({
        "success": False,
        "stage": "ai_ready",
        "attempts": rec["attempts"],
        "attempts_remaining": MAX_AUTOFIX_ATTEMPTS - rec["attempts"],
        "retcode": result["retcode"],
        "stdout_tail": result["stdout_tail"],
        "stderr_tail": result["stderr_tail"],
        "traceback": result["traceback"],
        "archived_to": result["archived_to"],
        "hint": (
            "Edit code/build_ai_ready.py via write_file to fix the error, then "
            "re-invoke save_processed with the same args. "
            f"Attempts remaining: {MAX_AUTOFIX_ATTEMPTS - rec['attempts']}."
        ),
    })


def _handle_suggest_pipeline(args, **kw):
    """Internal helper: recommend best-practice pipeline steps.

    Enhanced with:
    1. Data-adaptive routing: uses DataProfile to conditionally include/skip/modify steps
    2. Proven pipeline matching: finds similar validated pipelines from experience library
    3. Web search: when the scenario exceeds static knowledge
    """
    modality = args["modality"]
    paradigm = args.get("paradigm", "default")
    user_intent = args.get("user_intent", "")
    fingerprint = args.get("fingerprint")

    # --- Proven pipeline matching ---
    proven_matches = []
    try:
        from easybci_lib.tools.neural_processing.proven_match import match_proven_pipelines
        n_channels = fingerprint.get("n_channels", 0) if isinstance(fingerprint, dict) else 0
        freq_hz = fingerprint.get("frequency_hz", 0) if isinstance(fingerprint, dict) else 0
        duration_s = fingerprint.get("duration_seconds", 0) if isinstance(fingerprint, dict) else 0
        # T1.5 — pull cohort_tag from the previously-computed DataProfile so
        # the cohort similarity dimension actually receives a value (it has
        # been registered in proven_match_dimensions for a while but no
        # caller was passing it).
        _cohort = ""
        if isinstance(fingerprint, dict):
            _dp = fingerprint.get("data_profile") or {}
            _cohort = (_dp.get("cohort_tag") or args.get("cohort_tag") or "").strip()
        proven_matches = match_proven_pipelines(
            modality=modality,
            paradigm=paradigm,
            n_channels=n_channels,
            frequency_hz=freq_hz,
            duration_s=duration_s,
            top_n=3,
            cohort_tag=_cohort,
            analysis_goal=(args.get("analysis_goal") or "generic"),
        )
    except Exception as exc:
        logger.debug("Proven pipeline matching failed: %s", exc)

    # --- Data-adaptive routing ---
    routed = None
    data_profile_dict = None
    if isinstance(fingerprint, dict) and fingerprint.get("data_profile"):
        try:
            from easybci_lib.tools.neural_processing.profile.data_profile import DataProfile
            from easybci_lib.tools.neural_processing.preprocess.routing import route_pipeline

            # Reconstruct DataProfile from the profile dict
            dp = fingerprint["data_profile"]
            profile = DataProfile(
                powerline_present=dp.get("powerline", {}).get("present", False),
                powerline_freq=dp.get("powerline", {}).get("freq_hz", 0),
                powerline_amplitude_db=dp.get("powerline", {}).get("amplitude_db", 0),
                dominant_frequency=dp.get("frequency", {}).get("dominant_hz", 0),
                effective_bandwidth=dp.get("frequency", {}).get("effective_bandwidth_hz", 0),
                snr_per_band=dp.get("frequency", {}).get("snr_per_band", {}),
                drift_severity=dp.get("drift", {}).get("severity", 0),
                has_significant_drift=dp.get("drift", {}).get("significant", False),
                channel_consistency=dp.get("channels", {}).get("consistency", 1.0),
                n_bad_channels=dp.get("channels", {}).get("n_bad", 0),
                bad_channel_names=dp.get("channels", {}).get("bad_names", []),
                flat_channel_ratio=dp.get("channels", {}).get("flat_ratio", 0),
                artifact_ratio=dp.get("artifacts", {}).get("ratio", 0),
                has_extreme_amplitudes=dp.get("artifacts", {}).get("extreme_amplitudes", False),
                dynamic_range_db=dp.get("artifacts", {}).get("dynamic_range_db", 0),
                has_nans=dp.get("data", {}).get("has_nans", False),
                nan_ratio=dp.get("data", {}).get("nan_ratio", 0),
                sampling_rate=dp.get("data", {}).get("sampling_rate", 0),
                n_channels=dp.get("data", {}).get("n_channels", 0),
                duration_s=dp.get("data", {}).get("duration_s", 0),
                noise_score=dp.get("scores", {}).get("noise", 0),
                quality_score=dp.get("scores", {}).get("quality", 1.0),
            )

            routed = route_pipeline(profile, modality, paradigm)
            data_profile_dict = dp
        except Exception as exc:
            logger.debug("Adaptive routing failed: %s", exc)

    # Use routed pipeline if available, else fall back to static recommendations
    if routed:
        steps = routed.steps
    else:
        PIPELINE_RECOMMENDATIONS = {
            "eeg": {
                "motor_imagery": ["notch:50", "bandpass:0.5,40", "resample:256", "scale:robust"],
                "erp": ["notch:50", "bandpass:0.1,30", "resample:256", "scale:standard"],
                "ssvep": ["notch:50", "bandpass:5,45", "resample:256", "scale:robust"],
                "default": ["notch:50", "bandpass:0.5,40", "resample:256", "scale:robust"],
            },
            "seeg": {
                "default": ["bipolar_ref", "notch:50", "bandpass:1,200", "scale:robust"],
            },
            "ecog": {
                "default": ["bipolar_ref", "notch:50", "bandpass:1,200", "scale:robust"],
            },
            "meg": {
                "default": ["notch:50", "bandpass:1,100", "resample:500", "scale:robust"],
            },
            "spike": {
                "default": ["bin:100", "scale:standard"],
            },
        }
        mod_pipelines = PIPELINE_RECOMMENDATIONS.get(modality, PIPELINE_RECOMMENDATIONS["eeg"])
        steps = mod_pipelines.get(paradigm, mod_pipelines.get("default", []))

    # When analysis_goal=generic, override with the
    # broad-coverage Generic Safe Defaults table. The static
    # PIPELINE_RECOMMENDATIONS above are paradigm-tuned (and reasonable
    # given a real paradigm); generic mode means "no signal from user, give
    # me defaults that won't harm any downstream task" — that's what
    # generic_pipeline_defaults() encodes from §1.3.
    _goal_for_default = (args.get("analysis_goal") or "").strip()
    if _goal_for_default == "generic":
        try:
            from easybci_lib.tools.neural_processing.preprocess.generic_defaults import (
                generic_pipeline_defaults,
            )
            _gd = generic_pipeline_defaults(fingerprint if isinstance(fingerprint, dict) else {})
            steps = _gd["steps"]
        except Exception as exc:
            logger.debug("generic_pipeline_defaults skipped (%s); keeping static steps", exc)

    # --- Web search enhancement ---
    from easybci_lib.tools.neural_processing.research.complexity_classifier import classify_complexity

    has_exact_match = routed is not None or paradigm in (
        PIPELINE_RECOMMENDATIONS.get(modality, {}) if not routed else {}
    )
    has_proven = len(proven_matches) > 0

    level = classify_complexity(
        fingerprint=fingerprint,
        user_intent=user_intent,
        modality=modality,
        paradigm=paradigm,
        matched_skill=paradigm if has_exact_match else None,
        proven_match=has_proven,
    )

    web_evidence = None
    if level >= 2:
        try:
            web_evidence = _research_for_suggestion(modality, paradigm, user_intent, level)
        except Exception as exc:
            logger.debug("Web research failed during suggest_pipeline: %s", exc)

    # Build response
    result = {
        "success": True,
        "modality": modality,
        "paradigm": paradigm,
        "recommended_steps": steps,
        "description": " → ".join(steps),
        "complexity_level": level,
    }

    # Include adaptive routing details
    if routed:
        result["adaptive_routing"] = {
            "enabled": True,
            "adaptations_made": routed.adaptations_made,
            "profile_summary": routed.profile_summary,
            "decisions": [
                {"step": d.step, "action": d.action, "reason": d.reason}
                for d in routed.decisions
            ],
        }
    else:
        result["adaptive_routing"] = {"enabled": False, "reason": "No data profile available"}

    # Include proven pipeline matches
    if proven_matches:
        result["proven_matches"] = [m.to_dict() for m in proven_matches]
        best = proven_matches[0]
        # Strict analysis_goal gate for Reuse Mode: emit proven_recommendation
        # (which the SKILL.md Step 4 contract treats as a hard signal to Reuse)
        # only when the matched entry's analysis_goal exactly equals the query
        # goal. Legacy entries with an empty analysis_goal are visible in
        # proven_matches above for reference, but are NOT promoted to a Reuse
        # recommendation — that prevents replaying a feature_extraction
        # pipeline for an online_inference request.
        _query_goal = (args.get("analysis_goal") or "").strip().lower()
        _entry_goal = (best.entry.analysis_goal or "").strip().lower()
        _goal_ok = bool(_query_goal) and bool(_entry_goal) and _query_goal == _entry_goal
        if best.similarity > 0.6 and best.entry.steps and _goal_ok:
            result["proven_recommendation"] = {
                "name": best.entry.name,
                "steps": best.entry.steps,
                "similarity": round(best.similarity, 2),
                "analysis_goal": best.entry.analysis_goal,
                "reuse_contract": "full_flow_required",
                "reuse_note": (
                    "PROVEN PIPELINE MATCHED. Open pipeline/SKILL.md "
                    "Step 2.0 'PROVEN-PIPELINE REUSE CONTRACT' for the hard rules. "
                    "Lock steps + params + analysis_goal + web_evidence from this skill, "
                    "but Steps 0, 1, 1.5(verify), 5, 6, 6b, 7, 8, 9(patch reuse history), 10 "
                    "MUST run in full. Per-Step Rationale must be passed verbatim to "
                    "export_repo's `reasoning` arg — do NOT regenerate it."
                ),
                "note": (
                    f"Proven pipeline '{best.entry.name}' (similarity "
                    f"{best.similarity:.0%}) available as locked reference. "
                    f"Steps: {' → '.join(best.entry.steps)}"
                ),
            }
        elif best.similarity > 0.6 and best.entry.steps:
            # Similarity passes but goal mismatch / missing — surface a clear
            # downgrade reason so reasoning.md can explain why we fell back
            # to New-Plan Mode despite a top match.
            if not _entry_goal:
                _reason = (
                    f"proven match '{best.entry.name}' rejected: "
                    f"entry has no analysis_goal (legacy skill)"
                )
            elif not _query_goal:
                _reason = (
                    f"proven match '{best.entry.name}' rejected: "
                    f"query analysis_goal missing (required for Reuse gate)"
                )
            else:
                _reason = (
                    f"proven match '{best.entry.name}' rejected: "
                    f"goal mismatch ({_entry_goal} vs {_query_goal})"
                )
            result["proven_reuse_rejected"] = {
                "name": best.entry.name,
                "similarity": round(best.similarity, 2),
                "reason": _reason,
            }

    if web_evidence and web_evidence.get("confidence", 0) > 0.3:
        result["web_evidence"] = web_evidence
        result["note"] = (
            f"Standard recommendations provided. Web research (confidence: "
            f"{web_evidence['confidence']:.0%}) suggests additional considerations. "
            f"Review evidence before finalizing pipeline."
        )
    else:
        result["note"] = "Standard recommendations. Adjust based on data quality checks."

    # Surface the web_evidence prepared by _handle_plan_pipeline
    # (mandatory call when research_preprocessing is available). Falls back to
    # an "unavailable" envelope when the dispatch entry didn't run (e.g. tests
    # invoking _handle_suggest_pipeline directly).
    _dispatch_evidence = args.get("_web_evidence")
    if _dispatch_evidence is not None:
        question = _build_research_question(args)["question"]
        result["web_evidence"] = _shape_web_evidence_payload(
            _dispatch_evidence, question=question,
        )

    # --- Contradiction detection: paradigm requires events but data has none ---
    contradictions = _detect_paradigm_contradictions(paradigm, fingerprint)
    if contradictions:
        result["contradictions"] = contradictions
        result["has_contradictions"] = True

    # --- Label type classification (if sidecar/event info available) ---
    label_type_info = _classify_label_from_fingerprint(fingerprint)
    if label_type_info:
        result["label_type"] = label_type_info

    # T1.4 — surface the negatives prompt block (computed by
    # `_do_handle_plan_pipeline` from ExperienceStore.find_relevant_negatives)
    # so the LLM treats it as a hard hint when picking steps.
    _neg_block = args.get("_negatives_block")
    if _neg_block:
        result["negatives_hint"] = _neg_block

    return json.dumps(result)


def _research_for_suggestion(modality, paradigm, user_intent, level):
    """Run web research and return evidence dict (or None on failure)."""
    from easybci_lib.tools.neural_processing.research.query_builder import build_queries
    from easybci_lib.tools.neural_processing.research.search_cache import SearchCache
    from easybci_lib.tools.neural_processing.research.evidence_synthesizer import synthesize_evidence

    question = user_intent or f"{paradigm} {modality} preprocessing best practices"
    cache = SearchCache()
    cached = cache.get(modality, paradigm, question)
    if cached:
        return cached

    queries = build_queries(level=level, modality=modality, paradigm=paradigm, question=question)
    if not queries:
        return None

    search_results, _search_errors, provider_name, _discarded = _execute_research_searches(
        queries, modality=modality, paradigm=paradigm,
    )
    if not search_results:
        return None

    report = synthesize_evidence(
        search_results=search_results,
        modality=modality,
        paradigm=paradigm,
        question=question,
    )

    evidence = report.to_dict()
    evidence["provider"] = provider_name
    evidence["discarded"] = _discarded
    cache.put(modality, paradigm, question, evidence)
    return evidence


# Paradigms that require event-locked epoching (point events with onset times)
_PARADIGMS_REQUIRING_EVENTS = {
    "erp", "p300", "n400", "mmn", "n170", "n200",
    "motor_imagery", "mi",
    "ssvep",
    "oddball",
    "go_nogo", "gonogo",
    "flanker", "stroop",
}

# Paradigms that work without events (resting state, continuous, session-level)
_PARADIGMS_NO_EVENTS_OK = {
    "resting_state", "rest", "resting",
    "sleep", "sleep_staging",
    "fatigue", "drowsiness",
    "continuous", "default",
}


def _detect_paradigm_contradictions(paradigm: str, fingerprint) -> list:
    """Detect contradictions between paradigm requirements and data availability.

    Returns a list of warning dicts, empty if no contradictions found.
    """
    if not paradigm or not isinstance(fingerprint, dict):
        return []

    contradictions = []
    paradigm_lower = paradigm.lower().replace("-", "_").replace(" ", "_")

    # Check if paradigm needs events
    needs_events = paradigm_lower in _PARADIGMS_REQUIRING_EVENTS
    if not needs_events:
        return []

    # Determine event availability from fingerprint
    has_embedded_events = False
    has_sidecar_events = False
    n_events = 0

    # Check embedded events
    events_info = fingerprint.get("events")
    if events_info and isinstance(events_info, dict):
        n_events = events_info.get("n_events", 0)
        has_embedded_events = n_events > 0

    # Check event_sources report
    event_sources = fingerprint.get("event_sources")
    if isinstance(event_sources, dict):
        status = event_sources.get("status", "")
        if status == "none_detected":
            has_embedded_events = False
            has_sidecar_events = False
        elif status == "sidecar_only":
            has_sidecar_events = True
        elif status == "available":
            has_embedded_events = True
        sources = event_sources.get("sources", [])
        for src in sources:
            if src.get("source") == "sidecar":
                has_sidecar_events = True

    # Check sidecar_files directly
    sidecar_files = fingerprint.get("sidecar_files", [])
    for sf in sidecar_files:
        if sf.get("type_guess") == "events":
            has_sidecar_events = True
            break

    # Paradigm requires events but none found
    if needs_events and not has_embedded_events and not has_sidecar_events:
        contradictions.append({
            "type": "paradigm_event_mismatch",
            "severity": "error",
            "paradigm": paradigm,
            "message": (
                f"Paradigm '{paradigm}' requires event-locked epoching, but no events "
                f"were detected in the data (neither embedded annotations nor sidecar event files). "
                f"Segmentation will fail without event markers."
            ),
            "suggestions": [
                "Provide an external event file (CSV/TSV with onset + type columns)",
                "Check if events are stored in a separate .mat or .json file",
                "If this is continuous (no-event) data, consider changing paradigm to 'resting_state' or using sliding_windows",
            ],
        })

    # Events are very sparse for the paradigm
    if needs_events and has_embedded_events and n_events < 5:
        duration_s = fingerprint.get("duration_seconds", 0)
        contradictions.append({
            "type": "sparse_events",
            "severity": "warning",
            "paradigm": paradigm,
            "message": (
                f"Only {n_events} events found in {duration_s:.0f}s of data. "
                f"Paradigm '{paradigm}' typically requires more trials for meaningful analysis."
            ),
            "suggestions": [
                "Verify event extraction was complete (some events may be in sidecar files)",
                "Check if annotations use non-standard descriptions that were not parsed",
                f"Consider whether {n_events} trials is sufficient for your analysis",
            ],
        })

    return contradictions


def _classify_label_from_fingerprint(fingerprint) -> dict:
    """Attempt to classify label type from inspect fingerprint data.

    Returns label type info dict or None if insufficient data.
    """
    if not isinstance(fingerprint, dict):
        return None

    try:
        from easybci_lib.tools.neural_processing.io.label_classifier import classify_label_type, LabelType
    except ImportError:
        return None

    # Try classification from event_sources
    event_sources = fingerprint.get("event_sources")
    if isinstance(event_sources, dict):
        sources = event_sources.get("sources", [])
        if sources:
            # Build synthetic event list for classification
            for src in sources:
                if src.get("source") == "embedded" and src.get("n_events", 0) > 0:
                    # Embedded events with type distribution
                    n_events = src["n_events"]
                    types = src.get("types", ["unknown"])
                    synthetic_events = [
                        {"onset": float(i), "duration": 0.0, "type": types[i % len(types)]}
                        for i in range(min(n_events, 20))
                    ]
                    result = classify_label_type(synthetic_events)
                    if result["confidence"] > 0.5:
                        return {
                            "label_type_name": result["label_type_name"],
                            "strategy": result["strategy"],
                            "confidence": result["confidence"],
                        }

    # Check sidecar event files for label classification
    sidecar_files = fingerprint.get("sidecar_files", [])
    for sf in sidecar_files:
        if sf.get("type_guess") == "events" and sf.get("path"):
            try:
                n_samples = fingerprint.get("n_samples", 0)
                freq = fingerprint.get("frequency_hz", 0)
                duration = fingerprint.get("duration_seconds", 0)
                result = classify_label_type(
                    sf["path"],
                    n_samples=n_samples,
                    frequency=freq,
                    data_duration=duration,
                )
                if result["confidence"] > 0.5:
                    return {
                        "label_type_name": result["label_type_name"],
                        "strategy": result["strategy"],
                        "confidence": result["confidence"],
                        "source_file": sf.get("filename", ""),
                    }
            except Exception:
                pass

    return None


def _handle_propose_pipeline_evidence(args, **kw):
    """Evidence-driven propose: validates step-level param_evidence, hydrates
    missing entries from the parameter-uncertainty registry, and STAGES the
    full plan/ deliverable inside ``middle_process/proposal.staged.json``
    (proposal.json + goal.json + web_evidence.json + reasoning.md as text
    blobs in the envelope). Nothing lands under plan/ here — the actual files
    materialize only when the user confirms via ``mark_proposal_confirmed``,
    so iterative modify cycles never leave partial drafts behind.
    """
    # Contract: every proposal must reference a fresh inspection_report.
    err = _require_inspection_report(args)
    if err is not None:
        return json.dumps(err)

    from pathlib import Path as _Path
    from easybci_lib.tools.neural_processing.research.parameter_evidence import ParameterEvidence
    from easybci_lib.tools.neural_processing.research import parameter_registry as pr

    output_path = args.get("output_path") or ""
    if not output_path:
        return json.dumps({
            "success": False,
            "error": "output_path is required for evidence-driven propose",
        })

    modality = args.get("modality", "")
    paradigm = args.get("paradigm", "")
    # analysis_goal is validated by _handle_plan_pipeline before
    # dispatch — fall back to "generic" defensively in case this handler is
    # invoked through a non-dispatch path.
    analysis_goal = (args.get("analysis_goal") or "generic").strip() or "generic"
    rationale = args.get("rationale") or []
    raw_steps = args.get("steps") or []

    warnings: list = []
    registry_version = pr.registry_version_hash()

    for i, step in enumerate(raw_steps):
        if not isinstance(step, dict):
            return json.dumps({
                "success": False,
                "error": f"step {i}: expected mapping, got {type(step).__name__}",
            })
        params = step.get("params") or {}
        ev_block = step.get("param_evidence")
        if ev_block is None:
            ev_block = {}
            step["param_evidence"] = ev_block
        if not isinstance(ev_block, dict):
            return json.dumps({
                "success": False,
                "error": f"step {i}: param_evidence must be a mapping",
            })

        for pname, raw in list(ev_block.items()):
            if not isinstance(raw, dict) or "source" not in raw:
                return json.dumps({
                    "success": False,
                    "error": (
                        f"step {i}.{pname}: param_evidence entry missing required 'source'"
                    ),
                })

        for pname in list(ev_block.keys()):
            if pname not in params:
                warnings.append(
                    f"step {i}: param_evidence has extra key '{pname}' "
                    f"not in params; dropped"
                )
                del ev_block[pname]

        for pname, val in params.items():
            if pname in ev_block:
                continue
            d = pr.get_default(
                operator=step.get("operator", ""),
                parameter=pname,
                modality=modality,
                paradigm=paradigm,
            )
            ev_block[pname] = ParameterEvidence(
                operator=step.get("operator", ""),
                parameter=pname,
                value=val,
                source="empirical_default",
                confidence=1.0,
                default_origin=d.origin if d else "operator skill default",
                registry_version=registry_version,
            ).to_dict()
            warnings.append(
                f"step {i}: param_evidence missing for '{pname}'; "
                f"auto-filled empirical_default"
            )

        for pname, raw in ev_block.items():
            raw["registry_version"] = registry_version
            raw.setdefault("cache_key", "")

    work_dir_path = _Path(output_path)
    middle_dir = work_dir_path / "middle_process"
    middle_dir.mkdir(parents=True, exist_ok=True)

    # Register work_dir so the run's finally hook can finalize (best-effort)
    # even if the LLM stops before confirmation. With the post-confirm plan/
    # lifecycle, finalize will find no plan/ when no confirm happened — that's
    # by design (user never approved); recovery falls back to middle_process/.
    try:
        from easybci_lib.tools.neural_processing.export.finalize import set_current_work_dir
        set_current_work_dir(output_path)
    except Exception:
        pass

    # Shape the web_evidence payload — same content as before, but now it
    # rides in the staging envelope instead of going straight to disk. When
    # research_preprocessing ran (status=ok), the payload carries provider +
    # recommendations + applied_to_steps + confidence; otherwise it carries
    # {status: "unavailable", reason: ...} so the post-confirm deliverable can
    # still render the "unavailable" badge consistently.
    _web_evidence = args.get("_web_evidence") or {"status": "unavailable", "reason": "not invoked"}
    _evidence_payload = _shape_web_evidence_payload(
        _web_evidence,
        question=_build_research_question(args)["question"],
    )

    proposal = {
        "modality": modality,
        "paradigm": paradigm,
        "analysis_goal": analysis_goal,
        "registry_version": registry_version,
        "steps": [
            {
                "operator": s.get("operator", ""),
                "method": str(s.get("method", "") or ""),
                "params": s.get("params", {}) or {},
                "param_evidence": s.get("param_evidence", {}) or {},
            }
            for s in raw_steps
        ],
        "rationale": rationale,
        "web_evidence": _evidence_payload,
    }
    # Reuse Mode signal (P3): persist the proven-skill name so reuse_mode_guard
    # can recognize Reuse Mode from plan/proposal.json downstream. Only written
    # when non-empty — New-Plan Mode leaves the key absent (guard fail-opens).
    _reuse_source = (args.get("reuse_source") or "").strip()
    if _reuse_source:
        proposal["reuse_source"] = _reuse_source
    proposal_json_text = json.dumps(proposal, indent=2, default=str, ensure_ascii=False)

    goal_json_text = json.dumps({
        "analysis_goal": analysis_goal,
        "modality": modality,
        "paradigm": paradigm,
    }, indent=2, ensure_ascii=False)

    web_evidence_json_text = json.dumps(
        _evidence_payload, indent=2, ensure_ascii=False, default=str,
    )

    # Render reasoning.md to a string. We still run the full render_full_reasoning
    # + banner-prepend chain — only the destination changes from disk to the
    # staging envelope. mark_proposal_confirmed will materialize this text to
    # plan/reasoning.md when the user actually confirms.
    try:
        from easybci_lib.tools.neural_processing.quality.reasoning_writer import render_full_reasoning
        evidence_per_step: list = []
        for s in raw_steps:
            block = s.get("param_evidence", {}) or {}
            evidence_per_step.append({
                k: ParameterEvidence.from_dict(v) for k, v in block.items()
            })
        md = render_full_reasoning(
            title=f"{paradigm} {modality} — Reasoning",
            steps=raw_steps,
            rationales=rationale,
            evidence_per_step=evidence_per_step,
        )
        # Prepend the web-evidence banner so reviewers see, immediately under
        # the title, which search provider (if any) shaped the proposal.
        # Format mirrors `_write_reasoning_md` (the export-time fallback) so
        # the propose-time reasoning.md and the export-time fallback look the
        # same. Only render when the payload is non-trivial.
        if isinstance(_evidence_payload, dict) and _evidence_payload:
            if _evidence_payload.get("status") == "ok" and _evidence_payload.get("recommendations"):
                _provider = _evidence_payload.get("provider") or "unknown provider"
                _confidence = _evidence_payload.get("confidence")
                _applied = _evidence_payload.get("applied_to_steps") or []
                _applied_txt = f" applied to {', '.join(_applied)}" if _applied else ""
                _conf_txt = (
                    f" · confidence {float(_confidence):.2f}"
                    if isinstance(_confidence, (int, float)) else ""
                )
                md = (
                    f"> **Web evidence:** queried {_provider} for SOTA preprocessing"
                    f"{_conf_txt}{_applied_txt}. See `plan/web_evidence.json`.\n\n"
                ) + md
            elif _evidence_payload.get("reason"):
                _reason = _evidence_payload.get("reason", "unknown")
                md = (
                    f"> **Web evidence:** unavailable ({_reason}) — proposal "
                    "uses domain-skill defaults. Configure a web search "
                    "provider for SOTA parameter recommendations.\n\n"
                ) + md
        # Prepend a retracted-citation banner if any flagged citations are
        # recorded in the audit log. Best-effort, never blocks reasoning.md.
        try:
            if _latest_flagged_citation_ids is not None and _get_easybci_home is not None and _build_citation_banner is not None:
                flagged = _latest_flagged_citation_ids(
                    audit_log_path=_get_easybci_home() / "citation_audit.jsonl",
                )
                if flagged:
                    banner = _build_citation_banner([
                        ("registry", f"{len(flagged)} flagged citation(s) — see `easybci registry check`")
                    ])
                    md = banner + md
        except Exception:  # noqa: BLE001
            pass
        # T1.4 — append the negatives prompt block so reasoning.md reviewers
        # see what hints the LLM was given.  ``_negatives_block`` is filled
        # by ``_do_handle_plan_pipeline`` before suggest/propose dispatch.
        _neg_block_for_md = args.get("_negatives_block")
        if _neg_block_for_md:
            md = md + "\n\n## Domain context — known failure modes for similar data\n\n" + _neg_block_for_md.strip() + "\n"
        reasoning_md_text = md
    except Exception as exc:
        logger.error("reasoning_writer failed: %s", exc)
        reasoning_md_text = "# Reasoning (degraded; renderer failed)\n"

    # Stage everything in middle_process/proposal.staged.json — this envelope
    # is the single source of truth between propose and confirm. Each new
    # propose call overwrites it, so iterative modify cycles always land on
    # the latest version. mark_proposal_confirmed reads this envelope and
    # materializes plan/ files (and pipeline.yaml for the legacy path) only
    # when the user actually confirms.
    envelope = {
        "version": "1",
        "kind": "evidence",
        "modality": modality,
        "paradigm": paradigm,
        "analysis_goal": analysis_goal,
        "web_evidence": _evidence_payload,
        "root_files": {},
        "plan_files": {
            "proposal.json": proposal_json_text,
            "goal.json": goal_json_text,
            "web_evidence.json": web_evidence_json_text,
            "reasoning.md": reasoning_md_text,
        },
    }
    staged_path = middle_dir / "proposal.staged.json"
    staged_path.write_text(
        json.dumps(envelope, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    # A fresh propose invalidates any prior confirmation — drop the marker so
    # the agent has to re-confirm against the new staged proposal. Without
    # this clearing, a stale marker from a previous confirm could let
    # generate_code run against an outdated plan/ once the agent materializes.
    confirmed_marker = middle_dir / "proposal.confirmed"
    if confirmed_marker.exists():
        confirmed_marker.unlink()

    # Structured pipeline view for Step 7 CONFIRM. The legacy propose branch
    # returns a `viz` field and the SKILL.md CONFIRM step tells the LLM to
    # render it; the evidence branch historically returned only prose
    # (reasoning_preview), which let weaker models collapse the confirmation to
    # a bare "confirm?" without showing the pipeline. Emit the same structured
    # viz here (operator + params + rationale per step) so the numbered
    # pipeline is always presentable without the model having to parse prose.
    _viz_steps = []
    for _i, _s in enumerate(raw_steps):
        _step_viz = {
            "name": _s.get("operator", "") or "",
            "method": str(_s.get("method", "") or ""),
            "params": _s.get("params", {}) or {},
        }
        if _i < len(rationale) and rationale[_i]:
            _step_viz["rationale"] = rationale[_i]
        _viz_steps.append(_step_viz)

    return json.dumps({
        "success": True,
        "work_dir": output_path,
        "staged_path": str(staged_path),
        "registry_version": registry_version,
        "warnings": warnings,
        "awaiting_confirmation": True,
        "modality": modality,
        "paradigm": paradigm,
        "analysis_goal": analysis_goal,
        "web_evidence": _evidence_payload,
        "proposal": proposal,
        "reasoning_preview": reasoning_md_text,
        "viz": {
            "type": "pipeline_flow",
            "steps": _viz_steps,
        },
        "note": (
            "Proposal STAGED at middle_process/proposal.staged.json. "
            "plan/ will materialize only after mark_proposal_confirmed("
            "user_decision='confirm'). BEFORE calling mark_proposal_confirmed "
            "you MUST present the FULL pipeline to the user in chat: enumerate "
            "every step from the `viz.steps` field (or `proposal.steps`) as a "
            "numbered list — each with its operator, params, and rationale — "
            "then ask them to confirm / modify / abort. Never ask for "
            "confirmation without showing the steps; the expert cannot judge a "
            "pipeline they cannot see."
        ),
    }, default=str)


def _handle_propose_pipeline(args, **kw):
    """Internal helper: build full YAML config for user review.

    Two step forms accepted:
      - legacy:    steps = ["notch:50", "bandpass:1,40", ...]
      - evidence:  steps = [{operator, params, param_evidence?}, ...]
    """
    # Contract: every proposal must reference a fresh inspection_report.
    err = _require_inspection_report(args)
    if err is not None:
        return json.dumps(err)

    raw_steps = args.get("steps") or []
    if raw_steps and isinstance(raw_steps[0], dict):
        return _handle_propose_pipeline_evidence(args, **kw)

    import yaml as _yaml

    data_path = args["data_path"]
    register_source_path(data_path)
    steps = args["steps"]
    # Analysis_goal is already validated by _handle_plan_pipeline.
    # It flows into pipeline.yaml + plan/goal.json so finalize / build_mini_repo
    # can stamp it into reasoning.md banner and pipeline_record.json.
    analysis_goal = (args.get("analysis_goal") or "generic").strip() or "generic"
    # Enforce that the final pipeline never emits marker /
    # physio (EOG/ECG/Trigger) channels. Do it here so the YAML written below
    # AND the codegen path stay in sync (both consume `steps`).
    # Cleanup is goal-conditional.
    try:
        from easybci_lib.tools.neural_processing.codegen.generator import (
            _enforce_clean_output,
            cleanup_was_appended,
        )
        _orig_steps = list(steps)
        steps = _enforce_clean_output(steps, analysis_goal=analysis_goal)
        _output_cleanup_applied = cleanup_was_appended(_orig_steps, steps)
    except Exception as exc:
        logger.debug("_enforce_clean_output skipped (%s); using raw steps", exc)
        _output_cleanup_applied = False
    rationale = args.get("rationale", [])
    methods = args.get("methods", []) or []
    output_path = args.get("output_path", "")
    modality = args.get("modality", "auto")
    paradigm = args.get("paradigm", "")
    subject_id = args.get("subject_id", "")
    segment_method = args.get("segment_method", "sliding")
    segment_duration = args.get("segment_duration", 2.0)
    stride = args.get("stride", 1.0)
    output_format = args.get("output_format", "auto")
    # Resolve "auto" to a concrete format via format_policy. NWB is the
    # universal default for the preprocessed/ output across every modality;
    # pkl is only chosen when explicitly overridden. resolve_default_format
    # validates the override against {auto, pkl, nwb} and raises ValueError
    # for anything else (legacy hdf5/npz/mat are no longer allowed).
    try:
        from easybci_lib.tools.neural_processing.output.format_policy import (
            resolve_default_format,
        )
        chosen_format = resolve_default_format(modality, output_format)
    except ValueError as _vfe:
        return json.dumps({
            "success": False,
            "error": str(_vfe),
        })

    if not output_path:
        from pathlib import Path as _Path
        data_parent = _Path(data_path).parent
        input_stem = _Path(data_path).stem
        work_dir = str(data_parent.parent / f"{data_parent.name}_preprocess_work_dir")
        output_path = str(data_parent.parent / f"{data_parent.name}_preprocess_work_dir" / "results" / f"{input_stem}_preprocessed.{chosen_format}")
    else:
        from pathlib import Path as _Path
        # Reverse-engineer work_dir from the agent-supplied output_path. Three
        # legitimate shapes the agent passes:
        #   (1) <work_dir>/results/<file>.<ext> → work_dir = parent.parent
        #   (2) <work_dir>/<file>.<ext>         → work_dir = parent
        #   (3) <work_dir>                      → work_dir = output_path itself
        # Shape (3) used to fall through to (2), giving work_dir = parent
        # (the directory ABOVE work_dir). That dropped pipeline.yaml / plan/
        # one level too high and polluted the sibling tree — exactly what users
        # reported when output_path = ".../our_EEG_preprocess_work_dir".
        _out = _Path(output_path)
        _treat_as_dir = (
            _out.is_dir()
            or not _out.suffix
            or _out.name.endswith("_preprocess_work_dir")
        )
        if _treat_as_dir:
            work_dir = str(_out)
        elif _out.parent.name == "results":
            work_dir = str(_out.parent.parent)
        else:
            work_dir = str(_out.parent)

    config_dict = {
        "input": {
            "path": data_path,
            "modality": modality,
            "analysis_goal": analysis_goal,
        },
        "processing": {
            "steps": steps,
            "segment": {
                "method": segment_method,
                "duration": segment_duration,
                "stride": stride,
            },
        },
        "output": {
            "path": output_path,
            "format": chosen_format,
            "format_user_override": output_format,
        },
    }

    if paradigm:
        config_dict["input"]["paradigm"] = paradigm
    if subject_id:
        config_dict["input"]["subject"] = subject_id

    yaml_str = _yaml.dump(config_dict, default_flow_style=False, sort_keys=False, allow_unicode=True)

    # Compute the work_dir layout — no writes to plan/ here. propose now
    # stages the proposal inside middle_process/proposal.staged.json, and
    # plan/ (+ pipeline.yaml at work_dir root) only materializes when the
    # user confirms via mark_proposal_confirmed.
    _Path(work_dir).mkdir(parents=True, exist_ok=True)
    middle_dir = _Path(work_dir) / "middle_process"
    middle_dir.mkdir(parents=True, exist_ok=True)

    # analysis_goal payload that will land in plan/goal.json on
    # confirm. Same single source of truth (proposal → pipeline_record →
    # reasoning.md banner), but the on-disk file appears post-confirm only.
    _goal_payload = {
        "analysis_goal": analysis_goal,
        "modality": modality,
        "paradigm": paradigm,
        "output_cleanup_applied": bool(_output_cleanup_applied),
    }
    goal_json_text = json.dumps(_goal_payload, indent=2, ensure_ascii=False)

    # Shape the web_evidence payload (no disk write). When
    # research_preprocessing ran (status=ok), the payload carries provider +
    # recommendations + applied_to_steps + confidence; otherwise it carries
    # {status: "unavailable", reason: ...} so the post-confirm deliverable
    # still renders the "unavailable" badge consistently.
    _web_evidence = args.get("_web_evidence") or {"status": "unavailable", "reason": "not invoked"}
    _evidence_payload = _shape_web_evidence_payload(
        _web_evidence,
        question=_build_research_question(args)["question"],
    )
    web_evidence_json_text = json.dumps(
        _evidence_payload, indent=2, ensure_ascii=False, default=str,
    )

    # Render the legacy-form proposal payload. The string-step form produces
    # a minimal proposal.json (params:{raw}, param_evidence:{}) — keep that
    # shape so contract_check / finalize fallbacks read it the same way they
    # would after confirm.
    _proposal_steps = []
    for _i, _s in enumerate(steps):
        _op, _, _params_str = _s.partition(":")
        _method = ""
        if _i < len(methods) and isinstance(methods[_i], str):
            _method = methods[_i].strip()
        _proposal_steps.append({
            "operator": _op,
            "method": _method,
            "params": {"raw": _params_str} if _params_str else {},
            "param_evidence": {},
        })
    # Embed fingerprint into proposal.json so finalize / build_mini_repo can
    # populate the README "Input Data" table even after middle_process/ has
    # been cleaned up. Without this, the inspection_report.json sits in
    # middle_process/ and gets archived/removed by Step-14 cleanup before
    # finalize can extract n_channels / sampling_rate / duration.
    _proposal_data_info: dict = {}
    _proposal_input_path: str = ""
    try:
        _insp_path = args.get("inspection_report_path")
        if _insp_path and Path(_insp_path).is_file():
            _insp = json.loads(Path(_insp_path).read_text(encoding="utf-8"))
            _fp = (_insp or {}).get("fingerprint") or {}
            _proposal_data_info = {
                k: v for k, v in {
                    "n_channels": _fp.get("n_channels"),
                    "frequency_hz": _fp.get("sampling_freq_hz"),
                    "duration_seconds": _fp.get("duration_s"),
                    "format": _fp.get("format"),
                }.items() if v not in (None, "")
            }
            _proposal_input_path = str(_insp.get("data_path") or "")
    except Exception:
        pass
    _proposal_payload = {
        "modality": modality,
        "paradigm": paradigm,
        "analysis_goal": analysis_goal,
        "subject_id": subject_id,
        "steps": _proposal_steps,
        "rationale": rationale,
        "output_cleanup_applied": bool(_output_cleanup_applied),
        "web_evidence": _evidence_payload,
        "data_info": _proposal_data_info,
        "input_path": _proposal_input_path,
        "output_format": output_format,
        "chosen_format": chosen_format,
    }
    proposal_json_text = json.dumps(
        _proposal_payload, indent=2, ensure_ascii=False, default=str,
    )

    # Stage the proposal in middle_process/proposal.staged.json — single
    # source of truth between propose and confirm. Each new propose call
    # overwrites it, so iterative modify cycles always land on the latest
    # version. mark_proposal_confirmed reads this envelope and materializes
    # pipeline.yaml + plan/* only when the user actually confirms.
    # NOTE: the legacy string-step path intentionally does NOT pre-render
    # reasoning.md — that file appears post-confirm only via the export-time
    # fallback (_write_reasoning_md), matching the pre-refactor husk-path
    # behaviour. Use the evidence-driven step form to get reasoning.md at
    # confirm time.
    envelope = {
        "version": "1",
        "kind": "legacy",
        "modality": modality,
        "paradigm": paradigm,
        "analysis_goal": analysis_goal,
        "web_evidence": _evidence_payload,
        "root_files": {
            "pipeline.yaml": yaml_str,
        },
        "plan_files": {
            "proposal.json": proposal_json_text,
            "goal.json": goal_json_text,
            "web_evidence.json": web_evidence_json_text,
        },
    }
    staged_path = middle_dir / "proposal.staged.json"
    staged_path.write_text(
        json.dumps(envelope, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    # A fresh propose invalidates any prior confirmation — drop the marker
    # so the agent has to re-confirm against the new staged proposal.
    confirmed_marker = middle_dir / "proposal.confirmed"
    if confirmed_marker.exists():
        confirmed_marker.unlink()

    # Register work_dir with the finalize registry so the run's finally hook
    # can produce a (possibly partial) mini-repo if the LLM stops before
    # calling export_repo. Best-effort — never let registry errors break the
    # tool call.
    try:
        from easybci_lib.tools.neural_processing.export.finalize import set_current_work_dir
        set_current_work_dir(work_dir)
    except Exception:
        pass

    summary_parts = [f"{modality} data: {data_path}"]
    summary_parts.append(f"analysis_goal: {analysis_goal}")
    summary_parts.append(f"Steps: {' → '.join(steps)}")
    if segment_method != "none":
        summary_parts.append(f"Segment: {segment_method} ({segment_duration}s, stride {stride}s)")
    summary_parts.append(f"Output: {config_dict['output']['path']}")

    viz_steps = []
    for i, step in enumerate(steps):
        step_viz = {"name": step.split(":")[0], "params": step}
        if i < len(rationale) and rationale[i]:
            step_viz["rationale"] = rationale[i]
        viz_steps.append(step_viz)

    return json.dumps({
        "success": True,
        "work_dir": work_dir,
        "staged_path": str(staged_path),
        "yaml": yaml_str,
        "summary": " | ".join(summary_parts),
        "analysis_goal": analysis_goal,
        "output_cleanup_applied": bool(_output_cleanup_applied),
        "web_evidence": _evidence_payload,
        "awaiting_confirmation": True,
        "note": (
            "Proposal STAGED at middle_process/proposal.staged.json. "
            "plan/ + pipeline.yaml materialize only after mark_proposal_confirmed("
            "user_decision='confirm'). Present this proposal to the user "
            "in chat (use the `viz` field below), then call "
            "mark_proposal_confirmed with their decision."
        ),
        "negatives_hint": args.get("_negatives_block") or "",
        "viz": {
            "type": "pipeline_flow",
            "steps": viz_steps,
            "yaml": yaml_str,
        },
    })


def _handle_plan_pipeline(args, **kw):
    """Unified pipeline planning: dispatches to suggest or propose mode."""
    start_stage_if_active("plan")
    try:
        return _do_handle_plan_pipeline(args, **kw)
    finally:
        end_stage_if_active()


def _do_handle_plan_pipeline(args, **kw):
    """Unified pipeline planning: dispatches to suggest or propose mode."""
    # Archive any previously-finalized run on this work_dir (same session).
    _maybe_archive_prior_run(args, kw, phase="plan_pipeline")

    # Contract: every plan call must reference a fresh inspection_report.
    insp_err = _require_inspection_report(args)
    if insp_err is not None:
        return json.dumps(insp_err)

    # Contract: analysis_goal is required for all plan_pipeline /
    # suggest_pipeline / propose_pipeline calls. The schema marks it required
    # but server-side enforcement isn't guaranteed across LLM transports —
    # validate here so misuse fails loudly with a useful message instead of
    # silently degrading to "generic" downstream.
    goal = args.get("analysis_goal")
    if not goal or not isinstance(goal, str) or not goal.strip():
        return json.dumps({
            "success": False,
            "error": (
                "analysis_goal is required (one of: classification, "
                "source_localization, feature_extraction, clinical_screening, "
                "exploratory, generic). Infer from the user's natural-language "
                "intent; use 'generic' when the user gave no signal."
            ),
            "field": "analysis_goal",
        })
    allowed_goals = {
        "classification", "source_localization", "feature_extraction",
        "clinical_screening", "exploratory", "generic",
    }
    goal = goal.strip()
    args["analysis_goal"] = goal
    if goal not in allowed_goals:
        return json.dumps({
            "success": False,
            "error": f"analysis_goal={goal!r} not in {sorted(allowed_goals)}",
            "field": "analysis_goal",
        })
    mode = args.get("mode")
    if mode is None:
        # Auto-detect: if steps are provided, user wants propose mode
        if args.get("steps"):
            mode = "propose"
        else:
            mode = "suggest"

    # When research_preprocessing is detected as available,
    # call it once before dispatching to suggest/propose so the pipeline is
    # informed by current SOTA. The result lands on `args` as
    # ``_web_evidence`` and downstream handlers persist it into
    # ``plan/proposal.json:web_evidence`` + reasoning.md banner.
    available, reason = _research_preprocessing_available()
    if available:
        try:
            evidence = _call_research_preprocessing(_build_research_question(args))
        except Exception as exc:
            logger.warning("research_preprocessing raised; treating as unavailable: %s", exc)
            evidence = {"status": "unavailable", "reason": f"call failed: {exc}"}
    else:
        evidence = {"status": "unavailable", "reason": reason}
        # Attach per-provider diagnostics so the downstream payload shaping
        # can render a `(why?)` panel / banner footer with fix hints.
        try:
            from easybci_agent.web_search_registry import diagnose_active_provider
            _, _diag_errors = diagnose_active_provider("search")
            if _diag_errors:
                evidence["diagnostics"] = [
                    f"{e.provider}: {e.reason}" for e in _diag_errors
                ]
        except Exception:
            pass
    args.setdefault("_web_evidence", evidence)

    # T1.4 — render any relevant negative examples into a hint-tone prompt
    # block. The LLM sees this block in `_negatives_block` on the response;
    # the renderer (in suggest/propose) injects it into the user-visible
    # reasoning.md and into the system_prompt extension passed to
    # `_handle_plan_pipeline`. Best-effort: any failure here just leaves the
    # block empty, never aborts the plan.
    try:
        from easybci_lib.tools.neural_processing.experience import ExperienceStore
        from easybci_lib.tools.neural_processing.experience.prompt_block import (
            build_negatives_prompt_block,
        )
        from easybci_lib.constants import get_easybci_home as _get_home_for_neg
        _neg_store = ExperienceStore(store_dir=str(_get_home_for_neg() / "experience"))
        _negs = _neg_store.find_relevant_negatives(
            modality=args.get("modality", "") or "",
            paradigm=args.get("paradigm", "") or "",
            cohort_tag=(args.get("cohort_tag") or ""),
            analysis_goal=goal,
        )
        if _negs:
            args["_negatives_block"] = build_negatives_prompt_block(_negs)
    except Exception as exc:  # noqa: BLE001
        logger.debug("negatives prompt block skipped: %s", exc)

    if mode == "propose":
        return _handle_propose_pipeline(args, **kw)
    return _handle_suggest_pipeline(args, **kw)


def _shape_web_evidence_payload(evidence: dict, *, question: str = "") -> dict:
    """Coerce the raw research_preprocessing response into the canonical
    ``web_evidence`` shape stored in ``plan/web_evidence.json`` and surfaced
    via ``plan/proposal.json:web_evidence``.

    Three branches:
      - "unavailable": pass-through {"status": "unavailable", "reason": ...}
      - successful research call: extract recommendations / citations /
        confidence / provider, drop the verbose query log
      - anything else (no success flag): mark as unavailable with the raw
        error so reasoning.md can still tell the user what happened.

    The ``diagnostics`` (list[str]) field is preserved when present so the
    WebUI / CLI banner can render per-provider activation hints. ``reason``
    remains a string for backwards compatibility.
    """
    if not isinstance(evidence, dict) or evidence.get("status") == "unavailable":
        out = {
            "status": "unavailable",
            "reason": (evidence or {}).get("reason", "unknown"),
            "question": question,
        }
        diags = (evidence or {}).get("diagnostics")
        if diags:
            out["diagnostics"] = diags
        return out
    if not evidence.get("success") and evidence.get("status") != "ok":
        out = {
            "status": "unavailable",
            "reason": evidence.get("error") or "research_preprocessing returned failure",
            "question": question,
        }
        if evidence.get("diagnostics"):
            out["diagnostics"] = evidence["diagnostics"]
        return out
    payload = {
        "status": "ok",
        "question": question,
        "provider": evidence.get("provider"),
        "level": evidence.get("level"),
        "recommendations": evidence.get("recommendations") or [],
        "parameters_extracted": evidence.get("parameters_extracted") or [],
        "citations": evidence.get("citations") or [],
        "confidence": evidence.get("confidence"),
    }
    if evidence.get("conflicts") is not None:
        payload["conflicts"] = evidence["conflicts"]
    if evidence.get("applied_to_steps") is not None:
        payload["applied_to_steps"] = evidence["applied_to_steps"]
    # Surface relevance-gate drops so the user / reasoning.md can see what
    # was filtered. discarded is a list of dicts each with a "reason" field.
    discarded = (evidence or {}).get("discarded") or []
    if discarded:
        payload["discarded_count"] = len(discarded)
        reasons: dict[str, int] = {}
        for d in discarded:
            r = d.get("reason") or "unknown"
            reasons[r] = reasons.get(r, 0) + 1
        payload["discard_reasons"] = reasons
    return payload


def _research_preprocessing_available() -> tuple[bool, str]:
    """Return ``(available, reason)`` for the research_preprocessing must-call detection.

    Two conditions must hold:
      1. ``research_preprocessing`` is in the registry with a handler.
      2. A web-search provider is currently active (strict-availability:
         the provider is registered, supports search, AND its
         ``is_available()`` returns True).

    The empty ``reason`` is returned when both hold; otherwise an actionable
    diagnostic enumerating each registered provider's specific failure
    (``tavily: TAVILY_API_KEY environment variable not set; exa: EXA_API_KEY
    environment variable not set``) so reasoning.md / WebUI / CLI banner can
    render a fix path rather than a generic "unavailable".
    """
    entry = registry.get_entry("research_preprocessing")
    if entry is None or entry.handler is None:
        return False, "research_preprocessing tool not registered"
    try:
        from easybci_agent.web_search_registry import diagnose_active_provider
    except ImportError:
        return False, "easybci_agent.web_search_registry not importable"
    try:
        provider, errors = diagnose_active_provider("search")
    except Exception as exc:
        return False, f"web search provider lookup failed: {exc}"
    if provider is None:
        if errors:
            joined = "; ".join(f"{e.provider}: {e.reason}" for e in errors)
            return False, f"no usable web search backend ({joined})"
        return False, "no web search provider configured"
    return True, ""


def _build_research_question(args: dict) -> dict:
    """Compose the canonical research_preprocessing call args from a
    plan_pipeline call's args. Keeps the two contracts in lock-step."""
    paradigm = args.get("paradigm") or "general"
    modality = args.get("modality") or "unknown"
    goal = (args.get("analysis_goal") or "generic").strip() or "generic"
    fingerprint = args.get("fingerprint") or {}
    n_ch = fingerprint.get("n_channels") if isinstance(fingerprint, dict) else None
    fs = (
        fingerprint.get("frequency_hz") or fingerprint.get("frequency")
        if isinstance(fingerprint, dict)
        else None
    )
    bits = [f"{paradigm} preprocessing parameters for {goal}"]
    if n_ch:
        bits.append(f"{n_ch}-channel {modality}")
    elif modality:
        bits.append(modality.upper())
    if fs:
        bits.append(f"at {fs} Hz")
    bits.append(
        "SOTA recommendations for bandpass / notch / ICA / artifact rejection / segmentation"
    )
    question = "; ".join(bits)
    return {
        "question": question,
        "modality": modality,
        "paradigm": paradigm,
        "context": {
            "fingerprint": fingerprint,
            "analysis_goal": goal,
        },
    }


def _call_research_preprocessing(call_args: dict) -> dict:
    """Invoke ``research_preprocessing`` via the registry and parse the
    JSON envelope. Errors return ``{"status": "unavailable", "reason": ...}``
    so the caller never has to wrap in try/except.
    """
    entry = registry.get_entry("research_preprocessing")
    if entry is None or entry.handler is None:
        return {"status": "unavailable", "reason": "tool not registered"}
    try:
        raw = entry.handler(call_args)
    except Exception as exc:
        return {"status": "unavailable", "reason": f"handler raised: {exc}"}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (ValueError, TypeError) as exc:
            return {
                "status": "unavailable",
                "reason": f"non-JSON response: {exc}",
            }
    return {"status": "unavailable", "reason": "unrecognised response type"}


def _handle_list_data(args, **kw):
    directory = args.get("directory", ".")
    pattern = args.get("pattern", "*")

    NEURAL_EXTENSIONS = {
        ".edf", ".bdf", ".fif", ".set", ".cnt", ".gdf", ".ds",
        ".nwb", ".h5", ".hdf5",
        ".mat",
        ".csv", ".tsv",
    }

    dir_path = Path(directory).expanduser().resolve()
    if not dir_path.is_dir():
        return json.dumps({"error": f"Directory not found: {directory}"})

    files = []
    for f in sorted(dir_path.rglob(pattern)):
        if f.is_file() and f.suffix.lower() in NEURAL_EXTENSIONS:
            files.append(str(f))
            if len(files) >= 100:
                break

    return json.dumps({
        "total_files": len(files),
        "neural_files": files[:50],
        "directory": str(dir_path),
        "truncated": len(files) >= 100,
    })


def _handle_generate_code(args, **kw):
    """Write the full code bundle: pipeline.py + qc.py + run.py + requirements.txt
    (+ build_ai_ready.py iff events present OR label_config provided).

    Pre-existing files are archived to ``<work_dir>/middle_process/code/`` only
    when their content would change — byte-identical regenerations are no-ops.
    The archive location is the root ``middle_process/code/`` per the layout
    contract — ``<work_dir>/code/middle_process/`` is forbidden.
    """
    start_stage_if_active("codegen", with_daemon=True)
    try:
        return _do_handle_generate_code(args, **kw)
    finally:
        end_stage_if_active(with_daemon=True)


def _do_handle_generate_code(args, **kw):
    # Archive any previously-finalized run on this work_dir (same session).
    _maybe_archive_prior_run(args, kw, phase="generate_code")

    # Phase 1 → Phase 2 gate: <work_dir>/middle_process/proposal.confirmed is
    # the GROUND TRUTH. It is written only by mark_proposal_confirmed after
    # the user explicitly accepts; no other path produces it. We check the
    # marker first; the LLM-supplied ``proposal_confirmed=True`` parameter is
    # no longer required (it was a redundant declaration of what the marker
    # already proves). ``inspection_report_path`` is similarly auto-discovered
    # from work_dir by ``_require_inspection_report`` when the arg is absent.
    work_dir_check = _resolve_work_dir_from_args(args)
    if work_dir_check is None:
        return json.dumps({
            "success": False,
            "error": (
                "generate_code requires work_dir (or output_dir / output_path) "
                "so the proposal.confirmed marker and inspection_report.json "
                "can be located on disk."
            ),
            "fix_hint": (
                "Pass work_dir=<your_preprocess_work_dir>."
            ),
        })

    marker = work_dir_check / "middle_process" / "proposal.confirmed"
    if not marker.is_file():
        return json.dumps({
            "success": False,
            "error": (
                f"proposal.confirmed marker missing at {marker}. The user "
                "has not yet accepted the proposal — generate_code is the "
                "Phase 2 entry point and only runs post-confirmation."
            ),
            "fix_hint": (
                "Present the staged proposal to the user (see "
                "<work_dir>/middle_process/proposal.staged.json), then call "
                "mark_proposal_confirmed(work_dir=<work_dir>, "
                "user_decision='confirm') once they accept. THAT step "
                "materializes plan/ + writes this marker."
            ),
        })

    insp_err = _require_inspection_report(args)
    if insp_err is not None:
        return json.dumps(insp_err)

    from datetime import datetime as _dt
    from easybci_lib.tools.neural_processing.codegen.generator import (
        generate_build_ai_ready_script,
        generate_pipeline_script,
        generate_qc_script_v2,
        generate_requirements,
        generate_run_script_v2,
        generate_vis_script,
    )

    steps = args["steps"]
    data_info = args.get("data_info") or {}
    modality = args.get("modality", "eeg")
    analysis_goal = args.get("analysis_goal") or "generic"
    label_config = args.get("label_config")
    segment_duration = float(args.get("segment_duration", 2.0))
    stride = float(args.get("stride", 1.0))
    work_dir = args.get("work_dir") or args.get("output_dir")

    # output_format propagation: prefer explicit arg, otherwise read from
    # the staged proposal (written by _handle_propose_pipeline) under
    # middle_process/proposal.staged.json. Defaults to "auto" → resolved by
    # codegen via resolve_default_format(modality, output_format).
    output_format = args.get("output_format")
    if output_format is None and work_dir:
        try:
            _staged = Path(work_dir) / "middle_process" / "proposal.staged.json"
            if _staged.is_file():
                _staged_obj = json.loads(_staged.read_text(encoding="utf-8"))
                output_format = _staged_obj.get("output_format")
        except Exception:
            pass
    if output_format is None:
        output_format = "auto"

    if not work_dir:
        return json.dumps({
            "success": False,
            "error": (
                "generate_code requires work_dir (or legacy output_dir). "
                "Pass <your_preprocess_work_dir> so the bundle is written to "
                "<work_dir>/code/."
            ),
        })

    from easybci_lib.tools.neural_processing.preprocess.analysis_goals import REGISTRY

    events = data_info.get("events") or []
    has_events = bool(events) and len(events) > 0
    has_labels = bool(label_config)
    _goal_spec = REGISTRY.get(analysis_goal)
    _goal_allows_ai_ready = _goal_spec.produces_ai_ready if _goal_spec else True

    # preprocessed/ is NWB-only — output_format is no longer a meaningful
    # branch point. We keep the variable around as informational metadata
    # (echoed into pipeline_record), but codegen no longer consumes it.
    needs_ai_ready = (has_events or has_labels) and _goal_allows_ai_ready
    ai_ready_skipped_reason = None
    if (has_events or has_labels) and not _goal_allows_ai_ready:
        ai_ready_skipped_reason = "goal_not_ai_ready"

    # Load inspection_report (path validated by the gate above). Fail open if
    # the file is empty/malformed (e.g. the internal regen path passes a stub
    # path when no real inspection report exists — codegen still proceeds, just
    # without inspection-driven hints).
    inspection_report = None
    try:
        from easybci_lib.tools.neural_processing.io.inspection_report import (
            load_inspection_report,
        )
        insp_path = args.get("inspection_report_path")
        if insp_path and Path(insp_path).exists():
            inspection_report = load_inspection_report(Path(insp_path)).to_dict()
    except Exception as exc:  # noqa: BLE001
        logger.debug("inspection_report load skipped: %s", exc)

    files = {
        "pipeline.py": generate_pipeline_script(
            steps=steps, data_info=data_info, modality=modality,
            analysis_goal=analysis_goal,
            inspection_report=inspection_report,
        ),
        "qc.py": generate_qc_script_v2(
            steps=steps, data_info=data_info, modality=modality,
            analysis_goal=analysis_goal,
            inspection_report=inspection_report,
        ),
        "vis.py": generate_vis_script(
            modality=modality, analysis_goal=analysis_goal,
            inspection_report=inspection_report,
        ),
        "run.py": generate_run_script_v2(has_build_ai_ready=needs_ai_ready, has_vis=True),
        "requirements.txt": generate_requirements(),
    }
    if needs_ai_ready:
        files["build_ai_ready.py"] = generate_build_ai_ready_script(
            modality=modality, analysis_goal=analysis_goal,
            events_present=has_events, label_config=label_config,
            segment_duration=segment_duration, stride=stride,
        )

    code_dir = Path(work_dir) / "code"
    code_dir.mkdir(parents=True, exist_ok=True)
    # Archive into <work_dir>/middle_process/code/ (root middle_process, per
    # the layout contract — code/middle_process/ is forbidden).
    archive_dir = Path(work_dir) / "middle_process" / "code"
    written = []
    for name, body in files.items():
        target = code_dir / name
        if target.exists():
            try:
                existing = target.read_text(encoding="utf-8")
            except OSError:
                existing = None
            if existing == body:
                continue
            archive_dir.mkdir(parents=True, exist_ok=True)
            ts = _dt.now().strftime("%Y%m%d_%H%M%S")
            target.rename(archive_dir / f"{Path(name).stem}_{ts}{Path(name).suffix}")
        target.write_text(body, encoding="utf-8")
        written.append(name)

    # Static routing-safety check on the freshly-written code/. Multi-input
    # mode is mandatory for safety: any stage script that still derives
    # (sub, ses) from the raw stem is a regression of the multi-session fix.
    try:
        from easybci_lib.tools.neural_processing.codegen.code_standard_check import (
            run_routing_safety_check,
        )
        routing_violations = run_routing_safety_check(code_dir)
    except Exception as exc:  # noqa: BLE001
        logger.debug("routing-safety check unavailable: %s", exc)
        routing_violations = []
    if routing_violations:
        return json.dumps({
            "success": False,
            "error": (
                "generate_code produced scripts that violate the routing-safety "
                "contract (stem-based subject_id derivation or _find_preprocessed "
                "fallback). The routing table at middle_process/inputs_routing.json "
                "is the single source of truth — generated code MUST consume it."
            ),
            "routing_violations": routing_violations,
            "fix_hint": "Re-generate (delete code/pipeline.py and re-invoke).",
        })

    # AST safety scan: catch generated scripts that write back into protected
    # source data. Defense-in-depth — file_safety + approval already cover the
    # other surfaces, but pipeline.py is a subprocess and can side-step them.
    try:
        from easybci_lib.tools.neural_processing.codegen.safety_scan import (
            scan_script as _safety_scan_script,
            CodegenSafetyViolation as _CodegenSafetyViolation,
        )
        for _stage_name in ("pipeline", "qc", "vis", "build_ai_ready", "run"):
            _script_path = code_dir / f"{_stage_name}.py"
            if not _script_path.is_file():
                continue
            try:
                _safety_scan_script(str(_script_path), work_dir=str(work_dir))
            except _CodegenSafetyViolation as exc:
                return json.dumps({
                    "success": False,
                    "error": (
                        "generate_code produced a script that writes back into "
                        "registered source data. This violates the source-data "
                        "immutability contract — choose an output path inside "
                        "the work_dir mini-repo instead."
                    ),
                    "safety_violation": {
                        "file": exc.script_path,
                        "line": exc.line,
                        "target": exc.target,
                        "reason": exc.reason,
                    },
                    "fix_hint": (
                        f"Rewrite code/{_stage_name}.py so the listed write target "
                        "lands under preprocessed_output/ or middle_process/."
                    ),
                })
    except ImportError:
        pass

    payload = {
        "success": True,
        "written": written,
        "code_dir": str(code_dir),
        "has_build_ai_ready": needs_ai_ready,
        "analysis_goal": analysis_goal,
        "work_dir": str(work_dir),
        **_lint_generated_pipeline(code_dir / "pipeline.py"),
    }
    if ai_ready_skipped_reason is not None:
        payload["ai_ready_skipped_reason"] = ai_ready_skipped_reason
    return json.dumps(payload)


def _lint_generated_pipeline(pipeline_path: Path) -> dict:
    """T7 P-C — run the code-standard checker on the freshly-written
    pipeline.py.  Returns a dict snippet to merge into the generate_code
    success payload (or a structured error block when the script violates
    the standard so the agent can repair via write_file).
    """
    if not pipeline_path.exists():
        return {}
    try:
        from easybci_lib.tools.neural_processing.codegen.code_standard_check import (
            check_pipeline_code_standard,
            has_blocking_violations,
            violations_to_agent_error,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("code_standard_check unavailable: %s", exc)
        return {}
    violations = check_pipeline_code_standard(pipeline_path)
    if not violations:
        return {"code_standard": {"version": "1.0.0", "violations": []}}
    payload = {
        "code_standard": {
            "version": "1.0.0",
            "violations": [
                {"rule": v["rule"], "line": v["line"],
                 "message": v["message"], "blocking": v["blocking"]}
                for v in violations
            ],
        }
    }
    if has_blocking_violations(violations):
        payload["code_standard_error"] = violations_to_agent_error(
            violations, pipeline_path,
        )
    return payload


def _handle_export_repo(args, **kw):
    # If code_only=true, delegate to code generation helper
    if args.get("code_only"):
        return _handle_generate_code(args, **kw)

    from easybci_lib.tools.neural_processing.export.repo_builder import build_mini_repo

    output_dir = args.get("output_dir")
    if not output_dir:
        return json.dumps({
            "success": False,
            "error": (
                "export_repo requires output_dir (path for the mini-repo). "
                "Pass output_dir=<your_preprocess_work_dir>, or set "
                "code_only=true if you only want to (re)generate the code bundle "
                "into <work_dir>/code/."
            ),
        })
    output_error = check_output_path(output_dir)
    if output_error:
        return json.dumps({"success": False, "error": output_error})

    # Register the work_dir so the run's finally hook knows where to finalize.
    try:
        from easybci_lib.tools.neural_processing.export.finalize import set_current_work_dir
        set_current_work_dir(output_dir)
    except Exception:
        pass

    pipeline_record = args.get("pipeline_record", {})

    # Merge top-level reasoning into pipeline_record so it renders in reasoning.md
    top_level_reasoning = args.get("reasoning")
    if top_level_reasoning and isinstance(top_level_reasoning, dict):
        existing = pipeline_record.get("reasoning") or {}
        existing.update(top_level_reasoning)
        pipeline_record["reasoning"] = existing

    # Merge top-level step_states into pipeline_record for per-step explainability
    top_level_states = args.get("step_states")
    if top_level_states and isinstance(top_level_states, list):
        pipeline_record["step_states"] = top_level_states

    # Fix B2 — Backfill input_path / data_info from a pending QC payload that
    # matches this work_dir. The agent (especially the evidence-driven propose
    # path) frequently lands in export_repo without populating these args, which
    # leaves the README with "?" / "unknown" everywhere.  preprocess_neural
    # already stashed everything we need; reuse it here BEFORE build_mini_repo
    # so README / pipeline_record / config.yaml all see the same recovered values.
    _input_path = args.get("input_path", "")
    _data_info = args.get("data_info", {}) or {}
    _modality = args.get("modality", "eeg")
    _subject_id = args.get("subject_id", "")
    _peeked = _peek_qc_payload_for(output_dir)
    if _peeked:
        if not _input_path:
            _input_path = _peeked.get("data_path", "") or _input_path
        if not _subject_id:
            _subject_id = _peeked.get("subject_id", "") or _subject_id
        if not _data_info:
            states = _peeked.get("step_states") or []
            before = (states[0].get("before") if states and isinstance(states[0], dict) else {}) or {}
            channels = list(before.get("channels") or _peeked.get("channels") or [])
            recovered = {
                "n_channels": before.get("n_channels") or (len(channels) if channels else None),
                "frequency_hz": before.get("frequency") or _peeked.get("freq"),
                "duration_seconds": before.get("duration_s"),
                "channels": channels,
                "file": _peeked.get("data_path", ""),
            }
            _data_info = {k: v for k, v in recovered.items() if v not in (None, "", [])}

    result = build_mini_repo(
        output_dir=output_dir,
        steps=args["steps"],
        data_info=_data_info,
        pipeline_record=pipeline_record,
        input_path=_input_path,
        modality=_modality,
        segment_duration=args.get("segment_duration", 2.0),
        stride=args.get("stride", 1.0),
        subject_id=_subject_id,
        paradigm=args.get("paradigm", ""),
        pkl_path=args.get("pkl_path", ""),
        force=args.get("force", False),
        label_config=args.get("label_config"),
        split_config=args.get("split_config"),
    )

    # Fix A-2 — Drain any QC payloads still pending for this work_dir and
    # write their figures + per-session QC reports.  Defensive: when the
    # agent goes preprocess_neural → export_repo and skips quality_check,
    # the figures would otherwise stay in _PENDING_QC_PAYLOADS until the
    # process exits.  Idempotent w.r.t. quality_check (which already drained
    # its specific (data_path, modality) key) — this only catches whatever
    # quality_check missed.
    try:
        drained = _drain_qc_payloads_for(output_dir)
        if drained:
            artifacts = [_write_qc_artifacts(p) for p in drained]
            result.setdefault("post_export_qc_artifacts", [a for a in artifacts if a])
    except Exception as exc:
        logger.warning("Post-export QC artifact write failed: %s", exc)

    # If this is a fresh export (not cached), record the stage in experience store
    if result.get("success") and not result.get("cached"):
        try:
            from easybci_lib.tools.neural_processing.experience import ExperienceStore, create_processing_record
            store = ExperienceStore()
            record = create_processing_record(
                data_path=args.get("input_path", ""),
                modality=args.get("modality", "eeg"),
                paradigm=args.get("paradigm", ""),
                initial_steps=args["steps"],
                final_steps=args["steps"],
                success=True,
                stage="exported",
            )
            store.save_record(record)
        except Exception:
            pass

    # Post-export hygiene: clean middle_process/ as soon as the export succeeds
    # end-to-end, instead of waiting for the run-level finalize hook. Reuses
    # finalize's cleanup so env semantics + best-effort behavior stay identical.
    # Gating uses the REAL export signals (build_mini_repo does not surface a
    # contract_check / layout_repair key): Gate A = validate_mini_repo().ok,
    # Gate B = verify_and_repair().unrepairable is empty, Gate C = env-pin.
    _cleanup_reason = None
    _cleaned = False
    try:
        _wd_path = Path(output_dir) if isinstance(output_dir, str) else output_dir

        # Gate A: contract must be clean.
        from easybci_lib.tools.neural_processing.export.contract_check import (
            validate_mini_repo,
        )
        _cc_ok = bool(validate_mini_repo(str(_wd_path)).get("ok", False))

        # Gate B: layout must converge with no unrepairable residual. Run on the
        # hot tool-return path with allow_subprocess=False (matching the tool
        # boundary convention); the finalize hook re-runs it with subprocess
        # allowed as defense-in-depth. Idempotent.
        _vr_residual = []
        if _cc_ok:
            from easybci_lib.tools.neural_processing.export.layout_repair import (
                verify_and_repair,
            )
            _vr = verify_and_repair(
                _wd_path, analysis_goal=args.get("analysis_goal"),
                allow_subprocess=False, dry_run=False, write_report=True,
            )
            _vr_residual = list(_vr.unrepairable or [])

        from easybci_lib.tools.neural_processing.export.finalize import (
            _cleanup_middle_process,
            _should_keep_middle_process,
        )

        if not _cc_ok:
            _cleanup_reason = "contract_check_failed"
        elif _vr_residual:
            _cleanup_reason = "verify_and_repair_residual"
        elif _should_keep_middle_process():
            _cleanup_reason = "env_pinned"
        else:
            _mp = _wd_path / "middle_process"
            _existed = _mp.is_dir()
            _cleanup_middle_process(_wd_path, "ok")  # honors env-pin internally too
            _cleaned = _existed and not _mp.exists()
            if not _cleaned and _existed:
                _cleanup_reason = "cleanup_failed"
    except Exception:
        logger.exception("middle_process cleanup failed at export_repo tail — ignoring")
        _cleanup_reason = "cleanup_exception"

    # Append hygiene fields to the result (backward-compatible add).
    result["middle_process_cleaned"] = _cleaned
    result["cleanup_skipped_reason"] = _cleanup_reason

    return json.dumps(result, default=str)


def _handle_bin_spikes(args, **kw):
    from easybci_lib.tools.neural_processing.preprocess.spikes import bin_spikes

    data_path = args["data_path"]
    register_source_path(data_path)
    bin_frequency = args.get("bin_frequency", 100.0)

    data_dict = _load_cached(data_path, modality="spike")
    spike_trains = data_dict.get("spike_trains")
    if spike_trains is None:
        spike_trains = data_dict.get("data", [])

    result = bin_spikes(
        spike_trains,
        bin_frequency=bin_frequency,
        duration=data_dict.get("duration"),
        unit_names=data_dict.get("channels"),
    )
    return json.dumps({
        "success": True,
        "data_shape": list(result["data"].shape),
        "frequency": result["frequency"],
        "n_units": result["data"].shape[0],
    })


def _handle_batch_process(args, **kw):
    from easybci_lib.tools.neural_processing.batch.processor import batch_process
    import glob as _glob

    pattern = args["pattern"]
    steps = args["steps"]
    output_dir = args["output_dir"]

    matched_files = _glob.glob(pattern)
    for matched_path in matched_files:
        register_source_path(matched_path)

    output_error = check_output_path(output_dir)
    if output_error:
        return json.dumps({"success": False, "error": output_error})

    modality = args.get("modality", "auto")
    segment_duration = args.get("segment_duration", 2.0)
    stride = args.get("stride", 1.0)
    max_workers = args.get("max_workers", 4)

    # Environment recommendation for large batches
    try:
        from easybci_lib.tools.environments.advisor import recommend_environment
        import os
        total_size_mb = sum(os.path.getsize(f) for f in matched_files) / (1024 * 1024)
        env_rec = recommend_environment(
            data_size_mb=total_size_mb,
            n_subjects=len(matched_files),
            operation="batch",
        )
        if env_rec.recommended != "local":
            return json.dumps({
                "success": True,
                "environment_recommendation": env_rec.to_dict(),
                "message": (
                    f"Recommend using {env_rec.recommended} environment: {env_rec.reason} "
                    f"Set environment in config or proceed with local execution."
                ),
                "n_files": len(matched_files),
            }, default=str)
    except Exception:
        pass

    result = batch_process(
        pattern=pattern,
        steps=steps,
        output_dir=output_dir,
        modality=modality,
        segment_duration=segment_duration,
        stride=stride,
        max_workers=max_workers,
    )

    # Generate batch summary dashboard with group stats + exclusion recommendations
    try:
        from easybci_lib.tools.neural_processing.batch.summary import generate_batch_summary, save_batch_summary
        batch_results = result.get("results", [])
        if batch_results:
            summary_report = generate_batch_summary(
                batch_results=batch_results,
                output_dir=output_dir,
                pipeline=steps,
            )
            summary_paths = save_batch_summary(summary_report, output_dir)
            result["batch_summary"] = summary_report.to_dict()
            result["batch_summary_text"] = summary_report.summary_text
            result["batch_summary_paths"] = summary_paths
    except Exception as exc:
        logger.debug("Batch summary generation failed: %s", exc)

    return json.dumps(result, default=str)


# ---------------------------------------------------------------------------
# Registration — 10 primary tools + backward-compatible aliases
# ---------------------------------------------------------------------------

# 1. inspect_data (primary) + inspect_neural (alias)
registry.register(
    name="inspect_data",
    toolset="neural",
    schema=INSPECT_DATA_SCHEMA,
    handler=_handle_inspect_data,
    check_fn=_check_neural_requirements,
    emoji="\U0001f9e0",
)

registry.register(
    name="inspect_neural",
    toolset="neural",
    schema=INSPECT_DATA_SCHEMA,
    handler=_handle_inspect_data,
    check_fn=_check_neural_requirements,
    emoji="\U0001f9e0",
)

# 1b. deep_inspect — Phase 1 full-data scan (writes inspection_report.json)
registry.register(
    name="deep_inspect",
    toolset="neural",
    schema=DEEP_INSPECT_SCHEMA,
    handler=_handle_deep_inspect,
    check_fn=_check_neural_requirements,
    emoji="\U0001f50d",
)

# 1c. mark_proposal_confirmed — Phase 1 → Phase 2 hand-off marker
registry.register(
    name="mark_proposal_confirmed",
    toolset="neural",
    schema=MARK_PROPOSAL_CONFIRMED_SCHEMA,
    handler=_handle_mark_proposal_confirmed,
    check_fn=_check_neural_requirements,
    emoji="✅",
)

# 2. preprocess_neural
registry.register(
    name="preprocess_neural",
    toolset="neural",
    schema=PREPROCESS_NEURAL_SCHEMA,
    handler=_handle_preprocess_neural,
    check_fn=_check_neural_requirements,
    emoji="⚡",
)

# 3. quality_check
registry.register(
    name="quality_check",
    toolset="neural",
    schema=QUALITY_CHECK_SCHEMA,
    handler=_handle_quality_check,
    check_fn=_check_neural_requirements,
    emoji="✅",
)

# 3b. resume_preprocessing — explicit continue-where-you-left-off entrypoint
registry.register(
    name="resume_preprocessing",
    toolset="neural",
    schema=RESUME_PREPROCESSING_SCHEMA,
    handler=_handle_resume_preprocessing,
    check_fn=_check_neural_requirements,
    emoji="⏭",
)

# 4. segment_data
registry.register(
    name="segment_data",
    toolset="neural",
    schema=SEGMENT_DATA_SCHEMA,
    handler=_handle_segment_data,
    check_fn=_check_neural_requirements,
    emoji="✂️",
)

# 5. save_processed (primary) + confirm_output_format (alias)
registry.register(
    name="save_processed",
    toolset="neural",
    schema=SAVE_PROCESSED_SCHEMA,
    handler=_handle_save_processed,
    check_fn=_check_neural_requirements,
    emoji="\U0001f4be",
)

registry.register(
    name="confirm_output_format",
    toolset="neural",
    schema=SAVE_PROCESSED_SCHEMA,
    handler=_handle_save_processed,
    check_fn=_check_neural_requirements,
    emoji="\U0001f4e4",
)

# 6. plan_pipeline (primary) + suggest_pipeline, propose_pipeline (aliases)
registry.register(
    name="plan_pipeline",
    toolset="neural",
    schema=PLAN_PIPELINE_SCHEMA,
    handler=_handle_plan_pipeline,
    check_fn=_check_neural_requirements,
    emoji="\U0001f4a1",
)

registry.register(
    name="suggest_pipeline",
    toolset="neural",
    schema=PLAN_PIPELINE_SCHEMA,
    handler=_handle_plan_pipeline,
    check_fn=_check_neural_requirements,
    emoji="\U0001f4a1",
)

registry.register(
    name="propose_pipeline",
    toolset="neural",
    schema=PLAN_PIPELINE_SCHEMA,
    handler=_handle_plan_pipeline,
    check_fn=_check_neural_requirements,
    emoji="\U0001f4cb",
)

# 7. list_data
registry.register(
    name="list_data",
    toolset="neural",
    schema=LIST_DATA_SCHEMA,
    handler=_handle_list_data,
    check_fn=_check_neural_requirements,
    emoji="\U0001f4c2",
)

# 8. export_repo (primary) + generate_code (alias)
registry.register(
    name="export_repo",
    toolset="neural",
    schema=EXPORT_REPO_SCHEMA,
    handler=_handle_export_repo,
    check_fn=_check_neural_requirements,
    emoji="\U0001f4e6",
)


def _handle_generate_code_alias(args, **kw):
    # The generate_code alias's name implies "code only" — force the flag so
    # callers that omit it don't fall through to the full repo path (which
    # demands output_dir and would otherwise fail with KeyError).
    if not args.get("code_only"):
        args = {**args, "code_only": True}
    return _handle_export_repo(args, **kw)


registry.register(
    name="generate_code",
    toolset="neural",
    schema=EXPORT_REPO_SCHEMA,
    handler=_handle_generate_code_alias,
    check_fn=_check_neural_requirements,
    emoji="\U0001f4dd",
)

# 9. bin_spikes
registry.register(
    name="bin_spikes",
    toolset="neural",
    schema=BIN_SPIKES_SCHEMA,
    handler=_handle_bin_spikes,
    check_fn=_check_neural_requirements,
    emoji="\U0001f52c",
)

# 10. batch_process
registry.register(
    name="batch_process",
    toolset="neural",
    schema=BATCH_PROCESS_SCHEMA,
    handler=_handle_batch_process,
    check_fn=_check_neural_requirements,
    emoji="\U0001f504",
)


# ---------------------------------------------------------------------------
# 11. compare_pipelines — A/B pipeline comparison
# ---------------------------------------------------------------------------

COMPARE_PIPELINES_SCHEMA = {
    "name": "compare_pipelines",
    "description": (
        "Compare two preprocessing pipeline variants on the same data. "
        "Runs both pipelines, computes QC metrics for each, and generates a "
        "comparative report with a recommendation of which pipeline is better."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "data_path": {
                "type": "string",
                "description": "Path to the neural data file.",
            },
            "pipeline_a": {
                "type": "array",
                "items": {"type": "string"},
                "description": "First pipeline variant (list of step strings).",
            },
            "pipeline_b": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Second pipeline variant (list of step strings).",
            },
            "modality": {
                "type": "string",
                "description": "Data modality (auto, eeg, seeg, ecog, meg, spike, fnirs).",
                "default": "auto",
            },
        },
        "required": ["data_path", "pipeline_a", "pipeline_b"],
    },
}


def _handle_compare_pipelines(args, **kw):
    import json as _json
    from easybci_lib.tools.neural_processing.quality.comparison import compare_pipelines

    data_path = args["data_path"]
    register_source_path(data_path)
    pipeline_a = args["pipeline_a"]
    pipeline_b = args["pipeline_b"]
    modality = args.get("modality", "auto")

    data_dict = _load_cached(data_path, modality=modality)

    result = compare_pipelines(data_dict, pipeline_a, pipeline_b)

    response = result.to_dict()
    response["comparison_text"] = result.to_text()
    return _json.dumps(response, default=str)


registry.register(
    name="compare_pipelines",
    toolset="neural",
    schema=COMPARE_PIPELINES_SCHEMA,
    handler=_handle_compare_pipelines,
    check_fn=_check_neural_requirements,
    emoji="\U0001f4ca",
)

RESEARCH_PREPROCESSING_SCHEMA = {
    "name": "research_preprocessing",
    "description": (
        "Search academic sources and documentation for preprocessing best practices "
        "when the standard domain skills don't cover the scenario. Use when facing "
        "non-standard paradigms, unusual data characteristics, or repeated QC failures. "
        "Returns evidence-based pipeline recommendations with source citations."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "Specific preprocessing question to research (e.g. 'best preprocessing for auditory attention decoding EEG')",
            },
            "modality": {
                "type": "string",
                "enum": ["eeg", "seeg", "ecog", "meg", "spike", "fnirs", "ieeg"],
                "description": "Neural data modality",
            },
            "paradigm": {
                "type": "string",
                "description": "Processing paradigm or task (e.g. 'motor_imagery', 'neurofeedback', 'TMS-EEG')",
            },
            "context": {
                "type": "object",
                "description": "Additional context: data fingerprint, failed_steps, qc_issues, dataset_name",
            },
        },
        "required": ["question"],
    },
}


def _reuse_suppression_envelope(args: dict) -> str | None:
    """Shared Reuse-Mode gate for the research tools. Returns a soft-suppressed
    JSON envelope string when the current run is in Reuse Mode, else None.

    Locates work_dir from the args (rare for research calls) or falls back to
    finalize.get_current_work_dir() (set by export/propose). Defensive: any
    failure returns None so research proceeds normally.
    """
    try:
        _wd = _resolve_work_dir_from_args(args)
        if _wd is None:
            from easybci_lib.tools.neural_processing.export.finalize import (
                get_current_work_dir,
            )
            _cur = get_current_work_dir()
            _wd = Path(_cur) if _cur else None
        if _wd is None:
            return None
        from easybci_lib.tools.neural_processing.export.layout_repair import (
            reuse_mode_guard,
        )
        _reuse = reuse_mode_guard(_wd)
        if _reuse.active:
            from easybci_agent.i18n import t
            return json.dumps({
                "success": False,
                "error_kind": "reuse_mode_suppressed",
                "suppressed": True,
                "reuse_source": _reuse.source,
                "fix_hint": t(
                    "layout_repair.hygiene.reuse_suppressed",
                    reuse_source=_reuse.source or "<unknown>",
                ),
            })
    except Exception:
        logger.exception("reuse_mode_guard failed — proceeding without suppression")
    return None


def _handle_research_preprocessing(args, **kw):
    """Search web for preprocessing best practices and synthesize evidence."""
    _suppressed = _reuse_suppression_envelope(args)
    if _suppressed is not None:
        return _suppressed

    from easybci_lib.tools.neural_processing.research.complexity_classifier import classify_complexity
    from easybci_lib.tools.neural_processing.research.query_builder import build_queries
    from easybci_lib.tools.neural_processing.research.search_cache import SearchCache
    from easybci_lib.tools.neural_processing.research.evidence_synthesizer import synthesize_evidence

    question = args["question"]
    modality = args.get("modality", "")
    paradigm = args.get("paradigm", "")
    context = args.get("context", {})

    # Check cache first
    cache = SearchCache()
    cached = cache.get(modality, paradigm, question)
    if cached:
        return json.dumps({"success": True, "from_cache": True, **cached})

    # Classify complexity to determine search depth
    level = classify_complexity(
        fingerprint=context.get("fingerprint"),
        user_intent=question,
        modality=modality,
        paradigm=paradigm,
        matched_skill=context.get("matched_skill"),
        proven_match=context.get("proven_match", False),
        qc_failures=context.get("qc_failures", 0),
        failed_remedies=context.get("failed_remedies", []),
    )

    # Level 0 means no search needed
    if level == 0:
        return json.dumps({
            "success": True,
            "level": 0,
            "message": "Standard scenario — domain skills are sufficient, no web search needed.",
        })

    # Build queries
    queries = build_queries(
        level=level,
        modality=modality,
        paradigm=paradigm,
        question=question,
        context=context,
    )

    if not queries:
        return json.dumps({
            "success": False,
            "error": "Could not construct search queries for this scenario.",
        })

    # Execute searches via the web search registry
    search_results, search_errors, provider_name, discarded = _execute_research_searches(
        queries, modality=modality, paradigm=paradigm,
    )

    if not search_results:
        # Bubble each provider's real reason up so reasoning.md banner
        # and plan/web_evidence.json:reason can show "tavily: TAVILY_API_KEY
        # environment variable not set; exa: EXA_API_KEY environment variable not set"
        # instead of the generic "Web search unavailable" placeholder.
        joined_error = (
            "; ".join(search_errors)
            if search_errors
            else "Web search unavailable or returned no results. Falling back to domain skills."
        )
        return json.dumps({
            "success": False,
            "level": level,
            "error": joined_error,
            "diagnostics": search_errors,
        })

    # Synthesize evidence
    report = synthesize_evidence(
        search_results=search_results,
        modality=modality,
        paradigm=paradigm,
        question=question,
    )

    result = {
        "success": True,
        "level": level,
        "queries_executed": len(search_results),
        "provider": provider_name,
        "discarded": discarded,
        **report.to_dict(),
    }

    # Cache the result
    cache.put(modality, paradigm, question, result)

    return json.dumps(result, default=str)


def _sources_per_query(default: int = 8) -> int:
    """Per-query candidate count for research search, from web.research config."""
    try:
        from easybci_cli.config import load_config
        research = ((load_config() or {}).get("web") or {}).get("research") or {}
        return max(1, int(research.get("sources_per_query", default)))
    except Exception:  # noqa: BLE001
        return default


def _execute_research_searches(
    queries,
    *,
    modality: str = "",
    paradigm: str = "",
) -> tuple[list, list[str], str | None, list[dict]]:
    """Execute search queries via the web search provider registry.

    Returns ``(results, per_query_errors, provider_name, discarded)``.

    - ``results``: list of ``{"query", "purpose", "results": [...]}`` dicts for
      every query that returned at least one result that passed the relevance
      gate. Empty when every query failed or every result was filtered out.
    - ``per_query_errors``: per-provider/per-query diagnostic strings.
    - ``provider_name``: active search provider name, or None when no provider
      was usable.
    - ``discarded``: items dropped by :func:`relevance_filter.filter_results`,
      each carrying ``"reason"`` (``"blacklisted_domain"`` | ``"low_score"``)
      and, for low-score drops, the computed ``"score"``. ``modality`` /
      ``paradigm`` flow into the keyword-hit scorer; pass empty strings to
      fall back to whitelist/blacklist-only filtering.
    """
    if diagnose_active_provider is None:
        logger.warning("web_search_registry not available")
        return [], ["web_search_registry not importable"], None, []

    from easybci_lib.tools.neural_processing.research.relevance_filter import (
        filter_results,
    )

    provider, errors = diagnose_active_provider("search")
    if provider is None:
        if errors:
            return [], [f"{e.provider}: {e.reason}" for e in errors], None, []
        logger.warning("No web search provider available for research_preprocessing")
        return [], ["no web search provider configured"], None, []

    results: list = []
    per_query_errors: list[str] = []
    discarded: list[dict] = []

    def _run_one_query(sq):
        """Execute + filter a single query. Returns (result_set|None, dropped, error|None).

        Runs in a worker thread; never mutates shared state — the caller
        merges the returned pieces on the main thread.
        """
        try:
            response = provider.search(sq.query, limit=_sources_per_query())
        except Exception as exc:
            logger.debug("Search failed for query '%s': %s", sq.query, exc)
            return None, [], f"{provider.name}.search raised {exc!r}"
        if not (response and response.get("success")):
            err_msg = (response or {}).get("error") or "unknown error"
            logger.debug("Search returned error for '%s': %s", sq.query, err_msg)
            return None, [], f"{provider.name}: {err_msg}"

        raw_items = response.get("data", {}).get("web", []) or []
        kept, dropped = filter_results(
            raw_items,
            modality=modality, paradigm=paradigm, question=sq.query,
        )
        for d in dropped:
            d.setdefault("query", sq.query)
        result_set = None
        if kept:
            result_set = {
                "query": sq.query,
                "purpose": sq.purpose,
                "results": kept,
            }
        return result_set, dropped, None

    # Queries are independent HTTP round-trips; run them concurrently. The
    # provider's search() is a stateless HTTP call (Exa/Tavily hold no shared
    # mutable state), so this is thread-safe. Ordering is irrelevant —
    # synthesize_evidence re-dedupes and re-ranks.
    from concurrent.futures import ThreadPoolExecutor

    query_list = list(queries)
    if query_list:
        with ThreadPoolExecutor(max_workers=min(6, len(query_list))) as ex:
            for result_set, dropped, error in ex.map(_run_one_query, query_list):
                if error:
                    per_query_errors.append(error)
                discarded.extend(dropped)
                if result_set is not None:
                    results.append(result_set)

    # Optional enrichment: fetch full page content for the top result of each
    # query via the active extract provider. This is strictly additive — the
    # EvidenceSynthesizer already handles a missing "extracted_content" field,
    # so when no extract-capable provider is configured (or extraction fails)
    # the behavior is byte-for-byte identical to snippet-only synthesis.
    _enrich_with_extracted_content(results)

    return results, per_query_errors, provider.name, discarded


def _enrich_with_extracted_content(results: list) -> None:
    """Best-effort: populate each result set's "extracted_content" in place.

    Fully guarded — any absence/failure leaves the results untouched so the
    no-web and no-extract paths are unaffected. Only synchronous extract
    providers are used here (this is a sync function); async-only providers
    such as Firecrawl/Parallel are skipped to avoid event-loop juggling.
    """
    if not results:
        return
    try:
        import inspect

        from easybci_agent.web_search_registry import get_active_extract_provider

        extract_provider = get_active_extract_provider()
        if extract_provider is None or not extract_provider.supports_extract():
            return
        if inspect.iscoroutinefunction(extract_provider.extract):
            # Async-only extract provider — skip to keep this path sync-safe.
            return

        # Collect the single top URL per query (cap total to 2 to bound latency).
        top_urls: list[str] = []
        for result_set in results:
            web_results = result_set.get("results") or []
            if web_results:
                url = web_results[0].get("url")
                if url:
                    top_urls.append(url)
            if len(top_urls) >= 2:
                break
        if not top_urls:
            return

        documents = extract_provider.extract(top_urls)
        # Map URL -> extracted text for assignment back onto the result sets.
        by_url: dict[str, str] = {}
        for doc in documents or []:
            if not isinstance(doc, dict) or doc.get("error"):
                continue
            content = doc.get("content") or doc.get("raw_content") or ""
            if doc.get("url") and content:
                by_url[doc["url"]] = content

        for result_set in results:
            web_results = result_set.get("results") or []
            if web_results:
                url = web_results[0].get("url")
                if url and url in by_url:
                    result_set["extracted_content"] = by_url[url]
    except Exception as exc:
        logger.debug("Extract enrichment skipped: %s", exc)


# research_preprocessing does NOT require numpy — it only needs web search
registry.register(
    name="research_preprocessing",
    toolset="neural",
    schema=RESEARCH_PREPROCESSING_SCHEMA,
    handler=_handle_research_preprocessing,
    check_fn=None,
    emoji="\U0001f50d",
)


# ===========================================================================
# research_parameter — single-parameter, evidence-driven recommendation
# ===========================================================================
from easybci_lib.tools.neural_processing.research.search_cache import SearchCache as _SearchCache  # noqa: E402

RESEARCH_PARAMETER_SCHEMA = {
    "name": "research_parameter",
    "description": (
        "For a single preprocessing parameter, return an evidence-backed "
        "recommendation. Consults the parameter-uncertainty registry to decide "
        "whether to web-search or use the empirical default. Always returns a "
        "ParameterEvidence object (JSON) — never raises on search failure."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "operator":  {"type": "string"},
            "parameter": {"type": "string"},
            "modality":  {"type": "string",
                          "enum": ["eeg", "seeg", "ecog", "meg",
                                   "spike", "fnirs", "ieeg"]},
            "paradigm":  {"type": "string"},
            "context":   {"type": "object"},
        },
        "required": ["operator", "parameter", "modality"],
    },
}


def _handle_research_parameter(args, **kw):
    """Resolve one parameter via registry / cache / web search.

    Contract: never raises on search failure — every failure path returns a
    ParameterEvidence-backed JSON string with source=='empirical_default'
    (or 'registry_miss' for unknown operator/parameter).
    """
    _suppressed = _reuse_suppression_envelope(args)
    if _suppressed is not None:
        return _suppressed

    from easybci_lib.tools.neural_processing.research import parameter_registry as pr
    from easybci_lib.tools.neural_processing.research.parameter_evidence import (
        ParameterEvidence, Citation,
    )

    operator = args.get("operator", "")
    parameter = args.get("parameter", "")
    modality = args.get("modality", "")
    paradigm = args.get("paradigm", "")
    context = args.get("context") or {}

    registry_version = pr.registry_version_hash()

    entry = pr.lookup(operator, parameter)
    if entry is None:
        return ParameterEvidence(
            operator=operator, parameter=parameter, value=None,
            source="registry_miss", confidence=0.0,
            registry_version=registry_version, needs_user=True,
            summary=(
                f"No registry entry for {operator}.{parameter}. "
                "Please specify this value explicitly."
            ),
        ).to_dict_json()

    if not pr.needs_research(operator, parameter, fingerprint=context.get("fingerprint")):
        d = entry.get_default(modality=modality, paradigm=paradigm)
        return ParameterEvidence(
            operator=operator, parameter=parameter, value=d.value,
            source="empirical_default", confidence=1.0,
            default_origin=d.origin, registry_version=registry_version,
        ).to_dict_json()

    cache = _SearchCache()
    cached = cache.get_parameter(
        modality=modality, paradigm=paradigm,
        parameter=parameter, registry_version=registry_version,
    )
    if cached:
        return json.dumps(cached, default=str)

    try:
        from easybci_agent.web_search_registry import get_active_search_provider
    except ImportError:
        get_active_search_provider = lambda: None  # noqa: E731
    provider = get_active_search_provider()
    if provider is None:
        d = entry.get_default(modality=modality, paradigm=paradigm)
        return ParameterEvidence(
            operator=operator, parameter=parameter, value=d.value,
            source="empirical_default", confidence=1.0,
            default_origin=d.origin, registry_version=registry_version,
            fallback_reason="no_provider",
        ).to_dict_json()

    from easybci_lib.tools.neural_processing.research.evidence_synthesizer import synthesize_parameter

    question = pr.render_canonical_question(
        operator, parameter, modality=modality, paradigm=paradigm,
    )
    if not question:
        d = entry.get_default(modality=modality, paradigm=paradigm)
        return ParameterEvidence(
            operator=operator, parameter=parameter, value=d.value,
            source="empirical_default", confidence=1.0,
            default_origin=d.origin, registry_version=registry_version,
            fallback_reason="empty_question",
        ).to_dict_json()

    try:
        resp = provider.search(question, limit=_sources_per_query())
    except Exception as exc:  # noqa: BLE001 — contract: never raise
        d = entry.get_default(modality=modality, paradigm=paradigm)
        return ParameterEvidence(
            operator=operator, parameter=parameter, value=d.value,
            source="empirical_default", confidence=1.0,
            default_origin=d.origin, registry_version=registry_version,
            fallback_reason=f"search_error:{type(exc).__name__}:{str(exc)[:80]}",
        ).to_dict_json()

    web = ((resp or {}).get("data") or {}).get("web") or []
    search_results = [{"query": question, "results": web}]
    synth = synthesize_parameter(
        operator=operator, parameter=parameter,
        search_results=search_results,
        sanity_range=entry.sanity_range,
    )

    if synth is None:
        d = entry.get_default(modality=modality, paradigm=paradigm)
        return ParameterEvidence(
            operator=operator, parameter=parameter, value=d.value,
            source="empirical_default", confidence=1.0,
            default_origin=d.origin, registry_version=registry_version,
            fallback_reason="empty_results",
        ).to_dict_json()

    if synth.get("rejected_reason") == "out_of_range":
        d = entry.get_default(modality=modality, paradigm=paradigm)
        return ParameterEvidence(
            operator=operator, parameter=parameter, value=d.value,
            source="empirical_default", confidence=1.0,
            default_origin=d.origin, registry_version=registry_version,
            fallback_reason=f"out_of_range:{synth.get('raw_candidates', [])[:3]}",
            attempted_evidence=synth,
        ).to_dict_json()

    if synth.get("confidence", 0.0) < 0.3:
        d = entry.get_default(modality=modality, paradigm=paradigm)
        return ParameterEvidence(
            operator=operator, parameter=parameter, value=d.value,
            source="empirical_default", confidence=1.0,
            default_origin=d.origin, registry_version=registry_version,
            fallback_reason="low_confidence",
            attempted_evidence=synth,
        ).to_dict_json()

    citations = tuple(
        Citation(url=c.get("url", ""), title=c.get("title", ""),
                 snippet=c.get("snippet", ""))
        for c in synth.get("citations", [])
    )
    evidence = ParameterEvidence(
        operator=operator, parameter=parameter,
        value=synth["value"], source="web",
        confidence=float(synth["confidence"]),
        citations=citations,
        summary=synth.get("summary", ""),
        registry_version=registry_version,
    )
    cache.put_parameter(
        modality=modality, paradigm=paradigm,
        parameter=parameter, registry_version=registry_version,
        result=evidence.to_dict(),
    )
    return evidence.to_dict_json()


registry.register(
    name="research_parameter",
    toolset="neural",
    schema=RESEARCH_PARAMETER_SCHEMA,
    handler=_handle_research_parameter,
    check_fn=None,
    emoji="\U0001f4cf",
)

# repair_layout — LLM-facing hook into layout_repair.verify_and_repair.
# Registered without a check_fn so it's always available (no MNE / signal-lib
# imports needed to fix layout drift).
registry.register(
    name="repair_layout",
    toolset="neural",
    schema=REPAIR_LAYOUT_SCHEMA,
    handler=_handle_repair_layout,
    check_fn=None,
    emoji="\U0001f527",
)
