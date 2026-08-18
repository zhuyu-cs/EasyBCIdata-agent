"""Batch orchestration → standard reproducible mini-repo.

Both batch entry points (`batch_process` fixed-steps and
`batch_process_adaptive` reference-driven) route through `build_repro_repo`,
which produces the SAME canonical layout as the single-file flow instead of
hand-rolling load→preprocess→flat-save:

    <work_dir>/
    ├── plan/            proposal.json reasoning.md pipeline_record.json goal.json
    ├── code/            pipeline.py qc.py vis.py run.py requirements.txt
    ├── preprocessed_output/
    │   ├── preprocessed/sub-<id>/ses-<ses>/<stem>_preprocessed.nwb  (NWB-only)
    │   ├── figures/sub-<id>/ses-<ses>/*.png
    │   └── QC_out/sub-<id>/ses-<ses>/qc_report.json
    ├── middle_process/  inputs_routing.json + status sidecars + excluded_inputs.json
    └── README.md

The orchestrator OWNS ordering; the LLM is not in the loop:
  1. Populate the routing table SEQUENTIALLY via deep_inspect (one entry per
     input). Serial on purpose — deep_inspect writes a single shared
     inputs_routing.json and concurrent writers race on its .tmp (the
     [Errno 2] failure this fixes). Pre-filter OOM-oversized and
     out-of-adaptation-range inputs here (recorded, never silently dropped).
  2. Choose ONE step list: adaptive → build_adaptive_steps (per-file :auto
     markers resolved at runtime by the generated operators); fixed → verbatim.
  3. Scaffold the repo (build_mini_repo regenerates code/pipeline.py from the
     steps) + write qc.py / vis.py.
  4. Run pipeline.py → qc.py → vis.py via run_script (multi-input subprocess).
  5. Finalize (build_mini_repo status=ok, force) + contract-check.

Reproducibility contract: because the steps are `:auto` for the per-file
knobs (notch / resample / reject_by_labels), re-running `python code/pipeline.py
<work_dir>` reproduces the batch's heterogeneous results exactly.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional, Sequence

logger = logging.getLogger(__name__)


def _infer_modality_from_report(report: dict) -> str:
    fp = report.get("fingerprint", {}) if isinstance(report, dict) else {}
    return str(fp.get("modality") or "").lower()


def _has_ica_step(steps: Optional[Sequence[str]]) -> bool:
    return any(isinstance(s, str) and s.split(":", 1)[0].strip() == "ica"
               for s in (steps or []))


def _peak_for_report(report: dict, target_hz: Optional[float] = None,
                     steps: Optional[Sequence[str]] = None) -> tuple[float, float]:
    """Estimated peak-processing footprint and memory budget for one file (MB).

    Single source shared by ``_oom_excluded`` (single-file admission) and the
    per-file peak recorded into the routing table (reused by the batch scheduler
    and the cross-instance global gate). Routes through the authoritative
    ``estimate_peak_mb`` — float32, recipe-aware overhead, decimation-aware.
    Falls back to preload_full_mb × overhead when fingerprint metadata is thin.
    """
    from easybci_lib.tools.neural_processing.preprocess.memory_strategy import (
        _get_available_memory_mb, _MEMORY_BUDGET_RATIO, _NO_ICA_OVERHEAD_FACTOR,
        _PIPELINE_OVERHEAD_FACTOR, estimate_peak_mb,
    )
    fp = report.get("fingerprint", {}) if isinstance(report, dict) else {}
    mem = report.get("memory_estimate", {}) if isinstance(report, dict) else {}
    n_ch = int(fp.get("n_channels") or 0)
    fs = float(fp.get("sampling_freq_hz") or 0.0)
    dur = float(fp.get("duration_s") or 0.0)
    has_ica = _has_ica_step(steps)
    peak_mb = estimate_peak_mb(n_channels=n_ch, frequency=fs, duration_s=dur,
                               has_ica=has_ica, target_hz=target_hz)
    if peak_mb <= 0.0:  # fingerprint too thin — fall back to loader estimate
        overhead = _PIPELINE_OVERHEAD_FACTOR if has_ica else _NO_ICA_OVERHEAD_FACTOR
        peak_mb = float(mem.get("preload_full_mb") or 0.0) * overhead
    budget_mb = _get_available_memory_mb() * _MEMORY_BUDGET_RATIO
    return peak_mb, budget_mb


def _oom_excluded(report: dict, target_hz: Optional[float] = None,
                  steps: Optional[Sequence[str]] = None) -> Optional[dict]:
    """Return an exclusion record when this file's estimated peak footprint
    blows the memory budget, else None.

    Delegates the estimate to ``_peak_for_report`` (→ ``estimate_peak_mb``): the
    loader decimates on the fly (io/nk_backend._read_decimated_uV) so a
    261ch/2000Hz/4h recording peaks at ~14 GB (measured) instead of ~56 GB when a
    lower resample ``target_hz`` is set. Overhead is recipe-aware (8× with ICA's
    eigendecomposition, 3× without)."""
    from easybci_lib.tools.neural_processing.preprocess.memory_strategy import (
        _NO_ICA_OVERHEAD_FACTOR, _PIPELINE_OVERHEAD_FACTOR, safe_max_duration_s,
    )
    fp = report.get("fingerprint", {}) if isinstance(report, dict) else {}
    n_ch = int(fp.get("n_channels") or 0)
    fs = float(fp.get("sampling_freq_hz") or 0.0)
    dur = float(fp.get("duration_s") or 0.0)
    eff_fs = fs
    if target_hz and target_hz > 0 and fs > 0 and target_hz < fs:
        eff_fs = float(target_hz)
    has_ica = _has_ica_step(steps)
    overhead = _PIPELINE_OVERHEAD_FACTOR if has_ica else _NO_ICA_OVERHEAD_FACTOR
    peak_mb, budget_mb = _peak_for_report(report, target_hz=target_hz, steps=steps)
    if peak_mb > budget_mb:
        safe_s = safe_max_duration_s(n_ch, eff_fs, dur, overhead_factor=overhead)
        return {
            "reason": "oom_guard",
            "peak_processing_mb_estimate": round(peak_mb, 1),
            "memory_budget_mb": round(budget_mb, 1),
            "effective_sfreq_hz": eff_fs,
            "overhead_factor": overhead,
            "safe_max_duration_s": (round(safe_s, 1) if safe_s else None),
            "detail": (f"peak ~{peak_mb:.0f} MB > budget {budget_mb:.0f} MB "
                       f"(eff_fs={eff_fs:.0f}Hz, overhead={overhead:g}x); crop with "
                       "max_duration, add a lower resample target, or use a "
                       "larger environment"),
        }
    return None


def build_repro_repo(
    files: Sequence[str],
    *,
    work_dir: str,
    modality: str = "auto",
    analysis_goal: str = "generic",
    steps: Optional[Sequence[str]] = None,
    skill: Any = None,
    reject_keywords: Sequence[str] = (),
    adaptive: bool = False,
    paradigm: str = "",
    timeout: Optional[int] = None,
    preview: bool = False,
) -> dict:
    """Orchestrate a batch into the standard reproducible mini-repo.

    Parameters
    ----------
    files : list of raw input paths (already globbed + coverage-checked).
    work_dir : the mini-repo root (created if absent).
    adaptive : True → derive one self-adapting step list from `skill` via
        build_adaptive_steps; False → use `steps` verbatim (fixed-steps batch).
    skill : proven-pipeline entry (required when adaptive=True) — supplies
        steps/adaptation_slots/reject_keywords + gold n_channels/modality for
        the out-of-range check.
    reject_keywords : union'd into the adaptive reject_by_labels step.

    Returns a dict envelope: {success, work_dir, n_inputs, n_routed,
    n_excluded, excluded, steps, contract, stage_results}.
    """
    from easybci_lib.tools.neural_processing.io.deep_inspect import deep_inspect
    from easybci_lib.tools.neural_processing.io.routing_table import (
        load_routing_table,
    )

    wd = Path(work_dir)
    (wd / "middle_process").mkdir(parents=True, exist_ok=True)

    # Ensure the machine-global NK io_loader plugin exists so interactive
    # deep_inspect (and any generated pipeline run) reads Nihon Kohden correctly.
    # Best-effort — never block the batch on provisioning.
    try:
        from easybci_lib.tools.neural_processing.io.nk_loader_plugin import (
            ensure_global_plugin,
        )
        ensure_global_plugin()
    except Exception as exc:  # noqa: BLE001
        logger.warning("NK io_loader global provisioning failed: %s", exc)

    adapt_pipeline = None
    if adaptive:
        if skill is None:
            return {"success": False, "error": "adaptive=True requires a skill"}
        from easybci_lib.tools.neural_processing.proven_adapt import (
            adapt_pipeline as _adapt,
        )
        adapt_pipeline = _adapt

    # ---- Choose the step list UP FRONT (needed by the OOM guard) -----------
    # The recipe's concrete resample target is the load-time decimation hint:
    # the loader decimates on the fly so the guard can budget the decimated
    # footprint, not the native one. Must be known before the pre-filter loop.
    from easybci_lib.tools.neural_processing.codegen.generator import (
        build_adaptive_steps, resample_target_hz,
    )
    if adaptive:
        kw = list(getattr(skill, "reject_keywords", []) or []) + list(reject_keywords)
        repo_steps = build_adaptive_steps(skill, reject_keywords=kw)
    else:
        repo_steps = list(steps or [])
    if not repo_steps:
        return {"success": False, "error": "no steps to run"}
    load_target_hz = resample_target_hz(repo_steps)

    # ---- Step 1: populate routing SEQUENTIALLY + pre-filter ----------------
    excluded: list[dict] = []
    routed_modalities: list[str] = []
    for f in files:
        try:
            insp = deep_inspect(f, str(wd))
        except Exception as exc:  # deep_inspect shouldn't raise, but be safe
            excluded.append({"data_path": str(f), "reason": "inspect_failed",
                             "detail": str(exc)})
            continue
        report = insp.get("report", {}) if isinstance(insp, dict) else {}

        oom = _oom_excluded(report, target_hz=load_target_hz, steps=repo_steps)
        if oom is not None:
            _unroute(wd, f)
            excluded.append({"data_path": str(f), **oom})
            continue

        if adaptive and adapt_pipeline is not None:
            adapted = adapt_pipeline(
                list(skill.steps), list(skill.adaptation_slots), report,
                gold_n_channels=getattr(skill, "n_channels", 0),
                gold_modality=getattr(skill, "modality", ""),
                reject_keywords=list(getattr(skill, "reject_keywords", []) or []),
            )
            if adapted.out_of_range:
                _unroute(wd, f)
                excluded.append({"data_path": str(f), "reason": "out_of_range",
                                 "reasons": adapted.out_of_range_reasons})
                continue
        routed_modalities.append(_infer_modality_from_report(report))

        # Record the recipe-aware peak estimate on this file's routing entry so
        # the batch scheduler (Layer B) and the cross-instance memory gate
        # (Layer C, in the generated pipeline.py) reuse it without recomputing.
        peak_mb, _budget_mb = _peak_for_report(report, target_hz=load_target_hz,
                                                steps=repo_steps)
        if peak_mb > 0:
            _record_peak(wd, str(f), peak_mb)

    # Record exclusions loudly (never silent).
    excluded_path = wd / "middle_process" / "excluded_inputs.json"
    excluded_path.write_text(json.dumps(excluded, indent=2, ensure_ascii=False),
                             encoding="utf-8")

    table = load_routing_table(wd)
    n_routed = len(table.inputs) if table else 0
    if n_routed == 0:
        return {"success": False, "work_dir": str(wd), "n_inputs": len(files),
                "n_routed": 0, "n_excluded": len(excluded), "excluded": excluded,
                "error": "no inputs survived pre-filter (all OOM/out-of-range)"}

    # ---- Layer B: heterogeneous batch memory plan (advisory) ---------------
    # Data-review-driven + hardware-aware: derive parallel-vs-serial from each
    # routed file's recorded peak_mb + CPU count. Persisted for inspection and
    # surfaced in preview; execution stays serial by default (the in-pipeline
    # global gate makes any concurrency safe, so this never forces parallelism).
    memory_plan = _write_memory_plan(wd, table)

    # Resolve modality: explicit arg wins; else most-common routed modality.
    resolved_modality = modality
    if resolved_modality in ("", "auto") and routed_modalities:
        resolved_modality = max(set(routed_modalities), key=routed_modalities.count) or "auto"

    # ---- Preview gate: return the computed plan WITHOUT scaffolding/running --
    # Reuses the exact Step-1 computation above (deep_inspect + OOM guard +
    # out-of-range adapt check + routing population), so a preview can never
    # drift from what a confirm run does. Routing/excluded JSON are written and
    # reused by the subsequent confirm run (no double deep_inspect).
    if preview:
        return {
            "success": True, "preview": True, "work_dir": str(wd),
            "n_inputs": len(files), "n_routed": n_routed,
            "n_excluded": len(excluded), "excluded": excluded,
            "steps": repo_steps, "modality": resolved_modality,
            "analysis_goal": analysis_goal, "adaptive": adaptive,
            "memory_plan": memory_plan,
        }

    # ---- Step 3: scaffold repo (writes code/pipeline.py) + qc.py/vis.py ----
    scaffold = _scaffold_repo(wd, repo_steps, resolved_modality, analysis_goal,
                              paradigm, table)
    if not scaffold.get("ok"):
        return {"success": False, "work_dir": str(wd), "error": scaffold.get("error"),
                "steps": repo_steps}

    # ---- Step 4: run pipeline → qc → vis -----------------------------------
    from easybci_lib.tools.neural_processing.codegen.script_runner import run_script
    stage_results: dict[str, Any] = {}
    for stage in ("pipeline", "qc", "vis"):
        res = run_script(work_dir=str(wd), stage=stage, input_path=None, timeout=timeout)
        stage_results[stage] = {"ok": res.get("ok"), "retcode": res.get("retcode"),
                                "status": res.get("status")}
        if not res.get("ok"):
            # pipeline failure is fatal; qc/vis failures are surfaced but the
            # repo is still finalized so the user can inspect + repair.
            stage_results[stage]["stderr_tail"] = res.get("stderr_tail")
            if stage == "pipeline":
                return {"success": False, "work_dir": str(wd), "steps": repo_steps,
                        "n_routed": n_routed, "n_excluded": len(excluded),
                        "excluded": excluded, "stage_results": stage_results,
                        "error": "pipeline stage failed"}

    # ---- Step 5: finalize + contract-check ---------------------------------
    _finalize_repo(wd, repo_steps, resolved_modality, analysis_goal, paradigm, table)
    contract = _contract_check(wd, analysis_goal)

    result = {
        "success": bool(contract.get("ok")),
        "work_dir": str(wd),
        "n_inputs": len(files),
        "n_routed": n_routed,
        "n_excluded": len(excluded),
        "excluded": excluded,
        "steps": repo_steps,
        "modality": resolved_modality,
        "adaptive": adaptive,
        "contract": contract,
        "stage_results": stage_results,
    }
    # Surface the raw-vs-preprocessed footprint build_mini_repo stamped into
    # plan/pipeline_record.json, so the batch tool return shows it in chat.
    try:
        _rec = json.loads((wd / "plan" / "pipeline_record.json").read_text(encoding="utf-8"))
        if _rec.get("storage_footprint"):
            result["storage_footprint"] = _rec["storage_footprint"]
    except Exception:
        pass
    return result


def _unroute(work_dir: Path, data_path: str) -> None:
    """Remove any routing entry deep_inspect wrote for `data_path` (used when a
    file is excluded AFTER inspection). Idempotent."""
    from easybci_lib.tools.neural_processing.io.routing_table import (
        load_routing_table, save_routing_table,
    )
    table = load_routing_table(work_dir)
    if not table:
        return
    before = len(table.inputs)
    table.inputs = [e for e in table.inputs if e.data_path != str(data_path)]
    if len(table.inputs) != before:
        save_routing_table(work_dir, table)


def _record_peak(work_dir: Path, data_path: str, peak_mb: float) -> None:
    """Stamp the recipe-aware peak estimate (MB) onto this file's routing entry.

    Matched by ``data_path`` like :func:`_unroute`. Idempotent; a no-op when the
    entry is absent (deep_inspect failed to route it). Never raises — peak
    recording is advisory and must not break the batch."""
    from easybci_lib.tools.neural_processing.io.routing_table import (
        load_routing_table, save_routing_table,
    )
    try:
        table = load_routing_table(work_dir)
        if not table:
            return
        changed = False
        for e in table.inputs:
            if e.data_path == str(data_path):
                e.peak_mb = round(float(peak_mb), 1)
                changed = True
        if changed:
            save_routing_table(work_dir, table)
    except Exception as exc:  # advisory only — never break the batch
        logger.warning("could not record peak_mb for %s: %s", data_path, exc)


def _write_memory_plan(work_dir: Path, table) -> dict:
    """Compute the heterogeneous batch memory plan from routed peak_mb values and
    persist it to ``<work_dir>/middle_process/batch_memory_plan.json``.

    Advisory only: the plan records what parallelism the data + hardware would
    permit (large files → serial, small files → parallel, CPU-capped). The batch
    still executes serially; the in-pipeline global gate is what actually makes
    concurrency safe. Never raises — a plan-write failure must not break the
    batch. Returns the plan dict (also embedded in the preview envelope)."""
    from easybci_lib.tools.neural_processing.preprocess.memory_strategy import (
        _available_cpu_count, compute_strategy_from_peaks,
    )
    try:
        peaks = [e.peak_mb for e in (table.inputs if table else [])
                 if e.peak_mb is not None]
        n_known = len(peaks)
        # Files with no recorded peak (thin metadata) count as unknown; pass 0.0
        # so the scheduler pessimistically treats them as the batch max.
        n_missing = (len(table.inputs) if table else 0) - n_known
        peaks_arg = peaks + [0.0] * n_missing
        strat = compute_strategy_from_peaks(peaks_arg)
        cpu = _available_cpu_count()
        plan = {
            "mode": strat.mode,
            "max_workers": strat.max_workers,
            "cpu_count": cpu,
            "memory_budget_mb": strat.memory_budget_mb,
            "available_mb": round(strat.available_mb, 1),
            "n_files": (len(table.inputs) if table else 0),
            "n_peaks_known": n_known,
            "n_peaks_missing": n_missing,
            "peak_max_mb": (round(max(peaks), 1) if peaks else None),
            "total_estimated_mb": round(strat.total_estimated_mb, 1),
            "reason": strat.reason,
            "note": ("advisory — batch executes serially; the in-pipeline global "
                     "memory gate makes any concurrency safe"),
        }
        out = work_dir / "middle_process" / "batch_memory_plan.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(plan, indent=2, ensure_ascii=False),
                       encoding="utf-8")
        return plan
    except Exception as exc:  # advisory only — never break the batch
        logger.warning("could not write batch_memory_plan.json: %s", exc)
        return {}


def _data_info_from_table(table) -> dict:
    """Minimal data_info for build_mini_repo, from the first routed input's
    inspection report if available."""
    if not table or not table.inputs:
        return {}
    return {}


def _scaffold_repo(wd: Path, steps, modality, analysis_goal, paradigm, table, scenario="research") -> dict:
    """Run build_mini_repo (writes plan/ + code/pipeline.py + run.py +
    requirements.txt + README) and additionally write code/qc.py + code/vis.py
    (build_mini_repo deliberately omits them)."""
    try:
        from easybci_lib.tools.neural_processing.export.repo_builder import build_mini_repo
        from easybci_lib.tools.neural_processing.codegen.generator import (
            generate_qc_script_v2, generate_vis_script,
        )
    except Exception as exc:  # pragma: no cover
        return {"ok": False, "error": f"codegen/export import failed: {exc}"}

    data_info = _data_info_from_table(table)
    pipeline_record = {"analysis_goal": analysis_goal, "batch": True}
    input_path = table.inputs[0].data_path if (table and table.inputs) else ""
    try:
        build_mini_repo(
            output_dir=str(wd), steps=list(steps), data_info=data_info,
            pipeline_record=pipeline_record, input_path=input_path,
            modality=modality, paradigm=paradigm, analysis_goal=analysis_goal,
            status="ok", force=True,
        )
    except Exception as exc:
        return {"ok": False, "error": f"build_mini_repo failed: {exc}"}

    # qc.py + vis.py (mirror _handle_generate_code — repo_builder omits them).
    try:
        code_dir = wd / "code"
        code_dir.mkdir(parents=True, exist_ok=True)
        (code_dir / "qc.py").write_text(
            generate_qc_script_v2(steps=list(steps), data_info=data_info,
                                  modality=modality, analysis_goal=analysis_goal,
                                  scenario=scenario),
            encoding="utf-8")
        (code_dir / "vis.py").write_text(
            generate_vis_script(modality=modality, analysis_goal=analysis_goal),
            encoding="utf-8")
    except Exception as exc:
        return {"ok": False, "error": f"qc/vis codegen failed: {exc}"}
    return {"ok": True}


def _finalize_repo(wd: Path, steps, modality, analysis_goal, paradigm, table) -> None:
    """Re-run build_mini_repo after execution to consolidate intermediates +
    refresh README/record now that outputs exist."""
    try:
        from easybci_lib.tools.neural_processing.export.repo_builder import build_mini_repo
        build_mini_repo(
            output_dir=str(wd), steps=list(steps),
            data_info=_data_info_from_table(table),
            pipeline_record={"analysis_goal": analysis_goal, "batch": True},
            input_path=table.inputs[0].data_path if (table and table.inputs) else "",
            modality=modality, paradigm=paradigm, analysis_goal=analysis_goal,
            status="ok", force=True,
        )
    except Exception as exc:  # non-fatal — outputs already on disk
        logger.warning("batch finalize build_mini_repo failed: %s", exc)


def _contract_check(wd: Path, analysis_goal: str) -> dict:
    """Run both layout verifiers; return a structured result (never raises)."""
    from easybci_lib.tools.neural_processing.export.contract_check import (
        validate_mini_repo, verify_layout_strict_multi,
    )
    out: dict[str, Any] = {"ok": True, "violations": []}
    basic = validate_mini_repo(str(wd))
    if not basic.get("ok"):
        out["ok"] = False
        out["missing"] = basic.get("missing")
    try:
        verify_layout_strict_multi(str(wd), analysis_goal=analysis_goal)
    except Exception as exc:
        out["ok"] = False
        out["violations"].append(str(exc))
    return out
