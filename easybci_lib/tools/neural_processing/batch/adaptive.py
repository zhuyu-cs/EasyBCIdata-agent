"""Reference-driven adaptive batch: anchor each file to an enhanced proven skill,
recompute numeric params per file, run, QC, compare to soft baselines.

.. deprecated::
    The flat-save execution path here (``_run_pipeline`` → flat
    ``{stem}_preprocessed.nwb``) is NO LONGER wired to the
    ``batch_process_adaptive`` tool — it produced non-reproducible output with
    no ``code/``/``plan/``/figures/QC. The tool handler now routes through
    ``batch/orchestrate.py:build_repro_repo``, which emits the standard
    reproducible mini-repo and closes the per-file fidelity gap via the
    generated ``notch:auto``/``resample:auto``/``reject_by_labels`` operators.
    ``adapt_pipeline`` (per-file range check) and ``aggregate_label_diagnostics``
    are still used by the orchestrator; ``_run_pipeline``/``process_one_adaptive``/
    ``batch_process_adaptive`` below are retained only for reference/tests.

Coexists with batch/processor.py:batch_process (fixed-steps path) — that path is
untouched. Per-file isolation: one file's failure/warning never aborts the batch
(design 04 §4.1). Wraps the same numeric adapter as single-file reuse (P3).
"""
from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MAX_WORKERS_HARD_LIMIT = 8  # mirror batch/processor.py


def _deep_inspect(filepath: str, work_dir: str | None = None) -> dict:
    from easybci_lib.tools.neural_processing.io.deep_inspect import deep_inspect
    wd = work_dir or os.path.join(
        os.path.dirname(filepath) or ".", "_adaptive_inspect")
    return deep_inspect(filepath, wd)


def _run_pipeline(filepath: str, steps: list[str], output_dir: str,
                  modality: str, max_duration: float | None = None,
                  extra_reject_keywords: list[str] | None = None) -> dict:
    """Load → preprocess(steps) → save NWB → return processed data + bad counts.

    Reuses the same preprocess/save path as batch/processor._process_one_file.
    When max_duration is set, only the first N seconds are loaded (memory guard).
    extra_reject_keywords is passed through to the reject_by_labels step so an
    agent can extend rejection for an unfamiliar environment at call time.
    Returns {data, frequency, n_bad, n_total, output, grade, loaded_duration_s,
    unmatched_labels, suspicious_labels}.
    """
    from easybci_lib.tools.neural_processing.io.loader import load_neural
    from easybci_lib.tools.neural_processing.preprocess.pipeline import preprocess
    from easybci_lib.tools.neural_processing.output.nwb_writer import save_nwb

    loaded = load_neural(filepath, modality=modality, max_duration=max_duration)
    n_total = int(loaded["data"].shape[0])
    data_dict = preprocess(loaded, steps=steps,
                           extra_reject_keywords=list(extra_reject_keywords or []))
    data = data_dict["data"]
    n_bad = int(n_total - data.shape[0]) if data.shape[0] <= n_total else 0

    stem = os.path.splitext(os.path.basename(filepath))[0]
    out_path = Path(output_dir) / f"{stem}_preprocessed.nwb"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    save_nwb(
        payload={"data": data, "spike_times": data_dict.get("spike_times")},
        out_path=out_path,
        meta={
            "pipeline": steps,
            "frequency": data_dict.get("frequency"),
            "ch_names": data_dict.get("channels"),
            "modality": modality,
        },
        mne_info=data_dict.get("_mne_info"),
    )
    loaded_meta = loaded.get("meta", {}) if isinstance(loaded, dict) else {}
    dd_meta = data_dict.get("meta", {}) if isinstance(data_dict, dict) else {}
    return {"data": data, "frequency": float(data_dict.get("frequency") or 0.0),
            "n_bad": n_bad, "n_total": n_total, "output": str(out_path),
            "grade": "?",
            "cropped": bool(loaded_meta.get("cropped")),
            "loaded_duration_s": loaded_meta.get("loaded_duration_s"),
            "rejected_seconds": dd_meta.get("rejected_seconds"),
            "unmatched_labels": dd_meta.get("unmatched_labels") or [],
            "suspicious_labels": dd_meta.get("suspicious_labels") or []}


def process_one_adaptive(filepath: str, skill: Any, output_dir: str,
                         modality: str, max_duration: float | None = None,
                         extra_reject_keywords: list[str] | None = None) -> dict:
    """Inspect → adapt → run → QC + baseline for a single file.

    max_duration : float or None
        If set, crop the load to the first N seconds (explicit user cap).
        If None, an automatic memory guard runs: when deep_inspect's estimated
        peak processing footprint exceeds the memory budget, the file is
        SKIPPED (not silently truncated) with skipped_oom_guard=True so the
        batch continues and no OOM occurs. Pass max_duration to crop instead,
        or run in a larger environment (modal/docker).
    """
    from easybci_lib.tools.neural_processing.proven_adapt import adapt_pipeline
    from easybci_lib.tools.neural_processing.quality.baseline_compare import (
        extract_baseline_metrics, compare_to_baselines,
    )
    from easybci_lib.tools.neural_processing.preprocess.memory_strategy import (
        _get_available_memory_mb, _MEMORY_BUDGET_RATIO, _PIPELINE_OVERHEAD_FACTOR,
        safe_max_duration_s,
    )
    try:
        insp = _deep_inspect(filepath)
        report = insp.get("report", {}) if isinstance(insp, dict) else {}

        # Memory guard: refuse to load a recording whose peak footprint blows
        # the budget unless the caller opted into an explicit crop. Compute the
        # peak from the fingerprint (n_ch × fs × duration), NOT from
        # memory_estimate.peak_processing_mb_estimate — deep_inspect degrades to
        # 0.0 there precisely for oversized files (the MemoryError path), which
        # is the case this guard exists to catch. Fall back to preload_full_mb.
        if max_duration is None:
            fp = report.get("fingerprint", {}) if isinstance(report, dict) else {}
            mem = report.get("memory_estimate", {}) if isinstance(report, dict) else {}
            n_ch = int(fp.get("n_channels") or 0)
            fs = float(fp.get("sampling_freq_hz") or 0.0)
            dur = float(fp.get("duration_s") or 0.0)
            if n_ch > 0 and fs > 0 and dur > 0:
                peak_mb = (n_ch * fs * dur * 8 * _PIPELINE_OVERHEAD_FACTOR) / (1024 * 1024)
            else:  # fingerprint incomplete — fall back to loader estimate
                peak_mb = float(mem.get("preload_full_mb") or 0.0) * _PIPELINE_OVERHEAD_FACTOR
            budget_mb = _get_available_memory_mb() * _MEMORY_BUDGET_RATIO
            if peak_mb > budget_mb:
                safe_s = safe_max_duration_s(n_ch, fs, dur)
                return {
                    "input": filepath, "success": False, "out_of_range": False,
                    "skipped_oom_guard": True,
                    "peak_processing_mb_estimate": round(peak_mb, 1),
                    "memory_budget_mb": round(budget_mb, 1),
                    "safe_max_duration_s": (round(safe_s, 1) if safe_s else None),
                    "error": (
                        f"peak ~{peak_mb:.0f} MB > budget {budget_mb:.0f} MB; "
                        "pass max_duration to crop or run in a larger "
                        "environment (modal/docker)"
                    ),
                }

        adapted = adapt_pipeline(
            list(skill.steps), list(skill.adaptation_slots), report,
            gold_n_channels=getattr(skill, "n_channels", 0),
            gold_modality=getattr(skill, "modality", ""),
            reject_keywords=list(getattr(skill, "reject_keywords", []) or []),
        )
        if adapted.out_of_range:
            return {"input": filepath, "success": False, "out_of_range": True,
                    "reasons": adapted.out_of_range_reasons,
                    "error": "out of adaptation range for skill "
                             f"{getattr(skill, 'name', '?')}"}
        run = _run_pipeline(filepath, adapted.steps, output_dir, modality,
                            max_duration=max_duration,
                            extra_reject_keywords=extra_reject_keywords)
        measured = extract_baseline_metrics(
            run["data"], run["frequency"], n_bad=run["n_bad"], n_total=run["n_total"])
        baseline = compare_to_baselines(measured, getattr(skill, "qc_baselines", {}) or {})
        result = {
            "input": filepath, "success": True, "out_of_range": False,
            "output": run["output"], "grade": run.get("grade", "?"),
            "adapted_steps": adapted.steps,
            "adaptation_report": adapted.self_report,
            "measured_baselines": measured,
            "baseline": baseline,
            "rejected_seconds": run.get("rejected_seconds"),
            "unmatched_labels": run.get("unmatched_labels") or [],
            "suspicious_labels": run.get("suspicious_labels") or [],
        }
        if run.get("cropped"):
            result["memory_capped"] = True
            result["loaded_duration_s"] = run.get("loaded_duration_s")
        return result
    except Exception as exc:  # per-file isolation
        logger.warning("adaptive file failed %s: %s", filepath, exc)
        return {"input": filepath, "success": False, "out_of_range": False,
                "error": str(exc)}


def aggregate_label_diagnostics(results: list[dict]) -> dict:
    """Union per-file label diagnostics across a batch (deduped, sorted).

    Surfaces which labels went unmatched by the reject keywords and which of
    those look clinically suspicious — the signal an agent uses to top up the
    reject keyword list for an unfamiliar environment.
    """
    unmatched: set = set()
    suspicious: set = set()
    for r in results or []:
        if not isinstance(r, dict):
            continue
        for lab in r.get("unmatched_labels") or []:
            unmatched.add(str(lab))
        for lab in r.get("suspicious_labels") or []:
            suspicious.add(str(lab))
    return {
        "unmatched_labels": sorted(unmatched),
        "suspicious_labels": sorted(suspicious),
        "suspicious_count": len(suspicious),
    }


def _worker(args_tuple):
    filepath, skill, output_dir, modality, max_duration, extra_reject_keywords = args_tuple
    return process_one_adaptive(filepath, skill, output_dir, modality,
                                max_duration=max_duration,
                                extra_reject_keywords=extra_reject_keywords)


def batch_process_adaptive(files: list[str], skill: Any, output_dir: str,
                           modality: str = "auto", max_workers: int = 4,
                           max_duration: float | None = None,
                           extra_reject_keywords: list[str] | None = None) -> dict:
    """Run adaptive processing over `files`, isolating per-file failures.

    max_duration : float or None
        If set, every file's load is cropped to the first N seconds. If None,
        each file is subject to the automatic memory guard in
        process_one_adaptive (oversized files are skipped, not loaded, so the
        host never OOMs).
    extra_reject_keywords : list of str or None
        Additional reject keywords (union'd with the skill's + the built-in
        multilingual floor) so an agent can extend rejection for an unfamiliar
        environment without editing the skill.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    workers = max(1, min(max_workers, MAX_WORKERS_HARD_LIMIT, len(files) or 1))
    results: list[dict] = []
    if workers == 1:
        for f in files:
            results.append(process_one_adaptive(f, skill, output_dir, modality,
                                                 max_duration=max_duration,
                                                 extra_reject_keywords=extra_reject_keywords))
    else:
        task_args = [(f, skill, output_dir, modality, max_duration,
                      extra_reject_keywords) for f in files]
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_worker, a): a[0] for a in task_args}
            for fut in as_completed(futs):
                try:
                    results.append(fut.result())
                except Exception as exc:  # executor-level guard
                    results.append({"input": futs[fut], "success": False,
                                    "error": f"worker crashed: {exc}"})
    passed = sum(1 for r in results if r.get("success"))
    out = {"total": len(files), "passed": passed,
           "failed": len(files) - passed, "results": results,
           "skill": getattr(skill, "name", "?")}
    # Surface batch-wide label diagnostics so the agent can top up keywords.
    out["label_diagnostics"] = aggregate_label_diagnostics(results)
    return out
