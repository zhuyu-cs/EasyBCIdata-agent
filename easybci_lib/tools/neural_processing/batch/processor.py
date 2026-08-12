"""Batch processor — multi-subject parallel pipeline execution.

.. deprecated::
    ``batch_process`` / ``_process_one_file`` here write a partial layout
    (``preprocessed_output/{sub}/{ses}`` + comparison figures) with NO
    ``code/``/``plan/``/QC/README, so their output was not reproducible. The
    ``batch_process`` tool handler now routes through
    ``batch/orchestrate.py:build_repro_repo`` (standard mini-repo). The pure
    helpers ``_infer_subject_id``/``_infer_session_id``/``_expand_braces`` and
    the checkpoint helpers are still imported elsewhere and remain supported;
    ``batch_process``/``_process_one_file`` are retained for reference/tests.

Supports:
- Glob patterns (sub-*/eeg.fif)
- Brace expansion ({edf,fif,set})
- Memory-aware execution strategy: auto-selects parallel vs sequential
- Parallel processing via ProcessPoolExecutor (true parallelism, memory isolation)
- Summary report (which subjects passed/failed QC)
- Shared config across all subjects
"""

import glob
import json
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed, BrokenExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

from easybci_lib.tools.neural_processing.preprocess.memory_strategy import (
    compute_execution_strategy,
    format_strategy_report,
)

MAX_WORKERS_HARD_LIMIT = 8
DEFAULT_MAX_WORKERS = 4


def _expand_braces(pattern: str) -> List[str]:
    """Expand bash-style brace patterns into multiple glob patterns.

    Example: '/data/test.{edf,fif,set}' → ['/data/test.edf', '/data/test.fif', '/data/test.set']
    """
    match = re.search(r'\{([^}]+)\}', pattern)
    if not match:
        return [pattern]
    prefix = pattern[:match.start()]
    suffix = pattern[match.end():]
    alternatives = match.group(1).split(',')
    expanded = []
    for alt in alternatives:
        expanded.extend(_expand_braces(prefix + alt.strip() + suffix))
    return expanded


def _estimate_file_memory_mb(filepath: str) -> float:
    """Estimate memory footprint of loading a neural data file (MB)."""
    try:
        size_bytes = os.path.getsize(filepath)
        return (size_bytes * 8) / (1024 * 1024)  # 8x overhead: raw + MNE + ICA + peaks
    except OSError:
        return 500.0


def _compute_safe_workers(files: List[str], max_workers: int) -> int:
    """Compute safe concurrency using the memory strategy module."""
    strategy = compute_execution_strategy(files, max_workers=max_workers)
    return strategy.max_workers


def _process_one_file(args_tuple) -> Dict[str, Any]:
    """Process a single file — runs in a child process.

    Takes a tuple because ProcessPoolExecutor.map requires picklable args.
    """
    filepath, steps, output_dir, modality, segment_duration, stride = args_tuple
    subject_id = _infer_subject_id(filepath)
    session_id = _infer_session_id(filepath)
    sub_output_dir = os.path.join(output_dir, "preprocessed_output", subject_id, session_id)
    input_stem = os.path.splitext(os.path.basename(filepath))[0]
    sub_output_path = os.path.join(sub_output_dir, f"{input_stem}_preprocessed.nwb")

    try:
        from easybci_lib.tools.neural_processing.io.loader import load_neural
        from easybci_lib.tools.neural_processing.preprocess.pipeline import preprocess
        from easybci_lib.tools.neural_processing.quality.validators import validate_signal
        from easybci_lib.tools.neural_processing.output.nwb_writer import save_nwb

        data_dict = load_neural(filepath, modality=modality)

        # Capture before-snippet for figure generation (first 5s, ALL channels)
        import numpy as np
        raw_data = data_dict["data"]
        before_freq = data_dict.get("frequency", 256.0)
        channels = list(data_dict.get("channels", []))
        max_viz_samples = min(raw_data.shape[-1], int(before_freq * 5))
        if raw_data.ndim >= 2:
            before_snippet = raw_data[:, :max_viz_samples].copy()
        else:
            before_snippet = raw_data[:max_viz_samples].copy().reshape(1, -1)

        data_dict = preprocess(data_dict, steps=steps)

        qc = validate_signal(data_dict["data"], frequency=data_dict.get("frequency"))
        qc_passed = qc.get("passed", False)

        Path(sub_output_dir).mkdir(parents=True, exist_ok=True)
        # preprocessed/ is NWB-only since the format unification.
        save_nwb(
            payload={
                "data": data_dict["data"],
                "spike_times": data_dict.get("spike_times"),
            },
            out_path=Path(sub_output_path),
            meta={
                "subject_id": subject_id,
                "session_id": session_id,
                "pipeline": steps,
                "frequency": data_dict.get("frequency"),
                "ch_names": data_dict.get("channels"),
                "modality": modality,
            },
            mne_info=data_dict.get("_mne_info"),
        )

        # Generate comparison figures
        try:
            from easybci_lib.tools.neural_processing.quality.compare_viz import generate_comparison_figures
            from easybci_lib.tools.neural_processing.quality.final_view import FinalDataView
            after_view = FinalDataView.from_pipeline_result(
                after_data=data_dict["data"],
                channels=list(data_dict.get("channels", [])) or
                         [f"Ch{i}" for i in range(data_dict["data"].shape[0])],
                frequency=data_dict.get("frequency", before_freq),
                modality=data_dict.get("modality", "eeg"),
                enforce_data_only=True,
            )
            comp_figs = generate_comparison_figures(
                before_data=before_snippet, before_freq=before_freq,
                channels_before=channels, after_view=after_view,
                steps=steps, subject_id=subject_id,
            )
            fig_dir = Path(sub_output_dir) / "figures" / "comparison"
            fig_dir.mkdir(parents=True, exist_ok=True)
            for name, png_bytes in comp_figs.items():
                (fig_dir / name).write_bytes(png_bytes)
        except Exception:
            pass

        return {
            "subject_id": subject_id,
            "session_id": session_id,
            "input": filepath,
            "output": sub_output_path,
            "success": True,
            "qc_passed": qc_passed,
            "error": None,
        }
    except Exception as e:
        return {
            "subject_id": subject_id,
            "session_id": session_id,
            "input": filepath,
            "output": sub_output_path,
            "success": False,
            "qc_passed": False,
            "error": str(e),
        }


def batch_process(
    pattern: str,
    steps: List[str],
    output_dir: str,
    modality: str = "auto",
    segment_duration: float = 2.0,
    stride: float = 1.0,
    paradigm: str = "",
    max_workers: int = DEFAULT_MAX_WORKERS,
    progress_callback=None,
) -> Dict[str, Any]:
    """Process multiple data files matching a glob pattern.

    Parameters
    ----------
    pattern : str
        Glob pattern like "data/sub-*/eeg.fif" or "*.edf"
    steps : list of str
        Preprocessing steps to apply to all files
    output_dir : str
        Root output directory. Each subject gets a subdirectory.
    modality, segment_duration, stride, paradigm :
        Shared pipeline parameters
    max_workers : int
        Maximum parallel workers (capped at MAX_WORKERS_HARD_LIMIT=8).
        Actual concurrency may be lower based on available memory.
    progress_callback : callable, optional
        Called with (event, data) for progress tracking

    Returns
    -------
    dict with keys: total, passed, failed, results, report_path
    """
    patterns = _expand_braces(pattern)
    files = sorted(set(f for p in patterns for f in glob.glob(p, recursive=True)))
    if not files:
        return {"total": 0, "passed": 0, "failed": 0, "error": f"No files match: {pattern}"}

    # --- Batch checkpoint: skip already-completed subjects ---
    checkpoint_path = os.path.join(output_dir, ".batch_checkpoint.json")
    completed_subjects = _load_batch_checkpoint(checkpoint_path)

    if completed_subjects:
        original_count = len(files)
        files = [f for f in files if _infer_subject_id(f) not in completed_subjects]
        if len(files) < original_count:
            _emit(progress_callback, "batch_resume", {
                "skipped": original_count - len(files),
                "remaining": len(files),
            })

    safe_workers = _compute_safe_workers(files, max_workers)
    strategy = compute_execution_strategy(files, max_workers=max_workers)

    _emit(progress_callback, "batch_start", {
        "pattern": pattern,
        "total": len(files),
        "max_workers": safe_workers,
        "strategy": strategy.mode,
        "strategy_reason": strategy.reason,
    })

    results = []
    passed = 0
    failed = 0

    task_args = [
        (f, steps, output_dir, modality, segment_duration, stride)
        for f in files
    ]

    # Use ProcessPoolExecutor for true parallelism and memory isolation.
    # Each child process loads one file, processes it, saves output, then exits —
    # releasing all memory. No risk of accumulation across files.
    #
    # If strategy says sequential/chunked, we still use the pool (max_workers=1)
    # for OOM isolation — a killed child doesn't crash the parent.
    effective_workers = min(safe_workers, strategy.max_workers)

    with ProcessPoolExecutor(max_workers=effective_workers) as executor:
        futures = {executor.submit(_process_one_file, args): args[0] for args in task_args}

        for future in as_completed(futures):
            filepath = futures[future]
            try:
                result = future.result()
            except BrokenExecutor as e:
                result = {
                    "subject_id": _infer_subject_id(filepath),
                    "session_id": _infer_session_id(filepath),
                    "input": filepath,
                    "output": "",
                    "success": False,
                    "qc_passed": False,
                    "error": f"Process killed (likely OOM): {e}",
                }
            except Exception as e:
                result = {
                    "subject_id": _infer_subject_id(filepath),
                    "session_id": _infer_session_id(filepath),
                    "input": filepath,
                    "output": "",
                    "success": False,
                    "qc_passed": False,
                    "error": f"Unexpected error: {e}",
                }

            results.append(result)

            if result["success"] and result["qc_passed"]:
                passed += 1
            else:
                failed += 1

            # Checkpoint: record completed subjects for resume-on-failure
            if result["success"]:
                _save_batch_checkpoint(checkpoint_path, result["subject_id"])

            _emit(progress_callback, "batch_progress", {
                "subject": result["subject_id"],
                "success": result["success"],
                "done": len(results),
                "total": len(files),
            })

    results.sort(key=lambda r: r["subject_id"])

    report = _build_report(results, pattern, steps, output_dir)
    report_path = os.path.join(output_dir, "batch_report.json")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    md_path = os.path.join(output_dir, "batch_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(_build_report_md(report))

    _emit(progress_callback, "batch_complete", {
        "total": len(files), "passed": passed, "failed": failed,
    })

    return {
        "total": len(files),
        "passed": passed,
        "failed": failed,
        "results": results,
        "report_path": report_path,
        "report_md_path": md_path,
        "workers_used": safe_workers,
    }


def _infer_subject_id(filepath: str) -> str:
    """Infer subject ID from file path."""
    parts = Path(filepath).parts
    for part in parts:
        if part.startswith("sub-") or part.startswith("sub_"):
            return part
    return Path(filepath).stem


def _infer_session_id(filepath: str) -> str:
    """Infer session ID from file path or filename timestamp.

    Priority:
    1. BIDS-style ses-* in path components or filename
    2. Timestamp extracted from filename (YYYY_MM_DD_HHMM patterns) → ses-YYYYMMDDTHHMM
    3. Fallback: ses-001
    """
    p = Path(filepath)

    # Check path components and filename for BIDS ses-*
    for part in p.parts:
        if part.startswith("ses-"):
            return part
    stem = p.stem
    for segment in stem.split("_"):
        if segment.startswith("ses-"):
            return segment

    # Extract timestamp from filename patterns
    # Pattern: YYYY_MM_DD_HHMM (e.g., "2026_05_22_1623")
    m = re.search(r'(\d{4})_(\d{2})_(\d{2})_(\d{3,4})', stem)
    if m:
        year, month, day, time_part = m.groups()
        time_part = time_part.zfill(4)
        return f"ses-{year}{month}{day}T{time_part}"

    # Pattern: YYYYMMDD_HHMM or YYYYMMDDTHHMM
    m = re.search(r'(\d{8})[T_](\d{4})', stem)
    if m:
        date_part, time_part = m.groups()
        return f"ses-{date_part}T{time_part}"

    return "ses-001"


def _build_report(results: List[Dict], pattern: str, steps: List[str], output_dir: str) -> Dict:
    """Build structured batch report."""
    passed = [r for r in results if r["success"] and r["qc_passed"]]
    failed = [r for r in results if not r["success"] or not r["qc_passed"]]

    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "pattern": pattern,
        "pipeline": steps,
        "output_dir": output_dir,
        "summary": {
            "total": len(results),
            "passed": len(passed),
            "failed": len(failed),
            "pass_rate": f"{len(passed)/len(results)*100:.1f}%" if results else "0%",
        },
        "passed_subjects": [r["subject_id"] for r in passed],
        "failed_subjects": [
            {"subject_id": r["subject_id"], "error": r.get("error", "QC failed")}
            for r in failed
        ],
        "details": results,
    }


def _build_report_md(report: Dict) -> str:
    """Build markdown summary report."""
    lines = ["# Batch Processing Report\n"]
    lines.append(f"Generated: {report['generated_at']}\n")
    lines.append(f"Pattern: `{report['pattern']}`\n")
    lines.append(f"Pipeline: {' → '.join(report['pipeline'])}\n")

    s = report["summary"]
    lines.append(f"## Summary\n")
    lines.append(f"- Total: {s['total']}")
    lines.append(f"- Passed: {s['passed']} ({s['pass_rate']})")
    lines.append(f"- Failed: {s['failed']}")
    lines.append("")

    if report["passed_subjects"]:
        lines.append("## Passed\n")
        for sub in report["passed_subjects"]:
            lines.append(f"- {sub}")
        lines.append("")

    if report["failed_subjects"]:
        lines.append("## Failed\n")
        for item in report["failed_subjects"]:
            lines.append(f"- **{item['subject_id']}**: {item['error']}")
        lines.append("")

    return "\n".join(lines) + "\n"


def _emit(callback, event, data):
    if callback:
        try:
            callback(event, data)
        except Exception:
            pass


def _load_batch_checkpoint(checkpoint_path: str) -> set:
    """Load set of completed subject IDs from checkpoint file."""
    try:
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, encoding="utf-8") as f:
                data = json.load(f)
            return set(data.get("completed", []))
    except (OSError, json.JSONDecodeError, ValueError):
        pass
    return set()


def _save_batch_checkpoint(checkpoint_path: str, subject_id: str) -> None:
    """Append a subject ID to the checkpoint file (atomic)."""
    try:
        existing = _load_batch_checkpoint(checkpoint_path)
        existing.add(subject_id)
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        tmp_path = checkpoint_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump({"completed": sorted(existing)}, f)
        os.replace(tmp_path, checkpoint_path)
    except OSError:
        pass
