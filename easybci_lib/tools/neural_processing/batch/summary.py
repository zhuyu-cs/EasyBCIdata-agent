"""Batch summary dashboard — cross-subject QC aggregation and exclusion recommendations.

Provides:
1. Cross-subject QC summary table (pass/fail/reason per subject)
2. Group-level statistics (mean channel variance, SNR, artifact rates)
3. Automatic exclusion recommendations based on outlier detection

Integrates with batch_process() output to generate a comprehensive dashboard.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from easybci_lib.tools.neural_processing.batch.streaming_stats import (
    compute_streaming_metrics,
)

logger = logging.getLogger(__name__)


@dataclass
class SubjectQCSummary:
    """Per-subject QC summary entry."""
    subject_id: str
    passed: bool
    error: str = ""
    channel_variance_mean: float = 0.0
    channel_variance_std: float = 0.0
    snr_db: float = 0.0
    artifact_ratio: float = 0.0
    bad_channels: List[str] = field(default_factory=list)
    n_channels: int = 0
    frequency: float = 0.0
    duration_s: float = 0.0
    processing_time_s: float = 0.0
    outlier_flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject_id": self.subject_id,
            "passed": self.passed,
            "error": self.error,
            "channel_variance_mean": round(self.channel_variance_mean, 4),
            "channel_variance_std": round(self.channel_variance_std, 4),
            "snr_db": round(self.snr_db, 2),
            "artifact_ratio": round(self.artifact_ratio, 4),
            "bad_channels": self.bad_channels,
            "n_channels": self.n_channels,
            "frequency": self.frequency,
            "duration_s": round(self.duration_s, 1),
            "processing_time_s": round(self.processing_time_s, 2),
            "outlier_flags": self.outlier_flags,
        }


@dataclass
class GroupStats:
    """Group-level aggregate statistics across all subjects."""
    n_subjects: int = 0
    n_passed: int = 0
    n_failed: int = 0
    pass_rate: float = 0.0
    mean_channel_variance: float = 0.0
    std_channel_variance: float = 0.0
    mean_snr_db: float = 0.0
    std_snr_db: float = 0.0
    mean_artifact_ratio: float = 0.0
    std_artifact_ratio: float = 0.0
    mean_n_channels: float = 0.0
    mean_duration_s: float = 0.0
    total_processing_time_s: float = 0.0
    failure_reasons: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_subjects": self.n_subjects,
            "n_passed": self.n_passed,
            "n_failed": self.n_failed,
            "pass_rate": f"{self.pass_rate:.1%}",
            "mean_channel_variance": round(self.mean_channel_variance, 4),
            "std_channel_variance": round(self.std_channel_variance, 4),
            "mean_snr_db": round(self.mean_snr_db, 2),
            "std_snr_db": round(self.std_snr_db, 2),
            "mean_artifact_ratio": round(self.mean_artifact_ratio, 4),
            "std_artifact_ratio": round(self.std_artifact_ratio, 4),
            "mean_n_channels": round(self.mean_n_channels, 1),
            "mean_duration_s": round(self.mean_duration_s, 1),
            "total_processing_time_s": round(self.total_processing_time_s, 1),
            "failure_reasons": self.failure_reasons,
        }


@dataclass
class ExclusionRecommendation:
    """A recommendation to exclude a subject from analysis."""
    subject_id: str
    reason: str
    severity: str = "warning"  # "warning" or "critical"
    metric_name: str = ""
    metric_value: float = 0.0
    threshold: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject_id": self.subject_id,
            "reason": self.reason,
            "severity": self.severity,
            "metric_name": self.metric_name,
            "metric_value": round(self.metric_value, 4),
            "threshold": round(self.threshold, 4),
        }


@dataclass
class BatchSummaryReport:
    """Complete batch summary dashboard."""
    generated_at: str = ""
    pipeline: List[str] = field(default_factory=list)
    group_stats: GroupStats = field(default_factory=GroupStats)
    subjects: List[SubjectQCSummary] = field(default_factory=list)
    exclusion_recommendations: List[ExclusionRecommendation] = field(default_factory=list)
    summary_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "pipeline": self.pipeline,
            "group_stats": self.group_stats.to_dict(),
            "subjects": [s.to_dict() for s in self.subjects],
            "exclusion_recommendations": [e.to_dict() for e in self.exclusion_recommendations],
            "summary_text": self.summary_text,
        }

    def to_markdown(self) -> str:
        """Generate a human-readable Markdown report."""
        lines = ["# Batch Processing Summary Dashboard\n"]
        lines.append(f"Generated: {self.generated_at}")
        lines.append(f"Pipeline: `{' → '.join(self.pipeline)}`\n")

        # Group stats
        gs = self.group_stats
        lines.append("## Group Statistics\n")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Subjects | {gs.n_subjects} |")
        lines.append(f"| Passed QC | {gs.n_passed} ({gs.pass_rate:.0%}) |")
        lines.append(f"| Failed QC | {gs.n_failed} |")
        lines.append(f"| Mean SNR | {gs.mean_snr_db:.1f} ± {gs.std_snr_db:.1f} dB |")
        lines.append(f"| Mean Artifact Ratio | {gs.mean_artifact_ratio:.2%} ± {gs.std_artifact_ratio:.2%} |")
        lines.append(f"| Mean Ch. Variance | {gs.mean_channel_variance:.4f} ± {gs.std_channel_variance:.4f} |")
        lines.append(f"| Total Processing Time | {gs.total_processing_time_s:.0f}s |")
        lines.append("")

        if gs.failure_reasons:
            lines.append("### Failure Reasons\n")
            for reason, count in sorted(gs.failure_reasons.items(), key=lambda x: -x[1]):
                lines.append(f"- {reason}: {count} subjects")
            lines.append("")

        # Subject table
        lines.append("## Per-Subject QC\n")
        lines.append("| Subject | Status | SNR (dB) | Artifact % | Variance | Flags |")
        lines.append("|---------|--------|----------|------------|----------|-------|")
        for s in self.subjects:
            status = "✓" if s.passed else "✗"
            flags = ", ".join(s.outlier_flags) if s.outlier_flags else "—"
            lines.append(
                f"| {s.subject_id} | {status} | {s.snr_db:.1f} | "
                f"{s.artifact_ratio:.1%} | {s.channel_variance_mean:.4f} | {flags} |"
            )
        lines.append("")

        # Exclusion recommendations
        if self.exclusion_recommendations:
            lines.append("## Exclusion Recommendations\n")
            for rec in self.exclusion_recommendations:
                icon = "🔴" if rec.severity == "critical" else "🟡"
                lines.append(f"- {icon} **{rec.subject_id}**: {rec.reason}")
            lines.append("")

        # Summary text
        if self.summary_text:
            lines.append("## Summary\n")
            lines.append(self.summary_text)
            lines.append("")

        return "\n".join(lines)


def generate_batch_summary(
    batch_results: List[Dict[str, Any]],
    output_dir: str = "",
    pipeline: Optional[List[str]] = None,
    iqr_multiplier: float = 2.0,
) -> BatchSummaryReport:
    """Generate a comprehensive batch summary from individual processing results.

    Parameters
    ----------
    batch_results : list of dict
        Per-subject results from batch_process(). Each dict has:
        subject_id, success, qc_passed, error, output.
    output_dir : str
        Base output directory (to locate processed data for QC analysis).
    pipeline : list of str, optional
        Pipeline steps that were applied.
    iqr_multiplier : float
        IQR multiplier for outlier detection (default 2.0 = moderate sensitivity).

    Returns
    -------
    BatchSummaryReport with full dashboard data.
    """
    report = BatchSummaryReport(
        generated_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        pipeline=pipeline or [],
    )

    subjects: List[SubjectQCSummary] = []

    for result in batch_results:
        subject = SubjectQCSummary(
            subject_id=result.get("subject_id", "unknown"),
            passed=result.get("success", False) and result.get("qc_passed", False),
            error=result.get("error", "") or "",
        )

        # Try to load processed data for detailed QC metrics
        output_path = result.get("output", "")
        if output_path and result.get("success"):
            _populate_subject_metrics(subject, output_path, output_dir)

        subjects.append(subject)

    report.subjects = subjects

    # Compute group statistics
    report.group_stats = _compute_group_stats(subjects)

    # Detect outliers and generate exclusion recommendations
    report.exclusion_recommendations = _detect_outliers(subjects, iqr_multiplier)

    # Flag outlier subjects
    excluded_ids = {r.subject_id for r in report.exclusion_recommendations}
    for s in subjects:
        if s.subject_id in excluded_ids:
            reasons = [r.reason for r in report.exclusion_recommendations if r.subject_id == s.subject_id]
            s.outlier_flags = reasons

    # Generate summary text
    report.summary_text = _generate_summary_text(report)

    return report


def _populate_subject_metrics(
    subject: SubjectQCSummary,
    output_path: str,
    output_dir: str,
) -> None:
    """Load processed data and compute QC metrics for a subject."""
    try:
        path = Path(output_path)
        if path.suffix == ".nwb":
            # Stream metrics directly off the lazy h5py Dataset — never load the
            # whole recording (a single sEEG NWB is ~7 GB; the old data[:]/.T
            # here OOM-killed the host mid-batch). The IO context must stay open
            # for the whole reduction (h5py reads are lazy). See streaming_stats.
            from pynwb import NWBHDF5IO
            with NWBHDF5IO(str(path), "r") as _io:
                _nwb = _io.read()
                _acq = _nwb.acquisition
                _es_name = "preprocessed" if "preprocessed" in _acq else next(iter(_acq))
                _dset = _acq[_es_name].data  # h5py.Dataset, (n_samp, n_ch), lazy
                metrics = compute_streaming_metrics(_dset)
            subject.channel_variance_mean = metrics.channel_variance_mean
            subject.channel_variance_std = metrics.channel_variance_std
            subject.n_channels = metrics.n_channels
            subject.snr_db = metrics.snr_db
            subject.artifact_ratio = metrics.artifact_ratio
            _populate_bad_channels_from_qc(subject, path)
            return

        # .pkl / .npz hold already-epoched arrays, usually far smaller than raw
        # NWB. Load them (best-effort) and route through the same numeric core
        # via a transposed view so results stay consistent with the NWB path.
        if path.suffix == ".pkl":
            import pickle
            with open(path, "rb") as f:
                data_dict = pickle.load(f)
            data = data_dict.get("neural", data_dict.get("data"))
        elif path.suffix == ".npz":
            loaded = np.load(path, allow_pickle=True)
            data = loaded.get("data", loaded.get("arr_0"))
        else:
            return

        if data is None:
            return

        data = np.asarray(data)
        if data.ndim < 2:
            return
        if data.ndim == 3:
            data = data.reshape(-1, data.shape[-1])

        # data is (n_channels, n_samples); streaming source wants (n_samp, n_ch).
        metrics = compute_streaming_metrics(np.ascontiguousarray(data.T))
        subject.channel_variance_mean = metrics.channel_variance_mean
        subject.channel_variance_std = metrics.channel_variance_std
        subject.n_channels = metrics.n_channels
        subject.snr_db = metrics.snr_db
        subject.artifact_ratio = metrics.artifact_ratio

        # Also try to read QC result from json
        _populate_bad_channels_from_qc(subject, path)

    except Exception as exc:
        logger.debug("Failed to compute metrics for %s: %s", subject.subject_id, exc)


def _populate_bad_channels_from_qc(subject: SubjectQCSummary, path: Path) -> None:
    """Read sibling qc_result.json (if present) for a bad-channel list."""
    qc_path = path.parent / "qc_result.json"
    if qc_path.exists():
        try:
            qc_data = json.loads(qc_path.read_text(encoding="utf-8"))
            bad_chs = qc_data.get("bad_channels", [])
            if bad_chs:
                subject.bad_channels = bad_chs[:10]
        except (json.JSONDecodeError, OSError):
            pass


def _compute_group_stats(subjects: List[SubjectQCSummary]) -> GroupStats:
    """Compute aggregate statistics across all subjects."""
    gs = GroupStats(n_subjects=len(subjects))

    passed = [s for s in subjects if s.passed]
    failed = [s for s in subjects if not s.passed]
    gs.n_passed = len(passed)
    gs.n_failed = len(failed)
    gs.pass_rate = gs.n_passed / gs.n_subjects if gs.n_subjects > 0 else 0.0

    # Aggregate metrics from subjects with data
    variances = [s.channel_variance_mean for s in subjects if s.channel_variance_mean > 0]
    snrs = [s.snr_db for s in subjects if s.snr_db != 0]
    artifacts = [s.artifact_ratio for s in subjects if s.channel_variance_mean > 0]
    durations = [s.duration_s for s in subjects if s.duration_s > 0]
    n_channels = [s.n_channels for s in subjects if s.n_channels > 0]
    proc_times = [s.processing_time_s for s in subjects if s.processing_time_s > 0]

    if variances:
        gs.mean_channel_variance = float(np.mean(variances))
        gs.std_channel_variance = float(np.std(variances))
    if snrs:
        gs.mean_snr_db = float(np.mean(snrs))
        gs.std_snr_db = float(np.std(snrs))
    if artifacts:
        gs.mean_artifact_ratio = float(np.mean(artifacts))
        gs.std_artifact_ratio = float(np.std(artifacts))
    if durations:
        gs.mean_duration_s = float(np.mean(durations))
    if n_channels:
        gs.mean_n_channels = float(np.mean(n_channels))
    if proc_times:
        gs.total_processing_time_s = float(np.sum(proc_times))

    # Categorize failure reasons
    for s in failed:
        reason = _categorize_failure(s.error)
        gs.failure_reasons[reason] = gs.failure_reasons.get(reason, 0) + 1

    return gs


def _categorize_failure(error: str) -> str:
    """Categorize an error message into a high-level failure reason."""
    if not error:
        return "QC failed (unspecified)"

    error_lower = error.lower()
    if "memory" in error_lower:
        return "Memory error"
    if "nan" in error_lower or "inf" in error_lower:
        return "Non-finite values"
    if "flat" in error_lower or "zero variance" in error_lower:
        return "Flat/dead channels"
    if "file" in error_lower or "load" in error_lower or "read" in error_lower:
        return "File loading error"
    if "shape" in error_lower or "dimension" in error_lower:
        return "Data shape mismatch"
    if "singular" in error_lower or "covariance" in error_lower:
        return "Numerical instability"
    if "artifact" in error_lower:
        return "Excessive artifacts"
    return "Other error"


def _detect_outliers(
    subjects: List[SubjectQCSummary],
    iqr_multiplier: float = 2.0,
) -> List[ExclusionRecommendation]:
    """Detect outlier subjects using IQR-based method across multiple metrics."""
    recommendations: List[ExclusionRecommendation] = []

    # Only analyze subjects that have metrics
    valid = [s for s in subjects if s.channel_variance_mean > 0]
    if len(valid) < 3:
        # Not enough subjects for outlier detection
        # Still flag complete failures
        for s in subjects:
            if not s.passed and s.error:
                recommendations.append(ExclusionRecommendation(
                    subject_id=s.subject_id,
                    reason=f"Processing failed: {s.error[:80]}",
                    severity="critical",
                ))
        return recommendations

    # Metrics to check
    metrics = [
        ("channel_variance_mean", "high channel variance (noise)", "high"),
        ("snr_db", "low SNR", "low"),
        ("artifact_ratio", "high artifact ratio", "high"),
    ]

    for attr, description, direction in metrics:
        values = np.array([getattr(s, attr) for s in valid])
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1

        if iqr < 1e-10:
            continue

        if direction == "high":
            threshold = q3 + iqr_multiplier * iqr
            outlier_mask = values > threshold
        else:
            threshold = q1 - iqr_multiplier * iqr
            outlier_mask = values < threshold

        for i, is_outlier in enumerate(outlier_mask):
            if is_outlier:
                value = values[i]
                severity = "critical" if abs(value - np.median(values)) > 3 * iqr else "warning"
                recommendations.append(ExclusionRecommendation(
                    subject_id=valid[i].subject_id,
                    reason=f"Outlier: {description} ({value:.4f} vs group median {np.median(values):.4f})",
                    severity=severity,
                    metric_name=attr,
                    metric_value=float(value),
                    threshold=float(threshold),
                ))

    # Also flag subjects that completely failed processing
    for s in subjects:
        if not s.passed and s.error:
            already_flagged = any(r.subject_id == s.subject_id for r in recommendations)
            if not already_flagged:
                recommendations.append(ExclusionRecommendation(
                    subject_id=s.subject_id,
                    reason=f"Processing failed: {s.error[:80]}",
                    severity="critical",
                ))

    # Deduplicate by subject (keep highest severity)
    seen = {}
    for rec in recommendations:
        key = rec.subject_id
        if key not in seen or (rec.severity == "critical" and seen[key].severity != "critical"):
            seen[key] = rec
        elif key in seen and rec.metric_name != seen[key].metric_name:
            # Multiple flags — combine reasons
            existing = seen[key]
            existing.reason = f"{existing.reason}; {rec.reason}"

    return list(seen.values())


def _generate_summary_text(report: BatchSummaryReport) -> str:
    """Generate a natural language summary of the batch processing results."""
    gs = report.group_stats
    excl = report.exclusion_recommendations

    parts = []

    # Overall status
    if gs.pass_rate >= 0.9:
        parts.append(
            f"Batch processing completed successfully: {gs.n_passed}/{gs.n_subjects} subjects "
            f"passed QC ({gs.pass_rate:.0%} pass rate)."
        )
    elif gs.pass_rate >= 0.7:
        parts.append(
            f"Batch processing completed with some issues: {gs.n_passed}/{gs.n_subjects} subjects "
            f"passed QC ({gs.pass_rate:.0%}). {gs.n_failed} subjects failed."
        )
    else:
        parts.append(
            f"Batch processing had significant problems: only {gs.n_passed}/{gs.n_subjects} subjects "
            f"passed QC ({gs.pass_rate:.0%}). Review pipeline parameters."
        )

    # SNR summary
    if gs.mean_snr_db > 0:
        if gs.mean_snr_db >= 10:
            parts.append(f"Signal quality is good (mean SNR: {gs.mean_snr_db:.1f} dB).")
        elif gs.mean_snr_db >= 5:
            parts.append(f"Signal quality is moderate (mean SNR: {gs.mean_snr_db:.1f} dB).")
        else:
            parts.append(f"Signal quality is poor (mean SNR: {gs.mean_snr_db:.1f} dB). Consider additional denoising.")

    # Failure patterns
    if gs.failure_reasons:
        top_reason = max(gs.failure_reasons.items(), key=lambda x: x[1])
        parts.append(f"Most common failure: {top_reason[0]} ({top_reason[1]} subjects).")

    # Exclusion recommendations
    if excl:
        critical = [e for e in excl if e.severity == "critical"]
        warnings = [e for e in excl if e.severity == "warning"]
        if critical:
            ids = ", ".join(e.subject_id for e in critical[:5])
            parts.append(f"Recommend excluding {len(critical)} subject(s): {ids}.")
        if warnings:
            parts.append(f"Additionally, {len(warnings)} subject(s) show borderline metrics (review recommended).")

    return " ".join(parts)


def save_batch_summary(
    report: BatchSummaryReport,
    output_dir: str,
) -> Dict[str, str]:
    """Save the batch summary report to disk in JSON and Markdown formats.

    Also generates batch QC figures if processed data is available.

    Returns dict with paths to generated files.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    json_path = out / "batch_summary.json"
    md_path = out / "batch_summary.md"

    json_path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    md_path.write_text(report.to_markdown(), encoding="utf-8")

    result = {
        "json": str(json_path),
        "markdown": str(md_path),
    }

    # Generate batch QC visualizations
    try:
        from easybci_lib.tools.neural_processing.quality.visualize import generate_batch_qc_figures
        batch_results = []
        for subj in report.subjects:
            batch_results.append({
                "subject_id": subj.subject_id,
                "output_path": subj.output_path,
                "success": subj.status == "success",
                "frequency": subj.frequency if hasattr(subj, "frequency") else 0,
                "channels": subj.channels if hasattr(subj, "channels") else [],
            })
        if batch_results:
            qc_result = generate_batch_qc_figures(batch_results, output_dir)
            if qc_result.get("overview"):
                result["qc_overview"] = qc_result["overview"]
            result["qc_n_subjects_plotted"] = qc_result.get("n_subjects_plotted", 0)
    except Exception:
        pass

    return result


def summarize_adaptive(batch_result: dict) -> dict:
    """Build an auditable summary for an adaptive batch run.

    Per-file: QC grade + baseline status/warnings + which slots were adapted +
    success/out-of-range. Aggregate: counts. Advisory only — baseline warnings
    do NOT change pass/fail (design 04: no hard gate).
    """
    results = batch_result.get("results", []) or []
    per_file: list = []
    n_warn = n_oor = 0
    for r in results:
        if not r.get("success"):
            oor = bool(r.get("out_of_range"))
            n_oor += 1 if oor else 0
            per_file.append({
                "input": r.get("input", "?"), "status": "failed",
                "out_of_range": oor,
                "reasons": r.get("reasons") or ([r["error"]] if r.get("error") else []),
            })
            continue
        base = r.get("baseline", {}) or {}
        status = base.get("status", "no_baseline")
        warns = base.get("warnings", []) or []
        if status == "baseline_warning":
            n_warn += 1
        per_file.append({
            "input": r.get("input", "?"), "status": "ok",
            "grade": r.get("grade", "?"),
            "baseline_status": status,
            "baseline_warnings": [w.get("note", "") for w in warns],
            "adapted_slots": [s.get("param") for s in (r.get("adaptation_report") or [])],
            "output": r.get("output", ""),
        })
    n_passed = int(batch_result.get("passed", sum(1 for r in results if r.get("success"))))
    n_failed = int(batch_result.get("failed", len(results) - n_passed))
    lines = [
        f"Adaptive batch: skill={batch_result.get('skill', '?')}",
        f"  total={batch_result.get('total', len(results))} "
        f"passed={n_passed} failed={n_failed}",
        f"  baseline_warnings={n_warn} out_of_range={n_oor}",
    ]
    for row in per_file:
        if row["status"] == "failed":
            lines.append(f"  ✗ {row['input']}: {'; '.join(row.get('reasons', []))}")
        else:
            tag = "⚠ " if row["baseline_status"] == "baseline_warning" else "  "
            lines.append(
                f"{tag}{row['input']}: grade={row['grade']} "
                f"baseline={row['baseline_status']} "
                f"adapted={','.join(x for x in row['adapted_slots'] if x)}")
    return {
        "per_file": per_file, "n_passed": n_passed, "n_failed": n_failed,
        "n_baseline_warnings": n_warn, "n_out_of_range": n_oor,
        "summary_text": "\n".join(lines),
    }
