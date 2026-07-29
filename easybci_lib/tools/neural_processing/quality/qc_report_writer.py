"""Comprehensive QC report writer — generates detailed per-session quality reports.

Writes a human-readable Markdown QC report plus a machine-readable JSON sidecar
into QC_out/{session_id}/. The report includes signal metrics, per-step state
transitions, artifact statistics, and pass/fail grading.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def write_session_qc_report(
    output_dir: str,
    session_id: str,
    subject_id: str,
    data_path: str,
    steps: List[str],
    frequency_before: float,
    frequency_after: float,
    channels_before: List[str],
    channels_after: List[str],
    qc_metrics: Optional[Dict[str, Any]] = None,
    qc_feedback: Optional[Dict[str, Any]] = None,
    step_states: Optional[List[Dict[str, Any]]] = None,
    data_shape_before: Optional[List[int]] = None,
    data_shape_after: Optional[List[int]] = None,
) -> str:
    """Write a comprehensive QC report for a single session.

    Parameters
    ----------
    output_dir : str
        Directory to write report files into (e.g., QC_out/sub-{id}/ses-XXXX/).
    session_id : str
        Session identifier.
    subject_id : str
        Subject identifier.
    data_path : str
        Original data file path.
    steps : list of str
        Pipeline steps applied.
    frequency_before, frequency_after : float
        Sampling rates before/after processing.
    channels_before, channels_after : list of str
        Channel names before/after processing.
    qc_metrics : dict, optional
        Metrics from compute_qc_metrics().to_dict().
    qc_feedback : dict, optional
        NL feedback from generate_nl_feedback().to_dict().
    step_states : list of dict, optional
        Per-step state transitions from pipeline execution.
    data_shape_before, data_shape_after : list of int, optional
        Data array shapes before/after processing.

    Returns
    -------
    str
        Path to the written Markdown report.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- Build Markdown report ---
    lines = _build_markdown_report(
        session_id=session_id,
        subject_id=subject_id,
        data_path=data_path,
        steps=steps,
        frequency_before=frequency_before,
        frequency_after=frequency_after,
        channels_before=channels_before,
        channels_after=channels_after,
        qc_metrics=qc_metrics,
        qc_feedback=qc_feedback,
        step_states=step_states,
        data_shape_before=data_shape_before,
        data_shape_after=data_shape_after,
    )

    md_path = out / f"qc_report_{session_id}.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    # --- Write JSON sidecar (machine-readable) ---
    json_data = {
        "session_id": session_id,
        "subject_id": subject_id,
        "source_file": data_path,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pipeline_steps": steps,
        "sampling_rate": {"before": frequency_before, "after": frequency_after},
        "channels": {
            "before": channels_before,
            "after": channels_after,
            "n_before": len(channels_before),
            "n_after": len(channels_after),
        },
        "data_shape": {
            "before": data_shape_before,
            "after": data_shape_after,
        },
        "qc_metrics": qc_metrics,
        "qc_feedback": qc_feedback,
        "step_states": step_states,
    }
    json_path = out / f"qc_report_{session_id}.json"
    json_path.write_text(
        json.dumps(json_data, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.info("QC report written to %s", md_path)
    return str(md_path)


def _build_markdown_report(
    session_id: str,
    subject_id: str,
    data_path: str,
    steps: List[str],
    frequency_before: float,
    frequency_after: float,
    channels_before: List[str],
    channels_after: List[str],
    qc_metrics: Optional[Dict[str, Any]],
    qc_feedback: Optional[Dict[str, Any]],
    step_states: Optional[List[Dict[str, Any]]],
    data_shape_before: Optional[List[int]],
    data_shape_after: Optional[List[int]],
) -> List[str]:
    """Build the Markdown report content."""
    lines: List[str] = []

    # Header
    grade = "N/A"
    if qc_metrics and "overall" in qc_metrics:
        grade = qc_metrics["overall"].get("grade", "N/A")
    elif qc_feedback:
        grade = qc_feedback.get("grade", "N/A")

    lines.extend([
        f"# QC Report: {session_id}",
        "",
        f"**Subject:** {subject_id}  ",
        f"**Session:** {session_id}  ",
        f"**Source file:** `{Path(data_path).name}`  ",
        f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  ",
        f"**Overall Grade:** **{grade}**",
        "",
        "---",
        "",
    ])

    # Summary
    if qc_feedback and qc_feedback.get("summary"):
        lines.extend([
            "## Summary",
            "",
            qc_feedback["summary"],
            "",
        ])

    # Data Overview
    lines.extend([
        "## Data Overview",
        "",
        "| Property | Before | After |",
        "|----------|--------|-------|",
        f"| Sampling Rate | {frequency_before:.1f} Hz | {frequency_after:.1f} Hz |",
        f"| Channels | {len(channels_before)} | {len(channels_after)} |",
    ])
    if data_shape_before:
        lines.append(f"| Shape | {_shape_str(data_shape_before)} | {_shape_str(data_shape_after)} |")
    if data_shape_before and len(data_shape_before) >= 2:
        dur_before = data_shape_before[-1] / frequency_before if frequency_before > 0 else 0
        dur_after = (data_shape_after[-1] / frequency_after) if data_shape_after and frequency_after > 0 else 0
        lines.append(f"| Duration | {dur_before:.1f} s | {dur_after:.1f} s |")
    lines.append("")

    # Channels dropped
    dropped = set(channels_before) - set(channels_after)
    if dropped:
        n_before = len(channels_before) if channels_before else 0
        ratio = (len(dropped) / n_before) if n_before else 0.0
        lines.extend([
            "### Channels Dropped",
            "",
            f"The following {len(dropped)} channel(s) were removed during processing "
            f"({ratio:.0%} of {n_before} input channels):",
            "",
        ])
        for ch in sorted(dropped):
            lines.append(f"- `{ch}`")
        if ratio > 0.5:
            lines.extend([
                "",
                f"> **WARNING — high drop ratio ({ratio:.0%})**: verify data_unit and "
                f"consider re-running with a `scale` step before `drop_bads`.",
            ])
        lines.append("")

    # Pipeline Steps
    lines.extend([
        "## Pipeline Steps Applied",
        "",
        "| # | Step | Parameters |",
        "|---|------|------------|",
    ])
    for i, step in enumerate(steps, 1):
        if ":" in step:
            name, params = step.split(":", 1)
            lines.append(f"| {i} | {name} | {params} |")
        else:
            lines.append(f"| {i} | {step} | — |")
    lines.append("")

    # Per-step state transitions
    if step_states:
        lines.extend([
            "## Per-Step State Transitions",
            "",
        ])
        for state in step_states:
            step_name = state.get("step", "unknown")
            lines.append(f"### Step: `{step_name}`")
            lines.append("")
            before_info = state.get("before", {})
            after_info = state.get("after", {})
            if before_info or after_info:
                lines.append("| Metric | Before | After |")
                lines.append("|--------|--------|-------|")
                all_keys = sorted(set(list(before_info.keys()) + list(after_info.keys())))
                for key in all_keys:
                    b_val = _fmt_val(before_info.get(key))
                    a_val = _fmt_val(after_info.get(key))
                    lines.append(f"| {key} | {b_val} | {a_val} |")
                lines.append("")
            else:
                lines.append("_(no state data recorded)_")
                lines.append("")

    # QC Metrics
    if qc_metrics:
        lines.extend([
            "## Signal Quality Metrics",
            "",
        ])

        # SNR
        snr = qc_metrics.get("snr", {})
        if snr.get("improvement_db"):
            lines.extend([
                "### SNR (Signal-to-Noise Ratio)",
                "",
                "| Band | Before (dB) | After (dB) | Improvement |",
                "|------|-------------|------------|-------------|",
            ])
            before_snr = snr.get("before", {})
            after_snr = snr.get("after", {})
            improvement = snr.get("improvement_db", {})
            for band in sorted(improvement.keys()):
                b = before_snr.get(band, 0)
                a = after_snr.get(band, 0)
                imp = improvement[band]
                arrow = "+" if imp >= 0 else ""
                lines.append(f"| {band} | {b:.1f} | {a:.1f} | {arrow}{imp:.1f} dB |")
            lines.append("")

        # Artifact residual
        artifact = qc_metrics.get("artifact_residual", {})
        if artifact:
            lines.extend([
                "### Artifact Residual",
                "",
                f"- **Residual ratio:** {artifact.get('ratio', 0):.4f} "
                f"({artifact.get('ratio', 0) * 100:.2f}%)",
                f"- **Contaminated epochs (before):** {artifact.get('epochs_before', 0)}",
                f"- **Contaminated epochs (after):** {artifact.get('epochs_after', 0)}",
                f"- **Total epochs analyzed:** {artifact.get('total_epochs', 0)}",
                "",
            ])

        # Information retention
        retention = qc_metrics.get("information_retention", {})
        if retention:
            lines.extend([
                "### Information Retention",
                "",
                f"- **Waveform correlation:** {retention.get('waveform_correlation', 0):.4f}",
                f"- **Variance retention:** {retention.get('variance_retention', 0):.4f}",
                f"- **Spectral correlation:** {retention.get('spectral_correlation', 0):.4f}",
                "",
            ])

        # Channel consistency
        consistency = qc_metrics.get("channel_consistency", {})
        if consistency:
            lines.extend([
                "### Channel Consistency",
                "",
                f"- **Before:** {consistency.get('before', 0):.4f}",
                f"- **After:** {consistency.get('after', 0):.4f}",
                f"- **Improvement:** {consistency.get('improvement', 0):.4f}",
                "",
            ])

        # Overall
        overall = qc_metrics.get("overall", {})
        if overall:
            lines.extend([
                "### Overall Assessment",
                "",
                f"- **Score:** {overall.get('score', 0):.3f} / 1.000",
                f"- **Grade:** {overall.get('grade', 'N/A')}",
            ])
            warnings = overall.get("warnings", [])
            if warnings:
                lines.append(f"- **Warnings:** {len(warnings)}")
                for w in warnings:
                    lines.append(f"  - {w}")
            lines.append("")

    # Findings from NL feedback
    if qc_feedback and qc_feedback.get("findings"):
        lines.extend([
            "## Detailed Findings",
            "",
        ])
        for finding in qc_feedback["findings"]:
            severity = finding.get("severity", "info")
            icon = {"info": "ℹ️", "warning": "⚠️", "error": "❌"}.get(severity, "•")
            lines.append(f"### {icon} {finding.get('title', 'Finding')}")
            lines.append("")
            lines.append(finding.get("explanation", ""))
            options = finding.get("options", [])
            if options:
                lines.append("")
                lines.append("**Remediation options:**")
                for opt in options:
                    lines.append(f"- ({opt.get('id', '?')}) **{opt.get('label', '')}** — {opt.get('description', '')}")
            lines.append("")

    # Recommendation
    if qc_feedback and qc_feedback.get("recommended_action"):
        lines.extend([
            "## Recommendation",
            "",
            qc_feedback["recommended_action"],
            "",
        ])

    # Footer
    lines.extend([
        "---",
        "",
        f"_Report generated by EasyBCI-Data Agent | {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_",
    ])

    return lines


def _shape_str(shape: Optional[List[int]]) -> str:
    if not shape:
        return "—"
    return " × ".join(str(s) for s in shape)


def _fmt_val(val: Any) -> str:
    if val is None:
        return "—"
    if isinstance(val, float):
        return f"{val:.4g}"
    if isinstance(val, (list, tuple)):
        if len(val) <= 4:
            return str(val)
        return f"[{len(val)} items]"
    return str(val)
