"""Natural language QC feedback — human-readable quality reports.

Transforms technical QC metrics and validator issues into plain-language
explanations with concrete remediation options. Designed for researchers
who understand neuroscience but may not be signal processing experts.

Output format: structured report with sections for each finding and
numbered action options the user can select.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class RemediationOption:
    """A concrete action the user can choose."""
    id: int
    label: str
    description: str
    steps_to_add: List[str] = field(default_factory=list)
    steps_to_remove: List[str] = field(default_factory=list)
    steps_to_modify: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "id": self.id,
            "label": self.label,
            "description": self.description,
        }
        if self.steps_to_add:
            d["steps_to_add"] = self.steps_to_add
        if self.steps_to_remove:
            d["steps_to_remove"] = self.steps_to_remove
        if self.steps_to_modify:
            d["steps_to_modify"] = self.steps_to_modify
        return d


@dataclass
class QCFinding:
    """A single finding from QC analysis."""
    severity: str  # "info", "warning", "error"
    title: str
    explanation: str
    options: List[RemediationOption] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity,
            "title": self.title,
            "explanation": self.explanation,
            "options": [o.to_dict() for o in self.options],
        }


@dataclass
class NLQCReport:
    """Natural language QC report with findings and recommended actions."""
    summary: str
    grade: str
    findings: List[QCFinding] = field(default_factory=list)
    recommended_action: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary,
            "grade": self.grade,
            "findings": [f.to_dict() for f in self.findings],
            "recommended_action": self.recommended_action,
        }

    def to_text(self) -> str:
        """Render as plain text for display."""
        lines = [f"## QC Report (Grade: {self.grade})\n"]
        lines.append(self.summary)
        lines.append("")

        for i, finding in enumerate(self.findings, 1):
            icon = {"info": "ℹ️", "warning": "⚠️", "error": "❌"}.get(finding.severity, "•")
            lines.append(f"### {icon} {finding.title}")
            lines.append(finding.explanation)
            if finding.options:
                lines.append("\n**Options:**")
                for opt in finding.options:
                    lines.append(f"  ({opt.id}) {opt.label} — {opt.description}")
            lines.append("")

        if self.recommended_action:
            lines.append(f"**Recommended:** {self.recommended_action}")

        return "\n".join(lines)


def generate_nl_feedback(
    qc_metrics: Optional[Dict[str, Any]] = None,
    validator_result: Optional[Dict[str, Any]] = None,
    steps_applied: Optional[List[str]] = None,
    channels: Optional[List[str]] = None,
    data_profile: Optional[Dict[str, Any]] = None,
) -> NLQCReport:
    """Generate a natural language QC report from metrics and validation results.

    Parameters
    ----------
    qc_metrics : dict, optional
        Output of QCMetrics.to_dict() (enhanced metrics with SNR, retention, etc.)
    validator_result : dict, optional
        Output of validate_signal() (issues list + stats)
    steps_applied : list of str, optional
        Pipeline steps that were applied.
    channels : list of str, optional
        Channel names (for specific channel references in feedback).
    data_profile : dict, optional
        Output of DataProfile.to_dict() (pre-processing characteristics).

    Returns
    -------
    NLQCReport with human-readable findings and remediation options.
    """
    findings: List[QCFinding] = []
    option_counter = [1]

    if validator_result:
        _analyze_validator_issues(validator_result, findings, option_counter, channels)

    if qc_metrics:
        _analyze_snr(qc_metrics, findings, option_counter, steps_applied)
        _analyze_artifacts(qc_metrics, findings, option_counter, steps_applied)
        _analyze_retention(qc_metrics, findings, option_counter, steps_applied)
        _analyze_consistency(qc_metrics, findings, option_counter)

    grade = qc_metrics.get("overall", {}).get("grade", "?") if qc_metrics else "?"
    summary = _build_summary(findings, grade, qc_metrics)
    recommended = _build_recommendation(findings)

    return NLQCReport(
        summary=summary,
        grade=grade,
        findings=findings,
        recommended_action=recommended,
    )


def _next_id(counter: List[int]) -> int:
    val = counter[0]
    counter[0] += 1
    return val


def _analyze_validator_issues(
    result: Dict[str, Any],
    findings: List[QCFinding],
    counter: List[int],
    channels: Optional[List[str]],
) -> None:
    """Convert validator issues to natural language findings."""
    issues = result.get("issues", [])
    stats = result.get("stats", {})

    for issue in issues:
        check = issue.get("check", "")
        severity = issue.get("severity", "warning")

        if check == "nan":
            nan_ch = stats.get("nan_channels", 0)
            findings.append(QCFinding(
                severity=severity,
                title="Missing values (NaN) detected",
                explanation=(
                    f"Found NaN values in {nan_ch} channel(s). This typically happens with "
                    f"recording dropouts or corrupted segments. The affected data points cannot "
                    f"be used for analysis without treatment."
                ),
                options=[
                    RemediationOption(
                        id=_next_id(counter),
                        label="Interpolate missing values",
                        description="Replace NaN with interpolated values from neighboring time points",
                        steps_to_add=["fill_nan"],
                    ),
                    RemediationOption(
                        id=_next_id(counter),
                        label="Remove affected channels",
                        description=f"Drop the {nan_ch} channels containing NaN values entirely",
                        steps_to_add=["drop_bads"],
                    ),
                ],
            ))

        elif check == "flat":
            flat_ch = stats.get("flat_channels", 0)
            findings.append(QCFinding(
                severity=severity,
                title=f"Flat channels detected ({flat_ch})",
                explanation=(
                    f"{flat_ch} channel(s) show zero variance (flatlined signal). These channels "
                    f"are providing no useful information — likely a hardware connection issue "
                    f"during recording."
                ),
                options=[
                    RemediationOption(
                        id=_next_id(counter),
                        label="Remove flat channels",
                        description="Drop flatlined channels from the data",
                        steps_to_add=["drop_bads"],
                    ),
                    RemediationOption(
                        id=_next_id(counter),
                        label="Interpolate from neighbors",
                        description="Replace flat channels with spatially interpolated data from nearby electrodes",
                        steps_to_add=["interpolate_bads"],
                    ),
                ],
            ))

        elif check == "amplitude":
            extreme_ch = stats.get("extreme_channels", 0)
            max_amp = stats.get("max_amplitude", 0)
            findings.append(QCFinding(
                severity=severity,
                title=f"Extreme amplitude values in {extreme_ch} channel(s)",
                explanation=(
                    f"Peak amplitude reaches {max_amp:.0f}, which is abnormally high. "
                    f"This could indicate: (1) wrong unit scale (e.g., volts instead of microvolts), "
                    f"(2) movement artifacts, or (3) electrode pop-offs during recording."
                ),
                options=[
                    RemediationOption(
                        id=_next_id(counter),
                        label="Clip extreme values",
                        description="Limit amplitude to ±500 µV (standard EEG range)",
                        steps_to_add=["clip:500"],
                    ),
                    RemediationOption(
                        id=_next_id(counter),
                        label="Remove bad channels + ICA",
                        description="Remove worst channels, then use ICA to separate artifact components",
                        steps_to_add=["drop_bads", "ica"],
                    ),
                    RemediationOption(
                        id=_next_id(counter),
                        label="Apply robust scaling",
                        description="Use median/MAD normalization which is resistant to outliers",
                        steps_to_modify={"scale": "scale:robust"},
                    ),
                ],
            ))

        elif check == "variance":
            high_var = stats.get("high_variance_channels", 0)
            findings.append(QCFinding(
                severity=severity,
                title=f"Abnormal variance in {high_var} channel(s)",
                explanation=(
                    f"{high_var} channel(s) have variance more than 100x the typical level. "
                    f"These channels are likely contaminated by muscle artifacts (EMG) or "
                    f"environmental noise, and will dominate any downstream analysis."
                ),
                options=[
                    RemediationOption(
                        id=_next_id(counter),
                        label="Remove high-variance channels",
                        description="Drop channels with extreme variance automatically",
                        steps_to_add=["drop_bads"],
                    ),
                    RemediationOption(
                        id=_next_id(counter),
                        label="Apply ICA artifact removal",
                        description="Use Independent Component Analysis to separate and remove artifact sources",
                        steps_to_add=["ica"],
                    ),
                ],
            ))


def _analyze_snr(
    metrics: Dict[str, Any],
    findings: List[QCFinding],
    counter: List[int],
    steps_applied: Optional[List[str]],
) -> None:
    """Report on SNR changes from preprocessing."""
    snr = metrics.get("snr", {})
    improvement = snr.get("improvement_db", {})
    if not improvement:
        return

    degraded_bands = {k: v for k, v in improvement.items() if v < -3.0}
    improved_bands = {k: v for k, v in improvement.items() if v > 2.0}

    if degraded_bands:
        bands_str = ", ".join(f"{k} ({v:+.1f} dB)" for k, v in degraded_bands.items())
        findings.append(QCFinding(
            severity="warning",
            title="SNR degraded in some frequency bands",
            explanation=(
                f"Signal-to-noise ratio decreased in: {bands_str}. "
                f"This suggests the filter settings may be removing useful signal content "
                f"along with the noise."
            ),
            options=[
                RemediationOption(
                    id=_next_id(counter),
                    label="Widen bandpass filter",
                    description="Use a less aggressive frequency range to preserve more signal",
                    steps_to_modify={"bandpass": "bandpass:0.1,45"},
                ),
                RemediationOption(
                    id=_next_id(counter),
                    label="Skip bandpass, use notch only",
                    description="Remove only power line noise without bandpass filtering",
                    steps_to_remove=["bandpass"],
                ),
            ],
        ))
    elif improved_bands and not degraded_bands:
        avg_improvement = sum(improvement.values()) / len(improvement)
        if avg_improvement > 3:
            findings.append(QCFinding(
                severity="info",
                title=f"Good SNR improvement (+{avg_improvement:.1f} dB average)",
                explanation=(
                    f"Preprocessing improved signal quality across frequency bands. "
                    f"Best improvement in: {max(improvement, key=improvement.get)} band."
                ),
                options=[],
            ))


def _analyze_artifacts(
    metrics: Dict[str, Any],
    findings: List[QCFinding],
    counter: List[int],
    steps_applied: Optional[List[str]],
) -> None:
    """Report on residual artifact contamination."""
    artifact = metrics.get("artifact_residual", {})
    ratio = artifact.get("ratio", 0)
    before_count = artifact.get("epochs_before", 0)
    after_count = artifact.get("epochs_after", 0)

    if ratio > 0.3:
        findings.append(QCFinding(
            severity="error",
            title=f"High artifact contamination ({ratio*100:.0f}% of epochs)",
            explanation=(
                f"After preprocessing, {after_count} out of {artifact.get('total_epochs', 0)} "
                f"time windows still contain artifact-level amplitudes. "
                f"This level of contamination will likely impact analysis quality."
            ),
            options=[
                RemediationOption(
                    id=_next_id(counter),
                    label="Add ICA artifact removal",
                    description="Use ICA to identify and remove artifact components (eye blinks, muscle)",
                    steps_to_add=["ica"],
                ),
                RemediationOption(
                    id=_next_id(counter),
                    label="Apply aggressive clipping",
                    description="Clip amplitudes to ±200 µV to reduce artifact impact",
                    steps_to_add=["clip:200"],
                ),
                RemediationOption(
                    id=_next_id(counter),
                    label="Reject bad epochs",
                    description="Mark and exclude contaminated time windows from analysis",
                    steps_to_add=["reject_epochs:amplitude"],
                ),
            ],
        ))
    elif ratio > 0.1:
        findings.append(QCFinding(
            severity="warning",
            title=f"Moderate artifact residual ({ratio*100:.0f}% of epochs)",
            explanation=(
                f"Some artifact contamination remains ({after_count} windows affected). "
                f"This is moderate — may be acceptable depending on your analysis goals."
            ),
            options=[
                RemediationOption(
                    id=_next_id(counter),
                    label="Add ICA cleanup",
                    description="Apply ICA for finer artifact separation",
                    steps_to_add=["ica"],
                ),
            ],
        ))
    elif before_count > 0 and after_count == 0:
        findings.append(QCFinding(
            severity="info",
            title="Artifacts successfully removed",
            explanation=(
                f"All {before_count} artifact-contaminated windows from the raw data "
                f"have been cleaned by preprocessing."
            ),
            options=[],
        ))


def _analyze_retention(
    metrics: Dict[str, Any],
    findings: List[QCFinding],
    counter: List[int],
    steps_applied: Optional[List[str]],
) -> None:
    """Report on information retention (over-filtering detection)."""
    retention = metrics.get("information_retention", {})
    waveform_corr = retention.get("waveform_correlation", 1.0)
    variance_ret = retention.get("variance_retention", 1.0)

    if waveform_corr < 0.4:
        findings.append(QCFinding(
            severity="error",
            title="Severe signal distortion detected",
            explanation=(
                f"Waveform correlation between raw and processed data is only "
                f"{waveform_corr:.2f} (ideal: >0.7). The preprocessing pipeline "
                f"may be too aggressive — important signal features (ERPs, oscillations) "
                f"might have been removed along with the noise."
            ),
            options=[
                RemediationOption(
                    id=_next_id(counter),
                    label="Use gentler filter settings",
                    description="Widen the bandpass and reduce filter order",
                    steps_to_modify={"bandpass": "bandpass:0.5,45"},
                ),
                RemediationOption(
                    id=_next_id(counter),
                    label="Remove ICA step",
                    description="ICA sometimes removes signal components — skip it",
                    steps_to_remove=["ica"],
                ),
            ],
        ))
    elif waveform_corr < 0.6:
        findings.append(QCFinding(
            severity="warning",
            title=f"Moderate signal alteration (correlation: {waveform_corr:.2f})",
            explanation=(
                f"Preprocessing has substantially altered the signal shape. "
                f"If you're studying time-locked responses (ERPs), verify that "
                f"peak latencies and amplitudes are preserved."
            ),
            options=[
                RemediationOption(
                    id=_next_id(counter),
                    label="Widen bandpass filter",
                    description="Less aggressive filtering to preserve more of the original signal",
                    steps_to_modify={"bandpass": "bandpass:0.3,40"},
                ),
            ],
        ))

    if variance_ret < 0.3:
        findings.append(QCFinding(
            severity="warning",
            title="Large variance reduction",
            explanation=(
                f"Signal power dropped to {variance_ret*100:.0f}% of original. "
                f"While some reduction is expected from noise removal, this much loss "
                f"suggests potentially over-aggressive preprocessing."
            ),
            options=[],
        ))


def _analyze_consistency(
    metrics: Dict[str, Any],
    findings: List[QCFinding],
    counter: List[int],
) -> None:
    """Report on cross-channel consistency changes."""
    consistency = metrics.get("channel_consistency", {})
    before = consistency.get("before", 0)
    after = consistency.get("after", 0)
    improvement = consistency.get("improvement", 0)

    if improvement > 0.2:
        findings.append(QCFinding(
            severity="info",
            title="Channel consistency improved significantly",
            explanation=(
                f"Cross-channel variance uniformity improved from {before:.2f} to {after:.2f}. "
                f"This indicates bad channels and artifacts were effectively addressed."
            ),
            options=[],
        ))
    elif improvement < -0.1:
        findings.append(QCFinding(
            severity="warning",
            title="Channel consistency decreased",
            explanation=(
                f"Channels became less uniform after processing (consistency: {before:.2f} → {after:.2f}). "
                f"This could indicate that processing affected some channels differently than others."
            ),
            options=[
                RemediationOption(
                    id=_next_id(counter),
                    label="Add common average reference",
                    description="CAR helps equalize channels by removing shared noise",
                    steps_to_add=["car"],
                ),
            ],
        ))


def _build_summary(
    findings: List[QCFinding], grade: str, metrics: Optional[Dict[str, Any]]
) -> str:
    """Build a one-paragraph summary of the QC results."""
    n_errors = sum(1 for f in findings if f.severity == "error")
    n_warnings = sum(1 for f in findings if f.severity == "warning")
    n_info = sum(1 for f in findings if f.severity == "info")

    if n_errors == 0 and n_warnings == 0:
        return (
            f"Preprocessing completed successfully (Grade {grade}). "
            f"All quality checks passed with no issues detected."
        )

    parts = []
    if n_errors > 0:
        parts.append(f"{n_errors} critical issue{'s' if n_errors > 1 else ''}")
    if n_warnings > 0:
        parts.append(f"{n_warnings} warning{'s' if n_warnings > 1 else ''}")

    issues_str = " and ".join(parts)

    score = metrics.get("overall", {}).get("score", 0) if metrics else 0
    return (
        f"Preprocessing completed with {issues_str} (Grade {grade}, score {score:.2f}). "
        f"Review the findings below and choose remediation options if needed."
    )


def _build_recommendation(findings: List[QCFinding]) -> str:
    """Generate a single recommended course of action."""
    errors = [f for f in findings if f.severity == "error"]
    warnings = [f for f in findings if f.severity == "warning"]

    if not errors and not warnings:
        return "No action needed — data quality is good. Proceed with analysis."

    if errors:
        all_adds = []
        for f in errors:
            for opt in f.options[:1]:
                all_adds.extend(opt.steps_to_add)
        if all_adds:
            return f"Add {' → '.join(all_adds)} to address critical issues, then re-run preprocessing."
        return "Address the critical issues above before proceeding with analysis."

    return "Consider applying suggested fixes for the warnings to improve data quality."
