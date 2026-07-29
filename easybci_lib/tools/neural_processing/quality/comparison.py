"""A/B pipeline comparison — run two pipeline variants on the same data.

Computes QC metrics for both, generates a comparative report including
quantitative differences, and recommends which pipeline is better.
"""

import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PipelineComparison:
    """Result of comparing two pipeline variants."""
    pipeline_a: List[str]
    pipeline_b: List[str]
    metrics_a: Dict[str, Any] = field(default_factory=dict)
    metrics_b: Dict[str, Any] = field(default_factory=dict)
    differences: Dict[str, Any] = field(default_factory=dict)
    recommendation: str = ""
    winner: str = ""  # "A", "B", or "tie"
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pipeline_a": self.pipeline_a,
            "pipeline_b": self.pipeline_b,
            "metrics_a": self.metrics_a,
            "metrics_b": self.metrics_b,
            "differences": self.differences,
            "recommendation": self.recommendation,
            "winner": self.winner,
            "reasons": self.reasons,
        }

    def to_text(self) -> str:
        """Render as readable comparison report."""
        lines = ["## Pipeline A/B Comparison Report\n"]
        lines.append(f"**Pipeline A:** {' → '.join(self.pipeline_a)}")
        lines.append(f"**Pipeline B:** {' → '.join(self.pipeline_b)}")
        lines.append("")

        lines.append("### Quantitative Differences\n")
        lines.append("| Metric | Pipeline A | Pipeline B | Δ (B−A) |")
        lines.append("|--------|-----------|-----------|---------|")

        for metric, diff_info in self.differences.items():
            val_a = diff_info.get("a", "—")
            val_b = diff_info.get("b", "—")
            delta = diff_info.get("delta", "—")
            better = diff_info.get("better", "")
            marker = " ✓" if better else ""
            lines.append(f"| {metric} | {val_a} | {val_b}{marker} | {delta} |")

        lines.append("")
        lines.append(f"### Recommendation\n")
        lines.append(f"**Winner: Pipeline {self.winner}**")
        lines.append(self.recommendation)
        if self.reasons:
            lines.append("\n**Reasons:**")
            for r in self.reasons:
                lines.append(f"- {r}")

        return "\n".join(lines)


def compare_pipelines(
    data_dict: Dict[str, Any],
    pipeline_a: List[str],
    pipeline_b: List[str],
) -> PipelineComparison:
    """Run two pipeline variants on the same data and compare results.

    Parameters
    ----------
    data_dict : dict
        Loaded neural data (from load_neural).
    pipeline_a : list of str
        First pipeline variant steps.
    pipeline_b : list of str
        Second pipeline variant steps.

    Returns
    -------
    PipelineComparison with metrics, differences, and recommendation.
    """
    from easybci_lib.tools.neural_processing.preprocess.pipeline import preprocess
    from easybci_lib.tools.neural_processing.quality.metrics import compute_qc_metrics

    before_data = data_dict["data"].copy()
    frequency = data_dict["frequency"]
    channels = list(data_dict.get("channels", []))

    # Run pipeline A
    dict_a = copy.deepcopy(data_dict)
    result_a = preprocess(dict_a, steps=pipeline_a)
    after_a = result_a["data"]
    freq_a = result_a["frequency"]

    # Run pipeline B
    dict_b = copy.deepcopy(data_dict)
    result_b = preprocess(dict_b, steps=pipeline_b)
    after_b = result_b["data"]
    freq_b = result_b["frequency"]

    # Compute QC metrics for both
    metrics_a = compute_qc_metrics(
        before_data, after_a, frequency, freq_a,
        channels_before=channels,
        channels_after=result_a.get("channels", channels),
    )
    metrics_b = compute_qc_metrics(
        before_data, after_b, frequency, freq_b,
        channels_before=channels,
        channels_after=result_b.get("channels", channels),
    )

    dict_a_metrics = metrics_a.to_dict()
    dict_b_metrics = metrics_b.to_dict()

    # Compute differences
    differences = _compute_differences(dict_a_metrics, dict_b_metrics)

    # Determine winner
    winner, reasons, recommendation = _determine_winner(
        dict_a_metrics, dict_b_metrics, differences, pipeline_a, pipeline_b
    )

    return PipelineComparison(
        pipeline_a=pipeline_a,
        pipeline_b=pipeline_b,
        metrics_a=dict_a_metrics,
        metrics_b=dict_b_metrics,
        differences=differences,
        recommendation=recommendation,
        winner=winner,
        reasons=reasons,
    )


def _compute_differences(
    metrics_a: Dict[str, Any], metrics_b: Dict[str, Any]
) -> Dict[str, Any]:
    """Compute per-metric differences between the two pipelines."""
    diffs = {}

    # Overall score
    score_a = metrics_a.get("overall", {}).get("score", 0)
    score_b = metrics_b.get("overall", {}).get("score", 0)
    diffs["Overall Score"] = {
        "a": f"{score_a:.3f}",
        "b": f"{score_b:.3f}",
        "delta": f"{score_b - score_a:+.3f}",
        "better": "B" if score_b > score_a else ("A" if score_a > score_b else ""),
    }

    # Grade
    grade_a = metrics_a.get("overall", {}).get("grade", "?")
    grade_b = metrics_b.get("overall", {}).get("grade", "?")
    diffs["Grade"] = {
        "a": grade_a,
        "b": grade_b,
        "delta": f"{grade_a}→{grade_b}",
        "better": "",
    }

    # Artifact residual
    ar_a = metrics_a.get("artifact_residual", {}).get("ratio", 0)
    ar_b = metrics_b.get("artifact_residual", {}).get("ratio", 0)
    diffs["Artifact Residual"] = {
        "a": f"{ar_a*100:.1f}%",
        "b": f"{ar_b*100:.1f}%",
        "delta": f"{(ar_b - ar_a)*100:+.1f}%",
        "better": "B" if ar_b < ar_a else ("A" if ar_a < ar_b else ""),
    }

    # Waveform retention
    wr_a = metrics_a.get("information_retention", {}).get("waveform_correlation", 0)
    wr_b = metrics_b.get("information_retention", {}).get("waveform_correlation", 0)
    diffs["Waveform Retention"] = {
        "a": f"{wr_a:.3f}",
        "b": f"{wr_b:.3f}",
        "delta": f"{wr_b - wr_a:+.3f}",
        "better": "B" if wr_b > wr_a else ("A" if wr_a > wr_b else ""),
    }

    # Channel consistency
    cc_a = metrics_a.get("channel_consistency", {}).get("after", 0)
    cc_b = metrics_b.get("channel_consistency", {}).get("after", 0)
    diffs["Channel Consistency"] = {
        "a": f"{cc_a:.3f}",
        "b": f"{cc_b:.3f}",
        "delta": f"{cc_b - cc_a:+.3f}",
        "better": "B" if cc_b > cc_a else ("A" if cc_a > cc_b else ""),
    }

    # SNR improvement (average across bands)
    snr_a = metrics_a.get("snr", {}).get("improvement_db", {})
    snr_b = metrics_b.get("snr", {}).get("improvement_db", {})
    if snr_a and snr_b:
        avg_a = np.mean(list(snr_a.values()))
        avg_b = np.mean(list(snr_b.values()))
        diffs["Avg SNR Improvement"] = {
            "a": f"{avg_a:+.1f} dB",
            "b": f"{avg_b:+.1f} dB",
            "delta": f"{avg_b - avg_a:+.1f} dB",
            "better": "B" if avg_b > avg_a else ("A" if avg_a > avg_b else ""),
        }

    return diffs


def _determine_winner(
    metrics_a: Dict[str, Any],
    metrics_b: Dict[str, Any],
    differences: Dict[str, Any],
    pipeline_a: List[str],
    pipeline_b: List[str],
) -> tuple:
    """Determine which pipeline is better overall."""
    score_a = metrics_a.get("overall", {}).get("score", 0)
    score_b = metrics_b.get("overall", {}).get("score", 0)

    # Count how many metrics each pipeline wins
    a_wins = sum(1 for d in differences.values() if d.get("better") == "A")
    b_wins = sum(1 for d in differences.values() if d.get("better") == "B")

    reasons = []

    # Significant score difference?
    score_diff = abs(score_b - score_a)
    if score_diff < 0.02:
        # Effectively tied on overall score
        if a_wins > b_wins:
            winner = "A"
            reasons.append(f"Won {a_wins}/{a_wins + b_wins} individual metrics")
        elif b_wins > a_wins:
            winner = "B"
            reasons.append(f"Won {b_wins}/{a_wins + b_wins} individual metrics")
        else:
            winner = "tie"
            reasons.append("Both pipelines produce similar quality")
    else:
        winner = "B" if score_b > score_a else "A"
        reasons.append(f"Overall score: {max(score_a, score_b):.3f} vs {min(score_a, score_b):.3f}")

    # Add specific metric reasons
    ar_a = metrics_a.get("artifact_residual", {}).get("ratio", 0)
    ar_b = metrics_b.get("artifact_residual", {}).get("ratio", 0)
    if abs(ar_a - ar_b) > 0.05:
        better = "A" if ar_a < ar_b else "B"
        reasons.append(f"Pipeline {better} has lower artifact contamination")

    wr_a = metrics_a.get("information_retention", {}).get("waveform_correlation", 0)
    wr_b = metrics_b.get("information_retention", {}).get("waveform_correlation", 0)
    if abs(wr_a - wr_b) > 0.05:
        better = "A" if wr_a > wr_b else "B"
        reasons.append(f"Pipeline {better} preserves more signal waveform")

    # Build recommendation
    if winner == "tie":
        recommendation = (
            f"Both pipelines produce comparable results. "
            f"Pipeline A ({len(pipeline_a)} steps) and Pipeline B ({len(pipeline_b)} steps) "
            f"are effectively equivalent. Choose based on processing speed or simplicity."
        )
    else:
        winner_steps = pipeline_a if winner == "A" else pipeline_b
        loser_steps = pipeline_b if winner == "A" else pipeline_a
        recommendation = (
            f"Pipeline {winner} ({' → '.join(winner_steps)}) is recommended. "
            f"It outperforms the alternative on {max(a_wins, b_wins)} of "
            f"{a_wins + b_wins} quality metrics."
        )

    return winner, reasons, recommendation
