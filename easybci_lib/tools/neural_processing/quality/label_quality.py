"""Label quality assessment — per-type quality metrics for BCI labels.

Evaluates label quality based on label type (L1-L5):
- L1 (Event): ISI regularity, epoch overlap detection, class balance
- L2 (Segment): gap ratio, coverage, duration distribution
- L3 (Continuous): missing value ratio, anomalous jumps, frequency match
- L4 (Session): group balance, sample count per condition
- L5 (Hierarchical): level completeness, leaf distribution

Also provides cross-checks:
- Event-data time range consistency
- Label coverage relative to total recording
- Class distribution balance (entropy-based)
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


def assess_label_quality(
    labels: Any,
    label_type: str,
    frequency: Optional[float] = None,
    data_duration: Optional[float] = None,
    n_samples: Optional[int] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Assess quality of labels based on their type.

    Parameters
    ----------
    labels : varies by type
        - L1 (EVENT): list of dicts with "onset", "type"
        - L2 (SEGMENT): list of dicts with "start", "end", "label"
        - L3 (CONTINUOUS): ndarray (n_samples,) or (n_dims, n_samples)
        - L4 (SESSION): list of str (labels per file/session)
        - L5 (HIERARCHICAL): HierarchicalLabels object or nested dict
    label_type : str
        One of: "event", "segment", "continuous", "session", "hierarchical"
    frequency : float, optional
        Sampling rate of the data (needed for timing checks).
    data_duration : float, optional
        Total recording duration in seconds.
    n_samples : int, optional
        Total number of data samples.

    Returns
    -------
    Dict with:
        quality_score : float (0-1) — overall quality
        issues : list[dict] — detected problems with severity
        metrics : dict — type-specific quality metrics
        recommendations : list[str] — suggested actions
    """
    label_type_normalized = label_type.lower().strip()
    # Handle "L1", "L2", etc. by stripping the leading "l"
    if len(label_type_normalized) == 2 and label_type_normalized.startswith("l") and label_type_normalized[1].isdigit():
        label_type_normalized = label_type_normalized[1]

    dispatch = {
        "event": _assess_event_labels,
        "1": _assess_event_labels,
        "segment": _assess_segment_labels,
        "2": _assess_segment_labels,
        "continuous": _assess_continuous_labels,
        "3": _assess_continuous_labels,
        "session": _assess_session_labels,
        "4": _assess_session_labels,
        "hierarchical": _assess_hierarchical_labels,
        "5": _assess_hierarchical_labels,
    }

    handler = dispatch.get(label_type_normalized)
    if handler is None:
        return {
            "quality_score": 0.5,
            "issues": [{"severity": "warning", "message": f"Unknown label type: {label_type}"}],
            "metrics": {},
            "recommendations": ["Verify label type classification"],
        }

    return handler(labels, frequency=frequency, data_duration=data_duration, n_samples=n_samples, **kwargs)


def _assess_event_labels(
    events: List[Dict[str, Any]],
    frequency: Optional[float] = None,
    data_duration: Optional[float] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Assess L1 event label quality."""
    issues: List[Dict[str, Any]] = []
    recommendations: List[str] = []
    metrics: Dict[str, Any] = {}

    if not events:
        return {
            "quality_score": 0.0,
            "issues": [{"severity": "error", "message": "No events found"}],
            "metrics": {"n_events": 0},
            "recommendations": ["Check event source file or extraction method"],
        }

    # Extract onsets and types
    onsets = []
    types = []
    for ev in events:
        onset = ev.get("onset", ev.get("start"))
        if onset is not None:
            onsets.append(float(onset))
        etype = ev.get("type", ev.get("label", ev.get("trial_type", "unknown")))
        types.append(str(etype))

    n_events = len(onsets)
    metrics["n_events"] = n_events
    unique_types = list(set(types))
    metrics["n_types"] = len(unique_types)
    metrics["type_distribution"] = {t: types.count(t) for t in unique_types}

    score = 1.0

    # ISI analysis
    if len(onsets) >= 2:
        onsets_sorted = sorted(onsets)
        isis = np.diff(onsets_sorted)
        metrics["isi_mean_s"] = float(np.mean(isis))
        metrics["isi_std_s"] = float(np.std(isis))
        metrics["isi_min_s"] = float(np.min(isis))
        metrics["isi_max_s"] = float(np.max(isis))
        metrics["isi_cv"] = float(np.std(isis) / max(np.mean(isis), 1e-10))

        # Zero or negative intervals
        n_bad_isi = int(np.sum(isis <= 0))
        if n_bad_isi > 0:
            issues.append({
                "severity": "error",
                "message": f"{n_bad_isi} zero/negative ISI detected (duplicate or misordered events)",
            })
            score -= 0.3
            recommendations.append("Remove duplicate events or fix event ordering")

        # Very short intervals (< 100ms)
        n_very_short = int(np.sum((isis > 0) & (isis < 0.1)))
        if n_very_short > 0:
            issues.append({
                "severity": "warning",
                "message": f"{n_very_short} intervals shorter than 100ms (possible annotation errors)",
            })
            score -= 0.1

        # ISI regularity (CV > 1.0 means high variability)
        cv = float(np.std(isis) / max(np.mean(isis), 1e-10))
        if cv > 2.0:
            issues.append({
                "severity": "info",
                "message": f"High ISI variability (CV={cv:.2f}), may indicate mixed event types",
            })

    # Epoch overlap check
    if len(onsets) >= 2 and "epoch_duration" in kwargs:
        epoch_dur = kwargs["epoch_duration"]
        onsets_sorted = sorted(onsets)
        overlaps = sum(1 for i in range(len(onsets_sorted) - 1)
                       if onsets_sorted[i + 1] - onsets_sorted[i] < epoch_dur)
        if overlaps > 0:
            metrics["n_epoch_overlaps"] = overlaps
            issues.append({
                "severity": "warning",
                "message": f"{overlaps} epochs would overlap with duration={epoch_dur}s",
            })
            score -= 0.1

    # Time range check
    if data_duration and onsets:
        max_onset = max(onsets)
        min_onset = min(onsets)
        if max_onset > data_duration * 1.01:
            issues.append({
                "severity": "error",
                "message": f"Events extend beyond data ({max_onset:.2f}s > {data_duration:.2f}s). Time unit mismatch?",
            })
            score -= 0.3
            recommendations.append("Check if events use milliseconds while data uses seconds")
        if min_onset < -0.5:
            issues.append({
                "severity": "warning",
                "message": f"Events start before data (min onset={min_onset:.2f}s)",
            })
            score -= 0.1

        # Coverage
        if data_duration > 0:
            event_span = max_onset - max(0, min_onset)
            metrics["time_coverage"] = event_span / data_duration

    # Class balance
    if len(unique_types) > 1:
        counts = [types.count(t) for t in unique_types]
        balance = min(counts) / max(max(counts), 1)
        metrics["class_balance"] = balance
        if balance < 0.5:
            issues.append({
                "severity": "info",
                "message": f"Imbalanced classes (min/max ratio={balance:.2f})",
            })

    return {
        "quality_score": max(0.0, min(1.0, score)),
        "issues": issues,
        "metrics": metrics,
        "recommendations": recommendations,
    }


def _assess_segment_labels(
    segments: List[Dict[str, Any]],
    data_duration: Optional[float] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Assess L2 segment label quality."""
    issues: List[Dict[str, Any]] = []
    recommendations: List[str] = []
    metrics: Dict[str, Any] = {}

    if not segments:
        return {
            "quality_score": 0.0,
            "issues": [{"severity": "error", "message": "No segments found"}],
            "metrics": {"n_segments": 0},
            "recommendations": ["Check segment definition file"],
        }

    n_segments = len(segments)
    metrics["n_segments"] = n_segments

    # Extract timing and labels
    starts = []
    ends = []
    labels = []
    durations = []

    for seg in segments:
        s = float(seg.get("start", 0))
        e = float(seg.get("end", 0))
        starts.append(s)
        ends.append(e)
        durations.append(e - s)
        labels.append(str(seg.get("label", "unknown")))

    metrics["total_labeled_duration_s"] = sum(durations)
    metrics["mean_segment_duration_s"] = float(np.mean(durations))
    metrics["std_segment_duration_s"] = float(np.std(durations))

    score = 1.0

    # Invalid segments (end <= start)
    n_invalid = sum(1 for d in durations if d <= 0)
    if n_invalid > 0:
        issues.append({
            "severity": "error",
            "message": f"{n_invalid} segments with end <= start",
        })
        score -= 0.3

    # Gap analysis
    sorted_indices = sorted(range(n_segments), key=lambda i: starts[i])
    gaps = []
    overlaps = []
    for i in range(len(sorted_indices) - 1):
        curr_end = ends[sorted_indices[i]]
        next_start = starts[sorted_indices[i + 1]]
        gap = next_start - curr_end
        if gap > 0.01:
            gaps.append(gap)
        elif gap < -0.01:
            overlaps.append(abs(gap))

    total_gap = sum(gaps)
    metrics["n_gaps"] = len(gaps)
    metrics["total_gap_duration_s"] = total_gap
    metrics["n_overlaps"] = len(overlaps)

    if data_duration:
        gap_ratio = total_gap / data_duration
        coverage = sum(max(0, d) for d in durations) / data_duration
        metrics["gap_ratio"] = gap_ratio
        metrics["coverage"] = min(1.0, coverage)

        if coverage < 0.5:
            issues.append({
                "severity": "warning",
                "message": f"Low label coverage ({coverage:.1%} of data labeled)",
            })
            score -= 0.15
            recommendations.append("Consider gap segments as 'unlabeled' or 'rest'")

    if overlaps:
        issues.append({
            "severity": "warning",
            "message": f"{len(overlaps)} overlapping segments detected",
        })
        score -= 0.15
        recommendations.append("Resolve overlapping segments (may cause duplicate data)")

    # Label distribution
    unique_labels = list(set(labels))
    metrics["n_classes"] = len(unique_labels)
    metrics["label_distribution"] = {lbl: labels.count(lbl) for lbl in unique_labels}

    if len(unique_labels) > 1:
        counts = [labels.count(lbl) for lbl in unique_labels]
        balance = min(counts) / max(max(counts), 1)
        metrics["class_balance"] = balance

    return {
        "quality_score": max(0.0, min(1.0, score)),
        "issues": issues,
        "metrics": metrics,
        "recommendations": recommendations,
    }


def _assess_continuous_labels(
    labels: np.ndarray,
    frequency: Optional[float] = None,
    n_samples: Optional[int] = None,
    data_duration: Optional[float] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Assess L3 continuous label quality."""
    issues: List[Dict[str, Any]] = []
    recommendations: List[str] = []
    metrics: Dict[str, Any] = {}

    if not isinstance(labels, np.ndarray):
        labels = np.asarray(labels)

    if labels.size == 0:
        return {
            "quality_score": 0.0,
            "issues": [{"severity": "error", "message": "Empty label array"}],
            "metrics": {},
            "recommendations": ["Check label file loading"],
        }

    if labels.ndim == 1:
        n_label_samples = labels.shape[0]
        n_dims = 1
    else:
        n_dims = labels.shape[0]
        n_label_samples = labels.shape[-1]

    metrics["n_label_samples"] = n_label_samples
    metrics["n_dimensions"] = n_dims

    score = 1.0

    # Missing value analysis
    flat = labels.ravel() if labels.ndim > 1 else labels
    n_nan = int(np.sum(np.isnan(flat)))
    nan_ratio = n_nan / max(flat.size, 1)
    metrics["nan_ratio"] = nan_ratio
    metrics["n_nan_values"] = n_nan

    if nan_ratio > 0.1:
        issues.append({
            "severity": "warning",
            "message": f"High missing value ratio ({nan_ratio:.1%})",
        })
        score -= 0.2
        recommendations.append("Consider interpolation or forward-fill for NaN gaps")
    elif nan_ratio > 0.01:
        issues.append({
            "severity": "info",
            "message": f"Some missing values ({nan_ratio:.2%})",
        })

    # Frequency match check
    if n_samples is not None:
        ratio = n_label_samples / n_samples
        metrics["sample_ratio"] = ratio
        if abs(ratio - 1.0) < 0.01:
            metrics["frequency_match"] = "exact"
        elif ratio > 0.1:
            metrics["frequency_match"] = "resample_needed"
            if frequency:
                metrics["inferred_label_freq"] = frequency * ratio
        else:
            issues.append({
                "severity": "error",
                "message": f"Label length ({n_label_samples}) much shorter than data ({n_samples})",
            })
            score -= 0.3

    # Anomalous jumps detection (for 1D or per-dim)
    label_1d = flat if n_dims == 1 else labels[0] if labels.ndim > 1 else labels
    valid_mask = ~np.isnan(label_1d)
    if np.sum(valid_mask) > 10:
        valid_vals = label_1d[valid_mask]
        diffs = np.abs(np.diff(valid_vals))
        if len(diffs) > 0:
            median_diff = float(np.median(diffs))
            mad_diff = float(np.median(np.abs(diffs - median_diff)))
            threshold = median_diff + 10 * max(mad_diff, 1e-8)
            n_jumps = int(np.sum(diffs > threshold))
            metrics["n_anomalous_jumps"] = n_jumps
            if n_jumps > len(diffs) * 0.01:
                issues.append({
                    "severity": "warning",
                    "message": f"{n_jumps} anomalous jumps in label signal (possible sensor glitches)",
                })
                score -= 0.1

    # Value range
    valid_flat = flat[~np.isnan(flat)]
    if len(valid_flat) > 0:
        metrics["value_range"] = [float(np.min(valid_flat)), float(np.max(valid_flat))]
        metrics["value_mean"] = float(np.mean(valid_flat))
        metrics["value_std"] = float(np.std(valid_flat))

    return {
        "quality_score": max(0.0, min(1.0, score)),
        "issues": issues,
        "metrics": metrics,
        "recommendations": recommendations,
    }


def _assess_session_labels(
    labels: List[str],
    **kwargs: Any,
) -> Dict[str, Any]:
    """Assess L4 session-level label quality."""
    issues: List[Dict[str, Any]] = []
    recommendations: List[str] = []
    metrics: Dict[str, Any] = {}

    if not labels:
        return {
            "quality_score": 0.0,
            "issues": [{"severity": "error", "message": "No session labels"}],
            "metrics": {},
            "recommendations": ["Provide condition mapping or participants.tsv"],
        }

    unique = list(set(labels))
    counts = {lbl: labels.count(lbl) for lbl in unique}
    metrics["n_sessions"] = len(labels)
    metrics["n_conditions"] = len(unique)
    metrics["condition_counts"] = counts

    score = 1.0

    # Balance check
    if len(unique) > 1:
        count_vals = list(counts.values())
        balance = min(count_vals) / max(max(count_vals), 1)
        metrics["group_balance"] = balance

        if balance < 0.33:
            issues.append({
                "severity": "warning",
                "message": f"Highly imbalanced groups (min/max={balance:.2f})",
            })
            score -= 0.2
            recommendations.append("Consider oversampling minority condition or adjusting analysis")
    elif len(unique) == 1:
        issues.append({
            "severity": "warning",
            "message": "Only one condition — no contrast possible",
        })
        score -= 0.3
        recommendations.append("Need at least 2 conditions for classification/comparison")

    # Unknown labels
    n_unknown = labels.count("unknown") + labels.count("")
    if n_unknown > 0:
        metrics["n_unknown"] = n_unknown
        issues.append({
            "severity": "warning",
            "message": f"{n_unknown} sessions with unknown/empty labels",
        })
        score -= 0.15

    return {
        "quality_score": max(0.0, min(1.0, score)),
        "issues": issues,
        "metrics": metrics,
        "recommendations": recommendations,
    }


def _assess_hierarchical_labels(
    labels: Any,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Assess L5 hierarchical label quality."""
    issues: List[Dict[str, Any]] = []
    recommendations: List[str] = []
    metrics: Dict[str, Any] = {}

    score = 1.0

    # Handle HierarchicalLabels object
    if hasattr(labels, "levels") and hasattr(labels, "tree"):
        levels = labels.levels
        tree = labels.tree
    elif isinstance(labels, dict) and "levels" in labels:
        levels = labels["levels"]
        tree = labels.get("tree", [])
    else:
        return {
            "quality_score": 0.5,
            "issues": [{"severity": "warning", "message": "Cannot parse hierarchical structure"}],
            "metrics": {},
            "recommendations": ["Provide data as HierarchicalLabels or {levels, tree} dict"],
        }

    metrics["n_levels"] = len(levels)
    metrics["levels"] = levels
    metrics["n_leaves"] = len(tree)

    if not tree:
        return {
            "quality_score": 0.0,
            "issues": [{"severity": "error", "message": "Empty hierarchical tree"}],
            "metrics": metrics,
            "recommendations": ["Check label file parsing"],
        }

    # Level completeness — check each leaf has values for all levels
    incomplete_leaves = 0
    for leaf in tree:
        missing_levels = [lvl for lvl in levels if lvl not in leaf or leaf[lvl] in (None, "", "unknown")]
        if missing_levels:
            incomplete_leaves += 1

    completeness = 1.0 - (incomplete_leaves / len(tree))
    metrics["level_completeness"] = completeness
    if completeness < 0.9:
        issues.append({
            "severity": "warning",
            "message": f"{incomplete_leaves}/{len(tree)} leaves missing level values",
        })
        score -= 0.2

    # Timing coverage
    has_timing = sum(1 for leaf in tree if leaf.get("onset") is not None)
    timing_ratio = has_timing / len(tree)
    metrics["timing_coverage"] = timing_ratio
    if timing_ratio < 0.5:
        issues.append({
            "severity": "warning",
            "message": f"Only {timing_ratio:.0%} of leaves have timing info",
        })
        score -= 0.15
        recommendations.append("Add onset/duration to enable time-locked epoching")

    # Per-level distribution
    for lvl in levels:
        values = [str(leaf.get(lvl, "")) for leaf in tree if leaf.get(lvl)]
        unique_vals = list(set(values))
        metrics[f"{lvl}_n_unique"] = len(unique_vals)
        if len(unique_vals) <= 20:
            metrics[f"{lvl}_values"] = unique_vals

    # Leaf distribution balance per deepest level
    if levels:
        deepest = levels[-1]
        leaf_values = [str(leaf.get(deepest, "unknown")) for leaf in tree]
        unique_deep = list(set(leaf_values))
        if len(unique_deep) > 1:
            counts = [leaf_values.count(v) for v in unique_deep]
            balance = min(counts) / max(max(counts), 1)
            metrics["leaf_balance"] = balance

    return {
        "quality_score": max(0.0, min(1.0, score)),
        "issues": issues,
        "metrics": metrics,
        "recommendations": recommendations,
    }
