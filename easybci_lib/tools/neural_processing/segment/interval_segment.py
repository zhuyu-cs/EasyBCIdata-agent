"""Interval-based segmentation — cut data using [start, end] time ranges.

Unlike event-triggered epoching (fixed duration from point events), this handles
variable-length segments defined by explicit start/end boundaries. Common use cases:
- Sleep stage segments (30s epochs with stage labels)
- Emotion experiment blocks (variable duration video clips)
- Task/rest alternating blocks

Handles: gaps between intervals, variable-length segments (padding/truncation),
multi-dimensional labels per segment.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from easybci_lib.tools.neural_processing._core.timed_array import Frequency


def segment_by_intervals(
    data: np.ndarray,
    frequency: float,
    intervals: List[Dict[str, Any]],
    gap_handling: str = "mark_unlabeled",
    pad_to_max: bool = False,
    max_duration: Optional[float] = None,
    min_duration: Optional[float] = None,
) -> Dict[str, Any]:
    """Segment continuous data using time intervals with explicit start/end.

    Parameters
    ----------
    data : ndarray shape (n_channels, n_total_samples)
        Continuous recording.
    frequency : float
        Sampling rate in Hz.
    intervals : list of dict
        Each dict must have:
            "start": float — segment start time in seconds
            "end": float — segment end time in seconds
        Optional:
            "label": str — segment label/class
            "metadata": dict — additional per-segment info
    gap_handling : str
        How to handle gaps between intervals:
        - "mark_unlabeled": extract gap segments labeled "unlabeled"
        - "discard": ignore gaps
        - "as_rest": extract gap segments labeled "rest"
    pad_to_max : bool
        If True, pad shorter segments to max_duration with zeros.
        If False, segments have variable lengths (returned as list).
    max_duration : float, optional
        Maximum segment duration in seconds. Longer segments are truncated.
    min_duration : float, optional
        Minimum segment duration in seconds. Shorter segments are discarded.

    Returns
    -------
    Dict with:
        segments : list[ndarray] — each shape (n_channels, n_samples_i)
            or ndarray (n_segments, n_channels, n_padded_samples) if pad_to_max=True
        labels : list[str] — label for each segment
        starts : list[float] — start time of each segment
        ends : list[float] — end time of each segment
        durations : list[float] — duration of each segment
        frequency : float
        meta : dict
    """
    freq = Frequency(frequency)
    n_total = data.shape[-1]
    n_channels = data.shape[0] if data.ndim > 1 else 1

    if data.ndim == 1:
        data = data[np.newaxis, :]

    data_duration = n_total / frequency

    # Validate and sort intervals
    valid_intervals = _validate_intervals(intervals, data_duration, min_duration)

    # Handle gaps
    all_intervals = valid_intervals
    if gap_handling != "discard":
        gaps = _find_gaps(valid_intervals, data_duration)
        gap_label = "unlabeled" if gap_handling == "mark_unlabeled" else "rest"
        for gap in gaps:
            if min_duration and (gap["end"] - gap["start"]) < min_duration:
                continue
            all_intervals.append({
                "start": gap["start"],
                "end": gap["end"],
                "label": gap_label,
                "metadata": {"is_gap": True},
            })

    # Sort all intervals by start time
    all_intervals.sort(key=lambda x: x["start"])

    # Extract segments
    segments = []
    labels = []
    starts = []
    ends = []
    durations = []
    skipped = 0

    for interval in all_intervals:
        seg_start = interval["start"]
        seg_end = interval["end"]
        seg_duration = seg_end - seg_start

        # Apply max_duration truncation
        if max_duration and seg_duration > max_duration:
            seg_end = seg_start + max_duration
            seg_duration = max_duration

        start_idx = freq.to_ind(seg_start)
        end_idx = freq.to_ind(seg_end)

        # Bounds check
        start_idx = max(0, start_idx)
        end_idx = min(n_total, end_idx)

        if end_idx <= start_idx:
            skipped += 1
            continue

        seg = data[:, start_idx:end_idx].copy()
        segments.append(seg)
        labels.append(interval.get("label", "unknown"))
        starts.append(seg_start)
        ends.append(seg_end)
        durations.append(seg_duration)

    # Optionally pad to uniform length
    if pad_to_max and segments:
        if max_duration:
            target_samples = freq.to_ind(max_duration)
        else:
            target_samples = max(seg.shape[-1] for seg in segments)

        padded = np.zeros((len(segments), n_channels, target_samples), dtype=np.float32)
        for i, seg in enumerate(segments):
            seg_len = min(seg.shape[-1], target_samples)
            padded[i, :, :seg_len] = seg[:, :seg_len]
        segments_out: Any = padded
    else:
        segments_out = segments

    # Compute label statistics
    unique_labels = sorted(set(labels))
    label_counts = {lbl: labels.count(lbl) for lbl in unique_labels}
    total_labeled_duration = sum(d for d, lbl in zip(durations, labels) if lbl not in ("unlabeled", "rest"))

    meta = {
        "n_segments": len(segments),
        "n_skipped": skipped,
        "n_channels": n_channels,
        "gap_handling": gap_handling,
        "pad_to_max": pad_to_max,
        "unique_labels": unique_labels,
        "label_counts": label_counts,
        "total_labeled_duration_s": round(total_labeled_duration, 2),
        "data_duration_s": round(data_duration, 2),
        "coverage_ratio": round(total_labeled_duration / max(data_duration, 1e-6), 3),
    }

    if pad_to_max and segments:
        meta["padded_length_samples"] = target_samples

    return {
        "segments": segments_out,
        "labels": labels,
        "starts": starts,
        "ends": ends,
        "durations": durations,
        "frequency": frequency,
        "meta": meta,
    }


def _validate_intervals(
    intervals: List[Dict[str, Any]],
    data_duration: float,
    min_duration: Optional[float],
) -> List[Dict[str, Any]]:
    """Validate and clean interval list."""
    valid = []
    for iv in intervals:
        start = float(iv.get("start", 0))
        end = float(iv.get("end", 0))

        if end <= start:
            continue
        if start >= data_duration:
            continue
        if min_duration and (end - start) < min_duration:
            continue

        # Clamp to data range
        start = max(0.0, start)
        end = min(data_duration, end)

        valid.append({
            "start": start,
            "end": end,
            "label": iv.get("label", "unknown"),
            "metadata": iv.get("metadata", {}),
        })

    return valid


def _find_gaps(
    intervals: List[Dict[str, Any]],
    data_duration: float,
    min_gap: float = 0.1,
) -> List[Dict[str, float]]:
    """Find time gaps between sorted intervals."""
    if not intervals:
        if data_duration > min_gap:
            return [{"start": 0.0, "end": data_duration}]
        return []

    sorted_ivs = sorted(intervals, key=lambda x: x["start"])
    gaps = []

    # Gap before first interval
    if sorted_ivs[0]["start"] > min_gap:
        gaps.append({"start": 0.0, "end": sorted_ivs[0]["start"]})

    # Gaps between intervals
    for i in range(len(sorted_ivs) - 1):
        gap_start = sorted_ivs[i]["end"]
        gap_end = sorted_ivs[i + 1]["start"]
        if gap_end - gap_start >= min_gap:
            gaps.append({"start": gap_start, "end": gap_end})

    # Gap after last interval
    if sorted_ivs[-1]["end"] < data_duration - min_gap:
        gaps.append({"start": sorted_ivs[-1]["end"], "end": data_duration})

    return gaps
