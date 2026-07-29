"""Segmentation — cut continuous recordings into epochs.

Two approaches:
1. Event-triggered: each trigger event defines one segment
2. Sliding window: fixed-size overlapping windows across the recording

The fast overlap query from neuralset (_EventBucket with searchsorted)
is valuable for large event sets. We keep that pattern but drop the
heavy Segment/EventStore class hierarchy — just return numpy arrays.
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from easybci_lib.tools.neural_processing._core.timed_array import Frequency

logger = logging.getLogger(__name__)


def segment_data(
    data: np.ndarray,
    frequency: float,
    events: List[Dict[str, Any]],
    duration: float,
    offset: float = 0.0,
    baseline: Optional[Tuple[float, float]] = None,
) -> Dict[str, Any]:
    """Segment continuous data around events.

    Parameters
    ----------
    data : ndarray shape (n_channels, n_total_samples)
        Continuous recording.
    frequency : float
        Sampling rate.
    events : list of dict
        Each dict must have "start" (float, seconds). Optional: "label", "type".
    duration : float
        Segment duration in seconds.
    offset : float
        Start offset relative to event (negative = before event).
    baseline : tuple (start, end) or None
        Baseline window in seconds relative to segment start.
        If set, subtracts baseline mean from each segment.

    Returns
    -------
    dict with:
        segments : ndarray shape (n_events, n_channels, n_samples_per_segment)
        frequency : float
        events : list[dict] — input events (for label matching)
        meta : dict
    """
    freq = Frequency(frequency)
    n_seg_samples = freq.to_ind(duration)
    n_total = data.shape[-1]
    n_channels = data.shape[0] if data.ndim > 1 else 1

    if data.ndim == 1:
        data = data[np.newaxis, :]

    segments = []
    valid_events = []

    for event in events:
        seg_start_sec = event["start"] + offset
        start_idx = freq.to_ind(seg_start_sec)
        end_idx = start_idx + n_seg_samples

        # Skip out-of-bounds segments
        if start_idx < 0 or end_idx > n_total:
            continue

        seg = data[:, start_idx:end_idx].copy()

        # Baseline correction
        if baseline is not None:
            bl_start = freq.to_ind(baseline[0])
            bl_end = freq.to_ind(baseline[1])
            bl_start = max(0, bl_start)
            bl_end = min(n_seg_samples, bl_end)
            if bl_end > bl_start:
                bl_mean = seg[:, bl_start:bl_end].mean(axis=-1, keepdims=True)
                seg = seg - bl_mean

        segments.append(seg)
        valid_events.append(event)

    if not segments:
        segments_arr = np.zeros((0, n_channels, n_seg_samples), dtype=np.float32)
    else:
        segments_arr = np.stack(segments, axis=0).astype(np.float32)

    return {
        "segments": segments_arr,
        "frequency": frequency,
        "events": valid_events,
        "meta": {
            "n_segments": len(valid_events),
            "segment_duration": duration,
            "offset": offset,
            "n_channels": n_channels,
            "n_samples_per_segment": n_seg_samples,
            "baseline": baseline,
        },
    }


def sliding_windows(
    data: np.ndarray,
    frequency: float,
    window_duration: float,
    stride: float,
    drop_incomplete: bool = True,
) -> Dict[str, Any]:
    """Cut data into overlapping fixed-size windows.

    Parameters
    ----------
    data : ndarray shape (n_channels, n_total_samples)
        Continuous recording.
    frequency : float
        Sampling rate.
    window_duration : float
        Window size in seconds.
    stride : float
        Step size in seconds.
    drop_incomplete : bool
        Drop last window if shorter than window_duration.

    Returns
    -------
    dict with:
        segments : ndarray shape (n_windows, n_channels, n_samples_per_window)
        frequency : float
        starts : ndarray — start time of each window in seconds
        meta : dict
    """
    freq = Frequency(frequency)
    n_total = data.shape[-1]
    n_channels = data.shape[0] if data.ndim > 1 else 1

    if data.ndim == 1:
        data = data[np.newaxis, :]

    n_window = freq.to_ind(window_duration)
    n_stride = freq.to_ind(stride)

    if n_stride <= 0:
        n_stride = 1
        logger.warning("Stride too small: %ss at %sHz — auto-adjusted to 1 sample", stride, frequency)

    # Compute window starts
    if drop_incomplete:
        max_start = n_total - n_window
    else:
        max_start = n_total - 1

    if max_start < 0:
        logger.warning(
            "Data too short (%d samples) for window of %d samples — "
            "reducing window to fit data",
            n_total, n_window,
        )
        n_window = max(1, n_total)
        max_start = 0

    start_indices = np.arange(0, max_start + 1, n_stride)
    n_windows = len(start_indices)

    segments = np.zeros((n_windows, n_channels, n_window), dtype=np.float32)
    for i, idx in enumerate(start_indices):
        end = min(idx + n_window, n_total)
        seg_len = end - idx
        segments[i, :, :seg_len] = data[:, idx:end]

    starts = freq.to_sec(start_indices)

    return {
        "segments": segments,
        "frequency": frequency,
        "starts": starts,
        "meta": {
            "n_windows": n_windows,
            "window_duration": window_duration,
            "stride": stride,
            "drop_incomplete": drop_incomplete,
        },
    }
