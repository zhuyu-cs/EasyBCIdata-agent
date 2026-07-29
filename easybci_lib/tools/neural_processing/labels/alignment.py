"""Continuous label alignment — resample per-sample labels to data timebase.

Handles the case where labels are sampled at a different rate than the EEG/neural
data (e.g., cursor position at 60Hz while EEG is 256Hz). Provides interpolation,
nearest-neighbor, and window-mean resampling strategies.
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def align_continuous_labels(
    labels: np.ndarray,
    label_freq: float,
    target_freq: float,
    method: str = "interpolate",
    target_n_samples: Optional[int] = None,
    fill_value: float = np.nan,
) -> np.ndarray:
    """Resample continuous labels to match the target (data) timebase.

    Parameters
    ----------
    labels : ndarray
        Label array. Shape: (n_label_samples,) for single-dimensional labels,
        or (n_dims, n_label_samples) for multi-dimensional (e.g., x/y cursor position).
    label_freq : float
        Sampling rate of the labels in Hz.
    target_freq : float
        Target sampling rate (typically the EEG/neural data frequency).
    method : str
        Resampling method:
        - "interpolate": linear interpolation (best for continuous float targets)
        - "nearest": nearest-neighbor (best for categorical/integer labels)
        - "window_mean": sliding window average (best for noisy continuous targets)
    target_n_samples : int, optional
        Exact number of output samples. If None, computed from duration and target_freq.
    fill_value : float
        Value to use for positions that cannot be interpolated (edges, NaN segments).

    Returns
    -------
    ndarray — resampled labels with shape matching target timebase.
        For 1D input: shape (target_n_samples,)
        For 2D input: shape (n_dims, target_n_samples)
    """
    if label_freq <= 0 or target_freq <= 0:
        import logging
        logging.getLogger(__name__).warning(
            "Frequencies must be positive: label_freq=%s, target_freq=%s. Using abs values.",
            label_freq, target_freq,
        )
        label_freq = max(abs(label_freq), 1.0)
        target_freq = max(abs(target_freq), 1.0)

    labels = np.asarray(labels, dtype=np.float64)
    is_1d = labels.ndim == 1

    if is_1d:
        labels = labels[np.newaxis, :]

    n_dims, n_label_samples = labels.shape
    label_duration = n_label_samples / label_freq

    if target_n_samples is None:
        target_n_samples = int(round(label_duration * target_freq))

    if target_n_samples <= 0:
        return np.empty((0,) if is_1d else (n_dims, 0), dtype=np.float32)

    # If frequencies are equal and lengths match, no resampling needed
    if abs(label_freq - target_freq) < 0.01 and n_label_samples == target_n_samples:
        out = labels.astype(np.float32)
        return out[0] if is_1d else out

    # Source and target time vectors
    src_times = np.arange(n_label_samples) / label_freq
    tgt_times = np.arange(target_n_samples) / target_freq

    if method == "interpolate":
        result = _interpolate(labels, src_times, tgt_times, fill_value)
    elif method == "nearest":
        result = _nearest(labels, src_times, tgt_times, fill_value)
    elif method == "window_mean":
        result = _window_mean(labels, label_freq, target_freq, target_n_samples, fill_value)
    else:
        import logging
        logging.getLogger(__name__).warning(
            "Unknown alignment method '%s', falling back to 'nearest'. "
            "Available: interpolate, nearest, window_mean.", method,
        )
        result = _nearest(labels, src_times, tgt_times, fill_value)

    result = result.astype(np.float32)
    return result[0] if is_1d else result


def _interpolate(
    labels: np.ndarray,
    src_times: np.ndarray,
    tgt_times: np.ndarray,
    fill_value: float,
) -> np.ndarray:
    """Linear interpolation with NaN-aware handling."""
    n_dims = labels.shape[0]
    n_target = len(tgt_times)
    result = np.full((n_dims, n_target), fill_value, dtype=np.float64)

    for dim in range(n_dims):
        channel = labels[dim]
        # Handle NaN segments: interpolate valid sections separately
        valid_mask = ~np.isnan(channel)
        if not np.any(valid_mask):
            continue

        if np.all(valid_mask):
            result[dim] = np.interp(tgt_times, src_times, channel)
        else:
            # Interpolate only within valid regions
            valid_src_times = src_times[valid_mask]
            valid_values = channel[valid_mask]
            interpolated = np.interp(tgt_times, valid_src_times, valid_values)

            # Mark target samples that fall in NaN gaps as fill_value
            # Find NaN boundaries in source
            nan_starts = []
            nan_ends = []
            in_nan = False
            for i, v in enumerate(valid_mask):
                if not v and not in_nan:
                    nan_starts.append(src_times[i])
                    in_nan = True
                elif v and in_nan:
                    nan_ends.append(src_times[i])
                    in_nan = False
            if in_nan:
                nan_ends.append(src_times[-1] + 1.0 / len(src_times))

            result[dim] = interpolated
            for ns, ne in zip(nan_starts, nan_ends):
                gap_mask = (tgt_times >= ns) & (tgt_times < ne)
                result[dim, gap_mask] = fill_value

    return result


def _nearest(
    labels: np.ndarray,
    src_times: np.ndarray,
    tgt_times: np.ndarray,
    fill_value: float,
) -> np.ndarray:
    """Nearest-neighbor resampling (best for categorical labels)."""
    n_dims = labels.shape[0]
    n_target = len(tgt_times)
    result = np.full((n_dims, n_target), fill_value, dtype=np.float64)

    # Find nearest source index for each target time
    indices = np.searchsorted(src_times, tgt_times, side="right") - 1
    indices = np.clip(indices, 0, len(src_times) - 1)

    # Check if next index is actually closer
    next_indices = np.minimum(indices + 1, len(src_times) - 1)
    dist_left = np.abs(tgt_times - src_times[indices])
    dist_right = np.abs(tgt_times - src_times[next_indices])
    use_right = dist_right < dist_left
    indices[use_right] = next_indices[use_right]

    for dim in range(n_dims):
        result[dim] = labels[dim, indices]

    # Clip out-of-bounds: target times beyond source range
    beyond_start = tgt_times < src_times[0]
    beyond_end = tgt_times > src_times[-1]
    if np.any(beyond_start) or np.any(beyond_end):
        result[:, beyond_start | beyond_end] = fill_value

    return result


def _window_mean(
    labels: np.ndarray,
    label_freq: float,
    target_freq: float,
    target_n_samples: int,
    fill_value: float,
) -> np.ndarray:
    """Sliding window mean resampling (anti-aliased downsampling)."""
    n_dims, n_label_samples = labels.shape
    result = np.full((n_dims, target_n_samples), fill_value, dtype=np.float64)

    # Window size in source samples that maps to one target sample
    window_size = label_freq / target_freq

    for i in range(target_n_samples):
        # Center of this target sample in source-sample space
        center = i * window_size
        start = int(max(0, center - window_size / 2))
        end = int(min(n_label_samples, center + window_size / 2 + 1))

        if start >= end:
            continue

        window = labels[:, start:end]
        # Ignore NaN in mean
        with np.errstate(all="ignore"):
            means = np.nanmean(window, axis=1)
        nan_mask = np.isnan(means)
        means[nan_mask] = fill_value
        result[:, i] = means

    return result
