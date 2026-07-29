"""Bad channel/segment/epoch rejection — interactive data cleaning.

Provides functions for:
1. Marking and removing bad channels
2. Marking and removing bad segments from 3D epoch arrays
3. Auto-rejecting epochs based on statistical thresholds
4. Computing per-epoch quality statistics for UI display
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def mark_bad_channels(
    data_dict: Dict[str, Any],
    bad_channels: List[str],
) -> Dict[str, Any]:
    """Remove specified channels from data.

    Parameters
    ----------
    data_dict : dict with "data" (n_channels, n_samples), "channels"
    bad_channels : list of channel names to remove

    Returns
    -------
    Updated data_dict with bad channels removed.
    """
    channels = data_dict["channels"]
    data = data_dict["data"]

    keep_indices = [i for i, ch in enumerate(channels) if ch not in bad_channels]
    if len(keep_indices) == len(channels):
        logger.info("No matching channels to remove: %s", bad_channels)
        return data_dict

    removed = [ch for ch in channels if ch in bad_channels]
    logger.info("Removing %d bad channels: %s", len(removed), removed)

    data_dict["data"] = data[keep_indices]
    data_dict["channels"] = [channels[i] for i in keep_indices]

    # Update meta
    meta = data_dict.setdefault("meta", {})
    meta.setdefault("rejected_channels", []).extend(removed)
    if "ch_types" in meta:
        meta["ch_types"] = [meta["ch_types"][i] for i in keep_indices]

    # Invalidate MNE info cache
    data_dict.pop("_mne_info", None)
    return data_dict


def mark_bad_segments(
    segments: np.ndarray,
    bad_indices: List[int],
    labels: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Remove specified segments from a 3D epoch array.

    Parameters
    ----------
    segments : ndarray, shape (n_epochs, n_channels, n_samples)
    bad_indices : list of epoch indices to remove
    labels : ndarray of shape (n_epochs,) or None

    Returns
    -------
    dict with "segments" (kept epochs), "labels" (if provided), "removed_count"
    """
    if segments.ndim != 3:
        logger.warning("Expected 3D array (epochs, ch, samples), got %dD — skipping rejection", segments.ndim)
        result = {
            "segments": segments,
            "removed_count": 0,
            "kept_indices": list(range(segments.shape[0] if segments.ndim >= 1 else 0)),
            "removed_indices": [],
        }
        if labels is not None:
            result["labels"] = labels
        return result

    n_epochs = segments.shape[0]
    keep_mask = np.ones(n_epochs, dtype=bool)
    valid_bad = [i for i in bad_indices if 0 <= i < n_epochs]
    keep_mask[valid_bad] = False

    result = {
        "segments": segments[keep_mask],
        "removed_count": len(valid_bad),
        "kept_indices": np.where(keep_mask)[0].tolist(),
        "removed_indices": valid_bad,
    }

    if labels is not None:
        result["labels"] = labels[keep_mask]

    logger.info("Removed %d/%d segments", len(valid_bad), n_epochs)
    return result


def compute_epoch_stats(
    segments: np.ndarray,
    frequency: float,
) -> List[Dict[str, Any]]:
    """Compute per-epoch quality statistics for UI display.

    Parameters
    ----------
    segments : ndarray, shape (n_epochs, n_channels, n_samples)
    frequency : float

    Returns
    -------
    list of dicts, one per epoch:
        {epoch_idx, amplitude_range, max_amplitude, variance, has_artifact}
    """
    if segments.ndim != 3:
        logger.warning("Expected 3D array for epoch stats, got %dD — returning empty stats", segments.ndim)
        return []

    n_epochs = segments.shape[0]
    stats = []

    # Compute global variance for threshold
    all_variances = np.var(segments, axis=2).mean(axis=1)  # per-epoch mean variance
    median_var = np.median(all_variances)
    var_threshold = median_var * 5 if median_var > 0 else float("inf")

    for i in range(n_epochs):
        epoch = segments[i]
        amp_range = float(np.ptp(epoch))
        max_amp = float(np.max(np.abs(epoch)))
        variance = float(np.var(epoch))
        has_artifact = variance > var_threshold or max_amp > np.median(np.abs(segments)) * 10

        stats.append({
            "epoch_idx": i,
            "amplitude_range": amp_range,
            "max_amplitude": max_amp,
            "variance": variance,
            "has_artifact": has_artifact,
        })

    return stats


def auto_reject_epochs(
    segments: np.ndarray,
    frequency: float,
    threshold_std: float = 3.0,
    labels: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Automatically reject epochs with extreme statistics.

    Uses a z-score approach: epochs with variance > threshold_std standard
    deviations from the mean are rejected.

    Parameters
    ----------
    segments : ndarray, shape (n_epochs, n_channels, n_samples)
    frequency : float
    threshold_std : float, number of std devs for rejection
    labels : ndarray or None

    Returns
    -------
    dict with:
        segments : kept epochs
        labels : kept labels (if provided)
        kept_indices : list of kept epoch indices
        rejected_indices : list of rejected indices
        reasons : list of rejection reason strings
    """
    if segments.ndim != 3:
        logger.warning("Expected 3D array for auto-rejection, got %dD — returning input unchanged", segments.ndim)
        n = segments.shape[0] if segments.ndim >= 1 else 0
        result = {
            "segments": segments,
            "kept_indices": list(range(n)),
            "rejected_indices": [],
            "reasons": [],
        }
        if labels is not None:
            result["labels"] = labels
        return result

    n_epochs = segments.shape[0]
    if n_epochs < 3:
        return {
            "segments": segments,
            "labels": labels,
            "kept_indices": list(range(n_epochs)),
            "rejected_indices": [],
            "reasons": [],
        }

    # Per-epoch statistics
    epoch_variances = np.var(segments, axis=(1, 2))
    epoch_max_amp = np.max(np.abs(segments), axis=(1, 2))

    # Z-score based rejection
    var_mean = np.mean(epoch_variances)
    var_std = np.std(epoch_variances)
    amp_mean = np.mean(epoch_max_amp)
    amp_std = np.std(epoch_max_amp)

    rejected = []
    reasons = []

    for i in range(n_epochs):
        reject_reasons = []

        if var_std > 0 and abs(epoch_variances[i] - var_mean) > threshold_std * var_std:
            reject_reasons.append(
                f"variance={epoch_variances[i]:.4f} ({abs(epoch_variances[i] - var_mean)/var_std:.1f}σ)"
            )

        if amp_std > 0 and abs(epoch_max_amp[i] - amp_mean) > threshold_std * amp_std:
            reject_reasons.append(
                f"max_amplitude={epoch_max_amp[i]:.4f} ({abs(epoch_max_amp[i] - amp_mean)/amp_std:.1f}σ)"
            )

        if reject_reasons:
            rejected.append(i)
            reasons.append(f"epoch {i}: " + ", ".join(reject_reasons))

    kept_mask = np.ones(n_epochs, dtype=bool)
    kept_mask[rejected] = False

    result = {
        "segments": segments[kept_mask],
        "kept_indices": np.where(kept_mask)[0].tolist(),
        "rejected_indices": rejected,
        "reasons": reasons,
    }

    if labels is not None:
        result["labels"] = labels[kept_mask]

    logger.info(
        "Auto-rejected %d/%d epochs (threshold=%.1fσ)",
        len(rejected), n_epochs, threshold_std,
    )
    return result
