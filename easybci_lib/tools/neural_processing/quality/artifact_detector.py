"""Channel-level artifact detection — identifies bad segments per channel.

Detects three types of artifacts:
- Amplitude: exceeds threshold (default ±500µV for EEG scale, auto-calibrated)
- Flat: channel std < threshold within a sliding window
- High-frequency noise: anomalous power in 50-100Hz band (>3σ from mean)

Output is a structured annotation map usable by the rejection module
and renderable by the Web UI.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def detect_artifacts(
    data: np.ndarray,
    frequency: float,
    channels: Optional[List[str]] = None,
    amplitude_threshold: Optional[float] = None,
    flat_threshold: float = 0.5e-6,
    window_seconds: float = 1.0,
) -> Dict[str, Any]:
    """Detect artifacts across all channels.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_samples)
    frequency : float, sampling rate
    channels : list of str, channel names
    amplitude_threshold : float or None
        If None, auto-calibrated as 10× median absolute amplitude.
    flat_threshold : float
        Minimum std within a window to not be considered flat.
    window_seconds : float
        Sliding window size for flat detection.

    Returns
    -------
    dict with:
        artifacts : list of {channel, channel_idx, start_sec, end_sec, type, severity}
        summary : dict of channel → artifact count
        stats : dict with overall statistics
    """
    if data.ndim != 2:
        if data.ndim == 3:
            n_seg, n_ch, n_t = data.shape
            data = data.reshape(n_seg * n_ch, n_t)
        else:
            return {"artifacts": [], "summary": {}, "stats": {"error": "unexpected ndim"}}

    n_channels, n_samples = data.shape
    if channels is None:
        channels = [f"Ch{i}" for i in range(n_channels)]

    window_samples = max(1, int(window_seconds * frequency))
    artifacts: List[Dict[str, Any]] = []

    # Auto-calibrate amplitude threshold
    if amplitude_threshold is None:
        median_abs = np.median(np.abs(data))
        if median_abs > 0:
            amplitude_threshold = 10.0 * median_abs
        else:
            amplitude_threshold = 500e-6  # 500µV default

    # Per-channel detection
    for ch_idx in range(n_channels):
        ch_data = data[ch_idx]
        ch_name = channels[ch_idx] if ch_idx < len(channels) else f"Ch{ch_idx}"

        # 1. Amplitude artifacts
        amp_mask = np.abs(ch_data) > amplitude_threshold
        amp_segments = _mask_to_segments(amp_mask, frequency)
        for start, end in amp_segments:
            artifacts.append({
                "channel": ch_name,
                "channel_idx": ch_idx,
                "start_sec": start,
                "end_sec": end,
                "type": "amplitude",
                "severity": "high",
            })

        # 2. Flat detection (sliding window std)
        n_windows = max(1, n_samples // window_samples)
        for w in range(n_windows):
            w_start = w * window_samples
            w_end = min(w_start + window_samples, n_samples)
            window_data = ch_data[w_start:w_end]
            if np.std(window_data) < flat_threshold:
                artifacts.append({
                    "channel": ch_name,
                    "channel_idx": ch_idx,
                    "start_sec": w_start / frequency,
                    "end_sec": w_end / frequency,
                    "type": "flat",
                    "severity": "medium",
                })

    # 3. High-frequency noise (batch across channels)
    hf_artifacts = _detect_hf_noise(data, frequency, channels)
    artifacts.extend(hf_artifacts)

    # Build summary
    summary: Dict[str, int] = {}
    for art in artifacts:
        ch = art["channel"]
        summary[ch] = summary.get(ch, 0) + 1

    return {
        "artifacts": artifacts,
        "summary": summary,
        "stats": {
            "total_artifacts": len(artifacts),
            "channels_affected": len(summary),
            "amplitude_threshold_used": amplitude_threshold,
        },
    }


def _mask_to_segments(mask: np.ndarray, frequency: float) -> List[tuple]:
    """Convert a boolean mask to (start_sec, end_sec) segments."""
    segments = []
    if not np.any(mask):
        return segments

    diff = np.diff(mask.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1

    if mask[0]:
        starts = np.concatenate([[0], starts])
    if mask[-1]:
        ends = np.concatenate([ends, [len(mask)]])

    for s, e in zip(starts, ends):
        segments.append((float(s / frequency), float(e / frequency)))

    return segments


def _detect_hf_noise(
    data: np.ndarray,
    frequency: float,
    channels: List[str],
) -> List[Dict[str, Any]]:
    """Detect channels with anomalous high-frequency power."""
    artifacts = []

    # Only meaningful if sampling rate allows 50-100Hz band
    if frequency < 200:
        return artifacts

    from scipy.signal import welch

    n_channels = data.shape[0]
    hf_powers = np.zeros(n_channels)

    nperseg = min(int(frequency * 2), data.shape[1])
    if nperseg < 4:
        return artifacts

    for ch_idx in range(n_channels):
        freqs, psd = welch(data[ch_idx], fs=frequency, nperseg=nperseg)
        # 50-100Hz band
        hf_mask = (freqs >= 50) & (freqs <= 100)
        if np.any(hf_mask):
            hf_powers[ch_idx] = np.mean(psd[hf_mask])

    if np.std(hf_powers) == 0:
        return artifacts

    # Flag channels >3σ above mean
    mean_hf = np.mean(hf_powers)
    std_hf = np.std(hf_powers)
    threshold = mean_hf + 3 * std_hf

    for ch_idx in range(n_channels):
        if hf_powers[ch_idx] > threshold:
            ch_name = channels[ch_idx] if ch_idx < len(channels) else f"Ch{ch_idx}"
            artifacts.append({
                "channel": ch_name,
                "channel_idx": ch_idx,
                "start_sec": 0.0,
                "end_sec": float(data.shape[1] / frequency),
                "type": "hf_noise",
                "severity": "medium",
            })

    return artifacts
