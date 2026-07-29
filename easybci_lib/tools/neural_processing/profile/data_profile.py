"""Compute a quantified data profile from neural recording data.

The DataProfile captures signal characteristics that inform adaptive pipeline
routing — e.g., whether notch filtering is needed, how aggressive bandpass
should be, whether channels need rejection.

Design: pure functions operating on numpy arrays. No MNE dependency.
All metrics are computed from a data snippet (default: first 10s) to keep
inspection fast even for multi-GB files.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_STANDARD_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 100.0),
}

_POWERLINE_FREQS = [50.0, 60.0]


@dataclass
class DataProfile:
    """Quantified characteristics of a neural recording.

    All fields are designed to be directly usable as routing conditions.
    """

    # --- Power line ---
    powerline_freq: float = 0.0
    powerline_amplitude_db: float = 0.0
    powerline_present: bool = False

    # --- Frequency content ---
    dominant_frequency: float = 0.0
    effective_bandwidth: float = 0.0
    snr_per_band: Dict[str, float] = field(default_factory=dict)

    # --- Drift and stationarity ---
    drift_severity: float = 0.0
    has_significant_drift: bool = False

    # --- Channel quality ---
    channel_consistency: float = 0.0
    n_bad_channels: int = 0
    bad_channel_names: List[str] = field(default_factory=list)
    flat_channel_ratio: float = 0.0

    # --- Amplitude / artifacts ---
    artifact_ratio: float = 0.0
    has_extreme_amplitudes: bool = False
    dynamic_range_db: float = 0.0
    median_amplitude: float = 0.0

    # --- Data characteristics ---
    has_nans: bool = False
    nan_ratio: float = 0.0
    sampling_rate: float = 0.0
    n_channels: int = 0
    duration_s: float = 0.0

    # --- Summary scores (0-1, higher = worse problem) ---
    noise_score: float = 0.0
    quality_score: float = 1.0

    # --- Cohort label (optional) ---
    cohort_tag: str = ""  # empty = unspecified

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for JSON output."""
        return {
            "powerline": {
                "present": bool(self.powerline_present),
                "freq_hz": self.powerline_freq,
                "amplitude_db": round(self.powerline_amplitude_db, 1),
            },
            "frequency": {
                "dominant_hz": round(self.dominant_frequency, 1),
                "effective_bandwidth_hz": round(self.effective_bandwidth, 1),
                "snr_per_band": {k: round(v, 1) for k, v in self.snr_per_band.items()},
            },
            "drift": {
                "severity": round(self.drift_severity, 3),
                "significant": bool(self.has_significant_drift),
            },
            "channels": {
                "consistency": round(self.channel_consistency, 3),
                "n_bad": int(self.n_bad_channels),
                "bad_names": self.bad_channel_names[:10],
                "flat_ratio": round(self.flat_channel_ratio, 3),
            },
            "artifacts": {
                "ratio": round(self.artifact_ratio, 3),
                "extreme_amplitudes": bool(self.has_extreme_amplitudes),
                "dynamic_range_db": round(self.dynamic_range_db, 1),
            },
            "data": {
                "has_nans": bool(self.has_nans),
                "nan_ratio": round(self.nan_ratio, 4),
                "sampling_rate": self.sampling_rate,
                "n_channels": int(self.n_channels),
                "duration_s": round(self.duration_s, 1),
            },
            "scores": {
                "noise": round(self.noise_score, 2),
                "quality": round(self.quality_score, 2),
            },
            "cohort_tag": self.cohort_tag,
        }


def compute_profile(
    data: np.ndarray,
    frequency: float,
    channels: Optional[List[str]] = None,
    max_duration_s: float = 30.0,
    *,
    data_path: Optional[str] = None,
    cli_cohort_override: Optional[str] = None,
) -> DataProfile:
    """Compute a DataProfile from raw data.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_samples)
        Raw neural data (continuous, not epoched).
    frequency : float
        Sampling rate in Hz.
    channels : list of str, optional
        Channel names (for bad channel reporting).
    max_duration_s : float
        Maximum duration to analyze (uses first N seconds for speed).
    data_path : str, optional
        Path to the source recording.  When provided, ``cohort_tag`` is
        auto-resolved via :func:`cohort_resolver.resolve_cohort_tag`
        (BIDS ``participants.tsv`` > CLI override > empty).  Best-effort —
        any failure leaves ``cohort_tag`` as empty string.
    cli_cohort_override : str, optional
        CLI-set cohort tag (typically read by the caller from a prior
        ``data_profile.json`` written by ``easybci profile set-cohort``).
        Takes effect only when no BIDS source is found at ``data_path``.

    Returns
    -------
    DataProfile with all fields populated.
    """
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.ndim == 3:
        n_seg, n_ch, n_t = data.shape
        data = data.reshape(n_seg * n_ch, n_t)

    n_channels, n_samples = data.shape
    if channels is None:
        channels = [f"Ch{i}" for i in range(n_channels)]

    max_samples = int(max_duration_s * frequency)
    snippet = data[:, :min(n_samples, max_samples)]

    profile = DataProfile(
        sampling_rate=frequency,
        n_channels=n_channels,
        duration_s=n_samples / frequency if frequency > 0 else 0,
    )

    _check_nans(snippet, profile)

    work = np.nan_to_num(snippet, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)

    _analyze_powerline(work, frequency, profile)
    _analyze_frequency_content(work, frequency, profile)
    _analyze_drift(work, frequency, profile)
    _analyze_channels(work, channels, profile)
    _analyze_artifacts(work, frequency, profile)
    _compute_summary_scores(profile)

    # T1.5 — auto-resolve cohort_tag from BIDS participants.tsv (or CLI
    # override) when ``data_path`` is supplied.  Fail-open: any error leaves
    # cohort_tag empty rather than aborting the profile.
    if data_path is not None:
        try:
            from pathlib import Path as _Path
            from .cohort_resolver import resolve_cohort_tag
            profile.cohort_tag = resolve_cohort_tag(
                data_path=_Path(data_path),
                cli_override=cli_cohort_override,
            ) or ""
        except Exception:
            profile.cohort_tag = cli_cohort_override or ""

    return profile


def _check_nans(data: np.ndarray, profile: DataProfile) -> None:
    nan_count = np.isnan(data).sum()
    total = data.size
    profile.has_nans = nan_count > 0
    profile.nan_ratio = nan_count / total if total > 0 else 0.0


def _analyze_powerline(data: np.ndarray, fs: float, profile: DataProfile) -> None:
    """Detect power line interference and measure its amplitude."""
    nyquist = fs / 2.0
    if nyquist < 45:
        return

    n_channels, n_samples = data.shape
    if n_samples < int(fs * 2):
        return

    # Use median channel PSD for robustness
    from numpy.fft import rfft, rfftfreq

    n_fft = min(n_samples, int(fs * 4))
    freqs = rfftfreq(n_fft, 1.0 / fs)
    freq_resolution = fs / n_fft

    # Compute PSD (Welch-like: average over segments)
    n_segments = max(1, n_samples // n_fft)
    psd_accum = np.zeros(len(freqs))
    for seg_i in range(n_segments):
        start = seg_i * n_fft
        segment = data[:, start:start + n_fft]
        if segment.shape[1] < n_fft:
            break
        windowed = segment * np.hanning(n_fft)
        fft_mag = np.abs(rfft(windowed, axis=1)) ** 2
        psd_accum += np.median(fft_mag, axis=0)
    psd_accum /= n_segments

    psd_db = 10 * np.log10(psd_accum + 1e-20)

    best_pl_freq = 0.0
    best_pl_amplitude = 0.0

    for pl_freq in _POWERLINE_FREQS:
        if pl_freq >= nyquist:
            continue
        idx = np.argmin(np.abs(freqs - pl_freq))
        if idx < 3 or idx >= len(psd_db) - 3:
            continue

        peak_db = psd_db[idx]
        # Background: average of neighbors 3-8 bins away
        neighbors = np.concatenate([psd_db[max(0, idx - 8):idx - 2], psd_db[idx + 3:idx + 9]])
        if len(neighbors) == 0:
            continue
        background_db = np.median(neighbors)
        prominence = peak_db - background_db

        if prominence > best_pl_amplitude:
            best_pl_amplitude = prominence
            best_pl_freq = pl_freq

    profile.powerline_freq = best_pl_freq
    profile.powerline_amplitude_db = best_pl_amplitude
    # Threshold: >6 dB prominence means clearly visible power line peak
    profile.powerline_present = best_pl_amplitude > 6.0


def _analyze_frequency_content(data: np.ndarray, fs: float, profile: DataProfile) -> None:
    """Analyze spectral content and per-band SNR."""
    n_channels, n_samples = data.shape
    nyquist = fs / 2.0

    n_fft = min(n_samples, int(fs * 4))
    if n_fft < 64:
        return

    from numpy.fft import rfft, rfftfreq
    freqs = rfftfreq(n_fft, 1.0 / fs)

    segment = data[:, :n_fft]
    windowed = segment * np.hanning(n_fft)
    fft_mag = np.abs(rfft(windowed, axis=1)) ** 2
    psd_median = np.median(fft_mag, axis=0)
    psd_db = 10 * np.log10(psd_median + 1e-20)

    # Dominant frequency (peak in 1-100 Hz)
    valid_mask = (freqs >= 1.0) & (freqs <= min(100.0, nyquist - 1))
    if valid_mask.any():
        valid_psd = psd_db.copy()
        valid_psd[~valid_mask] = -np.inf
        profile.dominant_frequency = float(freqs[np.argmax(valid_psd)])

    # Effective bandwidth: frequency range containing 95% of total power
    cumulative_power = np.cumsum(psd_median[1:])
    total_power = cumulative_power[-1] if len(cumulative_power) > 0 else 1.0
    if total_power > 0:
        low_idx = np.searchsorted(cumulative_power, total_power * 0.025)
        high_idx = np.searchsorted(cumulative_power, total_power * 0.975)
        profile.effective_bandwidth = float(freqs[min(high_idx + 1, len(freqs) - 1)] - freqs[max(low_idx, 1)])

    # SNR per band: band power relative to broadband noise floor
    noise_floor = np.percentile(psd_db[valid_mask], 10) if valid_mask.any() else 0
    for band_name, (f_low, f_high) in _STANDARD_BANDS.items():
        if f_low >= nyquist:
            continue
        band_mask = (freqs >= f_low) & (freqs <= min(f_high, nyquist))
        if band_mask.any():
            band_power = np.mean(psd_db[band_mask])
            profile.snr_per_band[band_name] = float(band_power - noise_floor)


def _analyze_drift(data: np.ndarray, fs: float, profile: DataProfile) -> None:
    """Quantify low-frequency drift severity."""
    n_channels, n_samples = data.shape

    # Drift metric: ratio of power below 0.5 Hz to total power in 0.5-40 Hz
    # A simpler proxy: detrend and measure residual slow fluctuation
    window_s = min(10.0, n_samples / fs)
    window_samples = int(window_s * fs)
    if window_samples < 10:
        return

    segment = data[:, :window_samples]

    # Linear detrend per channel
    x = np.linspace(0, 1, window_samples)
    means = segment.mean(axis=1, keepdims=True)
    slopes = np.zeros(n_channels)
    for ch_i in range(n_channels):
        slopes[ch_i] = np.polyfit(x, segment[ch_i], 1)[0]

    # Drift severity: median absolute slope relative to signal std
    signal_std = np.std(segment, axis=1)
    signal_std[signal_std < 1e-12] = 1e-12
    relative_slopes = np.abs(slopes) / signal_std
    profile.drift_severity = float(np.median(relative_slopes))
    # Threshold: relative slope > 0.5 means drift is substantial relative to signal
    profile.has_significant_drift = profile.drift_severity > 0.5


def _analyze_channels(data: np.ndarray, channels: List[str], profile: DataProfile) -> None:
    """Assess per-channel quality: consistency, flat detection, bad identification."""
    n_channels, n_samples = data.shape

    channel_stds = np.std(data, axis=1)
    channel_means = np.mean(np.abs(data), axis=1)

    # Flat channels: std < 1% of median std
    median_std = np.median(channel_stds)
    if median_std > 0:
        flat_threshold = median_std * 0.01
        flat_mask = channel_stds < flat_threshold
    else:
        flat_mask = np.ones(n_channels, dtype=bool)

    n_flat = int(flat_mask.sum())
    profile.flat_channel_ratio = n_flat / n_channels if n_channels > 0 else 0.0

    # Bad channels: outlier variance (>5x or <0.1x median)
    bad_mask = flat_mask.copy()
    if median_std > 0:
        bad_mask |= (channel_stds > median_std * 5)
        bad_mask |= (channel_stds < median_std * 0.1)

    bad_indices = np.where(bad_mask)[0]
    profile.n_bad_channels = len(bad_indices)
    profile.bad_channel_names = [channels[i] for i in bad_indices[:20]]

    # Channel consistency: inverse coefficient of variation of channel stds
    # High consistency = channels have similar variance (good)
    if median_std > 0 and n_channels > 1:
        cv = np.std(channel_stds) / median_std
        profile.channel_consistency = float(max(0.0, 1.0 - min(cv, 3.0) / 3.0))
    else:
        profile.channel_consistency = 0.0


def _analyze_artifacts(data: np.ndarray, fs: float, profile: DataProfile) -> None:
    """Detect artifact contamination level."""
    n_channels, n_samples = data.shape

    # Amplitude analysis
    abs_data = np.abs(data)
    median_amp = float(np.median(abs_data))
    profile.median_amplitude = median_amp

    # Dynamic range
    p01 = np.percentile(data, 0.1)
    p999 = np.percentile(data, 99.9)
    dynamic_range = p999 - p01
    profile.dynamic_range_db = float(20 * np.log10(dynamic_range + 1e-20) - 20 * np.log10(median_amp + 1e-20)) if median_amp > 0 else 0.0

    # Extreme amplitudes: samples exceeding 10x median absolute amplitude
    threshold = median_amp * 10 if median_amp > 0 else np.inf
    extreme_mask = abs_data > threshold
    profile.has_extreme_amplitudes = bool(extreme_mask.any())

    # Artifact ratio: fraction of time windows containing extreme events
    window_samples = max(1, int(fs * 0.5))  # 500ms windows
    n_windows = max(1, n_samples // window_samples)
    artifact_windows = 0
    for w in range(n_windows):
        start = w * window_samples
        end = min(start + window_samples, n_samples)
        window = abs_data[:, start:end]
        if np.any(window > threshold):
            artifact_windows += 1
    profile.artifact_ratio = artifact_windows / n_windows


def _compute_summary_scores(profile: DataProfile) -> None:
    """Compute composite noise and quality scores (0-1)."""
    # Noise score: weighted combination of problems
    noise_components = []

    if profile.powerline_present:
        # Stronger powerline = higher noise contribution
        pl_score = min(1.0, profile.powerline_amplitude_db / 30.0)
        noise_components.append(pl_score * 0.2)
    else:
        noise_components.append(0.0)

    noise_components.append(min(1.0, profile.artifact_ratio) * 0.3)
    noise_components.append(min(1.0, profile.drift_severity) * 0.15)
    noise_components.append((1.0 - profile.channel_consistency) * 0.2)
    noise_components.append(min(1.0, profile.flat_channel_ratio * 5) * 0.15)

    profile.noise_score = min(1.0, sum(noise_components))
    profile.quality_score = max(0.0, 1.0 - profile.noise_score)
