"""Enhanced QC metrics — quantitative signal quality assessment.

Computes before/after preprocessing metrics to quantify:
- SNR per frequency band (dB improvement)
- Artifact residual rate (% epochs still contaminated)
- Information retention (ERP waveform correlation)
- Cross-channel consistency index (variance distribution change)

All metrics return numeric scores suitable for automated pass/fail decisions
and for inclusion in QC reports.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class QCMetrics:
    """Comprehensive QC metrics comparing pre- and post-processing signals."""

    # --- SNR ---
    snr_before: Dict[str, float] = field(default_factory=dict)
    snr_after: Dict[str, float] = field(default_factory=dict)
    snr_improvement_db: Dict[str, float] = field(default_factory=dict)

    # --- Artifact residual ---
    artifact_residual_ratio: float = 0.0
    artifact_epochs_before: int = 0
    artifact_epochs_after: int = 0
    total_epochs: int = 0

    # --- Information retention ---
    waveform_correlation: float = 1.0
    variance_retention: float = 1.0
    spectral_correlation: float = 1.0

    # --- Cross-channel consistency ---
    consistency_before: float = 0.0
    consistency_after: float = 0.0
    consistency_improvement: float = 0.0

    # --- Overall assessment ---
    overall_score: float = 1.0
    grade: str = "A"
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "snr": {
                "before": {k: round(v, 1) for k, v in self.snr_before.items()},
                "after": {k: round(v, 1) for k, v in self.snr_after.items()},
                "improvement_db": {k: round(v, 1) for k, v in self.snr_improvement_db.items()},
            },
            "artifact_residual": {
                "ratio": round(self.artifact_residual_ratio, 4),
                "epochs_before": self.artifact_epochs_before,
                "epochs_after": self.artifact_epochs_after,
                "total_epochs": self.total_epochs,
            },
            "information_retention": {
                "waveform_correlation": round(self.waveform_correlation, 4),
                "variance_retention": round(self.variance_retention, 4),
                "spectral_correlation": round(self.spectral_correlation, 4),
            },
            "channel_consistency": {
                "before": round(self.consistency_before, 4),
                "after": round(self.consistency_after, 4),
                "improvement": round(self.consistency_improvement, 4),
            },
            "overall": {
                "score": round(self.overall_score, 3),
                "grade": self.grade,
                "warnings": self.warnings,
            },
        }


_STANDARD_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 100.0),
}


def compute_qc_metrics(
    before: np.ndarray,
    after: np.ndarray,
    frequency_before: float,
    frequency_after: float,
    channels_before: Optional[List[str]] = None,
    channels_after: Optional[List[str]] = None,
) -> QCMetrics:
    """Compute comprehensive QC metrics comparing before/after preprocessing.

    Parameters
    ----------
    before : ndarray, shape (n_channels, n_samples)
        Raw data before preprocessing.
    after : ndarray, shape (n_channels, n_samples)
        Processed data after preprocessing.
    frequency_before : float
        Sampling rate before processing.
    frequency_after : float
        Sampling rate after processing (may differ if resampled).
    channels_before, channels_after : list of str, optional
        Channel names (for alignment when channels were dropped).

    Returns
    -------
    QCMetrics with all fields populated.
    """
    metrics = QCMetrics()

    before_2d = _ensure_2d(before)
    after_2d = _ensure_2d(after)

    _compute_snr(before_2d, after_2d, frequency_before, frequency_after, metrics)
    _compute_artifact_residual(after_2d, frequency_after, metrics, before_2d, frequency_before)
    _compute_information_retention(before_2d, after_2d, frequency_before, frequency_after, metrics)
    _compute_channel_consistency(before_2d, after_2d, metrics)
    _compute_overall(metrics)

    return metrics


def _ensure_2d(data: np.ndarray) -> np.ndarray:
    if data.ndim == 1:
        return data.reshape(1, -1)
    if data.ndim == 3:
        n_seg, n_ch, n_t = data.shape
        return data.reshape(n_seg * n_ch, n_t)
    return data


def _compute_snr(
    before: np.ndarray,
    after: np.ndarray,
    fs_before: float,
    fs_after: float,
    metrics: QCMetrics,
) -> None:
    """Compute per-band SNR before and after processing."""
    metrics.snr_before = _band_snr(before, fs_before)
    metrics.snr_after = _band_snr(after, fs_after)

    for band in metrics.snr_before:
        if band in metrics.snr_after:
            metrics.snr_improvement_db[band] = metrics.snr_after[band] - metrics.snr_before[band]


def _band_snr(data: np.ndarray, fs: float) -> Dict[str, float]:
    """Estimate SNR per frequency band using PSD analysis."""
    n_channels, n_samples = data.shape
    nyquist = fs / 2.0
    n_fft = min(n_samples, int(fs * 4))
    if n_fft < 64:
        return {}

    from numpy.fft import rfft, rfftfreq
    freqs = rfftfreq(n_fft, 1.0 / fs)

    segment = data[:, :n_fft]
    windowed = segment * np.hanning(n_fft)
    fft_mag = np.abs(rfft(windowed, axis=1)) ** 2
    psd_median = np.median(fft_mag, axis=0)
    psd_db = 10 * np.log10(psd_median + 1e-20)

    valid_mask = (freqs >= 0.5) & (freqs <= min(100.0, nyquist - 1))
    noise_floor = np.percentile(psd_db[valid_mask], 10) if valid_mask.any() else 0

    result = {}
    for band_name, (f_low, f_high) in _STANDARD_BANDS.items():
        if f_low >= nyquist:
            continue
        band_mask = (freqs >= f_low) & (freqs <= min(f_high, nyquist))
        if band_mask.any():
            band_power = float(np.mean(psd_db[band_mask]))
            result[band_name] = band_power - noise_floor
    return result


def _compute_artifact_residual(
    after: np.ndarray,
    fs_after: float,
    metrics: QCMetrics,
    before: np.ndarray,
    fs_before: float,
) -> None:
    """Compute artifact contamination ratio in processed data."""
    window_samples = max(1, int(fs_after * 0.5))
    n_channels, n_samples = after.shape
    n_windows = max(1, n_samples // window_samples)

    abs_after = np.abs(after)
    median_amp = np.median(abs_after)
    threshold = median_amp * 8 if median_amp > 0 else np.inf

    artifact_windows_after = 0
    for w in range(n_windows):
        start = w * window_samples
        end = min(start + window_samples, n_samples)
        if np.any(abs_after[:, start:end] > threshold):
            artifact_windows_after += 1

    # Also count before for comparison
    window_samples_b = max(1, int(fs_before * 0.5))
    n_samples_b = before.shape[1]
    n_windows_b = max(1, n_samples_b // window_samples_b)
    abs_before = np.abs(before)
    median_amp_b = np.median(abs_before)
    threshold_b = median_amp_b * 8 if median_amp_b > 0 else np.inf

    artifact_windows_before = 0
    for w in range(n_windows_b):
        start = w * window_samples_b
        end = min(start + window_samples_b, n_samples_b)
        if np.any(abs_before[:, start:end] > threshold_b):
            artifact_windows_before += 1

    metrics.artifact_epochs_before = artifact_windows_before
    metrics.artifact_epochs_after = artifact_windows_after
    metrics.total_epochs = n_windows
    metrics.artifact_residual_ratio = artifact_windows_after / n_windows if n_windows > 0 else 0.0


def _compute_information_retention(
    before: np.ndarray,
    after: np.ndarray,
    fs_before: float,
    fs_after: float,
    metrics: QCMetrics,
) -> None:
    """Quantify how much useful signal is retained after processing.

    Uses three complementary measures:
    - Waveform correlation (time domain)
    - Variance retention ratio
    - Spectral shape correlation
    """
    n_ch_before = before.shape[0]
    n_ch_after = after.shape[0]
    n_ch = min(n_ch_before, n_ch_after)

    # Align sample counts (if resampled, compare at same effective duration)
    dur_before = before.shape[1] / fs_before if fs_before > 0 else 0
    dur_after = after.shape[1] / fs_after if fs_after > 0 else 0
    common_dur = min(dur_before, dur_after)

    if common_dur <= 0:
        return

    # Resample shorter to match (simple decimation for comparison)
    n_samp_b = min(before.shape[1], int(common_dur * fs_before))
    n_samp_a = min(after.shape[1], int(common_dur * fs_after))

    # Use common time resolution for comparison
    n_compare = min(n_samp_b, n_samp_a, 10000)
    indices_b = np.linspace(0, n_samp_b - 1, n_compare, dtype=int)
    indices_a = np.linspace(0, n_samp_a - 1, n_compare, dtype=int)

    b_aligned = before[:n_ch, :][:, indices_b]
    a_aligned = after[:n_ch, :][:, indices_a]

    # Waveform correlation (per-channel, then median)
    correlations = []
    for ch in range(n_ch):
        b_ch = b_aligned[ch] - np.mean(b_aligned[ch])
        a_ch = a_aligned[ch] - np.mean(a_aligned[ch])
        norm_b = np.linalg.norm(b_ch)
        norm_a = np.linalg.norm(a_ch)
        if norm_b > 1e-12 and norm_a > 1e-12:
            corr = float(np.dot(b_ch, a_ch) / (norm_b * norm_a))
            correlations.append(max(0.0, corr))
    metrics.waveform_correlation = float(np.median(correlations)) if correlations else 1.0

    # Variance retention
    var_before = np.var(b_aligned, axis=1)
    var_after = np.var(a_aligned, axis=1)
    valid = var_before > 1e-20
    if valid.any():
        ratios = var_after[valid] / var_before[valid]
        metrics.variance_retention = float(np.median(np.clip(ratios, 0, 2)))
    else:
        metrics.variance_retention = 1.0

    # Spectral shape correlation
    metrics.spectral_correlation = _spectral_shape_correlation(
        before[:n_ch, :n_samp_b], after[:n_ch, :n_samp_a], fs_before, fs_after
    )


def _spectral_shape_correlation(
    before: np.ndarray, after: np.ndarray, fs_b: float, fs_a: float
) -> float:
    """Correlation of PSD shapes (normalized) between before and after."""
    from numpy.fft import rfft, rfftfreq

    n_fft_b = min(before.shape[1], int(fs_b * 2))
    n_fft_a = min(after.shape[1], int(fs_a * 2))
    if n_fft_b < 32 or n_fft_a < 32:
        return 1.0

    psd_b = np.median(np.abs(rfft(before[:, :n_fft_b] * np.hanning(n_fft_b), axis=1)) ** 2, axis=0)
    psd_a = np.median(np.abs(rfft(after[:, :n_fft_a] * np.hanning(n_fft_a), axis=1)) ** 2, axis=0)

    # Interpolate to common frequency axis
    n_common = min(len(psd_b), len(psd_a), 100)
    psd_b_interp = np.interp(np.linspace(0, 1, n_common), np.linspace(0, 1, len(psd_b)), psd_b)
    psd_a_interp = np.interp(np.linspace(0, 1, n_common), np.linspace(0, 1, len(psd_a)), psd_a)

    # Normalize and correlate
    psd_b_norm = psd_b_interp / (np.linalg.norm(psd_b_interp) + 1e-20)
    psd_a_norm = psd_a_interp / (np.linalg.norm(psd_a_interp) + 1e-20)

    return float(max(0.0, np.dot(psd_b_norm, psd_a_norm)))


def _compute_channel_consistency(
    before: np.ndarray, after: np.ndarray, metrics: QCMetrics
) -> None:
    """Measure cross-channel variance consistency improvement."""
    std_before = np.std(before, axis=1)
    std_after = np.std(after, axis=1)

    median_b = np.median(std_before)
    median_a = np.median(std_after)

    if median_b > 0 and before.shape[0] > 1:
        cv_b = np.std(std_before) / median_b
        metrics.consistency_before = float(max(0.0, 1.0 - min(cv_b, 3.0) / 3.0))
    if median_a > 0 and after.shape[0] > 1:
        cv_a = np.std(std_after) / median_a
        metrics.consistency_after = float(max(0.0, 1.0 - min(cv_a, 3.0) / 3.0))

    metrics.consistency_improvement = metrics.consistency_after - metrics.consistency_before


def _compute_overall(metrics: QCMetrics) -> None:
    """Compute overall score and grade from component metrics."""
    scores = []
    warnings = []

    # SNR improvement (positive is good)
    snr_vals = list(metrics.snr_improvement_db.values())
    if snr_vals:
        avg_snr_improvement = np.mean(snr_vals)
        snr_score = np.clip(0.5 + avg_snr_improvement / 20.0, 0, 1)
        scores.append(snr_score * 0.25)
        if avg_snr_improvement < -3:
            warnings.append(f"SNR decreased by {abs(avg_snr_improvement):.1f} dB on average")

    # Artifact residual (lower is better)
    artifact_score = 1.0 - min(1.0, metrics.artifact_residual_ratio * 5)
    scores.append(artifact_score * 0.25)
    if metrics.artifact_residual_ratio > 0.2:
        warnings.append(f"{metrics.artifact_residual_ratio*100:.0f}% of epochs still contain artifacts")

    # Information retention (higher is better, but some loss is expected)
    retention_score = (
        metrics.waveform_correlation * 0.4
        + min(1.0, metrics.variance_retention) * 0.3
        + metrics.spectral_correlation * 0.3
    )
    scores.append(retention_score * 0.30)
    if metrics.waveform_correlation < 0.5:
        warnings.append(f"Low waveform retention ({metrics.waveform_correlation:.2f}) — signal may be over-filtered")

    # Channel consistency improvement
    consistency_score = max(0, min(1.0, 0.5 + metrics.consistency_improvement))
    scores.append(consistency_score * 0.20)

    metrics.overall_score = sum(scores) if scores else 0.5
    metrics.warnings = warnings

    # Grade assignment
    s = metrics.overall_score
    if s >= 0.85:
        metrics.grade = "A"
    elif s >= 0.70:
        metrics.grade = "B"
    elif s >= 0.55:
        metrics.grade = "C"
    elif s >= 0.40:
        metrics.grade = "D"
    else:
        metrics.grade = "F"
