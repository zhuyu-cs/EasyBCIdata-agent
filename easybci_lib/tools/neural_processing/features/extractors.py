"""Feature extraction implementations for neural data.

Each extractor operates on epoched data (3D: n_epochs x n_channels x n_samples)
and produces an ML-ready FeatureResult (X matrix + optional labels + metadata).

Pipeline step format:
  extract_psd_bands:delta,theta,alpha,beta,gamma
  extract_csp:n_components=6
  extract_tfr:method=morlet,freqs=4-40
  extract_connectivity:method=plv,bands=alpha,beta
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


@dataclass
class FeatureResult:
    """ML-ready feature extraction output.

    Compatible with sklearn (X, y = result.X, result.y) and
    PyTorch (torch.from_numpy(result.X)).
    """
    X: np.ndarray  # (n_samples, n_features) or (n_samples, n_channels, n_timepoints)
    y: Optional[np.ndarray] = None  # (n_samples,) labels
    feature_names: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "X_shape": list(self.X.shape),
            "n_samples": self.X.shape[0],
            "n_features": self.X.shape[1] if self.X.ndim == 2 else self.X.shape[1:],
            "feature_names": self.feature_names[:50],
            "has_labels": self.y is not None,
        }
        if self.y is not None:
            unique_labels = np.unique(self.y)
            d["n_classes"] = len(unique_labels)
            d["label_counts"] = {str(l): int((self.y == l).sum()) for l in unique_labels[:20]}
        d.update(self.metadata)
        return d


def extract_psd_bands(
    data: np.ndarray,
    frequency: float,
    bands: Optional[List[str]] = None,
    labels: Optional[np.ndarray] = None,
    channels: Optional[List[str]] = None,
) -> FeatureResult:
    """Extract power spectral density features per frequency band.

    Parameters
    ----------
    data : ndarray
        Shape (n_epochs, n_channels, n_samples) for epoched data,
        or (n_channels, n_samples) for continuous (treated as single epoch).
    frequency : float
        Sampling rate in Hz.
    bands : list of str, optional
        Which bands to extract. Default: all standard bands.
    labels : ndarray, optional
        Per-epoch labels (n_epochs,).
    channels : list of str, optional
        Channel names for feature naming.

    Returns
    -------
    FeatureResult with X shape (n_epochs, n_channels * n_bands).
    """
    if data.ndim == 2:
        data = data[np.newaxis, :, :]

    n_epochs, n_channels, n_samples = data.shape

    if bands is None:
        bands = list(_STANDARD_BANDS.keys())
    band_ranges = [(b, _STANDARD_BANDS[b]) for b in bands if b in _STANDARD_BANDS]

    if channels is None:
        channels = [f"Ch{i}" for i in range(n_channels)]

    nyquist = frequency / 2.0
    n_fft = min(n_samples, int(frequency * 2))
    if n_fft < 16:
        n_fft = n_samples

    from numpy.fft import rfft, rfftfreq
    freqs = rfftfreq(n_fft, 1.0 / frequency)

    n_features = n_channels * len(band_ranges)
    X = np.zeros((n_epochs, n_features))
    feature_names = []

    for band_idx, (band_name, (f_low, f_high)) in enumerate(band_ranges):
        f_high_eff = min(f_high, nyquist - 0.5)
        if f_low >= nyquist:
            continue
        band_mask = (freqs >= f_low) & (freqs <= f_high_eff)

        for epoch_idx in range(n_epochs):
            segment = data[epoch_idx, :, :n_fft]
            windowed = segment * np.hanning(n_fft)
            psd = np.abs(rfft(windowed, axis=1)) ** 2

            band_power = np.mean(psd[:, band_mask], axis=1) if band_mask.any() else np.zeros(n_channels)
            # Log power for better ML scaling
            band_power_db = 10 * np.log10(band_power + 1e-20)

            col_start = band_idx * n_channels
            X[epoch_idx, col_start:col_start + n_channels] = band_power_db

        for ch_name in channels:
            feature_names.append(f"{ch_name}_{band_name}_power_db")

    return FeatureResult(
        X=X,
        y=labels,
        feature_names=feature_names,
        metadata={"method": "psd_bands", "bands": bands, "frequency": frequency},
    )


def extract_csp(
    data: np.ndarray,
    labels: np.ndarray,
    n_components: int = 6,
    frequency: float = 0.0,
    channels: Optional[List[str]] = None,
) -> FeatureResult:
    """Extract Common Spatial Pattern features.

    CSP finds spatial filters that maximize variance difference between classes.
    Requires binary or multi-class labels.

    Parameters
    ----------
    data : ndarray, shape (n_epochs, n_channels, n_samples)
        Epoched data.
    labels : ndarray, shape (n_epochs,)
        Class labels (required for CSP).
    n_components : int
        Number of CSP components (filters from both ends).
    frequency : float
        Sampling rate (for metadata only).

    Returns
    -------
    FeatureResult with X shape (n_epochs, n_components) — log-variance features.
    """
    if data.ndim != 3:
        import logging
        logging.getLogger(__name__).warning(
            "CSP requires 3D epoched data (n_epochs, n_ch, n_samples), got shape %s. "
            "Returning empty feature result.", data.shape,
        )
        from easybci_lib.tools.neural_processing.features.extractors import FeatureResult
        return FeatureResult(
            X=np.zeros((0, n_components)),
            y=np.array([]) if labels is None else labels,
            feature_names=[f"csp_{i}" for i in range(n_components)],
        )
    if labels is None or len(labels) != data.shape[0]:
        import logging
        logging.getLogger(__name__).warning(
            "CSP requires labels array matching n_epochs (%d). "
            "Returning empty feature result.", data.shape[0],
        )
        from easybci_lib.tools.neural_processing.features.extractors import FeatureResult
        return FeatureResult(
            X=np.zeros((data.shape[0], n_components)),
            y=np.zeros(data.shape[0]),
            feature_names=[f"csp_{i}" for i in range(n_components)],
        )

    n_epochs, n_channels, n_samples = data.shape
    n_components = min(n_components, n_channels)

    unique_classes = np.unique(labels)
    if len(unique_classes) < 2:
        import logging
        logging.getLogger(__name__).warning(
            "CSP requires at least 2 classes, got %d. Returning variance features as fallback.",
            len(unique_classes),
        )
        var_features = np.log(np.var(data, axis=2) + 1e-10)[:, :n_components]
        from easybci_lib.tools.neural_processing.features.extractors import FeatureResult
        return FeatureResult(
            X=var_features,
            y=labels,
            feature_names=[f"logvar_ch{i}" for i in range(var_features.shape[1])],
        )

    # Compute per-class covariance matrices
    covs_per_class = {}
    for cls in unique_classes:
        cls_mask = labels == cls
        cls_data = data[cls_mask]
        # Average covariance across epochs
        covs = []
        for epoch in cls_data:
            cov = np.cov(epoch)
            covs.append(cov / np.trace(cov))
        covs_per_class[cls] = np.mean(covs, axis=0)

    # Binary CSP: generalized eigenvalue problem
    if len(unique_classes) == 2:
        c1, c2 = unique_classes
        cov1 = covs_per_class[c1]
        cov2 = covs_per_class[c2]
        composite = cov1 + cov2

        # Solve generalized eigenvalue problem
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(np.linalg.solve(composite, cov1))
        except np.linalg.LinAlgError:
            # Regularize if singular
            reg = np.eye(n_channels) * 1e-6
            eigenvalues, eigenvectors = np.linalg.eigh(
                np.linalg.solve(composite + reg, cov1 + reg)
            )

        # Sort by eigenvalue (descending)
        sort_idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, sort_idx]

        # Select top and bottom components
        n_half = n_components // 2
        selected = np.concatenate([
            eigenvectors[:, :n_half],
            eigenvectors[:, -n_half:],
        ], axis=1)
        W = selected[:, :n_components]

    else:
        # Multi-class: one-vs-rest approach, take top components per class
        all_filters = []
        composite = sum(covs_per_class.values())
        for cls in unique_classes:
            cov_cls = covs_per_class[cls]
            try:
                eigenvalues, eigenvectors = np.linalg.eigh(np.linalg.solve(composite, cov_cls))
            except np.linalg.LinAlgError:
                reg = np.eye(n_channels) * 1e-6
                eigenvalues, eigenvectors = np.linalg.eigh(
                    np.linalg.solve(composite + reg, cov_cls + reg)
                )
            sort_idx = np.argsort(eigenvalues)[::-1]
            all_filters.append(eigenvectors[:, sort_idx[:2]])

        W = np.concatenate(all_filters, axis=1)[:, :n_components]

    # Apply spatial filters and compute log-variance features
    X = np.zeros((n_epochs, n_components))
    for epoch_idx in range(n_epochs):
        filtered = W.T @ data[epoch_idx]
        # Log-variance as feature
        X[epoch_idx] = np.log(np.var(filtered, axis=1) + 1e-20)

    feature_names = [f"csp_component_{i}" for i in range(n_components)]

    return FeatureResult(
        X=X,
        y=labels,
        feature_names=feature_names,
        metadata={
            "method": "csp",
            "n_components": n_components,
            "n_classes": len(unique_classes),
            "spatial_filters_shape": list(W.shape),
        },
    )


def extract_tfr(
    data: np.ndarray,
    frequency: float,
    method: str = "morlet",
    freq_range: Tuple[float, float] = (4.0, 40.0),
    n_freqs: int = 20,
    labels: Optional[np.ndarray] = None,
    channels: Optional[List[str]] = None,
) -> FeatureResult:
    """Extract time-frequency representation features.

    Parameters
    ----------
    data : ndarray
        Shape (n_epochs, n_channels, n_samples) or (n_channels, n_samples).
    frequency : float
        Sampling rate in Hz.
    method : str
        "morlet" (default) or "stft".
    freq_range : tuple of (low, high)
        Frequency range to analyze.
    n_freqs : int
        Number of frequency bins.
    labels : ndarray, optional
        Per-epoch labels.

    Returns
    -------
    FeatureResult with X shape (n_epochs, n_channels * n_freqs) — average power per band.
    """
    if data.ndim == 2:
        data = data[np.newaxis, :, :]

    n_epochs, n_channels, n_samples = data.shape
    f_low, f_high = freq_range
    f_high = min(f_high, frequency / 2.0 - 0.5)
    target_freqs = np.linspace(f_low, f_high, n_freqs)

    if method == "morlet":
        X = _morlet_tfr(data, frequency, target_freqs)
    else:
        X = _stft_tfr(data, frequency, target_freqs)

    # Flatten to (n_epochs, n_channels * n_freqs) for ML
    X_flat = X.reshape(n_epochs, -1)

    if channels is None:
        channels = [f"Ch{i}" for i in range(n_channels)]

    feature_names = []
    for ch in channels:
        for freq in target_freqs:
            feature_names.append(f"{ch}_{freq:.1f}Hz_power")

    return FeatureResult(
        X=X_flat,
        y=labels,
        feature_names=feature_names,
        metadata={
            "method": f"tfr_{method}",
            "freq_range": list(freq_range),
            "n_freqs": n_freqs,
            "tfr_shape": f"({n_epochs}, {n_channels}, {n_freqs})",
        },
    )


def _morlet_tfr(data: np.ndarray, fs: float, freqs: np.ndarray) -> np.ndarray:
    """Compute Morlet wavelet TFR — average power per frequency."""
    n_epochs, n_channels, n_samples = data.shape
    n_freqs = len(freqs)
    power = np.zeros((n_epochs, n_channels, n_freqs))

    t = np.arange(n_samples) / fs

    for f_idx, freq in enumerate(freqs):
        # Morlet wavelet: Gaussian-windowed complex sinusoid
        n_cycles = max(3, int(freq / 2))
        sigma_t = n_cycles / (2 * np.pi * freq)
        wavelet_duration = 3 * sigma_t
        wavelet_samples = min(int(wavelet_duration * fs * 2), n_samples)
        t_wav = np.arange(wavelet_samples) / fs - wavelet_samples / (2 * fs)

        wavelet = np.exp(2j * np.pi * freq * t_wav) * np.exp(-t_wav**2 / (2 * sigma_t**2))
        wavelet /= np.sqrt(np.sum(np.abs(wavelet)**2))

        for ep_idx in range(n_epochs):
            for ch_idx in range(n_channels):
                # Convolution for time-frequency decomposition
                conv = np.convolve(data[ep_idx, ch_idx], wavelet, mode='same')
                power[ep_idx, ch_idx, f_idx] = np.mean(np.abs(conv)**2)

    return power


def _stft_tfr(data: np.ndarray, fs: float, freqs: np.ndarray) -> np.ndarray:
    """Compute STFT-based TFR — average power per frequency band."""
    from numpy.fft import rfft, rfftfreq

    n_epochs, n_channels, n_samples = data.shape
    n_freqs = len(freqs)
    power = np.zeros((n_epochs, n_channels, n_freqs))

    window_size = min(n_samples, int(fs * 0.5))
    if window_size < 16:
        window_size = n_samples
    n_windows = max(1, n_samples // window_size)
    fft_freqs = rfftfreq(window_size, 1.0 / fs)

    for ep_idx in range(n_epochs):
        for ch_idx in range(n_channels):
            psd_accum = np.zeros(len(fft_freqs))
            for w in range(n_windows):
                start = w * window_size
                segment = data[ep_idx, ch_idx, start:start + window_size]
                if len(segment) < window_size:
                    break
                windowed = segment * np.hanning(window_size)
                psd_accum += np.abs(rfft(windowed))**2
            psd_accum /= max(1, n_windows)

            # Assign to target frequency bins
            for f_idx, target_f in enumerate(freqs):
                closest_idx = np.argmin(np.abs(fft_freqs - target_f))
                # Average over a small band around target frequency
                bw = max(1, int(2 * fs / window_size))
                low = max(0, closest_idx - bw)
                high = min(len(psd_accum), closest_idx + bw + 1)
                power[ep_idx, ch_idx, f_idx] = np.mean(psd_accum[low:high])

    return power


def extract_connectivity(
    data: np.ndarray,
    frequency: float,
    method: str = "plv",
    bands: Optional[List[str]] = None,
    labels: Optional[np.ndarray] = None,
    channels: Optional[List[str]] = None,
) -> FeatureResult:
    """Extract functional connectivity features.

    Computes pairwise connectivity between channels.

    Parameters
    ----------
    data : ndarray
        Shape (n_epochs, n_channels, n_samples) or (n_channels, n_samples).
    frequency : float
        Sampling rate in Hz.
    method : str
        "plv" (Phase Locking Value), "coh" (coherence), or "corr" (correlation).
    bands : list of str, optional
        Frequency bands to compute connectivity in. Default: ["alpha", "beta"].
    labels : ndarray, optional
        Per-epoch labels.

    Returns
    -------
    FeatureResult with X shape (n_epochs, n_pairs * n_bands).
    """
    if data.ndim == 2:
        data = data[np.newaxis, :, :]

    n_epochs, n_channels, n_samples = data.shape

    if bands is None:
        bands = ["alpha", "beta"]
    band_ranges = [(b, _STANDARD_BANDS[b]) for b in bands if b in _STANDARD_BANDS]

    if channels is None:
        channels = [f"Ch{i}" for i in range(n_channels)]

    # Number of channel pairs (upper triangle)
    n_pairs = n_channels * (n_channels - 1) // 2
    n_features = n_pairs * len(band_ranges)
    X = np.zeros((n_epochs, n_features))
    feature_names = []

    pair_indices = []
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            pair_indices.append((i, j))

    for band_idx, (band_name, (f_low, f_high)) in enumerate(band_ranges):
        for ep_idx in range(n_epochs):
            if method == "plv":
                conn = _compute_plv(data[ep_idx], frequency, f_low, f_high, pair_indices)
            elif method == "coh":
                conn = _compute_coherence(data[ep_idx], frequency, f_low, f_high, pair_indices)
            else:
                conn = _compute_correlation(data[ep_idx], pair_indices)

            col_start = band_idx * n_pairs
            X[ep_idx, col_start:col_start + n_pairs] = conn

        for (i, j) in pair_indices:
            feature_names.append(f"{channels[i]}-{channels[j]}_{band_name}_{method}")

    return FeatureResult(
        X=X,
        y=labels,
        feature_names=feature_names,
        metadata={
            "method": f"connectivity_{method}",
            "bands": bands,
            "n_pairs": n_pairs,
            "n_channels": n_channels,
        },
    )


def _compute_plv(
    data: np.ndarray, fs: float, f_low: float, f_high: float,
    pair_indices: List[Tuple[int, int]],
) -> np.ndarray:
    """Phase Locking Value between channel pairs in a frequency band."""
    n_channels, n_samples = data.shape
    nyquist = fs / 2.0
    f_high = min(f_high, nyquist - 0.5)

    # Bandpass via FFT
    from numpy.fft import fft, ifft, fftfreq
    freqs = fftfreq(n_samples, 1.0 / fs)
    band_mask = (np.abs(freqs) >= f_low) & (np.abs(freqs) <= f_high)

    # Get analytic signal (phase) for each channel in band
    phases = np.zeros((n_channels, n_samples))
    for ch in range(n_channels):
        spectrum = fft(data[ch])
        spectrum[~band_mask] = 0
        analytic = ifft(spectrum)
        phases[ch] = np.angle(analytic)

    # PLV for each pair
    plv = np.zeros(len(pair_indices))
    for idx, (i, j) in enumerate(pair_indices):
        phase_diff = phases[i] - phases[j]
        plv[idx] = np.abs(np.mean(np.exp(1j * phase_diff)))

    return plv


def _compute_coherence(
    data: np.ndarray, fs: float, f_low: float, f_high: float,
    pair_indices: List[Tuple[int, int]],
) -> np.ndarray:
    """Magnitude-squared coherence between channel pairs."""
    from numpy.fft import rfft, rfftfreq

    n_channels, n_samples = data.shape
    n_fft = min(n_samples, int(fs * 2))
    freqs = rfftfreq(n_fft, 1.0 / fs)
    band_mask = (freqs >= f_low) & (freqs <= min(f_high, fs / 2 - 0.5))

    # Compute FFT for each channel
    spectra = np.zeros((n_channels, len(freqs)), dtype=complex)
    segment = data[:, :n_fft] * np.hanning(n_fft)
    for ch in range(n_channels):
        spectra[ch] = rfft(segment[ch])

    coh = np.zeros(len(pair_indices))
    for idx, (i, j) in enumerate(pair_indices):
        if not band_mask.any():
            continue
        Sxy = spectra[i][band_mask] * np.conj(spectra[j][band_mask])
        Sxx = np.abs(spectra[i][band_mask])**2
        Syy = np.abs(spectra[j][band_mask])**2
        denom = np.sqrt(np.mean(Sxx) * np.mean(Syy))
        if denom > 1e-20:
            coh[idx] = np.abs(np.mean(Sxy)) / denom

    return coh


def _compute_correlation(
    data: np.ndarray, pair_indices: List[Tuple[int, int]]
) -> np.ndarray:
    """Pearson correlation between channel pairs."""
    corr = np.zeros(len(pair_indices))
    for idx, (i, j) in enumerate(pair_indices):
        r = np.corrcoef(data[i], data[j])[0, 1]
        corr[idx] = r if np.isfinite(r) else 0.0
    return corr
