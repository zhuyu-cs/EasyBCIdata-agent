"""Interactive QC visualization — Plotly-based dashboard + topomap + spectral.

Generates interactive plots as Plotly JSON for the Web UI (rendered via Plotly.js).
Falls back to matplotlib static PNGs if plotly is not installed.

Includes:
- Interactive PSD (hover per channel, zoom/pan)
- Interactive time series (zoom, channel toggle)
- Channel variance (clickable bars)
- Amplitude distribution
- Topographic map (band power spatial distribution)
- Spectral analysis (per-channel band power breakdown)
"""

import base64
import io
import logging
from typing import Dict, List, Optional

import numpy as np

from easybci_lib.tools.neural_processing._seed import EASYBCI_SEED

logger = logging.getLogger(__name__)

_BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 100),
}

_HAS_PLOTLY = None


def _check_plotly():
    global _HAS_PLOTLY
    if _HAS_PLOTLY is None:
        import importlib.util
        _HAS_PLOTLY = importlib.util.find_spec("plotly") is not None
    return _HAS_PLOTLY


def generate_interactive_qc(
    data: np.ndarray,
    frequency: float,
    channels: Optional[List[str]] = None,
    max_channels_display: int = 32,
) -> Dict[str, str]:
    """Generate interactive QC figures as Plotly JSON strings.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_samples)
    frequency : float
    channels : list of str
    max_channels_display : int

    Returns
    -------
    dict mapping figure_name → Plotly JSON string (or base64 PNG if no plotly)
    """
    if data.ndim == 3:
        data = data.mean(axis=0)

    n_ch = min(data.shape[0], max_channels_display)
    if channels is None:
        channels = [f"Ch{i}" for i in range(data.shape[0])]

    figures = {}

    if _check_plotly():
        figures["psd_plotly"] = _plotly_psd(data[:n_ch], frequency, channels[:n_ch])
        figures["timeseries_plotly"] = _plotly_timeseries(data[:n_ch], frequency, channels[:n_ch])
        figures["variance_plotly"] = _plotly_variance(data, channels)
        figures["amplitude_plotly"] = _plotly_amplitude(data)
    else:
        from easybci_lib.tools.neural_processing.quality.final_view import FinalDataView
        from easybci_lib.tools.neural_processing.quality.visualize import generate_qc_figures
        view = FinalDataView.from_pipeline_result(
            after_data=data,
            channels=channels or [f"Ch{i}" for i in range(data.shape[0])],
            frequency=frequency,
            modality="eeg",  # interactive_qc has no modality info; safe default
            enforce_data_only=False,  # interactive panel respects caller's choices
        )
        return generate_qc_figures(view, max_channels_display=max_channels_display)

    return figures


def generate_topomap(
    data: np.ndarray,
    frequency: float,
    channels: List[str],
    positions: Optional[np.ndarray] = None,
    band: str = "alpha",
) -> str:
    """Generate a topographic map of band power distribution.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_samples)
    frequency : float
    channels : list of str
    positions : ndarray shape (n_channels, 3) or None — 3D electrode positions
    band : str — one of "delta", "theta", "alpha", "beta", "gamma"

    Returns
    -------
    base64 PNG string of the topomap.
    """
    from scipy.signal import welch

    if band not in _BANDS:
        band = "alpha"
    f_low, f_high = _BANDS[band]

    # Compute band power per channel
    n_channels = data.shape[0]
    nperseg = min(int(frequency * 2), data.shape[1])
    band_power = np.zeros(n_channels)

    for ch_idx in range(n_channels):
        freqs, psd = welch(data[ch_idx], fs=frequency, nperseg=max(nperseg, 4))
        mask = (freqs >= f_low) & (freqs <= f_high)
        if np.any(mask):
            band_power[ch_idx] = np.mean(psd[mask])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    # Try MNE topomap if positions available
    if positions is not None and positions.shape[0] == n_channels:
        try:
            import mne
            info = mne.create_info(channels, frequency, ch_types="eeg")
            montage = mne.channels.make_dig_montage(
                ch_pos=dict(zip(channels, positions)),
                coord_frame="head",
            )
            info.set_montage(montage)
            mne.viz.plot_topomap(band_power, info, axes=ax, show=False)
            ax.set_title(f"{band.capitalize()} Power ({f_low}-{f_high} Hz)")
        except Exception:
            _fallback_bar_plot(ax, band_power, channels, band, f_low, f_high)
    else:
        # Try standard montage lookup
        try:
            import mne
            montage = mne.channels.make_standard_montage("standard_1020")
            montage_ch = montage.ch_names
            matched = [ch for ch in channels if ch in montage_ch]
            if len(matched) >= n_channels * 0.5:
                info = mne.create_info(matched, frequency, ch_types="eeg")
                info.set_montage(montage)
                matched_idx = [channels.index(ch) for ch in matched]
                mne.viz.plot_topomap(band_power[matched_idx], info, axes=ax, show=False)
                ax.set_title(f"{band.capitalize()} Power ({f_low}-{f_high} Hz)")
            else:
                _fallback_bar_plot(ax, band_power, channels, band, f_low, f_high)
        except Exception:
            _fallback_bar_plot(ax, band_power, channels, band, f_low, f_high)

    result = _fig_to_base64(fig)
    plt.close(fig)
    return result


def generate_spectral_analysis(
    data: np.ndarray,
    frequency: float,
    channels: Optional[List[str]] = None,
    max_channels: int = 16,
) -> Dict[str, str]:
    """Generate detailed spectral analysis figures.

    Returns per-channel PSD and band power comparison.
    """
    if data.ndim == 3:
        data = data.mean(axis=0)

    n_ch = min(data.shape[0], max_channels)
    if channels is None:
        channels = [f"Ch{i}" for i in range(data.shape[0])]

    from scipy.signal import welch

    nperseg = min(int(frequency * 2), data.shape[1])
    if nperseg < 4:
        return {}

    # Compute band powers for all channels
    all_band_powers = {band: np.zeros(n_ch) for band in _BANDS}

    for ch_idx in range(n_ch):
        freqs, psd = welch(data[ch_idx], fs=frequency, nperseg=nperseg)
        for band_name, (f_low, f_high) in _BANDS.items():
            mask = (freqs >= f_low) & (freqs <= f_high)
            if np.any(mask):
                all_band_powers[band_name][ch_idx] = np.mean(psd[mask])

    results = {}

    if _check_plotly():
        results["band_power_plotly"] = _plotly_band_power(
            all_band_powers, channels[:n_ch]
        )
    else:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        x = np.arange(n_ch)
        width = 0.15
        for i, (band_name, powers) in enumerate(all_band_powers.items()):
            ax.bar(x + i * width, powers, width, label=band_name)
        ax.set_xlabel("Channel")
        ax.set_ylabel("Power (V²/Hz)")
        ax.set_title("Band Power by Channel")
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(channels[:n_ch], rotation=45, ha="right", fontsize=7)
        ax.legend()
        plt.tight_layout()
        results["band_power"] = _fig_to_base64(fig)
        plt.close(fig)

    return results


# --- Plotly implementations ---

def _plotly_psd(data: np.ndarray, frequency: float, channels: List[str]) -> str:
    """Interactive PSD with Plotly."""
    import plotly.graph_objects as go
    from scipy.signal import welch

    nperseg = min(int(frequency * 2), data.shape[1])
    fig = go.Figure()

    for ch_idx, ch_name in enumerate(channels):
        freqs, psd = welch(data[ch_idx], fs=frequency, nperseg=max(nperseg, 4))
        fig.add_trace(go.Scatter(
            x=freqs.tolist(),
            y=(10 * np.log10(psd + 1e-30)).tolist(),
            mode="lines",
            name=ch_name,
            visible="legendonly" if ch_idx >= 8 else True,
        ))

    fig.update_layout(
        title="Power Spectral Density",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Power (dB)",
        hovermode="x unified",
        height=400,
    )
    return fig.to_json()


def _plotly_timeseries(data: np.ndarray, frequency: float, channels: List[str]) -> str:
    """Interactive time series with Plotly (first 5 seconds)."""
    import plotly.graph_objects as go

    max_samples = min(int(5 * frequency), data.shape[1])
    times = np.arange(max_samples) / frequency

    fig = go.Figure()
    n_show = min(8, len(channels))
    for ch_idx in range(n_show):
        fig.add_trace(go.Scatter(
            x=times.tolist(),
            y=data[ch_idx, :max_samples].tolist(),
            mode="lines",
            name=channels[ch_idx],
        ))

    fig.update_layout(
        title="Time Series (first 5s)",
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        hovermode="x unified",
        height=350,
    )
    return fig.to_json()


def _plotly_variance(data: np.ndarray, channels: List[str]) -> str:
    """Channel variance bar chart with Plotly."""
    import plotly.graph_objects as go

    variances = np.var(data, axis=1)
    fig = go.Figure(go.Bar(
        x=channels,
        y=variances.tolist(),
        marker_color=["red" if v > np.median(variances) * 5 else "steelblue" for v in variances],
    ))
    fig.update_layout(
        title="Channel Variance",
        xaxis_title="Channel",
        yaxis_title="Variance",
        height=300,
    )
    return fig.to_json()


def _plotly_amplitude(data: np.ndarray) -> str:
    """Amplitude distribution histogram with Plotly."""
    import plotly.graph_objects as go

    flat = data.flatten()
    # Subsample for performance
    if len(flat) > 100000:
        flat = np.random.default_rng(EASYBCI_SEED).choice(flat, 100000, replace=False)

    fig = go.Figure(go.Histogram(x=flat.tolist(), nbinsx=100))
    fig.update_layout(
        title="Amplitude Distribution",
        xaxis_title="Amplitude",
        yaxis_title="Count",
        height=300,
    )
    return fig.to_json()


def _plotly_band_power(band_powers: Dict[str, np.ndarray], channels: List[str]) -> str:
    """Band power grouped bar chart with Plotly."""
    import plotly.graph_objects as go

    fig = go.Figure()
    for band_name, powers in band_powers.items():
        fig.add_trace(go.Bar(
            name=band_name,
            x=channels,
            y=powers.tolist(),
        ))

    fig.update_layout(
        barmode="group",
        title="Band Power by Channel",
        xaxis_title="Channel",
        yaxis_title="Power (V²/Hz)",
        height=400,
    )
    return fig.to_json()


# --- Helpers ---

def _fallback_bar_plot(ax, band_power, channels, band, f_low, f_high):
    """Simple bar chart when topomap is not available."""
    n = len(band_power)
    ax.bar(range(n), band_power)
    ax.set_xlabel("Channel")
    ax.set_ylabel("Power")
    ax.set_title(f"{band.capitalize()} Power ({f_low}-{f_high} Hz)")
    if n <= 32:
        ax.set_xticks(range(n))
        ax.set_xticklabels(channels[:n], rotation=45, ha="right", fontsize=6)


def _fig_to_base64(fig) -> str:
    """Convert matplotlib figure to base64 PNG."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")
