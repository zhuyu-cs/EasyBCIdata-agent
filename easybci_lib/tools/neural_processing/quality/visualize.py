"""QC visualization — generate signal quality plots as base64 PNG.

Produces:
- PSD (Power Spectral Density) plot
- Channel variance bar chart
- Amplitude distribution histogram
- Data segment heatmap

All plots rendered via matplotlib, exported as base64 for web display.
Supports both single-file and batch (per-subject) QC generation.
"""

import base64
import io
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np

from easybci_lib.tools.neural_processing._seed import EASYBCI_SEED

if TYPE_CHECKING:
    from easybci_lib.tools.neural_processing.quality.final_view import FinalDataView

logger = logging.getLogger(__name__)


def generate_qc_figures(
    view: "FinalDataView",
    *,
    max_channels_display: int = 32,
) -> Dict[str, str]:
    """Generate QC visualization figures as base64 PNG strings.

    Parameters
    ----------
    view : FinalDataView
        Post-pipeline snapshot. Channel filtering and shape-alignment are
        already enforced by the view's constructor; this function never
        inspects the modality directly.
    max_channels_display : int
        Cap on channels rendered per figure (legend readability).

    Returns
    -------
    dict mapping figure_name → base64 PNG string
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = view.data
    channels = list(view.channels)
    frequency = view.frequency

    n_ch = min(data.shape[0], max_channels_display)
    figures: Dict[str, str] = {}
    figures["psd"] = _plot_psd(data[:n_ch], frequency, channels[:n_ch])
    figures["variance"] = _plot_channel_variance(data, channels)
    figures["amplitude"] = _plot_amplitude_dist(data)
    figures["timeseries"] = _plot_timeseries(data[:n_ch], frequency, channels[:n_ch])
    plt.close("all")
    return figures


def _plot_psd(data: np.ndarray, sfreq: float, channels: List[str]) -> str:
    """Power spectral density plot."""
    import matplotlib.pyplot as plt
    from scipy.signal import welch

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#1a1b26")
    ax.set_facecolor("#24283b")

    n_ch = data.shape[0]
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, min(n_ch, 8)))

    # Show a subset of channels to avoid clutter
    step = max(1, n_ch // 8)
    for i in range(0, n_ch, step):
        freqs, psd = welch(data[i], fs=sfreq, nperseg=min(1024, data.shape[1]))
        ax.semilogy(freqs, psd, color=colors[i // step % len(colors)],
                    alpha=0.7, linewidth=0.8, label=channels[i])

    ax.set_xlabel("Frequency (Hz)", color="#c0caf5")
    ax.set_ylabel("PSD (V²/Hz)", color="#c0caf5")
    ax.set_title("Power Spectral Density", color="#c0caf5")
    ax.tick_params(colors="#565f89")
    ax.legend(loc="upper right", fontsize=7, framealpha=0.5)
    ax.set_xlim(0, min(sfreq / 2, 100))
    ax.grid(True, alpha=0.2)

    for spine in ax.spines.values():
        spine.set_color("#3b4261")

    return _fig_to_base64(fig)


def _plot_channel_variance(data: np.ndarray, channels: List[str]) -> str:
    """Channel variance bar chart — highlights outliers."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 3))
    fig.patch.set_facecolor("#1a1b26")
    ax.set_facecolor("#24283b")

    variances = np.var(data, axis=1)
    median_var = np.median(variances)
    iqr = np.percentile(variances, 75) - np.percentile(variances, 25)
    threshold_high = median_var + 3 * iqr
    threshold_low = median_var - 3 * iqr

    colors = []
    for v in variances:
        if v > threshold_high or v < threshold_low:
            colors.append("#f7768e")  # red for outliers
        else:
            colors.append("#7aa2f7")  # blue for normal

    n_ch = len(variances)
    x = range(n_ch)
    ax.bar(x, variances, color=colors, width=0.8)

    ax.axhline(median_var, color="#9ece6a", linestyle="--", linewidth=1, label="Median")
    ax.axhline(threshold_high, color="#f7768e", linestyle=":", linewidth=1, label="Outlier threshold")

    ax.set_xlabel("Channel", color="#c0caf5")
    ax.set_ylabel("Variance", color="#c0caf5")
    ax.set_title("Channel Variance", color="#c0caf5")
    ax.tick_params(colors="#565f89")
    ax.legend(fontsize=7, framealpha=0.5)

    if n_ch <= 32:
        ax.set_xticks(x)
        ax.set_xticklabels(channels[:n_ch], rotation=45, ha="right", fontsize=6)

    for spine in ax.spines.values():
        spine.set_color("#3b4261")

    fig.tight_layout()
    return _fig_to_base64(fig)


def _plot_amplitude_dist(data: np.ndarray) -> str:
    """Amplitude distribution histogram."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3))
    fig.patch.set_facecolor("#1a1b26")
    ax.set_facecolor("#24283b")

    # Flatten and subsample for speed
    flat = data.ravel()
    if len(flat) > 100000:
        flat = np.random.default_rng(EASYBCI_SEED).choice(flat, 100000, replace=False)

    ax.hist(flat, bins=100, color="#7aa2f7", alpha=0.8, edgecolor="none")

    mean_val = np.mean(flat)
    std_val = np.std(flat)
    ax.axvline(mean_val, color="#9ece6a", linestyle="-", linewidth=1.5, label=f"Mean: {mean_val:.2e}")
    ax.axvline(mean_val + 3*std_val, color="#e0af68", linestyle="--", linewidth=1, label=f"±3σ")
    ax.axvline(mean_val - 3*std_val, color="#e0af68", linestyle="--", linewidth=1)

    ax.set_xlabel("Amplitude", color="#c0caf5")
    ax.set_ylabel("Count", color="#c0caf5")
    ax.set_title("Amplitude Distribution", color="#c0caf5")
    ax.tick_params(colors="#565f89")
    ax.legend(fontsize=7, framealpha=0.5)

    for spine in ax.spines.values():
        spine.set_color("#3b4261")

    fig.tight_layout()
    return _fig_to_base64(fig)


def _plot_timeseries(data: np.ndarray, sfreq: float, channels: List[str]) -> str:
    """Time series preview — first 5 seconds, offset channels."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#1a1b26")
    ax.set_facecolor("#24283b")

    n_ch = min(data.shape[0], 16)
    n_samples = min(int(5 * sfreq), data.shape[1])
    t = np.arange(n_samples) / sfreq

    # Normalize for display
    subset = data[:n_ch, :n_samples]
    offsets = np.arange(n_ch) * np.std(subset) * 4

    for i in range(n_ch):
        color = plt.cm.viridis(i / n_ch * 0.8 + 0.1)
        ax.plot(t, subset[i] + offsets[i], color=color, linewidth=0.5)

    ax.set_xlabel("Time (s)", color="#c0caf5")
    ax.set_title("Signal Preview (first 5s)", color="#c0caf5")
    ax.tick_params(colors="#565f89")
    ax.set_yticks(offsets)
    ax.set_yticklabels(channels[:n_ch], fontsize=6, color="#565f89")

    for spine in ax.spines.values():
        spine.set_color("#3b4261")

    fig.tight_layout()
    return _fig_to_base64(fig)


def _fig_to_base64(fig) -> str:
    """Convert matplotlib figure to base64 PNG string."""
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ---------------------------------------------------------------------------
# Batch QC — per-subject figure generation to disk
# ---------------------------------------------------------------------------


def _save_figure_to_file(fig, filepath: Path) -> bool:
    """Save a matplotlib figure to a PNG file. Returns True on success."""
    import matplotlib.pyplot as plt
    filepath.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(str(filepath), format="png", dpi=100, bbox_inches="tight",
                    facecolor=fig.get_facecolor(), edgecolor="none")
    except (OSError, ValueError) as exc:
        logger.debug("Failed to save figure %s: %s", filepath, exc)
        return False
    finally:
        plt.close(fig)
    return True


def generate_subject_qc_figures(
    view: "FinalDataView",
    subject_id: str,
    output_dir: str,
    *,
    max_channels_display: int = 32,
) -> Dict[str, str]:
    """Generate QC figures for a single subject and save to disk.

    Parameters
    ----------
    view : FinalDataView
        Post-pipeline snapshot (channel filtering already enforced).
    subject_id : str
        Subject identifier for file naming.
    output_dir : str
        Base output directory (figures go to {output_dir}/qc/{subject_id}/).
    max_channels_display : int
        Limit channels shown.

    Returns
    -------
    dict mapping plot_type → file path
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.signal import welch

    data = view.data
    channels = list(view.channels)
    frequency = view.frequency

    n_ch = min(data.shape[0], max_channels_display)
    subject_dir = Path(output_dir) / "qc" / subject_id
    subject_dir.mkdir(parents=True, exist_ok=True)

    saved_files: Dict[str, str] = {}

    # 1. PSD
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#1a1b26")
    ax.set_facecolor("#24283b")
    step = max(1, n_ch // 8)
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, min(n_ch, 8)))
    for i in range(0, n_ch, step):
        freqs, psd = welch(data[i], fs=frequency, nperseg=min(1024, data.shape[1]))
        ax.semilogy(freqs, psd, color=colors[i // step % len(colors)],
                    alpha=0.7, linewidth=0.8, label=channels[i])
    ax.set_xlabel("Frequency (Hz)", color="#c0caf5")
    ax.set_ylabel("PSD (V²/Hz)", color="#c0caf5")
    ax.set_title(f"PSD — {subject_id}", color="#c0caf5")
    ax.tick_params(colors="#565f89")
    ax.legend(loc="upper right", fontsize=7, framealpha=0.5)
    ax.set_xlim(0, min(frequency / 2, 100))
    ax.grid(True, alpha=0.2)
    for spine in ax.spines.values():
        spine.set_color("#3b4261")
    psd_path = subject_dir / f"{subject_id}_psd.png"
    if _save_figure_to_file(fig, psd_path):
        saved_files["psd"] = str(psd_path)

    # 2. Channel variance
    fig, ax = plt.subplots(figsize=(8, 3))
    fig.patch.set_facecolor("#1a1b26")
    ax.set_facecolor("#24283b")
    variances = np.var(data, axis=1)
    median_var = np.median(variances)
    iqr = np.percentile(variances, 75) - np.percentile(variances, 25)
    threshold_high = median_var + 3 * iqr
    bar_colors = ["#f7768e" if v > threshold_high else "#7aa2f7" for v in variances]
    ax.bar(range(len(variances)), variances, color=bar_colors, width=0.8)
    ax.axhline(median_var, color="#9ece6a", linestyle="--", linewidth=1)
    ax.set_xlabel("Channel", color="#c0caf5")
    ax.set_ylabel("Variance", color="#c0caf5")
    ax.set_title(f"Channel Variance — {subject_id}", color="#c0caf5")
    ax.tick_params(colors="#565f89")
    for spine in ax.spines.values():
        spine.set_color("#3b4261")
    fig.tight_layout()
    var_path = subject_dir / f"{subject_id}_channel_variance.png"
    if _save_figure_to_file(fig, var_path):
        saved_files["channel_variance"] = str(var_path)

    # 3. Amplitude distribution
    fig, ax = plt.subplots(figsize=(6, 3))
    fig.patch.set_facecolor("#1a1b26")
    ax.set_facecolor("#24283b")
    flat = data.ravel()
    if len(flat) > 100000:
        rng = np.random.default_rng(EASYBCI_SEED)
        flat = rng.choice(flat, 100000, replace=False)
    ax.hist(flat, bins=100, color="#7aa2f7", alpha=0.8, edgecolor="none")
    mean_val = np.mean(flat)
    std_val = np.std(flat)
    ax.axvline(mean_val, color="#9ece6a", linestyle="-", linewidth=1.5)
    ax.axvline(mean_val + 3 * std_val, color="#e0af68", linestyle="--", linewidth=1)
    ax.axvline(mean_val - 3 * std_val, color="#e0af68", linestyle="--", linewidth=1)
    ax.set_xlabel("Amplitude", color="#c0caf5")
    ax.set_ylabel("Count", color="#c0caf5")
    ax.set_title(f"Amplitude Distribution — {subject_id}", color="#c0caf5")
    ax.tick_params(colors="#565f89")
    for spine in ax.spines.values():
        spine.set_color("#3b4261")
    fig.tight_layout()
    amp_path = subject_dir / f"{subject_id}_amplitude_dist.png"
    if _save_figure_to_file(fig, amp_path):
        saved_files["amplitude_dist"] = str(amp_path)

    # 4. Time series preview
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#1a1b26")
    ax.set_facecolor("#24283b")
    n_display = min(n_ch, 16)
    n_samples = min(int(5 * frequency), data.shape[1])
    t = np.arange(n_samples) / frequency
    subset = data[:n_display, :n_samples]
    scale = np.std(subset) * 4 if np.std(subset) > 0 else 1
    offsets = np.arange(n_display) * scale
    for i in range(n_display):
        color = plt.cm.viridis(i / n_display * 0.8 + 0.1)
        ax.plot(t, subset[i] + offsets[i], color=color, linewidth=0.5)
    ax.set_xlabel("Time (s)", color="#c0caf5")
    ax.set_title(f"Signal Preview — {subject_id}", color="#c0caf5")
    ax.tick_params(colors="#565f89")
    ax.set_yticks(offsets)
    ax.set_yticklabels(channels[:n_display], fontsize=6, color="#565f89")
    for spine in ax.spines.values():
        spine.set_color("#3b4261")
    fig.tight_layout()
    ts_path = subject_dir / f"{subject_id}_timeseries.png"
    if _save_figure_to_file(fig, ts_path):
        saved_files["timeseries"] = str(ts_path)

    plt.close("all")
    return saved_files


def generate_batch_qc_figures(
    batch_results: List[Dict[str, Any]],
    output_dir: str,
    max_subjects_detail: int = 50,
) -> Dict[str, Any]:
    """Generate QC figures for all subjects in a batch run.

    Parameters
    ----------
    batch_results : list of dicts
        Results from batch_process(), each with keys:
        "subject_id", "output_path", "success", optionally "data", "frequency", "channels"
    output_dir : str
        Base output directory. Figures go to {output_dir}/qc/{subject_id}/
    max_subjects_detail : int
        Maximum number of subjects to generate full detail figures for.
        Beyond this limit, only the batch overview is generated.

    Returns
    -------
    dict with:
        "subjects": {subject_id: {plot_type: path}}
        "overview": path to batch_overview.png (or None)
        "n_subjects_plotted": int
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    qc_base = Path(output_dir) / "qc"
    qc_base.mkdir(parents=True, exist_ok=True)

    subject_figures: Dict[str, Dict[str, str]] = {}
    snr_values: List[float] = []
    artifact_ratios: List[float] = []
    subject_ids: List[str] = []

    n_plotted = 0
    for result in batch_results:
        if not result.get("success"):
            continue
        subject_id = result.get("subject_id", f"unknown_{n_plotted}")
        subject_ids.append(subject_id)

        # Try to load processed data for QC figure generation
        output_path = result.get("output_path", "")
        data = None
        frequency = result.get("frequency", 0)
        channels = result.get("channels", [])

        if output_path and Path(output_path).exists() and n_plotted < max_subjects_detail:
            try:
                loaded = np.load(output_path, allow_pickle=True)
                if isinstance(loaded, np.ndarray):
                    data = loaded
                elif hasattr(loaded, "files"):
                    # npz file
                    keys = list(loaded.files)
                    if "data" in keys:
                        data = loaded["data"]
                    elif keys:
                        data = loaded[keys[0]]
            except Exception:
                pass

            # Fallback: try pickle
            if data is None and output_path.endswith(".pkl"):
                try:
                    import pickle
                    with open(output_path, "rb") as f:
                        pkl_data = pickle.load(f)
                    if isinstance(pkl_data, dict):
                        data = pkl_data.get("data")
                        if frequency == 0:
                            frequency = pkl_data.get("frequency", 0)
                        if not channels:
                            channels = pkl_data.get("channels", [])
                    elif isinstance(pkl_data, np.ndarray):
                        data = pkl_data
                except Exception:
                    pass

        # Generate per-subject figures if we have data
        if data is not None and hasattr(data, "ndim") and data.ndim >= 2 and frequency > 0:
            from easybci_lib.tools.neural_processing.quality.final_view import FinalDataView
            ch_list = list(channels) if channels else [f"Ch{i}" for i in range(data.shape[0])]
            view = FinalDataView.from_pipeline_result(
                after_data=data, channels=ch_list,
                frequency=frequency, modality=result.get("modality", "eeg"),
                enforce_data_only=True,
            )
            figures = generate_subject_qc_figures(view, subject_id, output_dir)
            subject_figures[subject_id] = figures
            n_plotted += 1

            # Collect stats for overview
            snr_est = _estimate_snr(data)
            artifact_est = _estimate_artifact_ratio(data)
            snr_values.append(snr_est)
            artifact_ratios.append(artifact_est)
        else:
            # Can't generate figures but track the subject
            snr_values.append(0.0)
            artifact_ratios.append(0.0)

    # Generate batch overview figure
    overview_path = None
    if len(subject_ids) >= 2:
        overview_path = _generate_batch_overview(
            subject_ids, snr_values, artifact_ratios, qc_base
        )

    plt.close("all")
    return {
        "subjects": subject_figures,
        "overview": str(overview_path) if overview_path else None,
        "n_subjects_plotted": n_plotted,
        "n_subjects_total": len(subject_ids),
    }


def _estimate_snr(data: np.ndarray) -> float:
    """Quick SNR estimate in dB (signal power / noise floor)."""
    signal_power = np.mean(data ** 2)
    # Estimate noise as high-frequency content (diff approximation)
    noise = np.diff(data, axis=1)
    noise_power = np.mean(noise ** 2) / 2  # diff doubles noise power
    if noise_power <= 0:
        return 0.0
    return float(10 * np.log10(signal_power / noise_power))


def _estimate_artifact_ratio(data: np.ndarray) -> float:
    """Estimate fraction of samples that are likely artifacts (> 5 std)."""
    std = np.std(data)
    if std <= 0:
        return 0.0
    threshold = 5 * std
    n_artifact = np.sum(np.abs(data) > threshold)
    return float(n_artifact / data.size)


def _generate_batch_overview(
    subject_ids: List[str],
    snr_values: List[float],
    artifact_ratios: List[float],
    qc_base: Path,
) -> Optional[Path]:
    """Generate a batch overview figure showing cross-subject statistics."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor("#1a1b26")

    x = range(len(subject_ids))
    short_ids = [s[:12] for s in subject_ids]

    # SNR distribution
    ax1.set_facecolor("#24283b")
    colors_snr = ["#f7768e" if s < 5 else "#7aa2f7" for s in snr_values]
    ax1.bar(x, snr_values, color=colors_snr, width=0.7)
    if snr_values:
        median_snr = np.median(snr_values)
        ax1.axhline(median_snr, color="#9ece6a", linestyle="--", linewidth=1,
                    label=f"Median: {median_snr:.1f} dB")
    ax1.set_xlabel("Subject", color="#c0caf5")
    ax1.set_ylabel("SNR (dB)", color="#c0caf5")
    ax1.set_title("Signal-to-Noise Ratio", color="#c0caf5")
    ax1.tick_params(colors="#565f89")
    ax1.legend(fontsize=8, framealpha=0.5)
    if len(subject_ids) <= 30:
        ax1.set_xticks(list(x))
        ax1.set_xticklabels(short_ids, rotation=45, ha="right", fontsize=6)
    for spine in ax1.spines.values():
        spine.set_color("#3b4261")

    # Artifact ratio distribution
    ax2.set_facecolor("#24283b")
    colors_art = ["#f7768e" if a > 0.05 else "#7aa2f7" for a in artifact_ratios]
    ax2.bar(x, [r * 100 for r in artifact_ratios], color=colors_art, width=0.7)
    if artifact_ratios:
        median_art = np.median(artifact_ratios) * 100
        ax2.axhline(median_art, color="#9ece6a", linestyle="--", linewidth=1,
                    label=f"Median: {median_art:.2f}%")
    ax2.set_xlabel("Subject", color="#c0caf5")
    ax2.set_ylabel("Artifact (%)", color="#c0caf5")
    ax2.set_title("Artifact Ratio (>5σ samples)", color="#c0caf5")
    ax2.tick_params(colors="#565f89")
    ax2.legend(fontsize=8, framealpha=0.5)
    if len(subject_ids) <= 30:
        ax2.set_xticks(list(x))
        ax2.set_xticklabels(short_ids, rotation=45, ha="right", fontsize=6)
    for spine in ax2.spines.values():
        spine.set_color("#3b4261")

    fig.suptitle(f"Batch QC Overview ({len(subject_ids)} subjects)",
                 color="#c0caf5", fontsize=12, fontweight="bold")
    fig.tight_layout()

    overview_path = qc_base / "batch_overview.png"
    if _save_figure_to_file(fig, overview_path):
        return overview_path
    return None
