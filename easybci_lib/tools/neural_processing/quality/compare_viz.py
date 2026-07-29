"""Before/after comparison visualization for preprocessing pipelines.

Generates a time-domain comparison figure showing the signal before and after
preprocessing. This is the core evidence that processing improved signal quality.
The after-side is fixed by a ``FinalDataView`` snapshot (post-pipeline channel
filtering already enforced); the before-side is name-aligned against the view.
"""

import io
from pathlib import Path
from typing import Dict, List, Optional, Sequence, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from easybci_lib.tools.neural_processing.quality.final_view import FinalDataView


def generate_comparison_figures(
    before_data: np.ndarray,
    before_freq: float,
    channels_before: Sequence[str],
    after_view: "FinalDataView",
    steps: List[str],
    *,
    max_display_channels: int = 0,
    save_to_dir: Optional[str] = None,
    subject_id: str = "",
) -> Dict[str, bytes]:
    """Generate before/after comparison PNG figure.

    The after-side is fixed by ``after_view`` (post-pipeline snapshot, channels
    and shape pre-validated). The before-side is name-aligned against the
    view's channel set so dropped channels are surfaced in the title.

    Parameters
    ----------
    before_data : ndarray, shape (n_channels, n_samples)
        Raw data before preprocessing.
    before_freq : float
        Sampling rate of raw data.
    channels_before : sequence of str
        Channel names for ``before_data``.
    after_view : FinalDataView
        Post-pipeline snapshot — single source of truth for what figures show.
    steps : list of str
        Pipeline steps applied (e.g. ``["notch:50", "bandpass:1,40"]``).
    max_display_channels : int
        Max channels to display. 0 means show ALL surviving channels.
    save_to_dir : str, optional
        If provided, save PNG files to this directory in addition to returning
        bytes.
    subject_id : str
        Subject identifier for file naming.

    Returns
    -------
    dict mapping filename to PNG bytes
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures: Dict[str, bytes] = {}

    after_data = after_view.data
    after_channels = list(after_view.channels)
    after_freq = after_view.frequency

    # Name-aligned matching: display only channels present in BOTH before and
    # after, by name. The view is authoritative for the processed side, so
    # dropped channels are detected by set difference and surfaced in the
    # title.
    after_set = set(after_channels)
    keep_indices = [i for i, ch in enumerate(channels_before) if ch in after_set]
    removed_channels = [ch for ch in channels_before if ch not in after_set]

    if keep_indices:
        display_before = before_data[keep_indices]
        display_channels = [channels_before[i] for i in keep_indices]
    else:
        display_before = before_data
        display_channels = list(channels_before)
        removed_channels = []

    n_after = after_data.shape[0]
    n_ch = display_before.shape[0] if max_display_channels <= 0 else min(display_before.shape[0], max_display_channels)
    n_ch = min(n_ch, n_after)
    ch_names = display_channels[:n_ch] if display_channels else [f"Ch{i}" for i in range(n_ch)]

    prefix = f"{subject_id}_" if subject_id else ""
    fig_name = f"{prefix}timeseries_before_after.png"

    figures[fig_name] = _plot_timeseries_comparison(
        display_before[:n_ch], after_data[:n_ch],
        before_freq, after_freq, ch_names,
        removed_channels=removed_channels,
    )

    # Optionally persist to disk
    if save_to_dir:
        out_path = Path(save_to_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        for fname, png_bytes in figures.items():
            try:
                (out_path / fname).write_bytes(png_bytes)
            except OSError:
                pass

    plt.close("all")
    return figures


def _plot_timeseries_comparison(
    before: np.ndarray,
    after: np.ndarray,
    before_freq: float,
    after_freq: float,
    channels: List[str],
    removed_channels: Optional[List[str]] = None,
) -> bytes:
    """Side-by-side time-domain signal: top=before, bottom=after.

    Dynamically scales figure height based on channel count so all channels
    are clearly visible with proper spacing.
    """
    import matplotlib.pyplot as plt

    display_seconds = 5.0
    n_ch = before.shape[0]
    b_samples = min(before.shape[1], int(before_freq * display_seconds))
    a_samples = min(after.shape[1], int(after_freq * display_seconds))

    # Dynamic figure height: ~0.4 inches per channel per subplot, minimum 6
    panel_height = max(3.0, n_ch * 0.4)
    fig_height = panel_height * 2 + 1.5  # two panels + suptitle space
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(14, fig_height), sharex=False)

    # Before (top) — center each channel to remove DC offset before display
    t_before = np.arange(b_samples) / before_freq
    subset_b = before[:n_ch, :b_samples]
    centered_b = subset_b - np.mean(subset_b, axis=1, keepdims=True)
    ptp_b = np.ptp(centered_b, axis=1)
    scale_b = np.median(ptp_b) * 1.2 if np.median(ptp_b) > 0 else 1
    offsets_b = np.arange(n_ch) * scale_b

    for i in range(n_ch):
        ax_top.plot(t_before, centered_b[i] + offsets_b[i],
                    color="#3b82f6", linewidth=0.4, alpha=0.8)

    ax_top.set_ylabel("Channels", fontsize=10)
    ax_top.set_title("BEFORE Preprocessing (Raw Signal)", fontsize=11,
                     fontweight="bold", color="#1e40af")
    ax_top.set_yticks(offsets_b)
    ax_top.set_yticklabels(channels[:n_ch], fontsize=max(5, 8 - n_ch // 16))
    ax_top.set_xlim(0, display_seconds)
    ax_top.grid(True, alpha=0.2, axis="x")

    # After (bottom) — same per-channel centering approach
    t_after = np.arange(a_samples) / after_freq
    subset_a = after[:min(after.shape[0], n_ch), :a_samples]
    n_ch_after = subset_a.shape[0]
    centered_a = subset_a - np.mean(subset_a, axis=1, keepdims=True)
    ptp_a = np.ptp(centered_a, axis=1)
    scale_a = np.median(ptp_a) * 1.2 if np.median(ptp_a) > 0 else 1
    offsets_a = np.arange(n_ch_after) * scale_a

    for i in range(n_ch_after):
        ax_bot.plot(t_after, centered_a[i] + offsets_a[i],
                    color="#000000", linewidth=0.4, alpha=0.8)

    ax_bot.set_xlabel("Time (s)", fontsize=10)
    ax_bot.set_ylabel("Channels", fontsize=10)
    ax_bot.set_title("AFTER Preprocessing (Processed Signal)", fontsize=11,
                     fontweight="bold", color="#000000")
    ax_bot.set_yticks(offsets_a)
    ax_bot.set_yticklabels(channels[:n_ch_after], fontsize=max(5, 8 - n_ch_after // 16))
    ax_bot.set_xlim(0, display_seconds)
    ax_bot.grid(True, alpha=0.2, axis="x")

    suptitle = "Time-Domain Signal Comparison (first 5 seconds)"
    if removed_channels:
        preview = ", ".join(removed_channels[:6])
        if len(removed_channels) > 6:
            preview += f", … (+{len(removed_channels) - 6})"
        suptitle += f"\nRemoved {len(removed_channels)} channel(s): {preview}"
    fig.suptitle(suptitle, fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    return _fig_to_bytes(fig)


def _fig_to_bytes(fig) -> bytes:
    """Convert matplotlib figure to PNG bytes."""
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf.read()
