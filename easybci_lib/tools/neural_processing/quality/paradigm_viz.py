"""Paradigm-specific QC visualizations for the standard mini-repo contract.

The array-based ``preprocess_neural`` pipeline already emits before/after,
PSD, variance, amplitude and timeseries figures (see ``compare_viz.py`` and
``visualize.py``). This module adds the richer, spatially-aware figures that
the hand-written WebUI pipelines used to produce — most importantly per-band
scalp **topomaps** — so they land in the SAME
``preprocessed_output/figures/sub-{id}/{session_id}/`` directory and get consolidated
into the contract by ``export.repo_builder.build_mini_repo``.

Everything here is best-effort: if MNE / a usable montage / enough channels
are unavailable, each figure is skipped silently (matching the degradation
style of ``neural_tools.py``). The processed numpy array is all that's
required — no MNE Epochs/ICA objects are needed.
"""
from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from easybci_lib.tools.neural_processing.quality.final_view import FinalDataView

logger = logging.getLogger(__name__)

# Bands worth a scalp map for common EEG paradigms. mu (8-13) overlaps alpha
# and is the motor-imagery band of interest; beta carries motor/SSVEP energy.
_PARADIGM_BANDS = ("alpha", "beta", "theta")


def generate_paradigm_figures(
    view: "FinalDataView",
    fig_dir: str,
    stem: str,
    *,
    positions: Optional[np.ndarray] = None,
    bands: Optional[List[str]] = None,
) -> List[str]:
    """Write paradigm-specific topomap figures into *fig_dir*.

    Parameters
    ----------
    view : FinalDataView
        Post-pipeline snapshot (channel filtering already enforced).
    fig_dir : str
        Target directory (typically
        ``{work_dir}/preprocessed_output/figures/sub-{id}/{session_id}``).
    stem : str
        Filename stem (typically the input file stem), used as a prefix.
    positions : ndarray (n_channels, 3) or None
        Optional 3D electrode positions; falls back to a standard montage.
    bands : list of str or None
        Override the default band set.

    Returns
    -------
    list of str
        Filenames (not full paths) of figures successfully written. Empty if
        nothing could be produced — never raises.
    """
    written: List[str] = []
    try:
        data = view.data
        channels = list(view.channels)
        frequency = view.frequency

        if data.ndim != 2 or data.shape[0] < 2:
            return written
        if not channels or len(channels) < 2:
            return written

        from easybci_lib.tools.neural_processing.quality.interactive_qc import generate_topomap

        target = Path(fig_dir)
        target.mkdir(parents=True, exist_ok=True)

        n_ch = data.shape[0]
        ch_names = list(channels[:n_ch])

        for band in (bands or _PARADIGM_BANDS):
            try:
                b64 = generate_topomap(
                    data, frequency, ch_names,
                    positions=positions, band=band,
                )
                if not b64:
                    continue
                fname = f"{stem}_topomap_{band}.png"
                (target / fname).write_bytes(base64.b64decode(b64))
                written.append(fname)
            except Exception as exc:  # one bad band must not kill the rest
                logger.debug("Paradigm topomap (%s) failed: %s", band, exc)
    except Exception as exc:
        logger.debug("generate_paradigm_figures skipped: %s", exc)

    return written
