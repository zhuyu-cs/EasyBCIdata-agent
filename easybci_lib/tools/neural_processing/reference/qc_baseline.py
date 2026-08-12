"""Derive transferable QC soft-baselines from gold-standard products.

Mandatory (cheap, from bad_channels.csv-derived ratio):
  bad_channel_ratio  — {value, tolerance}
Best-effort (from final EDF via mne, only if present + loadable + small):
  band_power_shape        — per-band power fraction (shape, individual-invariant)
  channel_variance_scale  — {median, iqr} of per-channel variance
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from easybci_lib.tools.neural_processing.reference.recipe_parser import RecipeProfile

logger = logging.getLogger(__name__)

_MAX_EDF_MB = 200  # keep ingest fast; larger products skip spectral baselines
_BANDS = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13),
          "beta": (13, 30), "gamma": (30, 100)}


def _spectral_baselines(edf_path: Path) -> dict[str, Any] | None:
    try:
        size_mb = edf_path.stat().st_size / (1024 * 1024)
        if size_mb > _MAX_EDF_MB:
            logger.info("final EDF %.0fMB > %dMB cap — skipping spectral baselines",
                        size_mb, _MAX_EDF_MB)
            return None
        import numpy as np
        try:
            import mne
        except ImportError:
            return None
        raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose="ERROR")
        data = raw.get_data()  # (n_ch, n_samp), volts
        sfreq = float(raw.info["sfreq"])
        var = np.var(data, axis=1)
        var_med = float(np.median(var))
        q1, q3 = np.percentile(var, [25, 75])
        from scipy.signal import welch
        freqs, psd = welch(data, fs=sfreq, nperseg=min(4096, data.shape[1]))
        psd_mean = psd.mean(axis=0)
        total = float(psd_mean.sum()) or 1.0
        shape = {}
        for band, (lo, hi) in _BANDS.items():
            mask = (freqs >= lo) & (freqs < hi)
            shape[band] = round(float(psd_mean[mask].sum()) / total, 4)
        return {
            "band_power_shape": shape,
            "channel_variance_scale": {"median": var_med,
                                       "iqr": float(q3 - q1)},
        }
    except Exception as exc:  # noqa: BLE001 — best-effort, never block ingest
        logger.warning("spectral baseline extraction failed: %s", exc)
        return None


def build_qc_baselines(rp: RecipeProfile) -> dict[str, Any]:
    baselines: dict[str, Any] = {
        "bad_channel_ratio": {
            "value": float(rp.bad_channel_ratio),
            "tolerance": 0.15,
        },
    }
    spectral = None
    if rp.final_edf:
        spectral = _spectral_baselines(Path(rp.final_edf))
    if spectral:
        baselines.update(spectral)
    else:
        baselines["partial"] = True  # spectral baselines unavailable
    return baselines
