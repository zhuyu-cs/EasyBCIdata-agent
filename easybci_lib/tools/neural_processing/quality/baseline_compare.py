"""Soft QC baselines: extract transferable statistics from processed data and
compare them against a proven skill's qc_baselines.

Transferable = shape/ratio/magnitude that survive individual differences
(sEEG individual variability is large; absolute values would false-alarm).
Comparison is ADVISORY only — a deviation is a `baseline_warning`, never a
failure (design 04: no hard gate).

Band definitions mirror quality/metrics._STANDARD_BANDS so baselines extracted
here are comparable to the metrics the rest of the pipeline reports.
"""
from __future__ import annotations

from typing import Any

import numpy as np

# Mirror metrics._STANDARD_BANDS (kept local to avoid a cross-import cycle;
# if the two drift, that's a bug — see test_bands_match_metrics).
_BANDS = {
    "delta": (0.5, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0), "gamma": (30.0, 100.0),
}


def _band_power_shape(data: np.ndarray, fs: float) -> dict[str, float]:
    """Per-band power *fraction* (sums to 1) — individual-invariant shape."""
    if data.ndim != 2 or data.shape[0] == 0 or data.shape[1] < 64 or fs <= 0:
        return {}
    from numpy.fft import rfft, rfftfreq
    n_fft = min(data.shape[1], int(fs * 4))
    if n_fft < 64:
        return {}
    freqs = rfftfreq(n_fft, 1.0 / fs)
    seg = data[:, :n_fft] * np.hanning(n_fft)
    psd = (np.abs(rfft(seg, axis=1)) ** 2).mean(axis=0)  # linear power, avg ch
    nyq = fs / 2.0
    band_lin: dict[str, float] = {}
    for name, (lo, hi) in _BANDS.items():
        if lo >= nyq:
            band_lin[name] = 0.0
            continue
        mask = (freqs >= lo) & (freqs < min(hi, nyq))
        band_lin[name] = float(psd[mask].sum()) if mask.any() else 0.0
    total = sum(band_lin.values()) or 1.0
    return {k: round(v / total, 4) for k, v in band_lin.items()}


def extract_baseline_metrics(data: np.ndarray, fs: float, *,
                             n_bad: int, n_total: int) -> dict[str, Any]:
    """Extract the same transferable baselines P2 stores in qc_baselines."""
    ratio = (n_bad / n_total) if n_total > 0 else 0.0
    out: dict[str, Any] = {"bad_channel_ratio": round(ratio, 4)}
    out["band_power_shape"] = _band_power_shape(data, fs)
    if data.ndim == 2 and data.shape[0] > 0:
        var = np.var(data, axis=1)
        q1, q3 = np.percentile(var, [25, 75])
        out["channel_variance_scale"] = {
            "median": float(np.median(var)), "iqr": float(q3 - q1),
        }
    else:
        out["channel_variance_scale"] = {"median": 0.0, "iqr": 0.0}
    return out


_BAND_SHAPE_L1_TOL = 0.30   # sum |Δfraction| over 5 bands; sEEG-tolerant
_VAR_SCALE_FOLD_TOL = 3.0   # median within [1/3x, 3x] of baseline


def compare_to_baselines(measured: dict[str, Any],
                         baselines: dict[str, Any]) -> dict[str, Any]:
    """Advisory comparison of measured baselines vs a skill's qc_baselines.

    Returns {status, warnings:[{metric, measured, expected, note}], failed:False}.
    Deviations are soft warnings only — batch never fails a file on baseline.
    """
    warnings: list[dict] = []
    compared = 0

    # bad_channel_ratio: |Δ| > tolerance
    b = baselines.get("bad_channel_ratio")
    if isinstance(b, dict) and "value" in b and "bad_channel_ratio" in measured:
        compared += 1
        tol = float(b.get("tolerance", 0.15))
        exp = float(b["value"])
        got = float(measured["bad_channel_ratio"])
        if abs(got - exp) > tol:
            warnings.append({
                "metric": "bad_channel_ratio", "measured": round(got, 4),
                "expected": exp, "tolerance": tol,
                "note": f"bad-channel ratio {got:.3f} deviates from gold {exp:.3f}±{tol}",
            })

    # band_power_shape: L1 distance over bands
    bshape = baselines.get("band_power_shape")
    mshape = measured.get("band_power_shape")
    if isinstance(bshape, dict) and isinstance(mshape, dict) and bshape and mshape:
        compared += 1
        l1 = sum(abs(float(mshape.get(k, 0.0)) - float(v)) for k, v in bshape.items())
        if l1 > _BAND_SHAPE_L1_TOL:
            warnings.append({
                "metric": "band_power_shape", "measured": mshape, "expected": bshape,
                "note": f"band-power shape L1 distance {l1:.2f} > {_BAND_SHAPE_L1_TOL}",
            })

    # channel_variance_scale: fold-change of median
    bvar = baselines.get("channel_variance_scale")
    mvar = measured.get("channel_variance_scale")
    if isinstance(bvar, dict) and isinstance(mvar, dict) and bvar.get("median"):
        compared += 1
        exp_med = float(bvar["median"]) or 1e-12
        got_med = float(mvar.get("median", 0.0))
        fold = got_med / exp_med if exp_med else 0.0
        if fold and (fold > _VAR_SCALE_FOLD_TOL or fold < 1.0 / _VAR_SCALE_FOLD_TOL):
            warnings.append({
                "metric": "channel_variance_scale",
                "measured": round(got_med, 4), "expected": round(exp_med, 4),
                "note": f"variance magnitude {fold:.1f}x gold (outside 1/3x–3x)",
            })

    if compared == 0:
        return {"status": "no_baseline", "warnings": [], "failed": False}
    return {
        "status": "baseline_warning" if warnings else "within_baseline",
        "warnings": warnings, "failed": False,
    }
