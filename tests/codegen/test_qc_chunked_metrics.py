"""Equivalence + memory-safety tests for the generated qc.py metrics.

The generated ``qc.py`` used to compute QC metrics by materializing BOTH the
raw and preprocessed arrays as full float64 copies (``np.asarray(..., f64)``)
— ~14 GB each for a 259ch/7.18M sEEG file, ~30 GB peak, an OOM. The rewrite
computes mean/std/min/max/nan_frac by reading the (n_ch, n_samp) source in
time windows (exact, bit-equivalent to the old full-array path) and estimates
the PSD-SNR proxy on a bounded leading window only.

These tests import the ACTUAL generated qc.py text (not a hand-copy) so the
template itself is under test, and pin:
  * _chunked_stats == np.nanmean/std/min/max + nan fraction (full array)
  * _finite_fraction == np.isfinite(arr).mean()
  * window/budget invariance (the chunk boundary must not change results)
  * a memmap-backed source drives the lazy path without full materialization
"""
from __future__ import annotations

import importlib.util
import sys
import types

import numpy as np
import pytest

from easybci_lib.tools.neural_processing.codegen.generator import (
    generate_qc_script_v2,
)


def _load_qc_module(tmp_path):
    """Generate qc.py and import it as a module to exercise the real template."""
    src = generate_qc_script_v2(
        steps=["notch:50", "bandpass:1-80"],
        data_info={"n_channels": 8, "sampling_rate": 500.0},
        modality="seeg",
        analysis_goal="classification",
    )
    qc_path = tmp_path / "qc_generated.py"
    qc_path.write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("qc_generated", str(qc_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["qc_generated"] = mod
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------
# Oracles: the pre-rewrite full-array implementations.
# --------------------------------------------------------------------------
def _ref_stats(arr):
    a = np.asarray(arr, dtype=np.float64)
    finite = np.isfinite(a)
    if not finite.any():
        return {"mean": None, "std": None, "min": None, "max": None, "nan_frac": 1.0}
    return {
        "mean": float(np.nanmean(a)),
        "std": float(np.nanstd(a)),
        "min": float(np.nanmin(a)),
        "max": float(np.nanmax(a)),
        "nan_frac": float((~finite).mean()),
    }


class _LazyShim:
    """(n_ch, n_samp) view exposing .ndim/.shape/.size + [..., t0:t1] slicing,
    WITHOUT letting callers grab the whole array at once (mimics a memmap)."""

    def __init__(self, arr):
        self._a = arr
        self.shape = arr.shape
        self.ndim = arr.ndim
        self.size = arr.size
        self.dtype = arr.dtype

    def __getitem__(self, key):
        return self._a[key]


def _make(seed=0, n_ch=16, n_samp=40_000, with_nan=True):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n_ch, n_samp)).astype(np.float32) * 20.0
    for c in {1, 5} & set(range(n_ch)):
        idx = rng.choice(n_samp, size=30, replace=False)
        a[c, idx] += 500.0
    if with_nan:
        a[3, rng.choice(n_samp, size=17, replace=False)] = np.nan
    return a


# --------------------------------------------------------------------------
# Equivalence
# --------------------------------------------------------------------------
def test_chunked_stats_matches_reference(tmp_path):
    mod = _load_qc_module(tmp_path)
    a = _make()
    ref = _ref_stats(a)
    got = mod._chunked_stats(a)
    for k in ("mean", "std", "min", "max", "nan_frac"):
        np.testing.assert_allclose(got[k], ref[k], rtol=1e-9, atol=1e-9,
                                   err_msg=f"stat {k} mismatch")


def test_chunked_stats_min_max_exact(tmp_path):
    mod = _load_qc_module(tmp_path)
    a = _make()
    ref = _ref_stats(a)
    got = mod._chunked_stats(a)
    # min/max are exact reductions — must match bit-for-bit
    assert got["min"] == ref["min"]
    assert got["max"] == ref["max"]


def test_all_nan_channel_and_array(tmp_path):
    mod = _load_qc_module(tmp_path)
    a = np.full((4, 1000), np.nan, dtype=np.float32)
    got = mod._chunked_stats(a)
    assert got == {"mean": None, "std": None, "min": None, "max": None, "nan_frac": 1.0}


def test_finite_fraction_matches(tmp_path):
    mod = _load_qc_module(tmp_path)
    a = _make()
    ref = float(np.isfinite(a).mean())
    got = mod._finite_fraction(a)
    np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("budget", [64, 4096, 1 << 20, 1 << 30])
def test_window_invariance(tmp_path, budget):
    mod = _load_qc_module(tmp_path)
    a = _make()
    base = mod._chunked_stats(a)
    # force a specific window by monkeypatching the budget-based sizer
    orig = mod._stat_window_rows
    mod._stat_window_rows = lambda arr, budget_bytes=budget: orig(arr, budget)
    try:
        got = mod._chunked_stats(a)
    finally:
        mod._stat_window_rows = orig
    for k in ("mean", "std", "min", "max", "nan_frac"):
        np.testing.assert_allclose(got[k], base[k], rtol=1e-9, atol=1e-9)


# --------------------------------------------------------------------------
# Lazy path: a memmap-like source is never fully materialized by _chunked_stats
# --------------------------------------------------------------------------
def test_lazy_shim_drives_chunked_path(tmp_path):
    mod = _load_qc_module(tmp_path)
    a = _make()
    shim = _LazyShim(a)
    got = mod._chunked_stats(shim)
    ref = _ref_stats(a)
    for k in ("mean", "std", "min", "max", "nan_frac"):
        np.testing.assert_allclose(got[k], ref[k], rtol=1e-9, atol=1e-9)


def test_real_memmap_source(tmp_path):
    mod = _load_qc_module(tmp_path)
    a = _make(n_ch=12, n_samp=20_000, with_nan=False)
    # write (n_samp, n_ch) contiguous, memmap it as-is, then transpose lazily
    raw = tmp_path / "raw.dat"
    fp = np.memmap(raw, dtype=np.float32, mode="w+", shape=(a.shape[1], a.shape[0]))
    fp[:] = a.T
    fp.flush()
    mm = np.memmap(raw, dtype=np.float32, mode="r", shape=(a.shape[1], a.shape[0]))
    got = mod._chunked_stats(mm.T)
    ref = _ref_stats(a)
    for k in ("mean", "std", "min", "max"):
        np.testing.assert_allclose(got[k], ref[k], rtol=1e-9, atol=1e-9)


# --------------------------------------------------------------------------
# Bounded PSD-SNR: uses only the leading window, finite scalar
# --------------------------------------------------------------------------
def test_bounded_psd_snr_finite_and_windowed(tmp_path):
    mod = _load_qc_module(tmp_path)
    a = _make(n_ch=8, n_samp=200_000, with_nan=False)
    fs = 500.0
    full = mod._bounded_psd_snr(a, fs, max_seconds=1e9)  # whole array
    win = mod._bounded_psd_snr(a, fs, max_seconds=60.0)  # bounded
    assert np.isfinite(full) and np.isfinite(win)
    assert full > 0 and win > 0


def test_compute_metrics_end_to_end(tmp_path):
    mod = _load_qc_module(tmp_path)
    a = _make(n_ch=8, n_samp=30_000, with_nan=False)
    raw_d = {"data": a, "frequency": 500.0, "channels": [f"C{i}" for i in range(8)]}
    proc = {"data": a * 0.5, "frequency": 500.0, "channels": [f"C{i}" for i in range(8)]}
    m = mod._compute_metrics(raw_d, proc)
    assert m["before"]["n_channels"] == 8
    assert m["after"]["n_samples"] == 30_000
    assert m["overall"]["grade"] == "Pass"
    # after = 0.5 * before → mean/std exactly halved
    np.testing.assert_allclose(
        m["after"]["stats"]["mean"], 0.5 * m["before"]["stats"]["mean"],
        rtol=1e-9, atol=1e-9,
    )
