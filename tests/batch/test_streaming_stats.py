"""Equivalence tests for streaming (chunked) QC-metric computation.

The batch summary step used to load an entire preprocessed NWB into RAM
(``data[:]`` → ~7 GB float32 per file) just to compute three scalar QC
metrics — channel variance, SNR, artifact ratio — which OOM-killed the host
mid-batch. The new ``streaming_stats.compute_streaming_metrics`` computes the
SAME three metrics by reading the (n_samples, n_channels) source in time
windows and reducing per-channel in float64.

These tests pin numerical equivalence against inlined copies of the ORIGINAL
full-array implementations (the ``_ref_*`` oracles below), plus invariance to
window / channel-group sizing (the tricky bits: cross-window diff seam and
channel-group boundaries).
"""
from __future__ import annotations

import numpy as np
import pytest

from easybci_lib.tools.neural_processing.batch.streaming_stats import (
    compute_streaming_metrics,
)


# --------------------------------------------------------------------------
# Oracles: verbatim copies of the pre-fix full-array implementations
# (batch/summary.py channel-variance block, _estimate_snr_db,
#  _estimate_artifact_ratio). data is (n_channels, n_samples), time = axis -1.
# --------------------------------------------------------------------------
def _ref_channel_variance(data: np.ndarray) -> tuple[float, float]:
    ch_vars = np.var(data, axis=-1)
    if ch_vars.ndim > 1:
        ch_vars = ch_vars.mean(axis=0)
    return float(np.mean(ch_vars)), float(np.std(ch_vars))


def _ref_snr_db(data: np.ndarray) -> float:
    if data.ndim == 3:
        data = data.reshape(-1, data.shape[-1])
    total_var = np.var(data)
    noise_var = np.var(np.diff(data, axis=-1)) / 2.0
    signal_var = max(total_var - noise_var, 1e-12)
    if noise_var < 1e-12:
        return 30.0
    snr = signal_var / noise_var
    return float(10 * np.log10(max(snr, 1e-6)))


def _ref_artifact_ratio(data: np.ndarray) -> float:
    if data.ndim == 3:
        data = data.reshape(-1, data.shape[-1])
    median = np.median(data, axis=-1, keepdims=True)
    mad = np.median(np.abs(data - median), axis=-1, keepdims=True)
    mad = np.maximum(mad, 1e-12)
    threshold = 5.0 * mad
    artifacts = np.abs(data - median) > threshold
    return float(np.mean(artifacts))


# --------------------------------------------------------------------------
# Lazy sliceable shim: exposes .shape + [t0:t1, :] over a (n_samp, n_ch)
# array WITHOUT ever materializing the whole thing to the caller. Mirrors an
# h5py Dataset so tests drive the streaming path without needing HDF5.
# --------------------------------------------------------------------------
class _LazyShim:
    def __init__(self, arr_samp_ch: np.ndarray):
        # arr is (n_samples, n_channels) — same layout as NWB on disk
        self._a = arr_samp_ch
        self.shape = arr_samp_ch.shape
        self.dtype = arr_samp_ch.dtype

    def __getitem__(self, key):
        return self._a[key]


def _make_data(seed: int = 0, n_ch: int = 20, n_samp: int = 50_000):
    """(n_ch, n_samp) float32 with injected artifacts + one near-flat channel."""
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n_ch, n_samp)).astype(np.float32)
    # Inject strong artifacts on a few channels so per-channel 5*MAD fires.
    for c in {2, 7, 13} & set(range(n_ch)):
        idx = rng.choice(n_samp, size=40, replace=False)
        data[c, idx] += 50.0 * np.float32(data[c].std())
    # Near-flat channel → exercises mad→1e-12 / noise_var<1e-12 guards.
    flat = 5 if n_ch > 5 else n_ch - 1
    data[flat, :] = np.float32(1e-9) * rng.standard_normal(n_samp).astype(np.float32)
    return data


def _shim_from(data_ch_samp: np.ndarray) -> _LazyShim:
    # streaming source is (n_samples, n_channels) = data.T
    return _LazyShim(np.ascontiguousarray(data_ch_samp.T))


# --------------------------------------------------------------------------
# Equivalence
# --------------------------------------------------------------------------
def test_channel_variance_matches_reference():
    data = _make_data()
    ref_mean, ref_std = _ref_channel_variance(data)
    m = compute_streaming_metrics(_shim_from(data))
    np.testing.assert_allclose(m.channel_variance_mean, ref_mean, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(m.channel_variance_std, ref_std, rtol=1e-4, atol=1e-6)


def test_snr_db_matches_reference():
    data = _make_data()
    ref = _ref_snr_db(data)
    m = compute_streaming_metrics(_shim_from(data))
    np.testing.assert_allclose(m.snr_db, ref, rtol=1e-3, atol=1e-3)


def test_artifact_ratio_matches_reference():
    data = _make_data()
    ref = _ref_artifact_ratio(data)
    m = compute_streaming_metrics(_shim_from(data))
    np.testing.assert_allclose(m.artifact_ratio, ref, atol=1e-6)


def test_artifact_count_exact_when_far_from_threshold():
    """Artifacts far above 5*MAD → count must match reference EXACTLY."""
    rng = np.random.default_rng(3)
    data = rng.standard_normal((8, 20_000)).astype(np.float32)
    for c in (1, 4):
        idx = rng.choice(20_000, size=25, replace=False)
        data[c, idx] += 500.0  # far beyond any float32 median-boundary wobble
    ref = _ref_artifact_ratio(data)
    m = compute_streaming_metrics(_shim_from(data))
    assert m.artifact_ratio == pytest.approx(ref, abs=1e-12)


def test_reports_shape():
    data = _make_data(n_ch=17, n_samp=12_345)
    m = compute_streaming_metrics(_shim_from(data))
    assert m.n_channels == 17
    assert m.n_samples == 12_345


# --------------------------------------------------------------------------
# Invariance to window / channel-group sizing (seam-diff carry + group bounds)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("window_samples", [1, 7, 999, 4096, 50_000, 60_000])
def test_window_invariance(window_samples):
    data = _make_data()
    base = compute_streaming_metrics(_shim_from(data), window_samples=50_000)
    got = compute_streaming_metrics(_shim_from(data), window_samples=window_samples)
    np.testing.assert_allclose(got.channel_variance_mean, base.channel_variance_mean, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(got.snr_db, base.snr_db, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(got.artifact_ratio, base.artifact_ratio, rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("budget_mb", [0.01, 0.1, 1.0, 4096.0])
def test_channel_group_invariance(budget_mb):
    """Forcing G=1..all channels per sweep must not change results."""
    data = _make_data()
    base = compute_streaming_metrics(_shim_from(data), memory_budget_mb=4096.0)
    got = compute_streaming_metrics(_shim_from(data), memory_budget_mb=budget_mb)
    np.testing.assert_allclose(got.artifact_ratio, base.artifact_ratio, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(got.snr_db, base.snr_db, rtol=1e-6, atol=1e-6)


# --------------------------------------------------------------------------
# End-to-end: real NWB round-trip through _populate_subject_metrics
# --------------------------------------------------------------------------
def test_nwb_roundtrip_matches_reference(tmp_path):
    pytest.importorskip("pynwb")
    from easybci_lib.tools.neural_processing.output.nwb_writer import save_nwb
    from easybci_lib.tools.neural_processing.batch.summary import (
        SubjectQCSummary,
        _populate_subject_metrics,
    )

    data = _make_data(n_ch=12, n_samp=30_000)
    out = tmp_path / "sub-X" / "ses-001" / "DTX_preprocessed.nwb"
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {"sfreq": 500.0, "ch_names": [f"C{i}" for i in range(data.shape[0])]}
    save_nwb({"data": data, "meta": meta}, out, meta)

    subject = SubjectQCSummary(subject_id="X", passed=True)
    _populate_subject_metrics(subject, str(out), str(tmp_path))

    ref_var_mean, ref_var_std = _ref_channel_variance(data)
    np.testing.assert_allclose(subject.channel_variance_mean, ref_var_mean, rtol=1e-3, atol=1e-5)
    np.testing.assert_allclose(subject.snr_db, _ref_snr_db(data), rtol=1e-3, atol=1e-2)
    np.testing.assert_allclose(subject.artifact_ratio, _ref_artifact_ratio(data), atol=1e-4)
    assert subject.n_channels == 12
