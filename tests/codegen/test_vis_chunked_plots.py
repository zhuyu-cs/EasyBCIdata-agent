"""Equivalence + syntax tests for the generated vis.py plotting helpers.

After the memmap loader change, the generated vis.py must not fully
materialize the (n_ch, n_samp) array. Two helpers touched all channels:
  * _plot_channel_variance — rewritten to accumulate per-channel variance in
    time windows (bit-equivalent to np.var(data, axis=1)).
  * _plot_amplitude_dist — rewritten to sample from time-columns instead of
    flattening the whole array.
These tests import the ACTUAL generated vis.py and drive the helpers through a
lazy shim (mimicking a memmap) with a headless matplotlib backend.
"""
from __future__ import annotations

import importlib.util
import sys

import numpy as np
import pytest

from easybci_lib.tools.neural_processing.codegen.generator import (
    generate_qc_script_v2,
    generate_vis_script,
)


class _LazyShim:
    def __init__(self, arr):
        self._a = arr
        self.shape = arr.shape
        self.ndim = arr.ndim
        self.size = arr.size
        self.dtype = arr.dtype

    def __getitem__(self, key):
        return self._a[key]


def _load_vis_module(tmp_path, modality="seeg"):
    src = generate_vis_script(modality=modality, analysis_goal="classification")
    # force headless backend so savefig works in CI/tests
    src = "import matplotlib\nmatplotlib.use('Agg')\n" + src
    path = tmp_path / f"vis_{modality}.py"
    path.write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(f"vis_{modality}", str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"vis_{modality}"] = mod
    spec.loader.exec_module(mod)
    return mod


def _make(seed=0, n_ch=16, n_samp=40_000):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((n_ch, n_samp)).astype(np.float32) * 15.0)


@pytest.mark.parametrize("modality", ["seeg", "eeg"])
def test_channel_variance_plot_matches_np_var(tmp_path, modality):
    mod = _load_vis_module(tmp_path, modality)
    a = _make()
    ref = np.var(a.astype(np.float64), axis=1)

    captured = {}
    orig_bar = None

    # Intercept the variance vector by monkeypatching np.var? No — recompute the
    # same accumulation the plot uses by calling it and reading back via a stub
    # Axes. Simplest: monkeypatch plt.subplots to capture the bar heights.
    import matplotlib.pyplot as plt

    class _AxStub:
        def bar(self, xs, var, **kw):
            captured["var"] = np.asarray(var, dtype=np.float64)
        def __getattr__(self, name):
            return lambda *a, **k: None

    class _FigStub:
        def __getattr__(self, name):
            return lambda *a, **k: None

    real_subplots = plt.subplots
    plt.subplots = lambda *a, **k: (_FigStub(), _AxStub())
    try:
        mod._plot_channel_variance(_LazyShim(a), [f"C{i}" for i in range(a.shape[0])],
                                   tmp_path / "var.png")
    finally:
        plt.subplots = real_subplots

    np.testing.assert_allclose(captured["var"], ref, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("modality", ["seeg", "eeg"])
def test_amplitude_dist_runs_on_lazy_source(tmp_path, modality):
    mod = _load_vis_module(tmp_path, modality)
    a = _make(n_ch=12, n_samp=300_000)
    out = tmp_path / "amp.png"
    mod._plot_amplitude_dist(_LazyShim(a), out)
    assert out.is_file() and out.stat().st_size > 0


@pytest.mark.parametrize("modality", ["seeg", "eeg"])
def test_generated_vis_syntax_compiles(tmp_path, modality):
    src = generate_vis_script(modality=modality, analysis_goal="classification")
    compile(src, f"vis_{modality}.py", "exec")


def test_generated_qc_syntax_compiles():
    src = generate_qc_script_v2(
        steps=["notch:50"], data_info={"n_channels": 8, "sampling_rate": 500.0},
        modality="seeg", analysis_goal="classification",
    )
    compile(src, "qc.py", "exec")
