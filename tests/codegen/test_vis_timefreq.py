"""Tests for the invasive-only per-channel time-frequency spectrograms.

The GT reference (test_data/sEEG_preprocessing_gt/Code/02_process_nk_example.py
:make_spectrogram_figures) reads each channel as a single vector from a memmap,
builds a scipy.signal.spectrogram, and closes each figure immediately. We ported
that into the INVASIVE vis.py template only. These tests pin:
  * invasive vis.py exposes _plot_timefreq_per_channel and produces one PNG per
    channel with a {stem}_tf_NNN_{ch}.png name (contract_check-compatible prefix)
  * NO channel cap: 40 channels -> 40 TF PNGs
  * non-invasive vis.py is UNTOUCHED (no TF helper leaked in)
  * both templates still syntax-compile
"""
from __future__ import annotations

import importlib.util
import sys

import numpy as np
import pytest

from easybci_lib.tools.neural_processing.codegen.generator import (
    generate_vis_script,
)


def _load_vis_module(tmp_path, modality):
    src = generate_vis_script(modality=modality, analysis_goal="classification")
    src = "import matplotlib\nmatplotlib.use('Agg')\n" + src
    path = tmp_path / f"vis_{modality}.py"
    path.write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(f"vis_tf_{modality}", str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"vis_tf_{modality}"] = mod
    spec.loader.exec_module(mod)
    return mod


class _LazyShim:
    """(n_ch, n_samp) view: exposes .shape/.ndim + row/col slicing without
    ever handing back the whole array (mimics a memmap.T)."""

    def __init__(self, arr):
        self._a = arr
        self.shape = arr.shape
        self.ndim = arr.ndim
        self.size = arr.size
        self.dtype = arr.dtype

    def __getitem__(self, key):
        return self._a[key]


def _make(n_ch=40, n_samp=6000, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((n_ch, n_samp)).astype(np.float32) * 12.0)


@pytest.mark.parametrize("modality", ["seeg", "ecog", "ieeg", "dbs"])
def test_invasive_has_tf_helper(tmp_path, modality):
    mod = _load_vis_module(tmp_path, modality)
    assert hasattr(mod, "_plot_timefreq_per_channel")
    assert hasattr(mod, "_safe_name")


def test_tf_one_png_per_channel_no_cap(tmp_path):
    mod = _load_vis_module(tmp_path, "seeg")
    n_ch = 40
    a = _make(n_ch=n_ch, n_samp=6000)
    fig_dir = tmp_path / "figs"
    fig_dir.mkdir()
    channels = [f"LA{i}" for i in range(n_ch)]
    written = mod._plot_timefreq_per_channel(
        _LazyShim(a), 500.0, channels, fig_dir, "mystem"
    )
    # no channel cap: exactly one figure per channel
    assert len(written) == n_ch
    pngs = sorted(fig_dir.glob("*.png"))
    assert len(pngs) == n_ch
    # contract_check requires "<stem>_" prefix and no spaces
    for name in written:
        assert name.startswith("mystem_tf_")
        assert " " not in name
    # channel index zero-padded, safe channel name embedded
    assert (fig_dir / "mystem_tf_000_LA0.png").is_file()
    assert (fig_dir / f"mystem_tf_{n_ch - 1:03d}_LA{n_ch - 1}.png").is_file()


def test_tf_channel_name_sanitized(tmp_path):
    mod = _load_vis_module(tmp_path, "seeg")
    a = _make(n_ch=2, n_samp=3000)
    fig_dir = tmp_path / "figs2"
    fig_dir.mkdir()
    # names with spaces / slashes must be sanitized so filenames stay valid
    written = mod._plot_timefreq_per_channel(
        _LazyShim(a), 500.0, ["A 1", "B/2"], fig_dir, "s"
    )
    for name in written:
        assert " " not in name and "/" not in name.split("_tf_")[1]


def test_noninvasive_has_no_tf(tmp_path):
    mod = _load_vis_module(tmp_path, "eeg")
    # the per-channel TF helper must NOT leak into the non-invasive template
    assert not hasattr(mod, "_plot_timefreq_per_channel")


@pytest.mark.parametrize("modality", ["seeg", "ecog", "ieeg", "dbs", "spike", "eeg", "meg"])
def test_vis_syntax_compiles(modality):
    src = generate_vis_script(modality=modality, analysis_goal="classification")
    compile(src, f"vis_{modality}.py", "exec")
