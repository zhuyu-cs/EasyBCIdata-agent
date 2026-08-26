"""B6-extension: a pickled BARE ndarray (no dict wrapper) silently assumed
256 Hz with NO warning — the same latent bug B6 fixed for `.npy`. A wrong
sampling rate silently corrupts duration / resample decisions / event-onset
math, so it must be surfaced loudly (like the .npy and dict-without-fs paths).
"""

import pickle
import tempfile
import logging
from pathlib import Path

import numpy as np

from easybci_lib.tools.neural_processing.io.loader import _load_pkl


def _write_bare_ndarray_pkl():
    p = Path(tempfile.mkdtemp()) / "bare.pkl"
    arr = np.random.RandomState(0).randn(4, 500).astype(np.float32)
    with open(p, "wb") as f:
        pickle.dump(arr, f)
    return str(p)


def test_bare_ndarray_pkl_warns_on_assumed_rate(caplog):
    path = _write_bare_ndarray_pkl()
    with caplog.at_level(logging.WARNING):
        res = _load_pkl(path)
    assert res["data"].shape == (4, 500)
    assert res["frequency"] == 256.0  # still assumed, but…
    msgs = " ".join(r.getMessage().lower() for r in caplog.records)
    assert "sampling rate" in msgs or "256" in msgs, (
        "bare-ndarray pkl must warn about the assumed sampling rate"
    )
