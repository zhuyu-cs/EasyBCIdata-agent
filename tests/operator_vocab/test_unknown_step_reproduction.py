"""Reproduction + regression tests for the unknown-operator silent-skip bug.

Bug: LLM-authored pipelines occasionally use non-canonical operator names
(``highpass`` / ``lowpass`` / ``bad_channels``). The runtime engine
(``preprocess.pipeline.preprocess``) does not recognise them and SILENTLY skips
them (``logger.warning`` + ``continue``), dropping core preprocessing steps
without failing. Root cause: no single operator vocabulary + no pre-execution
name validation/normalization.

Fix direction (chosen): a single operator-vocabulary source of truth with a
synonym map that normalizes ALL known synonyms before execution, and fail-loud
(raise) at the bottom layers as a defense-in-depth backstop.
"""
from __future__ import annotations

import numpy as np
import pytest


def _make_data_dict(n_channels: int = 4, sfreq: float = 250.0, seconds: float = 4.0):
    n = int(sfreq * seconds)
    rng = np.random.default_rng(0)
    return {
        "data": rng.standard_normal((n_channels, n)).astype("float64"),
        "frequency": sfreq,
        "channels": [f"CH{i}" for i in range(n_channels)],
        "meta": {"bad_channels": []},
    }


# ── Phase 1 reproduction: prove the synonyms are silently skipped today ──────

def test_repro_highpass_silently_skipped_by_engine():
    """highpass is not a canonical engine operator → currently skipped."""
    from easybci_lib.tools.neural_processing.preprocess.pipeline import preprocess

    d = preprocess(_make_data_dict(), steps=["highpass:1"])
    applied = d["meta"]["preprocessing"]
    # BUG (pre-fix): highpass is dropped, so `applied` is empty.
    # POST-FIX: it is normalized to bandpass:1, and recorded as applied.
    assert applied, "highpass was silently skipped — core step lost"


def test_repro_bad_channels_silently_skipped_by_engine():
    from easybci_lib.tools.neural_processing.preprocess.pipeline import preprocess

    d = preprocess(_make_data_dict(), steps=["bad_channels"])
    applied = d["meta"]["preprocessing"]
    assert applied, "bad_channels was silently skipped — core step lost"
