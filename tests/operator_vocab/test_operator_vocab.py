"""Unit tests for the operator-vocabulary single source of truth."""
from __future__ import annotations

import pytest

from easybci_lib.tools.neural_processing.preprocess.operator_vocab import (
    CANONICAL_OPERATORS,
    OPERATOR_SYNONYMS,
    UnknownOperatorError,
    normalize_step,
    normalize_steps,
)


def test_canonical_set_covers_both_engines():
    # Names required by the runtime engine dispatcher
    for op in ("notch", "bandpass", "drop_bads", "drop_nondata_channels",
               "car", "ica", "resample", "reject_by_labels", "pick_channels",
               "bipolar_ref", "interpolate_bads", "hilbert", "scale", "clip",
               "fill_nan"):
        assert op in CANONICAL_OPERATORS
    # Names required by the codegen standalone _OPS
    for op in ("threshold_spike", "mua_binning"):
        assert op in CANONICAL_OPERATORS


def test_canonical_operators_are_not_synonyms():
    assert not (set(OPERATOR_SYNONYMS) & CANONICAL_OPERATORS)


def test_normalize_highpass_to_bandpass_single_sided():
    out, changed = normalize_step("highpass:1")
    assert out == "bandpass:1,"
    assert changed is True


def test_normalize_lowpass_to_bandpass_single_sided():
    out, changed = normalize_step("lowpass:40")
    assert out == "bandpass:,40"
    assert changed is True


def test_normalize_highpass_no_param_uses_default():
    out, changed = normalize_step("highpass")
    assert out == "bandpass:1.0,"
    assert changed is True


def test_normalize_bad_channels_to_drop_bads():
    out, changed = normalize_step("bad_channels")
    assert out == "drop_bads"
    assert changed is True


def test_canonical_step_passthrough_unchanged():
    out, changed = normalize_step("bandpass:1,40")
    assert out == "bandpass:1,40"
    assert changed is False


def test_case_and_whitespace_insensitive():
    out, changed = normalize_step("  HighPass:2 ")
    assert out == "bandpass:2,"
    assert changed is True


def test_truly_unknown_raises_with_suggestion():
    with pytest.raises(UnknownOperatorError) as exc:
        normalize_step("banpass:1,40")
    msg = str(exc.value)
    assert "banpass" in msg
    assert "bandpass" in msg  # nearest-match suggestion


def test_normalize_steps_batch_reports_changes():
    out, notes = normalize_steps(["highpass:1", "notch:50", "bad_channels"])
    assert out == ["bandpass:1,", "notch:50", "drop_bads"]
    # one note per normalized step (notch untouched)
    assert len(notes) == 2
    assert any("highpass" in n and "bandpass:1," in n for n in notes)
    assert any("bad_channels" in n and "drop_bads" in n for n in notes)
