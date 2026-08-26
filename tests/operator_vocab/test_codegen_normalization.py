"""Codegen path: synonyms are normalized BEFORE embedding into scripts.

Regression for the second silent-skip location (codegen standalone bundle):
- generate_pipeline_script / generate_qc_script_v2 must normalize synonyms to
  canonical names before ``repr()``-embedding them, so the generated script's
  EASYBCI_STEPS marker and dispatch only ever see canonical operators.
- The generated _apply_step must FAIL LOUD on an unknown op (no silent skip).
"""
from __future__ import annotations

from easybci_lib.tools.neural_processing.codegen.generator import (
    generate_pipeline_script,
    generate_qc_script_v2,
)

_DATA_INFO = {"n_channels": 4, "sampling_rate": 250.0, "channels": ["C1", "C2", "C3", "C4"]}


def test_highpass_normalized_in_generated_pipeline():
    script = generate_pipeline_script(
        steps=["highpass:1", "notch:50"],
        data_info=_DATA_INFO,
        modality="eeg",
        analysis_goal="generic",
    )
    # synonym rewritten to canonical single-sided bandpass
    assert "bandpass:1," in script
    # the synonym token must NOT survive into the embedded step list
    assert "highpass:1" not in script


def test_bad_channels_normalized_in_generated_pipeline():
    script = generate_pipeline_script(
        steps=["bad_channels", "car"],
        data_info=_DATA_INFO,
        modality="eeg",
        analysis_goal="generic",
    )
    assert "drop_bads" in script
    assert "bad_channels" not in script


def test_generated_apply_step_fails_loud_on_unknown():
    script = generate_pipeline_script(
        steps=["notch:50"],
        data_info=_DATA_INFO,
        modality="eeg",
        analysis_goal="generic",
    )
    # the generated bundle raises rather than printing + skipping
    assert "refusing to silently" in script
    assert "WARNING: unknown step" not in script


def test_qc_script_also_normalizes():
    script = generate_qc_script_v2(
        steps=["lowpass:40"],
        data_info=_DATA_INFO,
        modality="eeg",
        analysis_goal="generic",
    )
    assert "bandpass:,40" in script
    assert "lowpass:40" not in script
