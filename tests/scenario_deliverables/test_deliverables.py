from pathlib import Path

from easybci_lib.tools.neural_processing.preprocess.deliverables import (
    VALID_DELIVERABLES, DEFAULT_DELIVERABLES,
    normalize_deliverables, resolve_deliverables,
)


def test_valid_set_and_default():
    assert VALID_DELIVERABLES == {"preprocessed", "ai_ready"}
    assert DEFAULT_DELIVERABLES == ["preprocessed"]


def test_normalize_none_returns_default_copy():
    out = normalize_deliverables(None)
    assert out == ["preprocessed"]
    out.append("ai_ready")  # mutating result must not corrupt DEFAULT_DELIVERABLES
    assert DEFAULT_DELIVERABLES == ["preprocessed"]


def test_normalize_always_includes_preprocessed():
    assert normalize_deliverables(["ai_ready"]) == ["preprocessed", "ai_ready"]


def test_normalize_dedups_and_orders():
    assert normalize_deliverables(["ai_ready", "preprocessed", "ai_ready"]) == [
        "preprocessed", "ai_ready",
    ]


def test_normalize_unknown_raises():
    import pytest
    with pytest.raises(ValueError) as exc:
        normalize_deliverables(["edf"])
    assert "edf" in str(exc.value)


def test_resolve_reads_explicit_field():
    rec = {"deliverables": ["preprocessed", "ai_ready"]}
    assert resolve_deliverables(rec) == ["preprocessed", "ai_ready"]


def test_resolve_legacy_no_field_defaults_preprocessed(tmp_path: Path):
    assert resolve_deliverables({}, work_dir=tmp_path) == ["preprocessed"]


def test_resolve_legacy_infers_ai_ready_from_existing_output(tmp_path: Path):
    # legacy record without deliverables, but AI_ready artefacts already on disk
    ai = tmp_path / "preprocessed_output" / "AI_ready" / "01" / "ses-01"
    ai.mkdir(parents=True)
    (ai / "x_epochs.pkl").write_text("x")
    assert resolve_deliverables({}, work_dir=tmp_path) == ["preprocessed", "ai_ready"]
