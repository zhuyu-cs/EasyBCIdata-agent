"""Regression: goal × deliverables decoupling must not false-flag an NWB-only
run (deliverables=['preprocessed']) as missing code/build_ai_ready.py, even
when the analysis_goal's legacy produces_ai_ready hint is True."""
import json
from pathlib import Path

from easybci_lib.tools.neural_processing.export.layout_spec import resolve_for_goal
from easybci_lib.tools.neural_processing.export.layout_repair import (
    detect_violations, _read_deliverables_for_layout,
)


def test_resolve_for_goal_deliverables_overrides_goal_hint():
    # classification.produces_ai_ready == True, but deliverables says NWB-only
    r = resolve_for_goal("classification", ["preprocessed"])
    assert r.produces_ai_ready is False
    assert "build_ai_ready.py" not in r.code_files_required
    # explicit ai_ready request → expected again
    r2 = resolve_for_goal("classification", ["preprocessed", "ai_ready"])
    assert r2.produces_ai_ready is True
    assert "build_ai_ready.py" in r2.code_files_required


def test_resolve_for_goal_legacy_none_keeps_goal_hint():
    # deliverables=None (legacy callers) → unchanged behaviour (goal hint wins)
    r = resolve_for_goal("classification", None)
    assert r.produces_ai_ready is True


def _mk_run(wd: Path, deliverables, with_build_ai_ready: bool):
    (wd / "plan").mkdir(parents=True)
    (wd / "plan" / "proposal.json").write_text(json.dumps({
        "analysis_goal": "classification", "deliverables": deliverables,
    }))
    code = wd / "code"
    code.mkdir()
    for f in ("pipeline.py", "qc.py", "run.py", "requirements.txt"):
        (code / f).write_text("# stub")
    if with_build_ai_ready:
        (code / "build_ai_ready.py").write_text("# stub")


def test_no_false_missing_build_ai_ready_for_nwb_only_run(tmp_path):
    wd = tmp_path / "n_preprocess_work_dir"
    wd.mkdir()
    _mk_run(wd, ["preprocessed"], with_build_ai_ready=False)
    deliv = _read_deliverables_for_layout(wd)
    assert deliv == ["preprocessed"]
    resolved = resolve_for_goal("classification", deliv)
    viols = [v.kind for v in detect_violations(wd, resolved=resolved)]
    assert "missing_file:code/build_ai_ready.py" not in viols


def test_missing_build_ai_ready_still_flagged_when_ai_ready_requested(tmp_path):
    wd = tmp_path / "m_preprocess_work_dir"
    wd.mkdir()
    _mk_run(wd, ["preprocessed", "ai_ready"], with_build_ai_ready=False)
    deliv = _read_deliverables_for_layout(wd)
    resolved = resolve_for_goal("classification", deliv)
    viols = [v.kind for v in detect_violations(wd, resolved=resolved)]
    assert "missing_file:code/build_ai_ready.py" in viols
