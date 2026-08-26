import json
from pathlib import Path

from easybci_lib.tools.neural_processing.export.repo_builder import build_mini_repo


def test_pipeline_record_carries_scenario_deliverables_from_goal_json(tmp_path):
    """build_mini_repo recovers scenario/deliverables from plan/goal.json
    (written at confirm) even when the caller does not pass them explicitly,
    and stamps them into plan/pipeline_record.json (status=ok main path)."""
    wd = tmp_path / "x_preprocess_work_dir"
    plan = wd / "plan"
    plan.mkdir(parents=True)
    (plan / "goal.json").write_text(json.dumps({
        "analysis_goal": "classification",
        "scenario": "clinical",
        "deliverables": ["preprocessed", "ai_ready"],
    }))
    build_mini_repo(
        output_dir=str(wd), steps=["bandpass:1,40"], data_info={"n_channels": 4},
        pipeline_record={"status": "ok"}, modality="eeg", paradigm="erp",
        status="ok",
    )
    rec = json.loads((plan / "pipeline_record.json").read_text())
    assert rec["scenario"] == "clinical"
    assert rec["deliverables"] == ["preprocessed", "ai_ready"]


def test_pipeline_record_defaults_when_absent(tmp_path):
    """No scenario/deliverables anywhere → defaults research / [preprocessed]."""
    wd = tmp_path / "y_preprocess_work_dir"
    (wd / "plan").mkdir(parents=True)
    build_mini_repo(
        output_dir=str(wd), steps=["bandpass:1,40"], data_info={"n_channels": 4},
        pipeline_record={"status": "ok", "analysis_goal": "generic"},
        modality="eeg", paradigm="erp", status="ok",
    )
    rec = json.loads((wd / "plan" / "pipeline_record.json").read_text())
    assert rec["scenario"] == "research"
    assert rec["deliverables"] == ["preprocessed"]
