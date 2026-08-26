from easybci_lib.tools.neural_tools import PLAN_PIPELINE_SCHEMA


def test_schema_has_scenario_enum():
    props = PLAN_PIPELINE_SCHEMA["parameters"]["properties"]
    assert "scenario" in props
    assert set(props["scenario"]["enum"]) == {"research", "clinical", "deployment"}


def test_schema_has_deliverables_array():
    props = PLAN_PIPELINE_SCHEMA["parameters"]["properties"]
    assert "deliverables" in props
    assert props["deliverables"]["type"] == "array"
    assert set(props["deliverables"]["items"]["enum"]) == {"preprocessed", "ai_ready"}


def test_scenario_deliverables_not_required():
    req = PLAN_PIPELINE_SCHEMA["parameters"]["required"]
    assert "scenario" not in req
    assert "deliverables" not in req


import json as _json
from easybci_lib.tools import neural_tools


def _mk_inspection(tmp_path):
    """Minimal work_dir + legacy inspection_report so _require_inspection_report
    passes and the scenario/deliverables validation (which runs right after the
    goal check) is actually reached."""
    wd = tmp_path / "v_preprocess_work_dir"
    middle = wd / "middle_process"
    middle.mkdir(parents=True)
    report = {"data_path": str(wd / "raw.fif"), "modality": "eeg",
              "fingerprint": {"n_channels": 4, "sampling_freq_hz": 250}}
    (middle / "inspection_report.json").write_text(_json.dumps(report), "utf-8")
    return wd


def test_plan_rejects_bad_scenario(tmp_path):
    wd = _mk_inspection(tmp_path)
    out = neural_tools._do_handle_plan_pipeline(
        {"modality": "eeg", "analysis_goal": "classification",
         "scenario": "industrial", "mode": "suggest", "work_dir": str(wd)}
    )
    assert _json.loads(out)["field"] == "scenario"


def test_plan_rejects_unknown_deliverable(tmp_path):
    wd = _mk_inspection(tmp_path)
    out = neural_tools._do_handle_plan_pipeline(
        {"modality": "eeg", "analysis_goal": "classification",
         "deliverables": ["edf"], "mode": "suggest", "work_dir": str(wd)}
    )
    assert _json.loads(out)["field"] == "deliverables"
