import json
from pathlib import Path

from easybci_lib.tools import neural_tools


def _stage_envelope(wd: Path, deliverables, scenario="clinical"):
    middle = wd / "middle_process"
    middle.mkdir(parents=True, exist_ok=True)
    envelope = {
        "version": "1", "kind": "evidence",
        "modality": "eeg", "paradigm": "erp", "analysis_goal": "classification",
        "scenario": scenario, "deliverables": deliverables,
        "root_files": {},
        "plan_files": {
            "proposal.json": json.dumps({
                "analysis_goal": "classification", "scenario": scenario,
                "deliverables": deliverables, "steps": [],
            }),
            "goal.json": json.dumps({"analysis_goal": "classification"}),
        },
    }
    (middle / "proposal.staged.json").write_text(json.dumps(envelope), "utf-8")


def test_marker_persists_scenario_and_deliverables(tmp_path):
    wd = tmp_path / "s_preprocess_work_dir"
    wd.mkdir()
    _stage_envelope(wd, ["preprocessed", "ai_ready"], "clinical")
    # empty steps → presentation guard is skipped, confirm proceeds
    neural_tools._handle_mark_proposal_confirmed(
        {"work_dir": str(wd), "user_decision": "confirm"}
    )
    marker = json.loads(
        (wd / "middle_process" / "proposal.confirmed").read_text("utf-8")
    )
    assert marker["scenario"] == "clinical"
    assert marker["deliverables"] == ["preprocessed", "ai_ready"]
