import json
from pathlib import Path

from easybci_lib.tools import neural_tools


def _mk_inspection(work_dir: Path):
    middle = work_dir / "middle_process"
    middle.mkdir(parents=True, exist_ok=True)
    report = {"data_path": str(work_dir / "raw.fif"), "modality": "eeg",
              "fingerprint": {"n_channels": 4, "sampling_freq_hz": 250}}
    (middle / "inspection_report.json").write_text(json.dumps(report), "utf-8")


def _run_propose(work_dir: Path, deliverables=None, scenario="clinical"):
    _mk_inspection(work_dir)
    args = {
        "mode": "propose",
        "modality": "eeg",
        "paradigm": "erp",
        "analysis_goal": "classification",
        "scenario": scenario,
        "output_path": str(work_dir),
        "work_dir": str(work_dir),
        "steps": [{"operator": "bandpass", "params": {"l_freq": 1, "h_freq": 40}}],
        "rationale": ["Bandpass 1-40 Hz to retain ERP band while removing drift and high-freq noise. Standard for ERP classification. Applied with FIR zero-phase."],
    }
    if deliverables is not None:
        args["deliverables"] = deliverables
    neural_tools._do_handle_plan_pipeline(args)
    staged = json.loads(
        (work_dir / "middle_process" / "proposal.staged.json").read_text("utf-8")
    )
    return staged


def test_proposal_and_goal_carry_scenario_deliverables(tmp_path):
    wd = tmp_path / "subj_preprocess_work_dir"
    wd.mkdir()
    staged = _run_propose(wd, deliverables=["ai_ready"], scenario="clinical")
    proposal = json.loads(staged["plan_files"]["proposal.json"])
    goal = json.loads(staged["plan_files"]["goal.json"])
    assert proposal["scenario"] == "clinical"
    assert proposal["deliverables"] == ["preprocessed", "ai_ready"]
    assert goal["scenario"] == "clinical"
    assert goal["deliverables"] == ["preprocessed", "ai_ready"]
    # envelope top-level too (confirm reads these)
    assert staged["scenario"] == "clinical"
    assert staged["deliverables"] == ["preprocessed", "ai_ready"]


def test_proposal_default_deliverables_is_preprocessed_only(tmp_path):
    wd = tmp_path / "subj2_preprocess_work_dir"
    wd.mkdir()
    staged = _run_propose(wd, deliverables=None, scenario="research")
    proposal = json.loads(staged["plan_files"]["proposal.json"])
    assert proposal["deliverables"] == ["preprocessed"]
