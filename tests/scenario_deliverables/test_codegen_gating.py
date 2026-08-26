import json
from pathlib import Path

import pytest

from easybci_lib.tools import neural_tools


def _prep_confirmed_work_dir(wd: Path, deliverables, events=None):
    middle = wd / "middle_process"
    (middle / "inspect").mkdir(parents=True, exist_ok=True)
    # confirm marker with deliverables
    (middle / "proposal.confirmed").write_text(json.dumps({
        "user_decision": "confirm", "scenario": "research",
        "deliverables": deliverables,
    }), "utf-8")
    # minimal inspection_report.json (legacy mirror path)
    report = {"data_path": str(wd / "raw.fif"), "modality": "eeg",
              "fingerprint": {"n_channels": 4, "sampling_freq_hz": 250}}
    (middle / "inspection_report.json").write_text(json.dumps(report), "utf-8")
    # routing table
    routing = {"inputs": [{"file_id": "aaaa", "subject_id": "01",
                           "session_id": "01", "stem_safe": "raw"}]}
    if events:
        routing["inputs"][0]["events_path"] = str(wd / "events.tsv")
    (middle / "inputs_routing.json").write_text(json.dumps(routing), "utf-8")


def _args(wd: Path, data_info):
    return {
        "work_dir": str(wd), "code_only": True,
        "steps": ["bandpass:1,40"], "data_info": data_info,
        "modality": "eeg", "analysis_goal": "classification",
        "reasoning": {},
    }


def test_no_ai_ready_when_deliverables_preprocessed_only(tmp_path):
    wd = tmp_path / "a_preprocess_work_dir"
    wd.mkdir()
    _prep_confirmed_work_dir(wd, ["preprocessed"], events=True)
    out = json.loads(neural_tools._do_handle_generate_code(
        _args(wd, {"events": [{"onset": 1.0, "id": 1}]})
    ))
    assert out["success"] is True
    assert out["has_build_ai_ready"] is False
    assert not (wd / "code" / "build_ai_ready.py").exists()


def test_ai_ready_generated_when_requested_with_events(tmp_path):
    wd = tmp_path / "b_preprocess_work_dir"
    wd.mkdir()
    _prep_confirmed_work_dir(wd, ["preprocessed", "ai_ready"], events=True)
    out = json.loads(neural_tools._do_handle_generate_code(
        _args(wd, {"events": [{"onset": 1.0, "id": 1}]})
    ))
    assert out["has_build_ai_ready"] is True
    assert (wd / "code" / "build_ai_ready.py").exists()


def test_ai_ready_requested_but_no_events_returns_fix_hint(tmp_path):
    wd = tmp_path / "c_preprocess_work_dir"
    wd.mkdir()
    _prep_confirmed_work_dir(wd, ["preprocessed", "ai_ready"], events=False)
    out = json.loads(neural_tools._do_handle_generate_code(
        _args(wd, {})  # no events, no label_config
    ))
    assert out["success"] is False
    assert out.get("fix_hint")
    assert "ai_ready" in json.dumps(out)
