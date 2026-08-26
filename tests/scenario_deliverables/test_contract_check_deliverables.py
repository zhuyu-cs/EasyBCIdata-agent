from pathlib import Path

from easybci_lib.tools.neural_processing.export.contract_check import enumerate_pending


def _routing(wd: Path, events=True):
    m = wd / "middle_process"
    m.mkdir(parents=True, exist_ok=True)
    entry = {"file_id": "aa", "subject_id": "01", "session_id": "01", "stem_safe": "raw"}
    if events:
        entry["events_path"] = str(wd / "events.tsv")
    import json
    (m / "inputs_routing.json").write_text(json.dumps({"inputs": [entry]}), "utf-8")
    # NWB present so 'preprocessed' is satisfied; figures/qc irrelevant here
    nwb = wd / "preprocessed_output" / "preprocessed" / "sub-01" / "ses-01"
    nwb.mkdir(parents=True)
    (nwb / "raw_preprocessed.nwb").write_text("x")


def test_ai_ready_not_missing_when_not_in_deliverables(tmp_path):
    wd = tmp_path / "d_preprocess_work_dir"
    wd.mkdir()
    _routing(wd, events=True)
    snap = enumerate_pending(wd, deliverables=["preprocessed"])
    missing = snap["missing_by_entry"]["aa"]["missing"]
    assert "ai_ready" not in missing


def test_ai_ready_missing_when_in_deliverables_and_absent(tmp_path):
    wd = tmp_path / "e_preprocess_work_dir"
    wd.mkdir()
    _routing(wd, events=True)
    snap = enumerate_pending(wd, deliverables=["preprocessed", "ai_ready"])
    missing = snap["missing_by_entry"]["aa"]["missing"]
    assert "ai_ready" in missing
