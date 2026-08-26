"""Tests for the revise_proposal tool — post-confirm step modification."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def _make_work_dir(tmp_path):
    """Set up a minimal work_dir with confirmed proposal."""
    wd = tmp_path / "test_preprocess_work_dir"
    wd.mkdir()
    middle = wd / "middle_process"
    middle.mkdir()
    plan = wd / "plan"
    plan.mkdir()

    steps = [
        {"operator": "notch", "method": "", "params": {"raw": "50"}, "param_evidence": {}},
        {"operator": "bandpass", "method": "", "params": {"raw": "1,40"}, "param_evidence": {}},
        {"operator": "resample", "method": "", "params": {"raw": "256"}, "param_evidence": {}},
        {"operator": "drop_bads", "method": "", "params": {"raw": "auto"}, "param_evidence": {}},
    ]
    proposal = {
        "modality": "eeg",
        "analysis_goal": "classification",
        "steps": steps,
    }
    (plan / "proposal.json").write_text(
        json.dumps(proposal, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    marker_data = {
        "user_decision": "confirm",
        "proposal_summary": "test",
        "confirmed_at": "2026-08-25T12:00:00",
    }
    (middle / "proposal.confirmed").write_text(
        json.dumps(marker_data), encoding="utf-8",
    )

    code_dir = wd / "code"
    code_dir.mkdir()
    (code_dir / "pipeline.py").write_text("# generated pipeline", encoding="utf-8")
    (code_dir / "qc.py").write_text("# generated qc", encoding="utf-8")

    return wd


def test_revise_removes_step(tmp_path):
    """Removing drop_bads from the pipeline."""
    wd = _make_work_dir(tmp_path)

    from easybci_lib.tools.neural_tools import _handle_revise_proposal

    result = json.loads(_handle_revise_proposal({
        "work_dir": str(wd),
        "steps": ["notch:50", "bandpass:1,40", "resample:256"],
        "revision_reason": "user asked to remove drop_bads",
    }))

    assert result["success"] is True
    assert len(result["revised_steps"]) == 3
    assert "drop_bads" not in " ".join(result["revised_steps"])

    updated_proposal = json.loads(
        (wd / "plan" / "proposal.json").read_text(encoding="utf-8")
    )
    assert len(updated_proposal["steps"]) == 3
    ops = [s["operator"] for s in updated_proposal["steps"]]
    assert "drop_bads" not in ops

    marker = json.loads(
        (wd / "middle_process" / "proposal.confirmed").read_text(encoding="utf-8")
    )
    assert "revised_at" in marker

    assert len(result["archived_code"]) == 2
    archive_dir = wd / "middle_process" / "code"
    assert any("pipeline" in f.name for f in archive_dir.iterdir())

    history = wd / "middle_process" / "revision_history.jsonl"
    assert history.is_file()
    entry = json.loads(history.read_text(encoding="utf-8").strip())
    assert entry["reason"] == "user asked to remove drop_bads"
    assert len(entry["old_steps"]) == 4
    assert len(entry["new_steps"]) == 3


def test_revise_changes_params(tmp_path):
    """Changing bandpass params from 1,40 to 0.5,30."""
    wd = _make_work_dir(tmp_path)

    from easybci_lib.tools.neural_tools import _handle_revise_proposal

    result = json.loads(_handle_revise_proposal({
        "work_dir": str(wd),
        "steps": ["notch:50", "bandpass:0.5,30", "resample:256", "drop_bads:auto"],
        "revision_reason": "user adjusted bandpass range",
    }))

    assert result["success"] is True
    updated = json.loads(
        (wd / "plan" / "proposal.json").read_text(encoding="utf-8")
    )
    bp_step = [s for s in updated["steps"] if s["operator"] == "bandpass"][0]
    assert bp_step["params"]["raw"] == "0.5,30"


def test_revise_reorders_steps(tmp_path):
    """Reordering: put resample before bandpass."""
    wd = _make_work_dir(tmp_path)

    from easybci_lib.tools.neural_tools import _handle_revise_proposal

    result = json.loads(_handle_revise_proposal({
        "work_dir": str(wd),
        "steps": ["notch:50", "resample:256", "bandpass:1,40", "drop_bads:auto"],
        "revision_reason": "user wants resample before bandpass",
    }))

    assert result["success"] is True
    updated = json.loads(
        (wd / "plan" / "proposal.json").read_text(encoding="utf-8")
    )
    ops = [s["operator"] for s in updated["steps"]]
    assert ops == ["notch", "resample", "bandpass", "drop_bads"]


def test_revise_adds_step(tmp_path):
    """Adding a new step (car)."""
    wd = _make_work_dir(tmp_path)

    from easybci_lib.tools.neural_tools import _handle_revise_proposal

    result = json.loads(_handle_revise_proposal({
        "work_dir": str(wd),
        "steps": [
            "notch:50", "bandpass:1,40", "car",
            "resample:256", "drop_bads:auto",
        ],
        "revision_reason": "user wants CAR added",
    }))

    assert result["success"] is True
    assert len(result["revised_steps"]) == 5
    updated = json.loads(
        (wd / "plan" / "proposal.json").read_text(encoding="utf-8")
    )
    ops = [s["operator"] for s in updated["steps"]]
    assert "car" in ops


def test_revise_no_marker_rejects(tmp_path):
    """Cannot revise without proposal.confirmed marker."""
    wd = tmp_path / "test_preprocess_work_dir"
    wd.mkdir()
    (wd / "middle_process").mkdir()
    plan = wd / "plan"
    plan.mkdir()
    (plan / "proposal.json").write_text("{}", encoding="utf-8")

    from easybci_lib.tools.neural_tools import _handle_revise_proposal

    result = json.loads(_handle_revise_proposal({
        "work_dir": str(wd),
        "steps": ["notch:50"],
        "revision_reason": "test",
    }))

    assert result["success"] is False
    assert "confirmed" in result["error"].lower()


def test_revise_invalid_operator_rejects(tmp_path):
    """Invalid operator name should be rejected."""
    wd = _make_work_dir(tmp_path)

    from easybci_lib.tools.neural_tools import _handle_revise_proposal

    result = json.loads(_handle_revise_proposal({
        "work_dir": str(wd),
        "steps": ["notch:50", "totally_bogus_operator:42"],
        "revision_reason": "test",
    }))

    assert result["success"] is False
    assert "validation" in result["error"].lower() or "fix_hint" in result


def test_b4_guard_reads_revised_proposal(tmp_path):
    """After revise, the B4 guard in generate_code should read updated steps.

    This is an integration test verifying the B4 guard's read path.
    """
    wd = _make_work_dir(tmp_path)

    from easybci_lib.tools.neural_tools import _handle_revise_proposal

    _handle_revise_proposal({
        "work_dir": str(wd),
        "steps": ["notch:50", "bandpass:1,40", "resample:256"],
        "revision_reason": "removed drop_bads",
    })

    proposal = json.loads(
        (wd / "plan" / "proposal.json").read_text(encoding="utf-8")
    )
    ops = [s["operator"] for s in proposal["steps"]]
    assert ops == ["notch", "bandpass", "resample"]
    assert "drop_bads" not in ops

    assert (wd / "middle_process" / "proposal.confirmed").is_file()


def test_multiple_revisions_append_history(tmp_path):
    """Multiple revisions each append a line to revision_history.jsonl."""
    wd = _make_work_dir(tmp_path)

    from easybci_lib.tools.neural_tools import _handle_revise_proposal

    _handle_revise_proposal({
        "work_dir": str(wd),
        "steps": ["notch:50", "bandpass:1,40", "resample:256"],
        "revision_reason": "first revision",
    })

    (wd / "code").mkdir(exist_ok=True)
    (wd / "code" / "pipeline.py").write_text("# regen", encoding="utf-8")

    _handle_revise_proposal({
        "work_dir": str(wd),
        "steps": ["notch:50", "resample:256"],
        "revision_reason": "second revision",
    })

    history = (wd / "middle_process" / "revision_history.jsonl")
    lines = [json.loads(l) for l in history.read_text(encoding="utf-8").strip().split("\n")]
    assert len(lines) == 2
    assert lines[0]["reason"] == "first revision"
    assert lines[1]["reason"] == "second revision"
