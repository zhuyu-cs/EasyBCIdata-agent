"""Phase A Task 2: register_analysis_goal — agent self-extension of goals."""
import json
from easybci_lib.constants import get_easybci_home


def test_register_then_visible_and_usable():
    from easybci_lib.tools.neural_tools import _handle_register_analysis_goal
    res = json.loads(_handle_register_analysis_goal({
        "name": "drift_tracking",
        "display_name": {"en": "Drift Tracking", "zh": "漂移追踪"},
        "description": "cross-day stability preprocessing",
        "inject_drop_bads": True,
        "inject_drop_nondata": True,
        "rationale": "needed for a paradigm not in builtin REGISTRY",
    }))
    assert res["success"] is True, res
    p = get_easybci_home() / "skills" / "analysis_goals" / "drift_tracking.yaml"
    assert p.exists()
    # hot: loader merges it now
    from easybci_lib.tools.neural_processing.preprocess.analysis_goals import REGISTRY
    from easybci_lib.tools.neural_processing.preprocess.analysis_goals_loader import (
        load_and_merge_third_party)
    reg = dict(REGISTRY)
    load_and_merge_third_party(reg)
    assert "drift_tracking" in reg
    assert reg["drift_tracking"].notes  # rationale stored
    # next_action surfaced for weak models
    assert res["next_action"]["next_tool"] in ("plan_pipeline", "propose_pipeline")


def test_registered_goal_appears_in_tool_enum():
    from easybci_lib.tools.neural_tools import _handle_register_analysis_goal
    _handle_register_analysis_goal({"name": "my_new_goal", "description": "x"})
    from easybci_lib.tools import registry as _reg
    _reg.discover_builtin_tools()
    defs = _reg.registry.get_definitions({"plan_pipeline"}, quiet=True)
    enum = defs[0]["function"]["parameters"]["properties"]["analysis_goal"]["enum"]
    assert "my_new_goal" in enum


def test_reject_bad_name_with_fix_hint():
    from easybci_lib.tools.neural_tools import _handle_register_analysis_goal
    res = json.loads(_handle_register_analysis_goal({"name": "bad name!!", "description": "x"}))
    assert res["success"] is False
    assert res.get("fix_hint")


def test_missing_description_rejected():
    from easybci_lib.tools.neural_tools import _handle_register_analysis_goal
    res = json.loads(_handle_register_analysis_goal({"name": "no_desc"}))
    assert res["success"] is False


def test_conflict_with_builtin_surfaced():
    from easybci_lib.tools.neural_tools import _handle_register_analysis_goal
    res = json.loads(_handle_register_analysis_goal({
        "name": "classification", "description": "clash with builtin"}))
    assert res["success"] is False
    combined = (res.get("error", "") + res.get("fix_hint", "")).lower()
    assert "builtin" in combined


def test_tool_registered_and_discoverable():
    from easybci_lib.tools import registry as _reg
    _reg.discover_builtin_tools()
    names = {d["function"]["name"] for d in _reg.registry.get_definitions(
        {"register_analysis_goal"}, quiet=True)}
    assert "register_analysis_goal" in names


def _inspection_report(tmp_path):
    """Minimal inspection report so plan_pipeline reaches the analysis_goal gate."""
    wd = tmp_path / "sub_preprocess_work_dir"
    (wd / "middle_process").mkdir(parents=True, exist_ok=True)
    rep = wd / "middle_process" / "inspection_report.json"
    rep.write_text(json.dumps({"modality": "eeg", "channels": ["c1"],
                               "sampling_rate": 250}), encoding="utf-8")
    return str(rep), str(wd)


def test_registered_goal_accepted_by_plan_pipeline_validation(tmp_path):
    """The self-registered goal must pass plan_pipeline's allowed_goals gate,
    not just appear in the schema enum. Guards the floor-not-authority union."""
    from easybci_lib.tools.neural_tools import (
        _handle_register_analysis_goal, _do_handle_plan_pipeline)
    _handle_register_analysis_goal({"name": "custom_accept_me", "description": "x"})
    rep, wd = _inspection_report(tmp_path)
    res = json.loads(_do_handle_plan_pipeline({
        "analysis_goal": "custom_accept_me", "modality": "eeg",
        "work_dir": wd, "inspection_report_path": rep, "paradigm": "default"}))
    err = (res.get("error", "") or "")
    # Must NOT be rejected as an unknown goal (may fail later for other reasons).
    assert "not in [" not in err, f"self-registered goal was rejected: {err}"


def test_unknown_goal_rejection_surfaces_register_hint(tmp_path):
    from easybci_lib.tools.neural_tools import _do_handle_plan_pipeline
    rep, wd = _inspection_report(tmp_path)
    res = json.loads(_do_handle_plan_pipeline({
        "analysis_goal": "totally_bogus_goal_xyz", "modality": "eeg",
        "work_dir": wd, "inspection_report_path": rep, "paradigm": "default"}))
    assert res["success"] is False
    assert "register_analysis_goal" in (res.get("fix_hint", ""))


