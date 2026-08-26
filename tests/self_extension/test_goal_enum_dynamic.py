"""Phase A Task 1: analysis_goal schema enum is dynamic (builtin ∪ third-party)."""
import yaml
from easybci_lib.constants import get_easybci_home


def _write_third_party_goal():
    d = get_easybci_home() / "skills" / "analysis_goals"
    d.mkdir(parents=True, exist_ok=True)
    (d / "my_custom_goal.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "my_custom_goal",
                "display_name": {"en": "My Custom Goal", "zh": "自定义目标"},
                "description": "a third-party goal",
                "inject_drop_bads": True,
                "inject_drop_nondata": True,
            }
        ),
        encoding="utf-8",
    )


def _enum_for(tool_name):
    from easybci_lib.tools import registry as _reg
    _reg.discover_builtin_tools()
    defs = _reg.registry.get_definitions({tool_name}, quiet=True)
    assert defs, f"tool {tool_name} not defined (check_fn failed?)"
    return defs[0]["function"]["parameters"]["properties"]["analysis_goal"]["enum"]


def test_plan_pipeline_enum_includes_all_builtin_and_third_party():
    _write_third_party_goal()
    from easybci_lib.tools.neural_processing.preprocess.analysis_goals import REGISTRY
    enum = _enum_for("plan_pipeline")
    for g in REGISTRY:
        assert g in enum, f"builtin goal {g} missing from plan_pipeline enum"
    assert "my_custom_goal" in enum


def test_preprocess_neural_enum_dynamic():
    _write_third_party_goal()
    enum = _enum_for("preprocess_neural")
    assert "my_custom_goal" in enum
    # previously hardcoded 6 — now full 9 builtin present too
    assert "connectivity" in enum
    assert "online_inference" in enum


def test_save_processed_enum_dynamic():
    _write_third_party_goal()
    enum = _enum_for("save_processed")
    assert "my_custom_goal" in enum
    assert "phase_amplitude_coupling" in enum


def test_no_third_party_falls_back_to_full_builtin():
    # no yaml written this test (isolated home) — static/full builtin enum
    from easybci_lib.tools.neural_processing.preprocess.analysis_goals import REGISTRY
    enum = _enum_for("plan_pipeline")
    for g in REGISTRY:
        assert g in enum
