from easybci_lib.tools.neural_processing.preprocess.scenario import (
    SCENARIO_REGISTRY, ScenarioSpec, is_valid_scenario, get_scenario, DEFAULT_SCENARIO,
)


def test_registry_has_three_scenarios():
    assert set(SCENARIO_REGISTRY) == {"research", "clinical", "deployment"}


def test_default_scenario_is_research():
    assert DEFAULT_SCENARIO == "research"
    assert is_valid_scenario("research") is True


def test_invalid_scenario_rejected():
    assert is_valid_scenario("industrial") is False


def test_each_spec_has_required_fields():
    for name, spec in SCENARIO_REGISTRY.items():
        assert isinstance(spec, ScenarioSpec)
        assert spec.name == name
        assert "en" in spec.display_name and "zh" in spec.display_name
        assert spec.description
        assert isinstance(spec.default_deliverables, list)
        assert "preprocessed" in spec.default_deliverables
        assert isinstance(spec.param_bias_notes, str) and spec.param_bias_notes


def test_get_scenario_returns_spec():
    assert get_scenario("clinical").name == "clinical"


def test_get_scenario_unknown_raises():
    import pytest
    with pytest.raises(KeyError):
        get_scenario("nope")
