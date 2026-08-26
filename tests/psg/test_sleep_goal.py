from easybci_lib.tools.neural_processing.preprocess.analysis_goals import REGISTRY


def test_sleep_staging_goal_registered():
    assert "sleep_staging" in REGISTRY
    g = REGISTRY["sleep_staging"]
    assert g.produces_figures is True
    assert g.allow_ica is False
    assert g.inject_drop_nondata is False
    assert g.crystallize_eligible is True
    assert "zh" in g.display_name
