from easybci_lib.tools.neural_processing.io.channel_classifier import classify_channels

_PSG = ["C4-M1", "E1-M2", "EMG-L", "SpO2", "NPress", "Thor", "Abdo",
        "Position", "Pleth", "Sound", "Therm", "Pulse"]


def test_psg_aux_channels_are_kept_not_suggest_dropped():
    res = classify_channels(_PSG, modality="eeg", psg_context=True)
    # EEG stays data
    assert res["categories"]["C4-M1"] == "data"
    # Aux physio channels must NOT be suggested for drop in PSG context
    for aux in ["SpO2", "NPress", "Thor", "Abdo", "Position", "Pleth", "Sound", "Therm", "Pulse"]:
        assert aux not in res["suggest_drop"], f"{aux} should be kept for PSG"
    assert set(res["psg_aux"]) >= {"SpO2", "NPress", "Thor", "Abdo", "Position"}


def test_non_psg_context_unchanged():
    # Without psg_context, SpO2/Pleth keep prior physio suggest-drop behavior
    res = classify_channels(["C4-M1", "SpO2", "Pleth"], modality="eeg")
    assert "SpO2" in res["suggest_drop"]
