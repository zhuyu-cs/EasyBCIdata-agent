from pathlib import Path

import numpy as np

from easybci_lib.tools.neural_processing.io import compumedics_loader as cl


def test_parse_studycfg_returns_channel_specs(slp_dir):
    specs = cl.parse_studycfg(str(slp_dir / "STUDYCFG.XML"))
    assert [s.label for s in specs] == ["C4-M1", "SpO2", "Thor"]
    assert specs[0].rate == 200
    assert specs[0].sensitivity == 0.004
    assert specs[1].sensitivity is None      # SpO2 has none
    assert specs[2].filename == "CHANNEL3.DAT"


def test_matches_directory_only(slp_dir, tmp_path):
    assert cl.matches(str(slp_dir)) is True
    assert cl.matches(str(tmp_path / "foo.edf")) is False


def test_load_resamples_all_channels_to_common_rate(slp_dir):
    result = cl.load(str(slp_dir), target_hz=100.0)
    assert result["frequency"] == 100.0
    assert result["channels"] == ["C4-M1", "SpO2", "Thor"]
    # 120 s @ 100 Hz = 12000 samples, all channels aligned to one array
    assert result["data"].shape == (3, 12000)
    assert result["data"].dtype == np.float32
    m = result["meta"]
    assert m["format"] == "compumedics_slp"
    assert m["native_rates"] == {"C4-M1": 200, "SpO2": 1, "Thor": 25}
    assert m["data_unit"] == "V"
    assert m["hypnogram_path"].endswith("SLPSTAG.DAT")


def test_load_default_target_is_native_max(slp_dir):
    result = cl.load(str(slp_dir))       # no target_hz
    assert result["frequency"] == 200.0  # max native rate
    assert result["data"].shape[1] == 200 * 120


def test_inspect_only_skips_full_read(slp_dir):
    result = cl.load(str(slp_dir), inspect_only=True)
    assert result["meta"]["n_channels"] == 3
    assert result["duration"] == 120.0


def test_detect_backend_recognizes_slp_dir(slp_dir):
    from easybci_lib.tools.neural_processing.io.loader import _detect_backend
    assert _detect_backend(Path(str(slp_dir))) == "compumedics"


def test_load_neural_routes_slp_to_compumedics(slp_dir):
    from easybci_lib.tools.neural_processing.io.loader import load_neural
    result = load_neural(str(slp_dir), target_hz=100.0)
    assert result["meta"]["format"] == "compumedics_slp"
    assert result["data"].shape == (3, 12000)
