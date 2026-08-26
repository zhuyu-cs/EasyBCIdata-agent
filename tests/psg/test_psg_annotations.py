from pathlib import Path

from easybci_lib.tools.neural_processing.io import psg_annotations as pa


def test_parse_hypnogram_maps_bytes_to_stages(slp_dir):
    stages = pa.parse_hypnogram(str(slp_dir / "SLPSTAG.DAT"))
    # fixture wrote bytes [0x00, 0x02, 0x03, 0x80]
    assert stages == ["W", "N2", "N3", "unscored"]


def test_parse_hypnogram_missing_file_returns_empty(tmp_path):
    assert pa.parse_hypnogram(str(tmp_path / "nope.dat")) == []


def test_parse_hypnogram_all_unscored(tmp_path):
    p = tmp_path / "SLPSTAG.DAT"
    p.write_bytes(bytes([0x80] * 10))
    stages = pa.parse_hypnogram(str(p))
    assert stages == ["unscored"] * 10


def test_parse_events_unreadable_mdb_degrades(slp_dir):
    events, fix_hint = pa.parse_events(str(slp_dir / "EVENTS.MDB"))
    assert events == []
    assert fix_hint
    assert "pandas-access" in fix_hint or "mdbtools" in fix_hint


def test_parse_events_missing_file_degrades(tmp_path):
    events, fix_hint = pa.parse_events(str(tmp_path / "nope.mdb"))
    assert events == []
    assert fix_hint
