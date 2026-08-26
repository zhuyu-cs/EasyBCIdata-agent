"""Synthesized Compumedics .SLP fixture — tiny, self-contained, fast."""
import struct
from pathlib import Path

import numpy as np
import pytest

# 3 channels, 2 native rates, 4 epochs (120 s).
_DURATION_S = 120
_CHANNELS = [
    # (label, rate, type, sensitivity, unit)
    ("C4-M1", 200, 1, 0.004, "V"),   # EEG
    ("SpO2", 1, 3, None, "%"),       # oximetry
    ("Thor", 25, 1, 0.01, "V"),      # effort
]


def _studycfg_xml() -> str:
    chans = []
    for i, (label, rate, ctype, sens, unit) in enumerate(_CHANNELS, start=1):
        sens_xml = f"<Sensitivity>{sens}</Sensitivity>" if sens is not None else ""
        chans.append(
            f"<Channel><Label>{label}</Label><Rate>{rate}</Rate>"
            f"<Type>{ctype}</Type><Filename>CHANNEL{i}.DAT</Filename>"
            f"{sens_xml}<UnitOfMeasure>{unit}</UnitOfMeasure></Channel>"
        )
    return (
        '<?xml version="1.0" encoding="utf-8"?>'
        "<PSGStudyConfig><Device>Somte PSG V2</Device><EpochLength>30</EpochLength>"
        "<NotchValue>50</NotchValue>"
        f"<Channels>{''.join(chans)}</Channels></PSGStudyConfig>"
    )


@pytest.fixture
def slp_dir(tmp_path) -> Path:
    d = tmp_path / "STUDY01.SLP"
    d.mkdir()
    (d / "STUDYCFG.XML").write_text(_studycfg_xml(), encoding="utf-8")
    rng = np.random.default_rng(0)
    for i, (label, rate, *_rest) in enumerate(_CHANNELS, start=1):
        n = rate * _DURATION_S
        sig = rng.standard_normal(n).astype("<f4")
        (d / f"CHANNEL{i}.DAT").write_bytes(sig.tobytes())
    # hypnogram: 4 epochs, one unscored (0x80), rest = wake/N2/N3
    (d / "SLPSTAG.DAT").write_bytes(bytes([0x00, 0x02, 0x03, 0x80]))
    # events db present but empty/unreadable (mimics reference case)
    (d / "EVENTS.MDB").write_bytes(b"\x00" * 4096)
    return d
