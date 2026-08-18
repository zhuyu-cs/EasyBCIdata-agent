"""Compumedics ProFusion / Somte PSG ``.SLP`` study-bundle loader.

A ``.SLP`` is a *directory* of per-channel little-endian float32 binaries
(``CHANNELn.DAT``) described by ``STUDYCFG.XML``. Channels have mixed native
sampling rates; this loader reads each at its native rate, applies the XML
``Sensitivity`` scaling, then resamples every kept channel to one common rate
so the standard single-array easybci loader dict is preserved.

Source data is read-only (Rule 5): this module only reads.
"""
from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Channels the signal array should never contain (discrete/enum/derived). By
# XML <Type>: 1=sampled analog, 3=numeric (SpO2/pulse), 4=enum/step (Position).
# We keep 1 and 3 (numeric physio is meaningful for PSG); 4 goes to meta only.
_ENUM_TYPE = 4


@dataclass(frozen=True)
class ChannelSpec:
    label: str
    rate: int
    ctype: int
    filename: str
    sensitivity: Optional[float]
    unit: str


def matches(path: str) -> bool:
    p = Path(path)
    return p.is_dir() and (p / "STUDYCFG.XML").exists()


def parse_studycfg(xml_path: str) -> List[ChannelSpec]:
    root = ET.parse(xml_path).getroot()
    specs: List[ChannelSpec] = []
    for ch in root.findall(".//Channels/Channel"):
        def _txt(tag: str) -> Optional[str]:
            el = ch.find(tag)
            return el.text if el is not None and el.text is not None else None

        label = (_txt("Label") or "").strip()
        rate = int(_txt("Rate") or 0)
        ctype = int(_txt("Type") or 1)
        filename = (_txt("Filename") or "").strip()
        sens_raw = _txt("Sensitivity")
        sensitivity = float(sens_raw) if sens_raw not in (None, "") else None
        unit = (_txt("UnitOfMeasure") or "").strip()
        if label and filename:
            specs.append(ChannelSpec(label, rate, ctype, filename, sensitivity, unit))
    return specs


def _read_dat(path: Path, sensitivity: Optional[float]) -> np.ndarray:
    suffix = path.suffix.upper()
    if suffix == ".D16":
        raw = np.fromfile(path, dtype="<i2").astype(np.float32)
        if sensitivity is not None:
            raw *= np.float32(sensitivity)
    else:
        # .DAT float32 files are already in physical units.
        raw = np.fromfile(path, dtype="<f4")
    return raw.astype(np.float32, copy=False)


def _resample_to(sig: np.ndarray, src_hz: int, dst_hz: float) -> np.ndarray:
    if src_hz == dst_hz or sig.size == 0:
        return sig.astype(np.float32, copy=False)
    from math import gcd
    from scipy.signal import resample_poly
    up, down = int(round(dst_hz)), int(src_hz)
    g = gcd(up, down) or 1
    return resample_poly(sig, up // g, down // g).astype(np.float32, copy=False)


def load(path: str, inspect_only: bool = False, target_hz: Optional[float] = None) -> dict:
    d = Path(path)
    specs = parse_studycfg(str(d / "STUDYCFG.XML"))
    kept = [s for s in specs if s.ctype != _ENUM_TYPE]
    native_rates = {s.label: s.rate for s in kept}

    durations: List[float] = []
    for s in kept:
        f = d / s.filename
        if f.exists() and s.rate > 0:
            durations.append((f.stat().st_size // 4) / s.rate)
    # Use the longest channel duration (highest-rate channels carry the true
    # recording length; low-rate auxiliaries like SpO2 may be shorter).
    duration = float(max(durations)) if durations else 0.0

    common_hz = float(target_hz) if target_hz else float(
        max((s.rate for s in kept), default=1))

    meta = {
        "format": "compumedics_slp",
        "source_file": str(d),
        "device": _device(d),
        "native_rates": native_rates,
        "epoch_length_s": 30,
        "hypnogram_path": str(d / "SLPSTAG.DAT"),
        "events_path": str(d / "EVENTS.MDB"),
        "data_unit": "V",
        "n_channels": len(kept),
    }

    if inspect_only:
        meta["n_samples"] = int(duration * common_hz)
        return {
            "data": np.zeros((len(kept), min(int(common_hz), 1)), dtype=np.float32),
            "frequency": common_hz, "channels": [s.label for s in kept],
            "duration": duration, "meta": meta,
        }

    n_target = int(duration * common_hz)
    channels: list[str] = []
    rows: list[np.ndarray] = []
    for s in kept:
        f = d / s.filename
        if not f.exists():
            logger.warning("Compumedics: channel file missing, skipping: %s", s.filename)
            continue
        sig = _read_dat(f, s.sensitivity)
        sig = _resample_to(sig, s.rate, common_hz)
        if sig.size < n_target:
            sig = np.pad(sig, (0, n_target - sig.size))
        elif sig.size > n_target:
            sig = sig[:n_target]
        channels.append(s.label)
        rows.append(sig)

    data = (np.vstack(rows) if rows
            else np.zeros((0, n_target), dtype=np.float32)).astype(np.float32)
    meta["n_channels"] = len(channels)
    return {"data": data, "frequency": common_hz, "channels": channels,
            "duration": duration, "meta": meta}


def _device(d: Path) -> str:
    try:
        root = ET.parse(d / "STUDYCFG.XML").getroot()
        el = root.find("Device")
        return el.text.strip() if el is not None and el.text else ""
    except Exception:
        return ""
