"""PSG annotation parsers — AASM hypnogram + best-effort event DB.

Both parsers are defensive: a sleep study may be unscored (all epochs
``0x80``), the staging file may be missing or length-mismatched with the
signal, and the event DB may be empty or unreadable. Nothing here raises on
bad input — callers get empty results plus (for events) a ``fix_hint``.
Read-only (Rule 5).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)

# Compumedics ProFusion AASM staging byte → stage label. 0x80 = unscored.
_STAGE_MAP = {0x00: "W", 0x01: "N1", 0x02: "N2", 0x03: "N3",
              0x05: "REM", 0x80: "unscored"}


def parse_hypnogram(path: str) -> List[str]:
    """One label per 30 s epoch. Missing file → []. Unknown byte → 'unscored'."""
    p = Path(path)
    if not p.exists():
        return []
    try:
        raw = p.read_bytes()
    except OSError:
        logger.warning("Could not read hypnogram: %s", path)
        return []
    return [_STAGE_MAP.get(b, "unscored") for b in raw]


_EVENTS_FIX_HINT = (
    "Respiratory/arousal/desat events could not be extracted from EVENTS.MDB. "
    "Install the reader with `easybci` dependency `neural.mdb` (pip "
    "`pandas-access`) AND the system `mdbtools` package (provides `mdb-export`). "
    "The pipeline continues on signal + hypnogram without events."
)


def parse_events(mdb_path: str) -> Tuple[List[dict], str]:
    """Best-effort MS-Access event extraction.

    Returns ``(events, fix_hint)``. On any failure — missing file, no reader,
    empty/corrupt DB — returns ``([], hint)``. Never raises. Each event dict:
    ``{"onset": float_s, "duration": float_s, "label": str}``.
    """
    p = Path(mdb_path)
    if not p.exists() or p.stat().st_size == 0:
        return [], _EVENTS_FIX_HINT
    try:
        import pandas_access as mdb  # optional, needs mdbtools system binary
    except Exception:
        return [], _EVENTS_FIX_HINT
    try:
        tables = mdb.list_tables(str(p))
    except Exception:
        logger.info("EVENTS.MDB unreadable (no mdbtools or non-standard DB): %s", mdb_path)
        return [], _EVENTS_FIX_HINT
    events: List[dict] = []
    for tbl in tables:
        if "event" not in tbl.lower():
            continue
        try:
            df = mdb.read_table(str(p), tbl)
        except Exception:
            continue
        for _, row in df.iterrows():
            onset = _first_num(row, ("Start", "Onset", "StartTime", "Time"))
            dur = _first_num(row, ("Duration", "Length", "Dur"))
            label = _first_str(row, ("Type", "Event", "Name", "Label"))
            if onset is not None:
                events.append({"onset": float(onset),
                               "duration": float(dur or 0.0),
                               "label": label or "event"})
    if not events:
        return [], _EVENTS_FIX_HINT
    return events, ""


def _first_num(row, keys):
    for k in keys:
        if k in row and row[k] == row[k]:  # not NaN
            try:
                return float(row[k])
            except (TypeError, ValueError):
                continue
    return None


def _first_str(row, keys):
    for k in keys:
        if k in row and isinstance(row[k], str) and row[k].strip():
            return row[k].strip()
    return None
