"""Resolve cohort_tag for a recording, priority: BIDS participants.tsv > CLI > empty."""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_SUB_RE = re.compile(r"sub-([A-Za-z0-9]+)")


def _find_bids_root(data_path: Path) -> Optional[Path]:
    """Walk upward from data_path until we find a directory containing participants.tsv."""
    p = data_path if data_path.is_dir() else data_path.parent
    for ancestor in [p] + list(p.parents):
        if (ancestor / "participants.tsv").exists():
            return ancestor
    return None


def _extract_subject_id(data_path: Path) -> Optional[str]:
    for part in data_path.parts:
        m = _SUB_RE.fullmatch(part)
        if m:
            return part  # full "sub-XX" form, matches participant_id column
    return None


def _read_bids_group(bids_root: Path, subject_id: str) -> Optional[str]:
    tsv = bids_root / "participants.tsv"
    if not tsv.exists():
        return None
    try:
        with open(tsv, encoding="utf-8") as f:
            header = f.readline().rstrip("\n").split("\t")
            if "group" not in header:
                return None
            group_idx = header.index("group")
            id_idx = header.index("participant_id") if "participant_id" in header else 0
            for line in f:
                cols = line.rstrip("\n").split("\t")
                if cols[id_idx] == subject_id:
                    return cols[group_idx]
    except OSError as exc:
        logger.debug("participants.tsv read failed: %s", exc)
    return None


def resolve_cohort_tag(*, data_path: Path, cli_override: Optional[str]) -> str:
    """Return cohort_tag string ('' if no source). BIDS > CLI > empty."""
    bids_root = _find_bids_root(data_path)
    if bids_root is not None:
        sub_id = _extract_subject_id(data_path)
        if sub_id is not None:
            group = _read_bids_group(bids_root, sub_id)
            if group:
                return group.strip()
    if cli_override:
        return cli_override.strip()
    return ""
