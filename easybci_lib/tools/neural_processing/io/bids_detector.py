"""BIDS directory structure recognition.

Detects Brain Imaging Data Structure (BIDS) layout and automatically associates:
- _events.tsv with corresponding data files
- _channels.tsv for channel metadata
- participants.tsv for group labels
- Behavioral data in beh/ subdirectory

Reference: https://bids-specification.readthedocs.io/
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# BIDS modality directories
_BIDS_MODALITIES = {"eeg", "meg", "ieeg", "func", "anat", "beh", "perf"}

# BIDS standard file suffixes
_BIDS_DATA_SUFFIXES = {
    "_eeg": "eeg", "_meg": "meg", "_ieeg": "ieeg",
    "_bold": "func", "_T1w": "anat", "_T2w": "anat",
}

# BIDS sidecar associations (data suffix → list of companion suffixes)
_BIDS_SIDECARS = {
    "_eeg": ["_events.tsv", "_channels.tsv", "_electrodes.tsv", "_eeg.json"],
    "_meg": ["_events.tsv", "_channels.tsv", "_meg.json", "_coordsystem.json"],
    "_ieeg": ["_events.tsv", "_channels.tsv", "_electrodes.tsv", "_ieeg.json"],
}


def detect_bids_structure(path: str) -> Optional[Dict[str, Any]]:
    """Detect if a path is within a BIDS dataset and extract structure info.

    Parameters
    ----------
    path : str
        Path to a data file or directory within a potential BIDS dataset.

    Returns
    -------
    Dict with BIDS info, or None if not a BIDS dataset.
        Keys:
        - is_bids: bool
        - bids_root: str — root of the BIDS dataset
        - subject: str — subject ID (e.g., "sub-01")
        - session: str or None — session ID
        - task: str or None — task name
        - modality: str — detected modality
        - associated_files: dict — companion files found
        - participants: list or None — parsed participants.tsv
    """
    p = Path(path)

    # Walk up to find BIDS root (contains dataset_description.json)
    bids_root = _find_bids_root(p)
    if bids_root is None:
        # Not clearly BIDS — check if file naming follows BIDS convention
        if p.is_file() and _looks_like_bids_filename(p.name):
            return _infer_bids_from_filename(p)
        return None

    # Parse BIDS entities from path
    entities = _parse_bids_entities(p, bids_root)
    if not entities.get("subject"):
        return None

    # Find associated sidecar files
    associated = _find_bids_sidecars(p, bids_root, entities)

    # Load participants.tsv if it exists
    participants = _load_participants(bids_root, entities.get("subject"))

    # Load dataset description
    dataset_info = _load_dataset_description(bids_root)

    return {
        "is_bids": True,
        "bids_root": str(bids_root),
        "subject": entities.get("subject"),
        "session": entities.get("session"),
        "task": entities.get("task"),
        "run": entities.get("run"),
        "modality": entities.get("modality", "unknown"),
        "associated_files": associated,
        "participants": participants,
        "dataset_description": dataset_info,
    }


def _find_bids_root(p: Path) -> Optional[Path]:
    """Walk up the directory tree to find dataset_description.json."""
    current = p if p.is_dir() else p.parent
    for _ in range(6):  # max 6 levels up
        desc_file = current / "dataset_description.json"
        if desc_file.exists():
            return current
        # Also check for participants.tsv as an indicator
        if (current / "participants.tsv").exists():
            return current
        parent = current.parent
        if parent == current:
            break
        current = parent
    return None


def _looks_like_bids_filename(name: str) -> bool:
    """Check if filename follows BIDS naming convention (sub-XX_key-value_suffix.ext)."""
    return bool(re.match(r"sub-[a-zA-Z0-9]+(_[a-zA-Z]+-[a-zA-Z0-9]+)*_[a-zA-Z]+\.", name))


def _parse_bids_entities(p: Path, bids_root: Path) -> Dict[str, str]:
    """Parse BIDS key-value entities from path."""
    entities: Dict[str, str] = {}

    # Parse from filename
    stem = p.stem if p.is_file() else ""
    parts = stem.split("_")

    for part in parts:
        if part.startswith("sub-"):
            entities["subject"] = part
        elif part.startswith("ses-"):
            entities["session"] = part
        elif part.startswith("task-"):
            entities["task"] = part.replace("task-", "")
        elif part.startswith("run-"):
            entities["run"] = part.replace("run-", "")
        elif part.startswith("acq-"):
            entities["acquisition"] = part.replace("acq-", "")

    # Also try from directory structure
    rel_path = p.relative_to(bids_root) if p.is_relative_to(bids_root) else None
    if rel_path:
        for part in rel_path.parts:
            if part.startswith("sub-"):
                entities.setdefault("subject", part)
            elif part.startswith("ses-"):
                entities.setdefault("session", part)
            elif part in _BIDS_MODALITIES:
                entities["modality"] = part

    # Infer modality from filename suffix if not from directory
    if "modality" not in entities and stem:
        for suffix, mod in _BIDS_DATA_SUFFIXES.items():
            if stem.endswith(suffix.lstrip("_")):
                entities["modality"] = mod
                break

    return entities


def _find_bids_sidecars(
    data_path: Path,
    bids_root: Path,
    entities: Dict[str, str],
) -> Dict[str, Any]:
    """Find BIDS sidecar files associated with a data file."""
    associated: Dict[str, Any] = {}
    data_dir = data_path.parent if data_path.is_file() else data_path

    # Build the base prefix (everything before the modality suffix)
    if data_path.is_file():
        stem = data_path.stem
        # Remove the modality suffix to get the BIDS prefix
        base = stem
        for suffix in _BIDS_DATA_SUFFIXES:
            if stem.endswith(suffix.lstrip("_")):
                base = stem[: -len(suffix.lstrip("_"))]
                break
    else:
        # Build from entities
        parts = []
        if entities.get("subject"):
            parts.append(entities["subject"])
        if entities.get("session"):
            parts.append(entities["session"])
        if entities.get("task"):
            parts.append(f"task-{entities['task']}")
        if entities.get("run"):
            parts.append(f"run-{entities['run']}")
        base = "_".join(parts)

    if not base:
        return associated

    # Search for standard sidecars
    # 1. Events file
    events_path = data_dir / f"{base}_events.tsv"
    if not events_path.exists():
        # Try without run number
        candidates = list(data_dir.glob(f"*{entities.get('task', '')}*_events.tsv"))
        if candidates:
            events_path = candidates[0]
    if events_path.exists():
        associated["events"] = {
            "path": str(events_path),
            "format": "bids_events",
            "columns": _peek_tsv_columns(events_path),
        }

    # 2. Channels file
    channels_path = data_dir / f"{base}_channels.tsv"
    if not channels_path.exists():
        candidates = list(data_dir.glob("*_channels.tsv"))
        if candidates:
            channels_path = candidates[0]
    if channels_path.exists():
        associated["channels"] = {
            "path": str(channels_path),
            "columns": _peek_tsv_columns(channels_path),
        }

    # 3. Electrodes file
    electrodes_path = data_dir / f"{base}_electrodes.tsv"
    if not electrodes_path.exists():
        candidates = list(data_dir.glob("*_electrodes.tsv"))
        if candidates:
            electrodes_path = candidates[0]
    if electrodes_path.exists():
        associated["electrodes"] = {
            "path": str(electrodes_path),
        }

    # 4. JSON sidecar (modality-specific metadata)
    for ext in (".json",):
        json_path = data_dir / f"{base}_eeg{ext}"
        if not json_path.exists():
            # Try other modalities
            for mod in ("eeg", "meg", "ieeg"):
                json_path = data_dir / f"{base}_{mod}{ext}"
                if json_path.exists():
                    break
        if json_path.exists():
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    associated["json_sidecar"] = {
                        "path": str(json_path),
                        "content": json.load(f),
                    }
            except (json.JSONDecodeError, OSError):
                associated["json_sidecar"] = {"path": str(json_path)}

    # 5. Behavioral data (in beh/ sibling directory or same dir)
    beh_dir = data_dir.parent / "beh" if data_dir.name in _BIDS_MODALITIES else data_dir / "beh"
    if not beh_dir.exists() and data_dir.parent.parent.exists():
        beh_dir = data_dir.parent.parent / entities.get("subject", "") / "beh"

    if beh_dir.exists():
        beh_files = []
        for beh_file in beh_dir.iterdir():
            if beh_file.suffix in (".tsv", ".csv", ".json"):
                beh_files.append({
                    "path": str(beh_file),
                    "filename": beh_file.name,
                })
        if beh_files:
            associated["behavioral"] = beh_files

    return associated


def _load_participants(bids_root: Path, subject: Optional[str]) -> Optional[Dict[str, Any]]:
    """Load participants.tsv and extract info for the given subject."""
    participants_path = bids_root / "participants.tsv"
    if not participants_path.exists():
        return None

    try:
        participants: Dict[str, Any] = {"path": str(participants_path), "entries": []}
        with open(participants_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        if not lines:
            return None

        headers = [h.strip() for h in lines[0].split("\t")]
        for line in lines[1:]:
            parts = [p.strip() for p in line.split("\t")]
            if len(parts) >= len(headers):
                entry = dict(zip(headers, parts))
                participants["entries"].append(entry)

                # Highlight the current subject
                pid = entry.get("participant_id", "")
                if subject and (pid == subject or pid == subject.replace("sub-", "")):
                    participants["current_subject"] = entry

        participants["n_participants"] = len(participants["entries"])
        participants["columns"] = headers
        return participants

    except (OSError, UnicodeDecodeError):
        return None


def _load_dataset_description(bids_root: Path) -> Optional[Dict[str, Any]]:
    """Load dataset_description.json."""
    desc_path = bids_root / "dataset_description.json"
    if not desc_path.exists():
        return None
    try:
        with open(desc_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _peek_tsv_columns(path: Path) -> Optional[List[str]]:
    """Read the header row of a TSV file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
        if first_line:
            return [col.strip() for col in first_line.split("\t")]
    except (OSError, UnicodeDecodeError):
        pass
    return None


def _infer_bids_from_filename(p: Path) -> Optional[Dict[str, Any]]:
    """Try to infer BIDS structure from filename alone (no dataset_description.json found)."""
    entities = {}
    stem = p.stem
    parts = stem.split("_")

    for part in parts:
        if part.startswith("sub-"):
            entities["subject"] = part
        elif part.startswith("ses-"):
            entities["session"] = part
        elif part.startswith("task-"):
            entities["task"] = part.replace("task-", "")
        elif part.startswith("run-"):
            entities["run"] = part.replace("run-", "")

    if not entities.get("subject"):
        return None

    # Check for sidecars in same directory
    associated = {}
    data_dir = p.parent
    base = "_".join(part for part in parts if not any(
        part.endswith(s.lstrip("_")) for s in _BIDS_DATA_SUFFIXES
    ))

    events_path = data_dir / f"{base}_events.tsv"
    if events_path.exists():
        associated["events"] = {
            "path": str(events_path),
            "format": "bids_events",
            "columns": _peek_tsv_columns(events_path),
        }

    channels_path = data_dir / f"{base}_channels.tsv"
    if channels_path.exists():
        associated["channels"] = {
            "path": str(channels_path),
            "columns": _peek_tsv_columns(channels_path),
        }

    return {
        "is_bids": True,
        "bids_root": str(data_dir.parent) if entities.get("subject") else str(data_dir),
        "subject": entities.get("subject"),
        "session": entities.get("session"),
        "task": entities.get("task"),
        "run": entities.get("run"),
        "modality": entities.get("modality", "eeg"),
        "associated_files": associated,
        "participants": None,
        "dataset_description": None,
        "note": "BIDS structure inferred from filename (no dataset_description.json found)",
    }
