"""Directory file classification and relationship analysis.

Given a directory of files, classifies each by role (signal, event, behavior,
auxiliary, config) and infers relationships between files using name-based
fuzzy matching (shared subject IDs, session identifiers, run numbers).

Outputs a file relationship matrix showing which files are associated.
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# File role definitions by extension/pattern
_SIGNAL_EXTENSIONS = {
    ".edf", ".bdf", ".gdf", ".fif", ".set", ".cnt", ".vhdr", ".eeg", ".21e",
    ".ncs", ".nev", ".ns2", ".ns3", ".ns5", ".ns6", ".plx", ".nex",
    ".xdf", ".xdfz", ".snirf", ".nirs",
    ".mat", ".h5", ".hdf5",
}

_EVENT_EXTENSIONS = {".vmrk", ".evt", ".mrk"}
_EVENT_PATTERNS = re.compile(
    r"(event|marker|trigger|stim|annotation|label|trial)", re.IGNORECASE
)

_BEHAVIOR_PATTERNS = re.compile(
    r"(behav|response|reaction|rt|accuracy|performance|log|psychopy|eprime|e-prime)",
    re.IGNORECASE,
)

_CONFIG_EXTENSIONS = {".json", ".yaml", ".yml", ".xml", ".toml"}
_CONFIG_PATTERNS = re.compile(
    r"(config|setting|param|montage|electrode|channel_?loc)", re.IGNORECASE
)

_AUX_PATTERNS = re.compile(
    r"(readme|notes|info|description|metadata|sidecar)", re.IGNORECASE
)


class FileRole:
    SIGNAL = "signal"
    EVENT = "event"
    BEHAVIOR = "behavior"
    CONFIG = "config"
    AUXILIARY = "auxiliary"
    UNKNOWN = "unknown"


def classify_directory_files(
    directory: str,
    signal_extensions: Optional[set] = None,
) -> Dict[str, Any]:
    """Classify all files in a directory by role and infer relationships.

    Parameters
    ----------
    directory : str
        Path to directory to analyze.
    signal_extensions : set, optional
        Override signal file extensions.

    Returns
    -------
    Dict with:
        files: list of {path, name, role, subject_id, session, run}
        relationships: list of {signal, companions: [...]}
        subject_ids: list of detected subject identifiers
        summary: dict with counts per role
    """
    dir_path = Path(directory)
    if not dir_path.is_dir():
        return {
            "files": [],
            "relationships": [],
            "subject_ids": [],
            "summary": {},
            "error": f"Not a directory: {directory}",
        }

    sig_ext = signal_extensions or _SIGNAL_EXTENSIONS

    # Phase 1: Classify each file
    classified_files: List[Dict[str, Any]] = []
    try:
        entries = sorted(dir_path.rglob("*"))
    except (PermissionError, OSError):
        entries = []

    for entry in entries:
        if not entry.is_file():
            continue
        if entry.name.startswith("."):
            continue

        role = _classify_single_file(entry, sig_ext)
        ids = _extract_identifiers(entry.stem)

        classified_files.append({
            "path": str(entry),
            "name": entry.name,
            "role": role,
            "subject_id": ids.get("subject", ""),
            "session": ids.get("session", ""),
            "run": ids.get("run", ""),
            "stem_normalized": _normalize_stem(entry.stem),
        })

    # Phase 2: Build relationship matrix
    relationships = _build_relationships(classified_files)

    # Phase 3: Extract unique subject IDs
    subject_ids = sorted(set(
        f["subject_id"] for f in classified_files if f["subject_id"]
    ))

    # Summary counts
    role_counts: Dict[str, int] = {}
    for f in classified_files:
        role_counts[f["role"]] = role_counts.get(f["role"], 0) + 1

    return {
        "files": classified_files,
        "relationships": relationships,
        "subject_ids": subject_ids,
        "summary": role_counts,
    }


def _classify_single_file(path: Path, signal_extensions: set) -> str:
    """Classify a single file by its extension and name patterns."""
    suffix = path.suffix.lower()
    name = path.name
    stem = path.stem

    # Event files by extension
    if suffix in _EVENT_EXTENSIONS:
        return FileRole.EVENT

    # Signal files by extension
    if suffix in signal_extensions:
        # But check if the name suggests it's actually events in a .mat/.csv
        if _EVENT_PATTERNS.search(stem):
            return FileRole.EVENT
        return FileRole.SIGNAL

    # Tabular files — classify by name patterns
    if suffix in (".csv", ".tsv", ".txt"):
        if _EVENT_PATTERNS.search(stem):
            return FileRole.EVENT
        if _BEHAVIOR_PATTERNS.search(stem):
            return FileRole.BEHAVIOR
        # Generic tabular — could be events or behavior
        return FileRole.EVENT  # conservative default for tabular

    # Config files
    if suffix in _CONFIG_EXTENSIONS:
        if _CONFIG_PATTERNS.search(stem):
            return FileRole.CONFIG
        if _EVENT_PATTERNS.search(stem):
            return FileRole.EVENT
        return FileRole.CONFIG

    # Markdown/text — auxiliary
    if suffix in (".md", ".rst"):
        return FileRole.AUXILIARY

    # Check name patterns for remaining
    if _BEHAVIOR_PATTERNS.search(stem):
        return FileRole.BEHAVIOR
    if _EVENT_PATTERNS.search(stem):
        return FileRole.EVENT
    if _AUX_PATTERNS.search(stem):
        return FileRole.AUXILIARY

    return FileRole.UNKNOWN


def _extract_identifiers(stem: str) -> Dict[str, str]:
    """Extract subject/session/run identifiers from a filename stem.

    Handles patterns like:
    - sub-01_ses-01_run-01_eeg
    - S01_session1_run2
    - subject01_trial1
    - P001_REST_eyes_open
    """
    result: Dict[str, str] = {"subject": "", "session": "", "run": ""}

    # BIDS-style: sub-XX, ses-XX, run-XX. The separator is REQUIRED — BIDS is
    # always `sub-<label>` / `ses-<label>`. Making it optional (`sub[_-]?`) plus
    # a lazy `\w+?` misfires on plain words: "subaru" -> "aru",
    # "subject01" -> "ject01", "session1" -> "sion1". Word/no-separator forms
    # (S01 / subject01 / session1) are handled by the numeric fallbacks below.
    sub_match = re.search(r"sub[_-]([A-Za-z0-9]+)", stem, re.IGNORECASE)
    if sub_match:
        result["subject"] = sub_match.group(1)

    ses_match = re.search(r"ses[_-]([A-Za-z0-9]+)", stem, re.IGNORECASE)
    if ses_match:
        result["session"] = ses_match.group(1)

    run_match = re.search(r"run[_-]?(\d+)", stem, re.IGNORECASE)
    if run_match:
        result["run"] = run_match.group(1)

    # Fallback: S01/P01/subject01/subj01 patterns (no BIDS separator). \d+ so we
    # capture the number, not a word tail: "subject01" -> "01".
    if not result["subject"]:
        fallback = re.search(
            r"(?:^|[_\-])(?:subject|subj|sub|S|P)\s*(\d+)", stem, re.IGNORECASE
        )
        if fallback:
            result["subject"] = fallback.group(1)

    # Fallback: session1 / sess-2 word forms (no BIDS separator).
    if not result["session"]:
        ses_fb = re.search(
            r"(?:^|[_\-])(?:session|sess)\s*[_\-]?(\d+)", stem, re.IGNORECASE
        )
        if ses_fb:
            result["session"] = ses_fb.group(1)

    # Fallback: leading number pattern (01_, 001_)
    if not result["subject"]:
        leading = re.match(r"^(\d{2,4})[_\-]", stem)
        if leading:
            result["subject"] = leading.group(1)

    return result


def _normalize_stem(stem: str) -> str:
    """Normalize a filename stem for fuzzy matching.

    Strips common suffixes, lowercases, removes known role keywords.
    """
    normalized = stem.lower()
    # Remove common modality/role suffixes
    for pattern in (
        r"[_\-](eeg|meg|ecog|seeg|fnirs|emg|eog|ecg)",
        r"[_\-](event|marker|trigger|stim|behav|log|raw|processed)",
        r"[_\-](task|rest|baseline)",
    ):
        normalized = re.sub(pattern, "", normalized, flags=re.IGNORECASE)
    # Remove trailing numbers that look like part numbers
    normalized = re.sub(r"[_\-]\d+$", "", normalized)
    return normalized


def _build_relationships(
    files: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Build file relationship matrix: group companions with their signal files.

    Uses two matching strategies:
    1. Exact subject/session/run match (BIDS-like)
    2. Normalized stem similarity (fuzzy)
    """
    signal_files = [f for f in files if f["role"] == FileRole.SIGNAL]
    companion_files = [f for f in files if f["role"] != FileRole.SIGNAL]

    relationships: List[Dict[str, Any]] = []

    for sig in signal_files:
        companions: List[Dict[str, str]] = []

        for comp in companion_files:
            score = _match_score(sig, comp)
            if score > 0.5:
                companions.append({
                    "path": comp["path"],
                    "name": comp["name"],
                    "role": comp["role"],
                    "match_score": round(score, 2),
                })

        # Sort companions by match score descending
        companions.sort(key=lambda x: x["match_score"], reverse=True)

        relationships.append({
            "signal": sig["path"],
            "signal_name": sig["name"],
            "subject_id": sig["subject_id"],
            "companions": companions,
        })

    return relationships


def _match_score(signal_file: Dict, companion_file: Dict) -> float:
    """Compute relationship score between a signal file and a potential companion.

    Returns 0.0 (no match) to 1.0 (definite match).
    """
    score = 0.0

    # Strategy 1: Subject/session/run exact match (strongest signal)
    if signal_file["subject_id"] and companion_file["subject_id"]:
        if signal_file["subject_id"] == companion_file["subject_id"]:
            score += 0.6
            if signal_file["session"] and companion_file["session"]:
                if signal_file["session"] == companion_file["session"]:
                    score += 0.2
            if signal_file["run"] and companion_file["run"]:
                if signal_file["run"] == companion_file["run"]:
                    score += 0.1
            return min(score, 1.0)
        else:
            return 0.0  # Different subjects → no match

    # Strategy 2: Normalized stem similarity
    sig_stem = signal_file["stem_normalized"]
    comp_stem = companion_file["stem_normalized"]

    if not sig_stem or not comp_stem:
        return 0.0

    # Check prefix overlap
    common_prefix_len = _common_prefix_length(sig_stem, comp_stem)
    max_len = max(len(sig_stem), len(comp_stem))

    if max_len > 0:
        prefix_ratio = common_prefix_len / max_len
        if prefix_ratio > 0.4:
            score += prefix_ratio * 0.7

    # Check if stems are subsets of each other
    if sig_stem in comp_stem or comp_stem in sig_stem:
        score = max(score, 0.6)

    # Same directory bonus
    sig_parent = Path(signal_file["path"]).parent
    comp_parent = Path(companion_file["path"]).parent
    if sig_parent == comp_parent:
        score += 0.1

    return min(score, 1.0)


def _common_prefix_length(a: str, b: str) -> int:
    """Length of common prefix between two strings."""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n
