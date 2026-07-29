"""Sidecar file detection — discover event/behavioral/auxiliary files near neural data.

Scans the data file's directory (and parent) for companion files that contain
events, markers, labels, behavioral data, or auxiliary signals.
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Patterns for sidecar file discovery (case-insensitive matching)
_EVENT_PATTERNS = (
    r"event", r"marker", r"trigger", r"stimulus", r"stim",
    r"annotation", r"onset",
)
_LABEL_PATTERNS = (
    r"label", r"class", r"target", r"condition", r"trial_type",
)
_BEHAVIOR_PATTERNS = (
    r"behav", r"beh", r"response", r"reaction", r"rt",
    r"performance", r"log", r"psychopy", r"eprime", r"e-prime",
)
_AUX_SIGNAL_PATTERNS = (
    r"emg", r"eog", r"ecg", r"ekg", r"gsr", r"eda",
    r"eye", r"gaze", r"pupil", r"accel", r"gyro",
)

_SIDECAR_EXTENSIONS = {
    "event": {".csv", ".tsv", ".txt", ".json", ".mat", ".xlsx"},
    "signal": {".edf", ".bdf", ".fif", ".set", ".gdf", ".cnt", ".vhdr", ".xdf"},
}

# BIDS standard sidecar patterns
_BIDS_SUFFIXES = ("_events.tsv", "_channels.tsv", "_electrodes.tsv", "_beh.tsv")

# Directory listing cache: key=(dir_path, mtime) -> list of Path entries
_dir_listing_cache: Dict[Tuple[str, float], List[Path]] = {}
_DIR_CACHE_MAX_SIZE = 32


def _cached_dir_listing(scan_dir: Path) -> Optional[List[Path]]:
    """Return cached directory listing, refreshing if mtime changed."""
    try:
        dir_mtime = scan_dir.stat().st_mtime
    except OSError:
        return None

    cache_key = (str(scan_dir), dir_mtime)
    if cache_key in _dir_listing_cache:
        return _dir_listing_cache[cache_key]

    try:
        entries = list(scan_dir.iterdir())
    except (PermissionError, OSError):
        return None

    # Evict oldest entries if cache is full
    if len(_dir_listing_cache) >= _DIR_CACHE_MAX_SIZE:
        oldest_key = next(iter(_dir_listing_cache))
        del _dir_listing_cache[oldest_key]

    _dir_listing_cache[cache_key] = entries
    return entries


def clear_dir_cache() -> None:
    """Clear the directory listing cache (useful in tests or between sessions)."""
    _dir_listing_cache.clear()


def detect_sidecar_files(
    data_path: str,
    scan_parent: bool = True,
) -> Dict[str, Any]:
    """Detect companion files near a neural data file.

    Parameters
    ----------
    data_path : str
        Path to the primary data file.
    scan_parent : bool
        Also scan one directory level up.

    Returns
    -------
    Dict with:
        sidecar_files: List of detected companions with metadata
        data_type: "signal-only" | "signal+events" | "signal+events+behavior" | "multi-stream"
        relationships: Inferred relationships between files
    """
    path = Path(data_path)
    if not path.exists():
        return {"sidecar_files": [], "data_type": "signal-only", "relationships": {}}

    data_dir = path.parent
    data_stem = path.stem
    # Handle stems with spaces/special chars — use the first clean segment for matching
    stem_clean = re.sub(r"[^a-zA-Z0-9]", "_", data_stem).lower()
    # Also try matching on subject ID pattern (sub-XX, subXX, sXX)
    subject_match = re.search(r"(sub[_-]?\d+|s\d+)", data_stem, re.IGNORECASE)
    subject_id = subject_match.group(0) if subject_match else None

    dirs_to_scan = [data_dir]
    if scan_parent and data_dir.parent != data_dir:
        dirs_to_scan.append(data_dir.parent)
        # Also check sibling directories (e.g., eeg/ + beh/ in BIDS)
        for sibling in data_dir.parent.iterdir():
            if sibling.is_dir() and sibling != data_dir:
                sib_name = sibling.name.lower()
                if any(pat in sib_name for pat in ("beh", "behavior", "behav", "func", "aux")):
                    dirs_to_scan.append(sibling)

    sidecars: List[Dict[str, Any]] = []
    seen_paths = {path.resolve()}

    for scan_dir in dirs_to_scan:
        if not scan_dir.exists():
            continue
        entries = _cached_dir_listing(scan_dir)
        if entries is None:
            continue

        for entry in entries:
            if not entry.is_file():
                continue
            if entry.resolve() in seen_paths:
                continue
            # Skip very large binary files that aren't neural data
            if entry.suffix.lower() in (".zip", ".tar", ".gz", ".7z", ".rar"):
                continue

            classification = _classify_file(entry, data_stem, stem_clean, subject_id)
            if classification:
                info = {
                    "path": str(entry),
                    "filename": entry.name,
                    "type_guess": classification["type"],
                    "confidence": classification["confidence"],
                    "match_reason": classification["reason"],
                }
                # Get basic file stats
                try:
                    stat = entry.stat()
                    info["size_bytes"] = stat.st_size
                except OSError:
                    pass
                # For text files, peek at columns
                if entry.suffix.lower() in (".csv", ".tsv", ".txt"):
                    cols = _peek_columns(entry)
                    if cols:
                        info["columns"] = cols[:20]
                        info["n_columns"] = len(cols)

                sidecars.append(info)
                seen_paths.add(entry.resolve())

    # Determine data type
    has_events = any(s["type_guess"] == "events" for s in sidecars)
    has_behavior = any(s["type_guess"] == "behavior" for s in sidecars)
    has_aux_signal = any(s["type_guess"] == "aux_signal" for s in sidecars)

    if has_events and has_behavior:
        data_type = "signal+events+behavior"
    elif has_events:
        data_type = "signal+events"
    elif has_aux_signal:
        data_type = "multi-stream"
    else:
        data_type = "signal-only"

    # Build relationship map
    relationships: Dict[str, Any] = {"primary_signal": str(path)}
    event_files = [s["path"] for s in sidecars if s["type_guess"] == "events"]
    if event_files:
        relationships["event_sources"] = event_files
    behavior_files = [s["path"] for s in sidecars if s["type_guess"] == "behavior"]
    if behavior_files:
        relationships["behavior_sources"] = behavior_files
    aux_files = [s["path"] for s in sidecars if s["type_guess"] == "aux_signal"]
    if aux_files:
        relationships["aux_signals"] = aux_files
    channel_files = [s["path"] for s in sidecars if s["type_guess"] == "channels_meta"]
    if channel_files:
        relationships["channel_metadata"] = channel_files

    return {
        "sidecar_files": sorted(sidecars, key=lambda s: -s["confidence"]),
        "data_type": data_type,
        "relationships": relationships,
    }


def _classify_file(
    entry: Path,
    data_stem: str,
    stem_clean: str,
    subject_id: Optional[str],
) -> Optional[Dict[str, Any]]:
    """Classify a neighboring file by its likely role."""
    name_lower = entry.name.lower()
    stem_lower = entry.stem.lower()
    suffix = entry.suffix.lower()

    # BIDS standard sidecars (highest confidence)
    for bids_suffix in _BIDS_SUFFIXES:
        if name_lower.endswith(bids_suffix):
            if "_events" in bids_suffix:
                return {"type": "events", "confidence": 0.95, "reason": "BIDS events sidecar"}
            elif "_channels" in bids_suffix or "_electrodes" in bids_suffix:
                return {"type": "channels_meta", "confidence": 0.95, "reason": "BIDS channel metadata"}
            elif "_beh" in bids_suffix:
                return {"type": "behavior", "confidence": 0.95, "reason": "BIDS behavioral sidecar"}

    # Check if filename contains the same subject/stem as data file
    name_related = False
    if subject_id and subject_id.lower() in name_lower:
        name_related = True
    elif len(data_stem) > 3 and data_stem.lower()[:6] in name_lower:
        name_related = True

    # Skip if clearly unrelated and not a standard sidecar name
    if not name_related and suffix not in {".csv", ".tsv", ".json", ".txt"}:
        return None

    # Event patterns
    for pat in _EVENT_PATTERNS:
        if re.search(pat, name_lower):
            if suffix in _SIDECAR_EXTENSIONS["event"]:
                conf = 0.9 if name_related else 0.6
                return {"type": "events", "confidence": conf, "reason": f"filename contains '{pat}'"}

    # Label patterns
    for pat in _LABEL_PATTERNS:
        if re.search(pat, name_lower):
            if suffix in _SIDECAR_EXTENSIONS["event"]:
                conf = 0.85 if name_related else 0.55
                return {"type": "events", "confidence": conf, "reason": f"filename contains '{pat}' (labels)"}

    # Behavior patterns
    for pat in _BEHAVIOR_PATTERNS:
        if re.search(pat, name_lower):
            if suffix in _SIDECAR_EXTENSIONS["event"]:
                conf = 0.85 if name_related else 0.5
                return {"type": "behavior", "confidence": conf, "reason": f"filename contains '{pat}'"}

    # Auxiliary signal patterns
    for pat in _AUX_SIGNAL_PATTERNS:
        if re.search(pat, name_lower):
            if suffix in _SIDECAR_EXTENSIONS["signal"] | _SIDECAR_EXTENSIONS["event"]:
                conf = 0.8 if name_related else 0.45
                return {"type": "aux_signal", "confidence": conf, "reason": f"filename contains '{pat}'"}

    # Generic: same stem with known sidecar extension, related by name
    if name_related and suffix in (".csv", ".tsv", ".json", ".txt"):
        # Peek at content to classify
        cols = _peek_columns(entry)
        if cols:
            cols_lower = [c.lower() for c in cols]
            if any(k in cols_lower for k in ("onset", "start", "latency", "sample", "trigger")):
                return {"type": "events", "confidence": 0.75, "reason": "related file with event-like columns"}
            if any(k in cols_lower for k in ("rt", "response", "accuracy", "correct")):
                return {"type": "behavior", "confidence": 0.7, "reason": "related file with behavioral columns"}

    return None


def _peek_columns(filepath: Path, max_lines: int = 3) -> Optional[List[str]]:
    """Read the first few lines of a text file to extract column headers."""
    try:
        delimiter = "\t" if filepath.suffix.lower() == ".tsv" else ","
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = [p.strip().strip('"').strip("'") for p in line.split(delimiter)]
                # Check if this looks like a header (non-numeric first fields)
                if parts and not _is_numeric(parts[0]):
                    return parts
                break
    except (OSError, UnicodeDecodeError):
        pass
    return None


def _is_numeric(s: str) -> bool:
    """Check if a string looks like a number."""
    try:
        float(s)
        return True
    except ValueError:
        return False


def build_event_source_report(
    meta: Dict[str, Any],
    sidecar_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a unified event source report combining embedded + sidecar events.

    Parameters
    ----------
    meta : dict
        Metadata from the loader (may contain 'annotations').
    sidecar_result : dict
        Result from detect_sidecar_files().

    Returns
    -------
    Dict describing all event sources and their quality.
    """
    sources = []

    # Embedded events (from data file annotations)
    annotations = meta.get("annotations")
    if annotations and annotations.get("onset"):
        onsets = annotations["onset"]
        descriptions = annotations.get("description", [])
        n_events = len(onsets)
        unique_types = list(set(descriptions)) if descriptions else []

        # Compute event distribution stats
        intervals = []
        if len(onsets) > 1:
            sorted_onsets = sorted(onsets)
            intervals = [sorted_onsets[i+1] - sorted_onsets[i] for i in range(len(sorted_onsets)-1)]

        type_counts = {}
        for d in descriptions:
            type_counts[d] = type_counts.get(d, 0) + 1

        source_info = {
            "source": "embedded",
            "source_type": "annotations",
            "n_events": n_events,
            "n_types": len(unique_types),
            "types": unique_types[:20],
            "type_distribution": dict(sorted(type_counts.items(), key=lambda x: -x[1])[:10]),
        }
        if intervals:
            import numpy as np
            arr = np.array(intervals)
            source_info["interval_stats"] = {
                "mean_sec": round(float(np.mean(arr)), 3),
                "std_sec": round(float(np.std(arr)), 3),
                "min_sec": round(float(np.min(arr)), 3),
                "max_sec": round(float(np.max(arr)), 3),
            }
        sources.append(source_info)

    # Sidecar event files
    for sidecar in sidecar_result.get("sidecar_files", []):
        if sidecar["type_guess"] == "events":
            sources.append({
                "source": "sidecar",
                "source_type": "file",
                "path": sidecar["path"],
                "filename": sidecar["filename"],
                "confidence": sidecar["confidence"],
                "columns": sidecar.get("columns"),
                "match_reason": sidecar["match_reason"],
            })

    # Determine overall event status
    if not sources:
        status = "none_detected"
        coverage = "none"
    elif any(s["source"] == "embedded" for s in sources):
        status = "available"
        coverage = "full" if sources[0].get("n_events", 0) > 10 else "sparse"
    else:
        status = "sidecar_only"
        coverage = "unknown"

    return {
        "status": status,
        "coverage": coverage,
        "n_sources": len(sources),
        "sources": sources,
    }
