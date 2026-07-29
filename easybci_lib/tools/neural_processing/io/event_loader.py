"""Unified event/marker loader — load events from any external source.

Supports: CSV/TSV (BIDS-style), JSON, E-Prime txt, FieldTrip trl matrix,
EEGLAB event struct, and generic tabular formats.

Single entry point: load_events(path) → List[Dict] with standardized fields.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Standard output event structure:
# {"onset": float (seconds), "duration": float (seconds), "type": str, "metadata": dict}


def load_events(
    filepath: str,
    format: str = "auto",
    column_mapping: Optional[Dict[str, str]] = None,
    time_unit: str = "auto",
    frequency: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Load events from an external file into standardized format.

    Parameters
    ----------
    filepath : str
        Path to event/marker file.
    format : str
        "auto" (detect from extension/content), "csv", "tsv", "json",
        "eprime", "fieldtrip_trl", "eeglab_event", "bids".
    column_mapping : dict or None
        Override column name mapping. Keys: "onset", "duration", "type".
        Values: actual column names in the file.
        Example: {"onset": "latency", "type": "code", "duration": "dur"}
    time_unit : str
        "auto" (heuristic detection), "seconds", "milliseconds", "samples".
    frequency : float or None
        Sampling rate (required when time_unit is "samples" or for auto-detection).

    Returns
    -------
    List of event dicts, each with:
        onset: float — event start time in seconds
        duration: float — event duration in seconds (0.0 if instantaneous)
        type: str — event label/category
        metadata: dict — additional fields from the source file
    """
    path = Path(filepath)
    if not path.exists():
        logger.warning("Event file not found: %s — returning empty events.", filepath)
        return []

    if format == "auto":
        format = _detect_format(path)

    if format in ("csv", "tsv", "bids"):
        events = _load_tabular(filepath, column_mapping=column_mapping)
    elif format == "json":
        events = _load_json(filepath, column_mapping=column_mapping)
    elif format == "eprime":
        events = _load_eprime(filepath)
    elif format == "fieldtrip_trl":
        events = _load_fieldtrip_trl(filepath, frequency=frequency)
    elif format == "eeglab_event":
        events = _load_eeglab_event(filepath, frequency=frequency)
    elif format == "mat":
        events = _load_mat_events(filepath, frequency=frequency)
    else:
        logger.warning(
            "Unsupported event format '%s' for %s — returning empty events. "
            "Supported: csv, tsv, bids, json, eprime, fieldtrip_trl, eeglab_event, mat.",
            format, filepath,
        )
        return []

    if not events:
        logger.warning("No events loaded from %s", filepath)
        return []

    # Time unit conversion
    events = _normalize_time_units(events, time_unit, frequency)

    # Sort by onset
    events.sort(key=lambda e: e["onset"])

    # Validate
    _validate_events(events, filepath)

    return events


def detect_time_unit(
    onsets: List[float],
    data_duration: Optional[float] = None,
    frequency: Optional[float] = None,
) -> str:
    """Heuristic detection of time unit from onset values.

    Parameters
    ----------
    onsets : list of float
        Raw onset values from the event file.
    data_duration : float or None
        Known duration of the data in seconds (for sanity check).
    frequency : float or None
        Sampling rate of the data.

    Returns
    -------
    "seconds", "milliseconds", or "samples"
    """
    if not onsets:
        return "seconds"

    max_onset = max(onsets)
    min_onset = min(onsets)

    # If values are very large and dividing by frequency or 1000 gives reasonable range
    if data_duration and max_onset > data_duration * 1.5:
        # Check samples first (more specific when frequency is known)
        if frequency and max_onset / frequency <= data_duration * 1.1:
            # Verify values look like integer sample indices
            all_integer = all(abs(o - round(o)) < 0.01 for o in onsets[:50])
            if all_integer:
                return "samples"
        if max_onset / 1000.0 <= data_duration * 1.1:
            return "milliseconds"
        if frequency and max_onset / frequency <= data_duration * 1.1:
            return "samples"

    # Without known duration: use absolute thresholds
    if max_onset > 100000:
        # Probably milliseconds if in typical experiment range when /1000
        if max_onset / 1000.0 < 7200:  # less than 2 hours in seconds
            return "milliseconds"
        return "samples"

    if frequency and max_onset > 0:
        # If values look like integer sample indices
        all_integer = all(abs(o - round(o)) < 0.001 for o in onsets[:50])
        if all_integer and max_onset > 1000 and frequency > 0:
            as_seconds = max_onset / frequency
            if as_seconds < 7200:
                return "samples"

    return "seconds"


def _detect_format(path: Path) -> str:
    """Detect event file format from extension and content."""
    suffix = path.suffix.lower()

    if suffix == ".tsv":
        return "tsv"
    if suffix == ".csv":
        return "csv"
    if suffix == ".json":
        return "json"
    if suffix == ".mat":
        return "mat"

    # .txt — could be E-Prime or generic tabular
    if suffix == ".txt":
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                first_lines = [f.readline() for _ in range(5)]
            content = "".join(first_lines)
            if "Header Start" in content or "VersionPersist" in content:
                return "eprime"
            # Check if it's tab-delimited
            if "\t" in first_lines[0]:
                return "tsv"
            return "csv"
        except OSError:
            return "csv"

    if suffix in (".xlsx", ".xls"):
        return "csv"  # handled by pandas in _load_tabular

    return "csv"


def _load_tabular(
    filepath: str,
    column_mapping: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """Load events from CSV/TSV/Excel tabular format."""
    path = Path(filepath)
    suffix = path.suffix.lower()

    # Determine delimiter
    if suffix == ".tsv":
        delimiter = "\t"
    elif suffix in (".xlsx", ".xls"):
        return _load_excel_events(filepath, column_mapping)
    else:
        delimiter = ","
        # Auto-detect: peek at first data line
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        if "\t" in line and "," not in line:
                            delimiter = "\t"
                        break
        except OSError:
            pass

    # Read the file
    headers = None
    rows = []
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("%"):
                continue
            parts = [p.strip().strip('"').strip("'") for p in line.split(delimiter)]
            if headers is None:
                # Detect if first row is header (non-numeric first field)
                if parts and not _is_numeric(parts[0]):
                    headers = parts
                    continue
                else:
                    # No header — assign default names
                    headers = [f"col{i}" for i in range(len(parts))]
            rows.append(parts)

    if not rows:
        return []

    if headers is None:
        headers = [f"col{i}" for i in range(len(rows[0]))]

    # Resolve column mapping
    onset_col, duration_col, type_col = _resolve_columns(headers, column_mapping)

    events = []
    for row in rows:
        if len(row) < len(headers):
            row.extend([""] * (len(headers) - len(row)))

        row_dict = dict(zip(headers, row))
        onset_raw = row_dict.get(onset_col, "")
        duration_raw = row_dict.get(duration_col, "0") if duration_col else "0"
        type_raw = row_dict.get(type_col, "unknown") if type_col else "unknown"

        try:
            onset_val = float(onset_raw)
        except (ValueError, TypeError):
            continue

        try:
            duration_val = float(duration_raw) if duration_raw else 0.0
        except (ValueError, TypeError):
            duration_val = 0.0

        # Collect remaining fields as metadata
        metadata = {}
        for k, v in row_dict.items():
            if k not in (onset_col, duration_col, type_col):
                metadata[k] = v

        events.append({
            "onset": onset_val,
            "duration": duration_val,
            "type": str(type_raw).strip(),
            "metadata": metadata,
        })

    return events


def _load_excel_events(
    filepath: str,
    column_mapping: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """Load events from Excel file."""
    try:
        import pandas as pd
    except ImportError:
        from easybci_lib.tools.lazy_deps import ensure
        ensure("neural.pandas")
        import pandas as pd

    df = pd.read_excel(filepath)
    headers = list(df.columns)
    onset_col, duration_col, type_col = _resolve_columns(headers, column_mapping)

    events = []
    for _, row in df.iterrows():
        try:
            onset_val = float(row[onset_col])
        except (ValueError, TypeError, KeyError):
            continue

        duration_val = 0.0
        if duration_col and duration_col in row.index:
            try:
                duration_val = float(row[duration_col])
            except (ValueError, TypeError):
                pass

        type_val = str(row[type_col]) if type_col and type_col in row.index else "unknown"

        metadata = {k: str(v) for k, v in row.items()
                    if k not in (onset_col, duration_col, type_col) and pd.notna(v)}

        events.append({
            "onset": onset_val,
            "duration": duration_val,
            "type": type_val.strip(),
            "metadata": metadata,
        })

    return events


def _load_json(
    filepath: str,
    column_mapping: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """Load events from JSON file (array of objects or nested structure)."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Handle different JSON structures
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        # Look for an array field
        for key in ("events", "markers", "trials", "stimuli", "data"):
            if key in data and isinstance(data[key], list):
                items = data[key]
                break
        else:
            items = [data]
    else:
        return []

    onset_key = (column_mapping or {}).get("onset")
    duration_key = (column_mapping or {}).get("duration")
    type_key = (column_mapping or {}).get("type")

    # Auto-detect keys from first item
    if items and isinstance(items[0], dict):
        keys = list(items[0].keys())
        if not onset_key:
            onset_key = _find_best_match(keys, _ONSET_SYNONYMS)
        if not duration_key:
            duration_key = _find_best_match(keys, _DURATION_SYNONYMS)
        if not type_key:
            type_key = _find_best_match(keys, _TYPE_SYNONYMS)

    events = []
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            onset_val = float(item.get(onset_key, 0))
        except (ValueError, TypeError):
            continue

        duration_val = 0.0
        if duration_key and duration_key in item:
            try:
                duration_val = float(item[duration_key])
            except (ValueError, TypeError):
                pass

        type_val = str(item.get(type_key, "unknown")) if type_key else "unknown"

        metadata = {k: v for k, v in item.items()
                    if k not in (onset_key, duration_key, type_key)}

        events.append({
            "onset": onset_val,
            "duration": duration_val,
            "type": type_val,
            "metadata": metadata,
        })

    return events


def _load_eprime(filepath: str) -> List[Dict[str, Any]]:
    """Load events from E-Prime text export (.txt).

    E-Prime exports use a block structure with key-value pairs separated by
    'LogFrame Start' and 'LogFrame End' delimiters.
    """
    events = []
    current_block: Dict[str, str] = {}
    in_block = False

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if "LogFrame Start" in line:
                in_block = True
                current_block = {}
            elif "LogFrame End" in line:
                if in_block and current_block:
                    event = _parse_eprime_block(current_block)
                    if event:
                        events.append(event)
                in_block = False
                current_block = {}
            elif in_block and ":" in line:
                key, _, val = line.partition(":")
                current_block[key.strip()] = val.strip()

    return events


def _parse_eprime_block(block: Dict[str, str]) -> Optional[Dict[str, Any]]:
    """Extract onset/duration/type from an E-Prime log block."""
    # E-Prime typically uses these field names (case-insensitive search)
    onset_val = None
    type_val = "unknown"
    duration_val = 0.0

    for key, val in block.items():
        key_lower = key.lower()
        if onset_val is None and any(k in key_lower for k in ("onset", "stimulusonset", "targetonset", "fixationonset")):
            try:
                onset_val = float(val)
            except ValueError:
                pass
        if any(k in key_lower for k in ("duration", "stimulusduration")):
            try:
                duration_val = float(val)
            except ValueError:
                pass
        if any(k in key_lower for k in ("procedure", "condition", "trialtype", "stimulus")):
            if val and val.lower() not in ("", "null", "none"):
                type_val = val

    if onset_val is None:
        return None

    metadata = {k: v for k, v in block.items()
                if k.lower() not in ("onset", "duration")}

    return {
        "onset": onset_val,
        "duration": duration_val,
        "type": type_val,
        "metadata": metadata,
    }


def _load_fieldtrip_trl(
    filepath: str,
    frequency: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Load events from FieldTrip trl matrix in .mat file.

    trl format: [begin_sample, end_sample, offset] per row.
    May also have additional columns (trial type, etc.).
    """
    import scipy.io

    try:
        mat = scipy.io.loadmat(filepath, squeeze_me=True)
    except NotImplementedError:
        import h5py
        with h5py.File(filepath, "r") as f:
            trl = f["trl"][:] if "trl" in f else None
            if trl is None:
                for key in f.keys():
                    obj = f[key]
                    if hasattr(obj, "shape") and obj.ndim == 2 and obj.shape[1] >= 3:
                        trl = obj[:]
                        break
        if trl is None:
            logger.warning("No trl matrix found in %s (v7.3) — returning empty events.", filepath)
            return []
        mat = {"trl": trl}

    # Find trl matrix
    trl = None
    if "trl" in mat:
        trl = np.asarray(mat["trl"])
    elif "cfg" in mat:
        cfg = mat["cfg"]
        if hasattr(cfg, "dtype") and cfg.dtype.names and "trl" in cfg.dtype.names:
            trl = np.asarray(cfg["trl"].item())

    if trl is None:
        logger.warning("No trl matrix found in %s — returning empty events.", filepath)
        return []

    if trl.ndim == 1:
        trl = trl.reshape(1, -1)

    freq = frequency or 1.0
    events = []
    for i in range(trl.shape[0]):
        begin_sample = int(trl[i, 0])
        end_sample = int(trl[i, 1])
        offset = int(trl[i, 2]) if trl.shape[1] > 2 else 0

        onset_sec = float(begin_sample) / freq
        duration_sec = float(end_sample - begin_sample) / freq

        type_val = str(int(trl[i, 3])) if trl.shape[1] > 3 else "trial"

        events.append({
            "onset": onset_sec,
            "duration": duration_sec,
            "type": type_val,
            "metadata": {
                "begin_sample": begin_sample,
                "end_sample": end_sample,
                "offset": offset,
                "time_unit_source": "samples",
            },
        })

    return events


def _load_eeglab_event(
    filepath: str,
    frequency: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Load events from EEGLAB .set file's EEG.event struct."""
    import scipy.io

    try:
        mat = scipy.io.loadmat(filepath, squeeze_me=True)
    except NotImplementedError:
        logger.warning("v7.3 .set files not yet supported for event extraction: %s — returning empty.", filepath)
        return []

    eeg = mat.get("EEG")
    if eeg is None or not hasattr(eeg, "dtype"):
        logger.warning("No EEG struct found in %s — returning empty events.", filepath)
        return []

    if "event" not in (eeg.dtype.names or ()):
        logger.warning("No event field in EEG struct: %s — returning empty events.", filepath)
        return []

    event_struct = eeg["event"].item()
    if not hasattr(event_struct, "dtype") or event_struct.dtype.names is None:
        logger.warning("EEG.event is not a struct array in %s — returning empty events.", filepath)
        return []

    fields = event_struct.dtype.names
    n_events = event_struct.shape[0] if event_struct.ndim > 0 else 1

    freq = frequency
    if freq is None and "srate" in (eeg.dtype.names or ()):
        freq = float(eeg["srate"])
    freq = freq or 1.0

    events = []
    for i in range(n_events):
        ev = event_struct[i] if n_events > 1 else event_struct

        # latency field (in samples, 1-indexed in EEGLAB)
        latency = 0.0
        if "latency" in fields:
            latency = float(ev["latency"]) - 1  # convert to 0-indexed
        onset_sec = latency / freq

        # duration
        duration_sec = 0.0
        if "duration" in fields:
            try:
                duration_sec = float(ev["duration"]) / freq
            except (ValueError, TypeError):
                pass

        # type
        type_val = "unknown"
        if "type" in fields:
            type_val = str(ev["type"])

        metadata = {}
        for f in fields:
            if f not in ("latency", "duration", "type"):
                try:
                    metadata[f] = str(ev[f]) if ev[f] is not None else ""
                except Exception:
                    pass

        metadata["time_unit_source"] = "samples"
        metadata["latency_samples"] = latency

        events.append({
            "onset": onset_sec,
            "duration": duration_sec,
            "type": type_val.strip(),
            "metadata": metadata,
        })

    return events


def _load_mat_events(
    filepath: str,
    frequency: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Load events from generic .mat file — try FieldTrip trl, then EEGLAB event."""
    try:
        return _load_fieldtrip_trl(filepath, frequency=frequency)
    except (ValueError, KeyError):
        pass
    try:
        return _load_eeglab_event(filepath, frequency=frequency)
    except (ValueError, KeyError):
        pass
    logger.warning(
        "Cannot extract events from %s: not recognized as FieldTrip trl or EEGLAB event format. "
        "Returning empty events.",
        filepath,
    )
    return []


# --- Column resolution helpers ---

_ONSET_SYNONYMS = (
    "onset", "start", "start_time", "time", "latency",
    "sample", "trigger_time", "stimulus_onset", "event_time",
    "onset_sec", "onset_ms", "onset_s", "t", "timestamp",
)
_DURATION_SYNONYMS = (
    "duration", "dur", "length", "end", "stop",
    "duration_sec", "duration_ms",
)
_TYPE_SYNONYMS = (
    "type", "trial_type", "condition", "label", "class",
    "event_type", "code", "trigger", "value", "stimulus",
    "category", "marker", "description",
)


def _resolve_columns(
    headers: List[str],
    mapping: Optional[Dict[str, str]] = None,
) -> Tuple[str, Optional[str], Optional[str]]:
    """Resolve onset/duration/type column names from headers."""
    if mapping:
        onset_col = mapping.get("onset", _find_best_match(headers, _ONSET_SYNONYMS))
        duration_col = mapping.get("duration", _find_best_match(headers, _DURATION_SYNONYMS))
        type_col = mapping.get("type", _find_best_match(headers, _TYPE_SYNONYMS))
    else:
        onset_col = _find_best_match(headers, _ONSET_SYNONYMS)
        duration_col = _find_best_match(headers, _DURATION_SYNONYMS)
        type_col = _find_best_match(headers, _TYPE_SYNONYMS)

    if not onset_col:
        # Fallback: use first numeric column
        onset_col = headers[0] if headers else "col0"

    return onset_col, duration_col, type_col


def _find_best_match(headers: List[str], synonyms: Tuple[str, ...]) -> Optional[str]:
    """Find the best matching column header from synonym list."""
    headers_lower = {h.lower().strip(): h for h in headers}
    for syn in synonyms:
        if syn in headers_lower:
            return headers_lower[syn]
    # Partial match
    for syn in synonyms:
        for h_lower, h_orig in headers_lower.items():
            if syn in h_lower:
                return h_orig
    return None


def _normalize_time_units(
    events: List[Dict[str, Any]],
    time_unit: str,
    frequency: Optional[float],
) -> List[Dict[str, Any]]:
    """Convert event times to seconds based on detected/specified unit."""
    if not events:
        return events

    onsets = [e["onset"] for e in events]

    if time_unit == "auto":
        time_unit = detect_time_unit(onsets, frequency=frequency)

    if time_unit == "seconds":
        return events

    if time_unit == "milliseconds":
        for e in events:
            e["onset"] /= 1000.0
            e["duration"] /= 1000.0
            e.setdefault("metadata", {})["original_unit"] = "milliseconds"
        return events

    if time_unit == "samples":
        if not frequency or frequency <= 0:
            logger.warning(
                "Time unit detected as 'samples' but no frequency provided. "
                "Treating values as seconds."
            )
            return events
        for e in events:
            e["onset"] /= frequency
            e["duration"] /= frequency
            e.setdefault("metadata", {})["original_unit"] = "samples"
        return events

    return events


def _validate_events(events: List[Dict[str, Any]], filepath: str) -> None:
    """Log warnings for suspicious event data."""
    if not events:
        return

    # Check for negative onsets
    neg_count = sum(1 for e in events if e["onset"] < 0)
    if neg_count > 0:
        logger.warning(
            "%s: %d events have negative onset times (pre-recording markers?)",
            filepath, neg_count,
        )

    # Check for zero-interval events (likely duplicates)
    onsets = [e["onset"] for e in events]
    if len(onsets) > 1:
        diffs = np.diff(sorted(onsets))
        zero_gaps = int(np.sum(diffs < 1e-6))
        if zero_gaps > 0:
            logger.warning(
                "%s: %d pairs of events with zero time gap (possible duplicates)",
                filepath, zero_gaps,
            )


def _is_numeric(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False
