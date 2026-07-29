"""Label type classifier — automatically identify label taxonomy level (L1-L5).

Given a label source (file path, array, or loaded event list), determines:
- L1 (EVENT): precise onset + type per trial
- L2 (SEGMENT): start/end time ranges
- L3 (CONTINUOUS): per-sample labels aligned to data timebase
- L4 (SESSION): whole-file/session-level labels, no time dimension
- L5 (HIERARCHICAL): nested multi-level labels

Output includes the detected type and a recommended processing strategy.
"""

import json
import logging
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


class LabelType(Enum):
    EVENT = "L1_event"
    SEGMENT = "L2_segment"
    CONTINUOUS = "L3_continuous"
    SESSION = "L4_session"
    HIERARCHICAL = "L5_hierarchical"
    UNKNOWN = "unknown"


_PROCESSING_STRATEGIES = {
    LabelType.EVENT: "event_locked_epoching",
    LabelType.SEGMENT: "interval_based_extraction",
    LabelType.CONTINUOUS: "sliding_windows_with_label_alignment",
    LabelType.SESSION: "sliding_windows_with_session_broadcast",
    LabelType.HIERARCHICAL: "hierarchical_parse_then_epoch",
    LabelType.UNKNOWN: "manual_inspection_required",
}


def classify_label_type(
    label_source: Union[str, List[Dict], np.ndarray, Dict],
    n_samples: Optional[int] = None,
    frequency: Optional[float] = None,
    data_duration: Optional[float] = None,
) -> Dict[str, Any]:
    """Classify a label source into the L1-L5 taxonomy.

    Parameters
    ----------
    label_source : str, list, ndarray, or dict
        - str: path to a label/event file
        - list of dicts: already-loaded events (from event_loader)
        - ndarray: raw label array
        - dict: structured label data (possibly hierarchical)
    n_samples : int, optional
        Number of samples in the corresponding data (for L3 detection).
    frequency : float, optional
        Sampling rate of the data.
    data_duration : float, optional
        Duration of the data in seconds.

    Returns
    -------
    Dict with:
        label_type: LabelType enum value
        label_type_name: str (e.g., "L1_event")
        confidence: float (0-1)
        strategy: str — recommended processing approach
        details: dict — type-specific diagnostic info
        warnings: list[str] — potential issues detected
    """
    warnings: List[str] = []

    if isinstance(label_source, str):
        return _classify_from_file(
            label_source, n_samples, frequency, data_duration, warnings
        )
    elif isinstance(label_source, np.ndarray):
        return _classify_from_array(
            label_source, n_samples, frequency, data_duration, warnings
        )
    elif isinstance(label_source, list):
        return _classify_from_event_list(
            label_source, n_samples, frequency, data_duration, warnings
        )
    elif isinstance(label_source, dict):
        return _classify_from_dict(
            label_source, n_samples, frequency, data_duration, warnings
        )
    else:
        return _build_result(LabelType.UNKNOWN, 0.0, {}, warnings)


def _classify_from_file(
    filepath: str,
    n_samples: Optional[int],
    frequency: Optional[float],
    data_duration: Optional[float],
    warnings: List[str],
) -> Dict[str, Any]:
    """Classify label type from a file path."""
    path = Path(filepath)
    if not path.exists():
        warnings.append(f"File not found: {filepath}")
        return _build_result(LabelType.UNKNOWN, 0.0, {}, warnings)

    suffix = path.suffix.lower()

    # JSON files may be hierarchical
    if suffix == ".json":
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            return _classify_from_dict(
                data, n_samples, frequency, data_duration, warnings
            )
        except (json.JSONDecodeError, OSError) as e:
            warnings.append(f"Failed to parse JSON: {e}")
            return _build_result(LabelType.UNKNOWN, 0.0, {}, warnings)

    # NumPy files — likely continuous labels
    if suffix in (".npy", ".npz"):
        try:
            if suffix == ".npy":
                arr = np.load(filepath, mmap_mode="r")
            else:
                npz = np.load(filepath, allow_pickle=False)
                # Use largest array
                arr = max((npz[k] for k in npz.files), key=lambda a: a.size)
            return _classify_from_array(
                arr, n_samples, frequency, data_duration, warnings
            )
        except Exception as e:
            warnings.append(f"Failed to load numpy file: {e}")
            return _build_result(LabelType.UNKNOWN, 0.0, {}, warnings)

    # Tabular files (CSV/TSV/TXT) — peek at structure
    if suffix in (".csv", ".tsv", ".txt"):
        return _classify_tabular_file(
            filepath, n_samples, frequency, data_duration, warnings
        )

    # .mat files
    if suffix == ".mat":
        return _classify_mat_file(
            filepath, n_samples, frequency, data_duration, warnings
        )

    warnings.append(f"Unrecognized label file extension: {suffix}")
    return _build_result(LabelType.UNKNOWN, 0.3, {}, warnings)


def _classify_from_array(
    arr: np.ndarray,
    n_samples: Optional[int],
    frequency: Optional[float],
    data_duration: Optional[float],
    warnings: List[str],
) -> Dict[str, Any]:
    """Classify label type from a numpy array."""
    if arr.ndim == 0:
        # Scalar — session-level label
        return _build_result(
            LabelType.SESSION, 0.9,
            {"value": str(arr.item()), "reason": "scalar label value"},
            warnings,
        )

    # Flatten if shape like (1, N) or (N, 1)
    effective = arr.squeeze()

    if effective.ndim == 0:
        return _build_result(
            LabelType.SESSION, 0.9,
            {"value": str(effective.item()), "reason": "single-element array"},
            warnings,
        )

    # Check if array length matches n_samples (L3: continuous)
    if n_samples is not None and effective.ndim >= 1:
        longest_dim = max(effective.shape)
        ratio = longest_dim / n_samples if n_samples > 0 else 0

        if 0.9 <= ratio <= 1.1:
            return _build_result(
                LabelType.CONTINUOUS, 0.92,
                {
                    "array_shape": list(arr.shape),
                    "n_samples": n_samples,
                    "length_ratio": round(ratio, 3),
                    "dtype": str(arr.dtype),
                    "reason": "array length matches data n_samples",
                },
                warnings,
            )

        # Check if length matches a different sampling rate
        if frequency and data_duration:
            for candidate_freq in [30, 60, 90, 120, 240, 500, 1000]:
                expected_len = int(data_duration * candidate_freq)
                if 0.95 <= longest_dim / max(expected_len, 1) <= 1.05:
                    return _build_result(
                        LabelType.CONTINUOUS, 0.85,
                        {
                            "array_shape": list(arr.shape),
                            "inferred_label_freq": candidate_freq,
                            "data_freq": frequency,
                            "reason": f"array length matches {candidate_freq}Hz over data duration",
                        },
                        warnings,
                    )

    # Small array with few unique values — likely session or event-count
    if effective.ndim == 1 and len(effective) <= 5:
        return _build_result(
            LabelType.SESSION, 0.7,
            {
                "values": effective.tolist(),
                "reason": "very short array (<=5 elements), likely metadata labels",
            },
            warnings,
        )

    # 2D array with shape (N, 2) or (N, 3+) — could be events or segments
    if effective.ndim == 2:
        n_rows, n_cols = effective.shape
        if n_cols == 2:
            # [start, end] pairs → L2 segment
            return _build_result(
                LabelType.SEGMENT, 0.75,
                {
                    "n_intervals": n_rows,
                    "n_cols": n_cols,
                    "reason": "2-column array interpreted as [start, end] intervals",
                },
                warnings,
            )
        if n_cols >= 3:
            # [onset, duration, type] or [begin, end, type] → L1 or L2
            col0_range = float(effective[:, 0].max() - effective[:, 0].min())
            col1_vals = effective[:, 1]
            # If col1 mostly zeros → point events (L1)
            if np.mean(col1_vals == 0) > 0.8:
                return _build_result(
                    LabelType.EVENT, 0.8,
                    {"n_events": n_rows, "reason": "3+ column array with mostly-zero durations"},
                    warnings,
                )
            else:
                return _build_result(
                    LabelType.SEGMENT, 0.75,
                    {"n_intervals": n_rows, "reason": "3+ column array with non-zero durations"},
                    warnings,
                )

    # Default: if long 1D with many unique values → likely continuous
    if effective.ndim == 1 and len(effective) > 100:
        n_unique = len(np.unique(effective[:1000]))
        if n_unique > 50:
            return _build_result(
                LabelType.CONTINUOUS, 0.6,
                {
                    "array_length": len(effective),
                    "n_unique_sample": n_unique,
                    "reason": "long array with many unique values (likely continuous regression target)",
                },
                warnings,
            )
        else:
            # Few unique values in a long array — repeated class labels per window/trial?
            return _build_result(
                LabelType.EVENT, 0.5,
                {
                    "array_length": len(effective),
                    "n_unique": n_unique,
                    "reason": "long array with few unique values (likely per-trial class labels)",
                },
                warnings,
            )

    return _build_result(LabelType.UNKNOWN, 0.3, {"array_shape": list(arr.shape)}, warnings)


def _classify_from_event_list(
    events: List[Dict],
    n_samples: Optional[int],
    frequency: Optional[float],
    data_duration: Optional[float],
    warnings: List[str],
) -> Dict[str, Any]:
    """Classify label type from a list of event dicts (event_loader output)."""
    if not events:
        return _build_result(LabelType.UNKNOWN, 0.0, {"reason": "empty event list"}, warnings)

    # Check if events have onset + duration
    has_onset = all("onset" in e for e in events)
    has_duration = all("duration" in e for e in events)

    if not has_onset:
        # No time information — session-level?
        if all("type" in e or "label" in e for e in events):
            if len(events) <= 3:
                return _build_result(
                    LabelType.SESSION, 0.7,
                    {"n_entries": len(events), "reason": "few entries without onset → session labels"},
                    warnings,
                )
        return _build_result(LabelType.UNKNOWN, 0.3, {}, warnings)

    # Has onset — check durations
    durations = [e.get("duration", 0.0) for e in events]
    nonzero_durations = [d for d in durations if d > 0]
    fraction_with_duration = len(nonzero_durations) / len(events) if events else 0

    if fraction_with_duration > 0.8:
        # Most events have duration → L2 (segment labels)
        return _build_result(
            LabelType.SEGMENT, 0.88,
            {
                "n_events": len(events),
                "mean_duration": round(float(np.mean(nonzero_durations)), 3),
                "reason": f"{fraction_with_duration*100:.0f}% events have non-zero duration → segment labels",
            },
            warnings,
        )
    else:
        # Point events → L1
        types = [e.get("type", "unknown") for e in events]
        unique_types = list(set(types))
        return _build_result(
            LabelType.EVENT, 0.9,
            {
                "n_events": len(events),
                "n_types": len(unique_types),
                "types": unique_types[:10],
                "fraction_with_duration": round(fraction_with_duration, 2),
                "reason": "events with precise onsets and mostly zero durations → point event labels",
            },
            warnings,
        )


def _classify_from_dict(
    data: Union[Dict, List],
    n_samples: Optional[int],
    frequency: Optional[float],
    data_duration: Optional[float],
    warnings: List[str],
) -> Dict[str, Any]:
    """Classify label type from a dict or parsed JSON structure."""
    if isinstance(data, list):
        # Array of items
        if not data:
            return _build_result(LabelType.UNKNOWN, 0.0, {}, warnings)
        if isinstance(data[0], dict):
            return _classify_from_event_list(data, n_samples, frequency, data_duration, warnings)
        # List of scalars — session labels
        if len(data) <= 5:
            return _build_result(
                LabelType.SESSION, 0.7,
                {"values": data[:10], "reason": "short list of scalar labels"},
                warnings,
            )
        return _build_result(LabelType.UNKNOWN, 0.3, {}, warnings)

    if not isinstance(data, dict):
        return _build_result(LabelType.UNKNOWN, 0.2, {}, warnings)

    # Check for hierarchical nesting
    if _is_hierarchical(data):
        depth = _measure_nesting_depth(data)
        return _build_result(
            LabelType.HIERARCHICAL, 0.85,
            {
                "nesting_depth": depth,
                "top_keys": list(data.keys())[:10],
                "reason": "nested dict/list structure with multiple levels",
            },
            warnings,
        )

    # Check for known structures
    # Dict with event-like arrays
    if "onset" in data or "onsets" in data:
        onset_key = "onset" if "onset" in data else "onsets"
        onsets = data[onset_key]
        if isinstance(onsets, (list, np.ndarray)) and len(onsets) > 0:
            events_as_list = []
            types = data.get("type", data.get("types", data.get("trial_type", ["unknown"] * len(onsets))))
            durations = data.get("duration", data.get("durations", [0.0] * len(onsets)))
            for i, o in enumerate(onsets):
                events_as_list.append({
                    "onset": float(o),
                    "duration": float(durations[i]) if i < len(durations) else 0.0,
                    "type": str(types[i]) if i < len(types) else "unknown",
                })
            return _classify_from_event_list(events_as_list, n_samples, frequency, data_duration, warnings)

    # Single key-value pairs without nesting — session metadata
    if all(not isinstance(v, (dict, list)) for v in data.values()):
        return _build_result(
            LabelType.SESSION, 0.75,
            {"keys": list(data.keys())[:10], "reason": "flat dict with no nesting → session metadata"},
            warnings,
        )

    # Dict with an events/trials array inside
    for key in ("events", "trials", "markers", "stimuli"):
        if key in data and isinstance(data[key], list):
            return _classify_from_event_list(
                data[key], n_samples, frequency, data_duration, warnings
            )

    return _build_result(LabelType.UNKNOWN, 0.3, {"keys": list(data.keys())[:10]}, warnings)


def _classify_tabular_file(
    filepath: str,
    n_samples: Optional[int],
    frequency: Optional[float],
    data_duration: Optional[float],
    warnings: List[str],
) -> Dict[str, Any]:
    """Classify a CSV/TSV file by peeking at its structure."""
    path = Path(filepath)
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","

    headers = None
    n_data_rows = 0
    sample_rows: List[List[str]] = []

    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("%"):
                    continue
                parts = [p.strip().strip('"').strip("'") for p in line.split(delimiter)]

                if headers is None:
                    try:
                        float(parts[0])
                        headers = [f"col{j}" for j in range(len(parts))]
                        n_data_rows += 1
                        if len(sample_rows) < 5:
                            sample_rows.append(parts)
                    except ValueError:
                        headers = parts
                else:
                    n_data_rows += 1
                    if len(sample_rows) < 5:
                        sample_rows.append(parts)

                if i > 10000:
                    # Estimate total rows
                    break
    except OSError as e:
        warnings.append(f"Cannot read file: {e}")
        return _build_result(LabelType.UNKNOWN, 0.0, {}, warnings)

    if not headers:
        return _build_result(LabelType.UNKNOWN, 0.2, {"reason": "empty or unreadable file"}, warnings)

    headers_lower = [h.lower() for h in headers]

    # Check for onset/start columns → L1 or L2
    has_onset = any(
        k in headers_lower
        for k in ("onset", "start", "start_time", "time", "latency", "sample", "timestamp")
    )
    has_end = any(
        k in headers_lower
        for k in ("end", "stop", "end_time", "stop_time")
    )
    has_duration = any(
        k in headers_lower
        for k in ("duration", "dur", "length")
    )
    has_type = any(
        k in headers_lower
        for k in ("type", "trial_type", "condition", "label", "class", "event_type", "category")
    )

    # L2: has start + end columns
    if has_onset and has_end:
        return _build_result(
            LabelType.SEGMENT, 0.9,
            {
                "columns": headers,
                "n_rows": n_data_rows,
                "reason": "has onset/start + end/stop columns → interval labels",
            },
            warnings,
        )

    # L1: has onset + type but no end
    if has_onset and has_type and not has_end:
        # Check if durations are mostly nonzero (→ L2 instead)
        if has_duration and sample_rows:
            dur_col_idx = next(
                (i for i, h in enumerate(headers_lower) if h in ("duration", "dur", "length")),
                None,
            )
            if dur_col_idx is not None:
                try:
                    sample_durs = [float(r[dur_col_idx]) for r in sample_rows if len(r) > dur_col_idx]
                    if sample_durs and sum(d > 0 for d in sample_durs) / len(sample_durs) > 0.8:
                        return _build_result(
                            LabelType.SEGMENT, 0.85,
                            {
                                "columns": headers,
                                "n_rows": n_data_rows,
                                "reason": "onset+duration+type with mostly nonzero durations → segment labels",
                            },
                            warnings,
                        )
                except (ValueError, IndexError):
                    pass

        return _build_result(
            LabelType.EVENT, 0.88,
            {
                "columns": headers,
                "n_rows": n_data_rows,
                "reason": "has onset + type columns → point event labels",
            },
            warnings,
        )

    # Only onset, no type — still likely events
    if has_onset:
        return _build_result(
            LabelType.EVENT, 0.7,
            {
                "columns": headers,
                "n_rows": n_data_rows,
                "reason": "has onset column but no explicit type column",
            },
            warnings,
        )

    # Check for continuous-like structure: single column of values, many rows
    if len(headers) <= 3 and n_data_rows > 100:
        # Could be per-sample labels
        if n_samples and 0.9 <= n_data_rows / n_samples <= 1.1:
            return _build_result(
                LabelType.CONTINUOUS, 0.85,
                {
                    "columns": headers,
                    "n_rows": n_data_rows,
                    "n_samples": n_samples,
                    "reason": "row count matches data n_samples → per-sample labels",
                },
                warnings,
            )
        if frequency and data_duration:
            for candidate_freq in [30, 60, 120, 250, 500]:
                expected = int(data_duration * candidate_freq)
                if expected > 0 and 0.95 <= n_data_rows / expected <= 1.05:
                    return _build_result(
                        LabelType.CONTINUOUS, 0.75,
                        {
                            "columns": headers,
                            "n_rows": n_data_rows,
                            "inferred_label_freq": candidate_freq,
                            "reason": f"row count matches {candidate_freq}Hz label stream",
                        },
                        warnings,
                    )

    # Few rows, only type/label columns → session labels
    if n_data_rows <= 5 and has_type:
        return _build_result(
            LabelType.SESSION, 0.7,
            {"columns": headers, "n_rows": n_data_rows, "reason": "very few rows with type column"},
            warnings,
        )

    # Fallback
    if n_data_rows <= 10:
        return _build_result(
            LabelType.SESSION, 0.5,
            {"columns": headers, "n_rows": n_data_rows, "reason": "few rows, no clear time structure"},
            warnings,
        )

    return _build_result(
        LabelType.UNKNOWN, 0.3,
        {"columns": headers, "n_rows": n_data_rows, "reason": "cannot determine label type from structure"},
        warnings,
    )


def _classify_mat_file(
    filepath: str,
    n_samples: Optional[int],
    frequency: Optional[float],
    data_duration: Optional[float],
    warnings: List[str],
) -> Dict[str, Any]:
    """Classify .mat file label structure."""
    try:
        import scipy.io
        mat = scipy.io.loadmat(filepath, squeeze_me=True)
    except (ImportError, NotImplementedError, OSError) as e:
        warnings.append(f"Cannot read .mat file: {e}")
        return _build_result(LabelType.UNKNOWN, 0.3, {}, warnings)

    # Check for FieldTrip trl
    if "trl" in mat:
        trl = np.asarray(mat["trl"])
        return _build_result(
            LabelType.SEGMENT, 0.9,
            {"n_trials": trl.shape[0], "trl_cols": trl.shape[1] if trl.ndim == 2 else 1,
             "reason": "FieldTrip trl matrix (begin/end sample pairs)"},
            warnings,
        )

    # Check for EEGLAB event struct
    if "EEG" in mat and hasattr(mat["EEG"], "dtype"):
        eeg = mat["EEG"]
        if eeg.dtype.names and "event" in eeg.dtype.names:
            return _build_result(
                LabelType.EVENT, 0.9,
                {"reason": "EEGLAB EEG.event struct detected"},
                warnings,
            )

    # Look for arrays matching n_samples
    for key, val in mat.items():
        if key.startswith("_"):
            continue
        if isinstance(val, np.ndarray):
            result = _classify_from_array(val, n_samples, frequency, data_duration, [])
            if result["confidence"] > 0.6:
                return result

    return _build_result(LabelType.UNKNOWN, 0.4, {"mat_keys": [k for k in mat if not k.startswith("_")]}, warnings)


def _is_hierarchical(data: Dict) -> bool:
    """Check if a dict has nested structure indicating L5 hierarchy."""
    if not isinstance(data, dict):
        return False

    nested_count = 0
    for val in data.values():
        if isinstance(val, dict):
            # Check for further nesting
            for v2 in val.values():
                if isinstance(v2, (dict, list)):
                    nested_count += 1
                    break
        elif isinstance(val, list) and val and isinstance(val[0], dict):
            # List of dicts with further nesting
            sample = val[0]
            for v2 in sample.values():
                if isinstance(v2, (dict, list)) and v2:
                    nested_count += 1
                    break

    return nested_count >= 2


def _measure_nesting_depth(data: Any, max_depth: int = 10) -> int:
    """Measure the maximum nesting depth of a structure."""
    if max_depth <= 0:
        return 0
    if isinstance(data, dict):
        if not data:
            return 1
        return 1 + max(_measure_nesting_depth(v, max_depth - 1) for v in data.values())
    elif isinstance(data, list):
        if not data:
            return 1
        # Sample first element
        return 1 + _measure_nesting_depth(data[0], max_depth - 1)
    return 0


def _build_result(
    label_type: LabelType,
    confidence: float,
    details: Dict[str, Any],
    warnings: List[str],
) -> Dict[str, Any]:
    """Build standardized classification result."""
    return {
        "label_type": label_type,
        "label_type_name": label_type.value,
        "confidence": confidence,
        "strategy": _PROCESSING_STRATEGIES[label_type],
        "details": details,
        "warnings": warnings,
    }
