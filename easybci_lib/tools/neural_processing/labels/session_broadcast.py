"""Session-level label broadcast — attach per-file/session labels to windowed data.

Handles L4 (coarse-grained) labels where the entire recording or session shares
one label (e.g., rest/task, patient/control, drug condition). After sliding_windows
produces segments, this module broadcasts the session label to every window.

Also supports batch mode: given a mapping {file_path → label}, processes multiple
files and produces a unified dataset with condition labels.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


def attach_session_label(
    windowed_data: Dict[str, Any],
    label: Union[str, Dict[str, Any]],
    label_source: str = "explicit",
) -> Dict[str, Any]:
    """Broadcast a session-level label to all windows in segmented data.

    Parameters
    ----------
    windowed_data : dict
        Output from sliding_windows() or segment_data(). Must have "segments" key.
    label : str or dict
        The session label to broadcast. If str, used as-is.
        If dict, can contain multiple label dimensions (e.g., {"condition": "rest", "group": "control"}).
    label_source : str
        How the label was determined: "explicit", "filename", "metadata", "mapping".

    Returns
    -------
    Same dict with added keys:
        labels : list[str] — one label per segment (the broadcast label)
        label_dimensions : dict — if label was multi-dimensional
        label_meta : dict — source information
    """
    segments = windowed_data.get("segments")
    if segments is None:
        import logging
        logging.getLogger(__name__).warning(
            "windowed_data has no 'segments' key — returning unmodified with empty labels."
        )
        result = dict(windowed_data)
        result["labels"] = []
        result["label_meta"] = {"source": "broadcast", "error": "no segments key"}
        return result

    if isinstance(segments, np.ndarray):
        n_segments = segments.shape[0]
    elif isinstance(segments, list):
        n_segments = len(segments)
    else:
        import logging
        logging.getLogger(__name__).warning(
            "Unexpected segments type %s — attempting len().", type(segments),
        )
        try:
            n_segments = len(segments)
        except TypeError:
            n_segments = 0

    result = dict(windowed_data)

    if isinstance(label, str):
        result["labels"] = [label] * n_segments
        result["label_dimensions"] = {"primary": [label] * n_segments}
    elif isinstance(label, dict):
        primary_key = next(iter(label))
        result["labels"] = [str(label[primary_key])] * n_segments
        result["label_dimensions"] = {
            k: [v] * n_segments for k, v in label.items()
        }
    else:
        result["labels"] = [str(label)] * n_segments
        result["label_dimensions"] = {"primary": [str(label)] * n_segments}

    result["label_meta"] = {
        "type": "session_broadcast",
        "source": label_source,
        "label_value": label,
        "n_segments_labeled": n_segments,
    }

    return result


def label_from_filename(filepath: str) -> Optional[str]:
    """Extract a label from a filename using common BCI naming patterns.

    Recognizes patterns like:
    - sub01_rest.edf → "rest"
    - sub01_task-mi.gdf → "mi"
    - EEG_fatigue_session2.set → "fatigue"
    - alert_sub03.bdf → "alert"

    Returns None if no clear label can be extracted.
    """
    stem = Path(filepath).stem.lower()

    known_conditions = {
        "rest", "task", "alert", "drowsy", "fatigue", "sleep",
        "baseline", "active", "passive", "control", "patient",
        "placebo", "drug", "pre", "post", "training", "test",
        "eyes_open", "eyes_closed", "eyesopen", "eyesclosed",
    }

    # Pattern: task-<label> (BIDS style)
    m = re.search(r"task-([a-zA-Z0-9]+)", stem)
    if m:
        return m.group(1)

    # Pattern: condition is a standalone segment of the filename
    parts = re.split(r"[_\-\s]+", stem)
    for part in parts:
        if part in known_conditions:
            return part

    # Pattern: condition after "cond" or "state"
    m = re.search(r"(?:cond|state|condition)[_\-]?([a-zA-Z]+)", stem)
    if m:
        return m.group(1)

    return None


def load_condition_mapping(
    mapping_source: Union[str, Dict[str, str]],
) -> Dict[str, str]:
    """Load a file→label mapping from various sources.

    Parameters
    ----------
    mapping_source : str or dict
        - If dict: used directly as {filename_or_path: label}
        - If str path to .json: load JSON object
        - If str path to .csv/.tsv: expect columns (filename, label)

    Returns
    -------
    Dict mapping filename (stem or full) to label string.
    """
    if isinstance(mapping_source, dict):
        return mapping_source

    p = Path(mapping_source)
    if not p.exists():
        import logging
        logging.getLogger(__name__).warning("Condition mapping not found: %s — returning empty.", mapping_source)
        return {}

    if p.suffix == ".json":
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
        import logging
        logging.getLogger(__name__).warning(
            "JSON condition mapping expected {filename: label} dict, got %s — returning empty.", type(data),
        )
        return {}

    if p.suffix in (".csv", ".tsv"):
        sep = "\t" if p.suffix == ".tsv" else ","
        mapping: Dict[str, str] = {}
        with open(p, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if not lines:
            return {}
        for line in lines[1:] if _looks_like_header(lines[0]) else lines:
            parts = [x.strip() for x in line.strip().split(sep)]
            if len(parts) >= 2:
                mapping[parts[0]] = parts[1]
        return mapping

    import logging
    logging.getLogger(__name__).warning("Unsupported mapping format: %s — returning empty.", p.suffix)
    return {}


def batch_with_session_labels(
    file_data_pairs: List[Dict[str, Any]],
    condition_map: Dict[str, str],
    window_duration: float = 4.0,
    stride: float = 2.0,
    frequency: Optional[float] = None,
) -> Dict[str, Any]:
    """Process multiple files with session labels into a unified dataset.

    Parameters
    ----------
    file_data_pairs : list of dict
        Each dict has: "filepath" (str), "data" (ndarray), "frequency" (float).
    condition_map : dict
        Mapping from filename/path to condition label.
    window_duration : float
        Sliding window size in seconds.
    stride : float
        Sliding window step in seconds.
    frequency : float, optional
        If set, resample all files to this frequency before windowing.

    Returns
    -------
    Dict with:
        segments : ndarray (n_total_windows, n_channels, n_samples)
        labels : list[str] — condition label per window
        file_indices : list[int] — which file each window came from
        file_labels : list[str] — label for each file
        meta : dict
    """
    from easybci_lib.tools.neural_processing.segment.segment import sliding_windows

    all_segments = []
    all_labels = []
    all_file_indices = []
    file_labels = []

    for idx, entry in enumerate(file_data_pairs):
        filepath = entry["filepath"]
        data = entry["data"]
        freq = entry.get("frequency", frequency)

        if freq is None:
            import logging
            logging.getLogger(__name__).warning(
                "No frequency for %s — defaulting to 256.0 Hz.", filepath,
            )
            freq = 256.0

        # Resolve label
        label = _resolve_label(filepath, condition_map)
        file_labels.append(label)

        # Window the data
        windowed = sliding_windows(
            data=data,
            frequency=freq,
            window_duration=window_duration,
            stride=stride,
        )

        n_windows = windowed["segments"].shape[0]
        all_segments.append(windowed["segments"])
        all_labels.extend([label] * n_windows)
        all_file_indices.extend([idx] * n_windows)

    if not all_segments:
        return {
            "segments": np.zeros((0,), dtype=np.float32),
            "labels": [],
            "file_indices": [],
            "file_labels": file_labels,
            "meta": {"n_files": 0, "n_total_windows": 0},
        }

    combined_segments = np.concatenate(all_segments, axis=0)

    unique_labels = sorted(set(all_labels))
    label_counts = {lbl: all_labels.count(lbl) for lbl in unique_labels}

    return {
        "segments": combined_segments,
        "labels": all_labels,
        "file_indices": all_file_indices,
        "file_labels": file_labels,
        "meta": {
            "n_files": len(file_data_pairs),
            "n_total_windows": len(all_labels),
            "window_duration": window_duration,
            "stride": stride,
            "unique_labels": unique_labels,
            "label_counts": label_counts,
            "label_source": "condition_map",
        },
    }


def _resolve_label(filepath: str, condition_map: Dict[str, str]) -> str:
    """Resolve label for a file from condition map (tries full path, filename, stem)."""
    if filepath in condition_map:
        return condition_map[filepath]

    name = Path(filepath).name
    if name in condition_map:
        return condition_map[name]

    stem = Path(filepath).stem
    if stem in condition_map:
        return condition_map[stem]

    # Try case-insensitive match
    lower_map = {k.lower(): v for k, v in condition_map.items()}
    if filepath.lower() in lower_map:
        return lower_map[filepath.lower()]
    if name.lower() in lower_map:
        return lower_map[name.lower()]
    if stem.lower() in lower_map:
        return lower_map[stem.lower()]

    # Fallback: try to extract from filename
    inferred = label_from_filename(filepath)
    if inferred:
        return inferred

    return "unknown"


def _looks_like_header(line: str) -> bool:
    """Check if a CSV/TSV line looks like a header."""
    lower = line.lower().strip()
    return any(kw in lower for kw in ("filename", "file", "path", "label", "condition", "class"))
