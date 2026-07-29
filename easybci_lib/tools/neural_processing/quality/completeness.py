"""Completeness checks — verify segments have all required data.

Catches problems like:
- Segments shorter than expected (edge-of-recording truncation)
- Missing modalities in multi-modal batches
- Label/data count mismatch
"""

from typing import Any, Dict, List, Optional

import numpy as np


def check_segments_complete(
    batch: Dict[str, Any],
    required_modalities: Optional[List[str]] = None,
    required_labels: Optional[List[str]] = None,
    min_segments: int = 1,
) -> Dict[str, Any]:
    """Verify a batch dict is complete and consistent.

    Parameters
    ----------
    batch : dict
        Standard batch format: {"data": {...}, "labels": {...}, "meta": {...}}
    required_modalities : list of str or None
        Modalities that must be present in data.
    required_labels : list of str or None
        Label keys that must be present.
    min_segments : int
        Minimum acceptable segment count.

    Returns
    -------
    dict with:
        passed : bool
        issues : list of str
        summary : dict
    """
    issues = []
    data = batch.get("data", {})
    labels = batch.get("labels", {})
    meta = batch.get("meta", {})

    # Check data exists
    if not data:
        issues.append("No data modalities present")
        return {"passed": False, "issues": issues, "summary": {}}

    # Segment counts per modality
    seg_counts = {}
    for mod, arr in data.items():
        if isinstance(arr, np.ndarray):
            seg_counts[mod] = arr.shape[0]
        else:
            issues.append(f"Modality '{mod}' is not ndarray (type: {type(arr).__name__})")

    # Check consistency across modalities
    unique_counts = set(seg_counts.values())
    if len(unique_counts) > 1:
        issues.append(
            f"Inconsistent segment counts across modalities: {seg_counts}"
        )

    n_segments = max(seg_counts.values()) if seg_counts else 0

    # Min segments
    if n_segments < min_segments:
        issues.append(
            f"Only {n_segments} segments, expected at least {min_segments}"
        )

    # Required modalities
    if required_modalities:
        missing = [m for m in required_modalities if m not in data]
        if missing:
            issues.append(f"Missing required modalities: {missing}")

    # Required labels
    if required_labels:
        missing_labels = [l for l in required_labels if l not in labels]
        if missing_labels:
            issues.append(f"Missing required labels: {missing_labels}")

    # Label/data shape consistency
    for label_name, label_arr in labels.items():
        if isinstance(label_arr, np.ndarray) and label_arr.shape[0] != n_segments:
            issues.append(
                f"Label '{label_name}' has {label_arr.shape[0]} entries but data has {n_segments} segments"
            )

    # Check for empty segments (all zeros)
    empty_segments = {}
    for mod, arr in data.items():
        if isinstance(arr, np.ndarray) and arr.ndim >= 2:
            per_seg_sum = np.abs(arr).reshape(arr.shape[0], -1).sum(axis=-1)
            n_empty = int((per_seg_sum == 0).sum())
            if n_empty > 0:
                empty_segments[mod] = n_empty

    if empty_segments:
        issues.append(f"Empty (all-zero) segments detected: {empty_segments}")

    summary = {
        "n_segments": n_segments,
        "modalities": list(data.keys()),
        "labels": list(labels.keys()),
        "segment_counts": seg_counts,
        "empty_segments": empty_segments,
    }

    return {
        "passed": len(issues) == 0,
        "issues": issues,
        "summary": summary,
    }
