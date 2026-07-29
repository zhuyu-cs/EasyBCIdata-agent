"""Batch builder — assemble multi-modal segments into a unified batch dict.

Takes outputs from segment_data/sliding_windows + label encoding and
combines them into the standardized format for save_pkl().
"""

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


def build_batch(
    segments: Dict[str, Dict[str, Any]],
    labels: Optional[Dict[str, np.ndarray]] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble segmented data from multiple modalities into one batch.

    Parameters
    ----------
    segments : dict
        Modality name → segmentation result dict (from segment_data or sliding_windows).
        Each must have key "segments" (ndarray) and "frequency" (float).
        Example: {"eeg": {...}, "spike": {...}}
    labels : dict or None
        Pre-encoded label arrays. Keys are label names, values are ndarrays of shape (n_segments,).
        If None, attempts to extract labels from segment events.
    meta : dict or None
        Additional metadata to include. Merged with auto-collected info.

    Returns
    -------
    dict ready for save_pkl():
        {"data": {...}, "labels": {...}, "meta": {...}}
    """
    data = {}
    sampling_rates = {}
    channels = {}
    n_segments = None

    for modality, seg_result in segments.items():
        seg_array = seg_result["segments"]
        data[modality] = seg_array
        sampling_rates[modality] = seg_result["frequency"]

        if n_segments is None:
            n_segments = seg_array.shape[0]

        # Extract channel info if available in meta
        seg_meta = seg_result.get("meta", {})
        if "channels" in seg_result:
            channels[modality] = seg_result["channels"]

    if n_segments is None:
        n_segments = 0

    # Build labels dict
    if labels is None:
        labels = _extract_labels_from_events(segments, n_segments)

    # Build meta
    batch_meta = {
        "n_segments": n_segments,
        "modalities": list(segments.keys()),
        "sampling_rates": sampling_rates,
    }
    if channels:
        batch_meta["channels"] = channels

    # Merge segment-level meta from first modality
    first_mod = list(segments.keys())[0] if segments else None
    if first_mod and "meta" in segments[first_mod]:
        for k, v in segments[first_mod]["meta"].items():
            if k not in batch_meta:
                batch_meta[k] = v

    # Merge user-provided meta
    if meta:
        batch_meta.update(meta)

    return {
        "data": data,
        "labels": labels,
        "meta": batch_meta,
    }


def _extract_labels_from_events(
    segments: Dict[str, Dict[str, Any]], n_segments: int
) -> Dict[str, np.ndarray]:
    """Try to extract labels from event dicts in segmentation results."""
    labels = {}

    # Find first modality with events
    for modality, seg_result in segments.items():
        events = seg_result.get("events", [])
        if not events:
            continue

        # Extract label fields present in events
        sample_event = events[0]
        for key in ("label", "type", "class", "category"):
            if key in sample_event:
                raw_labels = [e.get(key, "") for e in events]
                # Pad if fewer events than segments (out-of-bounds trimmed)
                if len(raw_labels) < n_segments:
                    raw_labels.extend([""] * (n_segments - len(raw_labels)))
                labels[key] = np.array(raw_labels[:n_segments])
        break

    return labels


def merge_batches(batches: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Concatenate multiple batch dicts along the segment axis.

    All batches must have the same modalities and compatible shapes
    (same n_channels, n_samples).
    """
    if not batches:
        logger.warning("No batches to merge — returning empty result")
        return {"data": {}, "labels": {}, "meta": {"n_segments": 0, "n_batches_merged": 0}}
    if len(batches) == 1:
        return batches[0]

    merged_data = {}
    merged_labels = {}

    # Merge data arrays per modality
    modalities = list(batches[0]["data"].keys())
    for mod in modalities:
        arrays = [b["data"][mod] for b in batches if mod in b["data"]]
        merged_data[mod] = np.concatenate(arrays, axis=0)

    # Merge labels
    label_keys = set()
    for b in batches:
        label_keys.update(b.get("labels", {}).keys())

    for key in label_keys:
        arrays = [b["labels"][key] for b in batches if key in b.get("labels", {})]
        if arrays:
            merged_labels[key] = np.concatenate(arrays, axis=0)

    # Merge meta (take first, update counts)
    merged_meta = dict(batches[0].get("meta", {}))
    merged_meta["n_segments"] = merged_data[modalities[0]].shape[0] if modalities else 0
    merged_meta["n_batches_merged"] = len(batches)

    return {
        "data": merged_data,
        "labels": merged_labels,
        "meta": merged_meta,
    }
