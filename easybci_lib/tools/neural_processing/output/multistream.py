"""Multi-stream output — preserve modality correspondence after alignment.

After multi-stream alignment (e.g., EEG + EMG + eye-tracking from XDF),
this module writes output that preserves:
1. Per-stream data arrays (all on the same timebase)
2. Stream metadata (original frequency, alignment offset, quality)
3. Master clock reference information
4. Inter-stream correspondence mapping

Output formats supported: HDF5 (recommended for multi-stream), pkl, npz.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


def save_multistream_output(
    streams: Dict[str, np.ndarray],
    stream_meta: List[Dict[str, Any]],
    master_stream: str,
    output_path: Union[str, Path],
    labels: Optional[Dict[str, np.ndarray]] = None,
    alignment_report: Optional[Dict[str, Any]] = None,
    segments: Optional[Dict[str, np.ndarray]] = None,
    global_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Save multi-stream aligned data with modality correspondence metadata.

    Parameters
    ----------
    streams : dict of stream_name → ndarray
        Aligned data arrays. All on the same timebase (master clock).
        Each array shape: (n_channels, n_samples) or (n_segments, n_channels, n_samples).
    stream_meta : list of dict
        Per-stream metadata: {name, original_freq, aligned_freq, offset_applied,
        alignment_method, quality, stream_type, n_channels, channel_names}.
    master_stream : str
        Name of the master stream (others aligned to it).
    output_path : str or Path
        Output file path. Extension determines format (.h5, .pkl, .npz).
    labels : dict, optional
        Label arrays per stream or global.
    alignment_report : dict, optional
        Full alignment report from align_to_master().
    segments : dict, optional
        Segmented data per stream (if segmentation was applied).
    global_meta : dict, optional
        Global metadata (subject_id, paradigm, pipeline steps, etc.)

    Returns
    -------
    Dict with: path, format, streams_saved, total_bytes.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ext = output_path.suffix.lower()

    # Build correspondence mapping
    correspondence = _build_correspondence_map(stream_meta, master_stream)

    # Assemble metadata
    meta = {
        "format_version": "2.0_multistream",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "master_stream": master_stream,
        "n_streams": len(streams),
        "stream_names": list(streams.keys()),
        "correspondence": correspondence,
        **(global_meta or {}),
    }

    if alignment_report:
        meta["alignment"] = alignment_report

    if ext in (".h5", ".hdf5"):
        result = _save_hdf5_multistream(
            output_path, streams, stream_meta, meta, labels, segments
        )
    elif ext == ".npz":
        result = _save_npz_multistream(
            output_path, streams, stream_meta, meta, labels, segments
        )
    else:
        result = _save_pkl_multistream(
            output_path, streams, stream_meta, meta, labels, segments
        )

    return result


def _build_correspondence_map(
    stream_meta: List[Dict[str, Any]],
    master_stream: str,
) -> Dict[str, Any]:
    """Build inter-stream correspondence metadata.

    Documents how each stream relates to the master clock:
    - Time offset applied
    - Resampling ratio
    - Alignment quality
    - Channel-level mapping (if applicable)
    """
    correspondence: Dict[str, Any] = {
        "master": master_stream,
        "aligned_timebase": True,
        "streams": {},
    }

    for meta in stream_meta:
        name = meta.get("name", "")
        original_freq = meta.get("original_freq", meta.get("original_frequency", 0))
        aligned_freq = meta.get("aligned_freq", meta.get("frequency", original_freq))
        offset = meta.get("offset_applied", meta.get("offset_applied_s", 0.0))
        quality = meta.get("quality", meta.get("alignment_quality", 1.0))
        method = meta.get("alignment_method", meta.get("method", "none"))

        is_master = (name == master_stream)

        correspondence["streams"][name] = {
            "is_master": is_master,
            "original_frequency_hz": original_freq,
            "aligned_frequency_hz": aligned_freq,
            "resample_ratio": round(aligned_freq / max(original_freq, 1e-6), 4),
            "time_offset_s": round(offset, 6),
            "alignment_method": method if not is_master else "reference",
            "alignment_quality": round(quality, 3),
            "stream_type": meta.get("stream_type", "unknown"),
            "n_channels": meta.get("n_channels", 0),
        }

    return correspondence


def _save_hdf5_multistream(
    path: Path,
    streams: Dict[str, np.ndarray],
    stream_meta: List[Dict[str, Any]],
    meta: Dict[str, Any],
    labels: Optional[Dict[str, np.ndarray]],
    segments: Optional[Dict[str, np.ndarray]],
) -> Dict[str, Any]:
    """Save multi-stream data as HDF5 with group-per-stream structure."""
    try:
        import h5py
    except ImportError:
        logger.warning("h5py not available — falling back to pkl format.")
        return _save_pkl_multistream(
            path.with_suffix(".pkl"), streams, stream_meta, meta, labels, segments
        )

    with h5py.File(str(path), "w") as f:
        # Global metadata
        f.attrs["meta"] = json.dumps(meta, default=str)
        f.attrs["format_version"] = "2.0_multistream"
        f.attrs["master_stream"] = meta["master_stream"]

        # Per-stream groups
        streams_grp = f.create_group("streams")
        for name, data in streams.items():
            grp = streams_grp.create_group(name)
            grp.create_dataset("data", data=data, compression="gzip")

            # Find matching stream metadata
            smeta = next((m for m in stream_meta if m.get("name") == name), {})
            grp.attrs["stream_type"] = smeta.get("stream_type", "unknown")
            grp.attrs["original_frequency"] = smeta.get("original_freq", 0)
            grp.attrs["aligned_frequency"] = smeta.get("aligned_freq", 0)
            grp.attrs["offset_applied_s"] = smeta.get("offset_applied", 0.0)
            grp.attrs["alignment_quality"] = smeta.get("quality", 1.0)
            if smeta.get("channel_names"):
                grp.attrs["channels"] = json.dumps(smeta["channel_names"])

        # Labels
        if labels:
            lbl_grp = f.create_group("labels")
            for name, lbl_arr in labels.items():
                lbl_grp.create_dataset(name, data=lbl_arr, compression="gzip")

        # Segmented data
        if segments:
            seg_grp = f.create_group("segments")
            for name, seg_arr in segments.items():
                seg_grp.create_dataset(name, data=seg_arr, compression="gzip")

        # Correspondence map
        f.attrs["correspondence"] = json.dumps(meta.get("correspondence", {}), default=str)

    total_bytes = path.stat().st_size
    return {
        "path": str(path),
        "format": "hdf5_multistream",
        "streams_saved": list(streams.keys()),
        "total_bytes": total_bytes,
    }


def _save_npz_multistream(
    path: Path,
    streams: Dict[str, np.ndarray],
    stream_meta: List[Dict[str, Any]],
    meta: Dict[str, Any],
    labels: Optional[Dict[str, np.ndarray]],
    segments: Optional[Dict[str, np.ndarray]],
) -> Dict[str, Any]:
    """Save multi-stream data as compressed npz."""
    arrays: Dict[str, np.ndarray] = {}

    for name, data in streams.items():
        arrays[f"stream_{name}"] = data

    if labels:
        for name, lbl_arr in labels.items():
            arrays[f"labels_{name}"] = lbl_arr

    if segments:
        for name, seg_arr in segments.items():
            arrays[f"segments_{name}"] = seg_arr

    np.savez_compressed(str(path), **arrays)

    # Save metadata as companion JSON
    meta_path = path.with_suffix(".meta.json")
    with open(str(meta_path), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)

    total_bytes = path.stat().st_size
    return {
        "path": str(path),
        "format": "npz_multistream",
        "streams_saved": list(streams.keys()),
        "total_bytes": total_bytes,
        "meta_path": str(meta_path),
    }


def _save_pkl_multistream(
    path: Path,
    streams: Dict[str, np.ndarray],
    stream_meta: List[Dict[str, Any]],
    meta: Dict[str, Any],
    labels: Optional[Dict[str, np.ndarray]],
    segments: Optional[Dict[str, np.ndarray]],
) -> Dict[str, Any]:
    """Save multi-stream data as pickle dict."""
    import pickle

    if path.suffix != ".pkl":
        path = path.with_suffix(".pkl")

    output = {
        "streams": streams,
        "stream_meta": stream_meta,
        "labels": labels or {},
        "segments": segments or {},
        "meta": meta,
    }

    with open(str(path), "wb") as f:
        pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)

    total_bytes = path.stat().st_size
    return {
        "path": str(path),
        "format": "pkl_multistream",
        "streams_saved": list(streams.keys()),
        "total_bytes": total_bytes,
    }
