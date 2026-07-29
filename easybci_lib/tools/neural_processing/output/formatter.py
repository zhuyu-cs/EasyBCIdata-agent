"""Pkl formatter — serialize processed data to standardized pkl + meta.

Output schema:
{
    "data": {modality: ndarray, ...},
    "labels": {name: ndarray, ...},
    "meta": {
        "subject_id": str,
        "session": str,
        "paradigm": str,
        "sampling_rates": {modality: float, ...},
        "channels": {modality: [str, ...], ...},
        "n_segments": int,
        "segment_duration": float,
        "created_at": str (ISO),
        "pipeline": [str, ...],  # steps applied
        ...
    }
}
"""

import json
import logging
import pickle
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


def save_pkl(
    data: Dict[str, np.ndarray],
    labels: Dict[str, np.ndarray],
    meta: Dict[str, Any],
    output_path: Union[str, Path],
    save_meta_json: bool = True,
) -> Path:
    """Save processed data as pkl file with optional JSON meta sidecar.

    Parameters
    ----------
    data : dict
        Modality name → ndarray. E.g. {"eeg": (n_seg, n_ch, n_t), "spike": (n_seg, n_units, n_bins)}
    labels : dict
        Label name → ndarray. E.g. {"task": (n_seg,), "subject": (n_seg,)}
    meta : dict
        Metadata dict (will be augmented with timestamps).
    output_path : str or Path
        Destination .pkl file path.
    save_meta_json : bool
        If True, write a .meta.json sidecar alongside the pkl.

    Returns
    -------
    Path to the saved pkl file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Augment meta
    meta = dict(meta)
    meta.setdefault("created_at", time.strftime("%Y-%m-%dT%H:%M:%S"))
    meta.setdefault("format_version", "1.0")

    # Add shape info for quick inspection without loading full arrays
    meta["shapes"] = {
        "data": {k: list(v.shape) for k, v in data.items()},
        "labels": {k: list(v.shape) for k, v in labels.items()},
    }

    payload = {
        "data": data,
        "labels": labels,
        "meta": meta,
    }

    with open(output_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    if save_meta_json:
        meta_path = output_path.with_suffix(".meta.json")
        # JSON-safe copy (no ndarrays)
        meta_json = _make_json_safe(meta)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_json, f, indent=2)

    return output_path


def load_pkl(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a pkl file back into the standard dict format.

    Returns dict with keys: "data", "labels", "meta".
    """
    path = Path(path)
    with open(path, "rb") as f:
        payload = pickle.load(f)

    if not isinstance(payload, dict) or "data" not in payload:
        logger.warning("Invalid pkl format at %s — expected dict with 'data' key, returning raw data", path)
        return payload if isinstance(payload, dict) else {"data": payload, "labels": {}, "meta": {}}

    return payload


def save_npz(
    data: Dict[str, np.ndarray],
    labels: Dict[str, np.ndarray],
    meta: Dict[str, Any],
    output_path: Union[str, Path],
    save_meta_json: bool = True,
) -> Path:
    """Save processed data as compressed NumPy .npz file with JSON meta sidecar.

    Keys are prefixed: data_{modality}, labels_{name}.
    """
    output_path = Path(output_path)
    if output_path.suffix != ".npz":
        output_path = output_path.with_suffix(".npz")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    meta = dict(meta)
    meta.setdefault("created_at", time.strftime("%Y-%m-%dT%H:%M:%S"))
    meta.setdefault("format_version", "1.0")
    meta["shapes"] = {
        "data": {k: list(v.shape) for k, v in data.items()},
        "labels": {k: list(v.shape) for k, v in labels.items()},
    }

    arrays = {}
    for k, v in data.items():
        arrays[f"data_{k}"] = v
    for k, v in labels.items():
        arrays[f"labels_{k}"] = v

    np.savez_compressed(str(output_path), **arrays)

    if save_meta_json:
        meta_path = output_path.with_suffix(".meta.json")
        meta_json = _make_json_safe(meta)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_json, f, indent=2)

    return output_path


def load_npz(path: Union[str, Path]) -> Dict[str, Any]:
    """Load an npz file back into the standard dict format."""
    path = Path(path)
    npz = np.load(str(path), allow_pickle=False)

    data = {}
    labels = {}
    for key in npz.files:
        if key.startswith("data_"):
            data[key[5:]] = npz[key]
        elif key.startswith("labels_"):
            labels[key[7:]] = npz[key]

    meta_path = path.with_suffix(".meta.json")
    meta = {}
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    return {"data": data, "labels": labels, "meta": meta}


def save_mat(
    data: Dict[str, np.ndarray],
    labels: Dict[str, np.ndarray],
    meta: Dict[str, Any],
    output_path: Union[str, Path],
    save_meta_json: bool = True,
) -> Path:
    """Save processed data as MATLAB .mat file.

    Structure: data_{modality}, labels_{name}, meta (as struct).
    """
    from scipy.io import savemat

    output_path = Path(output_path)
    if output_path.suffix != ".mat":
        output_path = output_path.with_suffix(".mat")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    meta = dict(meta)
    meta.setdefault("created_at", time.strftime("%Y-%m-%dT%H:%M:%S"))
    meta.setdefault("format_version", "1.0")
    meta["shapes"] = {
        "data": {k: list(v.shape) for k, v in data.items()},
        "labels": {k: list(v.shape) for k, v in labels.items()},
    }

    mat_dict = {}
    for k, v in data.items():
        mat_dict[f"data_{k}"] = v
    for k, v in labels.items():
        mat_dict[f"labels_{k}"] = v
    mat_dict["meta"] = _make_json_safe(meta)

    savemat(str(output_path), mat_dict, do_compression=True)

    if save_meta_json:
        meta_path = output_path.with_suffix(".meta.json")
        meta_json = _make_json_safe(meta)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_json, f, indent=2)

    return output_path


def load_mat(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a .mat file back into the standard dict format."""
    from scipy.io import loadmat

    path = Path(path)
    raw = loadmat(str(path), squeeze_me=True)

    data = {}
    labels = {}
    meta = {}
    for key, val in raw.items():
        if key.startswith("__"):
            continue
        if key.startswith("data_"):
            data[key[5:]] = np.asarray(val)
        elif key.startswith("labels_"):
            labels[key[7:]] = np.asarray(val)
        elif key == "meta":
            meta = val if isinstance(val, dict) else {}

    meta_path = path.with_suffix(".meta.json")
    if not meta and meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    return {"data": data, "labels": labels, "meta": meta}


SUPPORTED_FORMATS = ("pkl", "nwb")

FORMAT_DESCRIPTIONS = {
    "pkl": "Python pickle — default for AI_ready/, dict {data, labels, meta}, compatible with all downstream tools",
    "nwb": "NWB — default for preprocessed/; universal standard for neural data across modalities",
}


def collect_meta(
    subject_id: str = "",
    paradigm: str = "",
    pipeline_steps: Optional[list] = None,
    **extra,
) -> Dict[str, Any]:
    """Assemble a metadata dict for batch output.

    Parameters
    ----------
    subject_id : str
    paradigm : str
    pipeline_steps : list of str
    **extra : additional metadata fields

    Returns
    -------
    dict suitable for the 'meta' field of the batch payload.
    """
    meta: Dict[str, Any] = {
        "subject_id": subject_id,
        "paradigm": paradigm,
        "pipeline": pipeline_steps or [],
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    meta.update(extra)
    return meta


def build_batch(
    segments: Dict[str, Dict[str, Any]],
    labels: Optional[Dict[str, np.ndarray]] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble multi-modal segments into a unified batch dict.

    Parameters
    ----------
    segments : dict
        Mapping modality name → segmentation result dict.  Each result is
        expected to have a "segments" key with an ndarray of shape
        (n_segments, n_channels, n_samples).
    labels : dict or None
        Label arrays.  If None, an empty dict is used.
    meta : dict or None
        Metadata.  Augmented with per-modality shape info.

    Returns
    -------
    dict with keys "data", "labels", "meta" ready for save_output/save_pkl.
    """
    data: Dict[str, np.ndarray] = {}
    for mod_name, seg_result in segments.items():
        if isinstance(seg_result, dict):
            data[mod_name] = seg_result.get("segments", np.array([]))
        else:
            data[mod_name] = np.asarray(seg_result)

    meta = dict(meta) if meta else {}
    meta["n_segments"] = max((v.shape[0] for v in data.values() if v.size > 0), default=0)
    meta["modalities"] = list(data.keys())

    return {
        "data": data,
        "labels": labels if labels is not None else {},
        "meta": meta,
    }


def save_output(
    data: Dict[str, np.ndarray],
    labels: Dict[str, np.ndarray],
    meta: Dict[str, Any],
    output_path: Union[str, Path],
    fmt: str = "pkl",
    save_meta_json: bool = True,
) -> Path:
    """Unified save dispatcher — routes to format-specific writer.

    Parameters
    ----------
    fmt : str
        Output format: "pkl" (AI_ready/ default) or "nwb" (preprocessed/ default).
    """
    if fmt not in SUPPORTED_FORMATS:
        logger.warning("Unsupported format '%s' — falling back to pkl. Choose from: %s", fmt, SUPPORTED_FORMATS)
        fmt = "pkl"

    if fmt == "pkl":
        return save_pkl(data, labels, meta, output_path, save_meta_json=save_meta_json)
    elif fmt == "nwb":
        from easybci_lib.tools.neural_processing.output.nwb_writer import save_nwb
        # Adapt formatter's multi-modal Dict[str, ndarray] form to save_nwb's
        # single-array payload form. AI_ready segmented data is normally written
        # as .pkl (Spec D2); this NWB route exists for callers that explicitly
        # request fmt="nwb" with continuous (n_ch, n_samp) data.
        if isinstance(data, dict):
            if len(data) == 1:
                arr = next(iter(data.values()))
            else:
                raise ValueError(
                    f"NWB output: multi-modal save not supported "
                    f"(got modalities {list(data.keys())})"
                )
        else:
            arr = data
        # Enrich meta from any embedded mne.Info — the writer also handles
        # this internally, but doing it here keeps the merged meta on the
        # payload for downstream consumers.
        mne_info = None
        if isinstance(meta, dict):
            mne_info = meta.get("_mne_info")
        if mne_info is not None:
            from easybci_lib.tools.neural_processing.output.meta import meta_from_mne_info
            info_meta = meta_from_mne_info(mne_info)
            merged_meta = {**info_meta, **(meta or {})}  # upstream wins on conflicts
        else:
            merged_meta = meta or {}
        payload = {"data": arr, "labels": labels, "meta": merged_meta}
        return save_nwb(payload, Path(output_path), merged_meta, mne_info=mne_info)


def _make_json_safe(obj):
    """Recursively convert non-JSON-serializable types."""
    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_json_safe(x) for x in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        return str(obj)
