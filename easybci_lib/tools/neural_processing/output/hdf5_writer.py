"""HDF5 and BIDS-compatible output formats.

Extends the default pkl output with:
- HDF5: efficient storage for large datasets (>4GB), supports compression
- BIDS: Brain Imaging Data Structure directory layout for sharing/archiving
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np

logger = logging.getLogger(__name__)


def save_hdf5(
    data: Dict[str, np.ndarray],
    labels: Dict[str, np.ndarray],
    meta: Dict[str, Any],
    output_path: Union[str, Path],
    compression: str = "gzip",
) -> Path:
    """Save processed data as HDF5 file.

    Structure:
        /data/{modality}  — datasets
        /labels/{name}    — datasets
        /meta             — JSON string attribute

    Parameters
    ----------
    data : dict of modality → ndarray
    labels : dict of name → ndarray
    meta : dict of metadata
    output_path : path to .h5 file
    compression : "gzip" (default), "lzf", or None

    Returns
    -------
    Path to saved file.
    """
    import h5py

    output_path = Path(output_path)
    if output_path.suffix not in (".h5", ".hdf5"):
        output_path = output_path.with_suffix(".h5")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    meta = dict(meta)
    meta.setdefault("created_at", time.strftime("%Y-%m-%dT%H:%M:%S"))
    meta.setdefault("format_version", "1.0")

    with h5py.File(str(output_path), "w") as f:
        # Data group
        data_grp = f.create_group("data")
        for name, arr in data.items():
            data_grp.create_dataset(
                name, data=arr, compression=compression,
                chunks=True if arr.size > 1000 else None,
            )

        # Labels group
        labels_grp = f.create_group("labels")
        for name, arr in labels.items():
            labels_grp.create_dataset(name, data=arr, compression=compression)

        # Meta as JSON attribute
        meta_safe = _make_json_safe(meta)
        f.attrs["meta"] = json.dumps(meta_safe, ensure_ascii=False)

    logger.info("Saved HDF5: %s (%.1f MB)", output_path, output_path.stat().st_size / 1e6)
    return output_path


def load_hdf5(path: Union[str, Path]) -> Dict[str, Any]:
    """Load an HDF5 file back into the standard dict format."""
    import h5py

    path = Path(path)
    with h5py.File(str(path), "r") as f:
        data = {name: ds[:] for name, ds in f["data"].items()}
        labels = {name: ds[:] for name, ds in f["labels"].items()}
        meta = json.loads(f.attrs.get("meta", "{}"))

    return {"data": data, "labels": labels, "meta": meta}


def save_bids(
    data: Dict[str, np.ndarray],
    labels: Dict[str, np.ndarray],
    meta: Dict[str, Any],
    output_dir: Union[str, Path],
    subject_id: str = "01",
    session: str = "01",
    task: str = "task",
) -> Path:
    """Save in BIDS-compatible directory structure.

    Layout:
        output_dir/
        ├── dataset_description.json
        ├── participants.tsv
        └── sub-{id}/
            └── ses-{session}/
                └── eeg/
                    ├── sub-{id}_ses-{session}_task-{task}_desc-processed_eeg.npy
                    ├── sub-{id}_ses-{session}_task-{task}_events.tsv
                    └── sub-{id}_ses-{session}_task-{task}_eeg.json

    Returns path to the subject directory.
    """
    output_dir = Path(output_dir)
    sub_id = f"sub-{subject_id}"
    ses_id = f"ses-{session}"
    modality = meta.get("modality", "eeg") if isinstance(meta.get("modality"), str) else "eeg"

    # BIDS directory
    bids_dir = output_dir / sub_id / ses_id / modality
    bids_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"{sub_id}_{ses_id}_task-{task}"

    # Dataset description (root level)
    desc_path = output_dir / "dataset_description.json"
    if not desc_path.exists():
        desc = {
            "Name": meta.get("paradigm", "EasyBCIdata processed"),
            "BIDSVersion": "1.8.0",
            "GeneratedBy": [{"Name": "EasyBCIdata", "Version": "1.0"}],
            "License": "CC0",
        }
        desc_path.write_text(json.dumps(desc, indent=2))

    # Participants TSV
    participants_path = output_dir / "participants.tsv"
    if not participants_path.exists():
        participants_path.write_text("participant_id\n")
    existing = participants_path.read_text()
    if sub_id not in existing:
        with open(participants_path, "a", encoding="utf-8") as f:
            f.write(f"{sub_id}\n")

    # Data files
    for mod_name, arr in data.items():
        npy_path = bids_dir / f"{prefix}_desc-processed_{mod_name}.npy"
        np.save(str(npy_path), arr)

    # Labels as events.tsv
    if labels:
        events_path = bids_dir / f"{prefix}_events.tsv"
        first_label = next(iter(labels.values()))
        lines = ["onset\tduration\ttrial_type\n"]
        for i, val in enumerate(first_label):
            lines.append(f"{i}\t1.0\t{val}\n")
        events_path.write_text("".join(lines))

    # Sidecar JSON
    sidecar = {
        "SamplingFrequency": meta.get("sampling_rates", {}).get(modality, 256),
        "TaskName": task,
        "PowerLineFrequency": 50,
        "EEGReference": "average",
    }
    if meta.get("channels"):
        ch_info = meta["channels"]
        if isinstance(ch_info, dict):
            ch_info = next(iter(ch_info.values()), [])
        sidecar["EEGChannelCount"] = len(ch_info) if isinstance(ch_info, list) else 0

    sidecar_path = bids_dir / f"{prefix}_{modality}.json"
    sidecar_path.write_text(json.dumps(sidecar, indent=2))

    logger.info("Saved BIDS: %s", bids_dir)
    return bids_dir


def _make_json_safe(obj):
    """Recursively make an object JSON-serializable."""
    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_json_safe(v) for v in obj]
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
