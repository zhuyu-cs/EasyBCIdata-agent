"""Meta collector — gather metadata from various processing stages.

Provides a structured way to accumulate pipeline metadata so the final
pkl output documents exactly what was done to produce it.
"""

import time
from typing import Any, Dict, List, Optional, Union

import numpy as np


def collect_meta(
    subject_id: Optional[str] = None,
    session: Optional[str] = None,
    paradigm: Optional[str] = None,
    pipeline_steps: Optional[List[str]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a metadata dict for pkl output.

    Parameters
    ----------
    subject_id : str or None
        Subject identifier.
    session : str or None
        Session/run identifier.
    paradigm : str or None
        Experimental paradigm (e.g. "motor_imagery", "p300", "ssvep").
    pipeline_steps : list of str or None
        Ordered list of processing steps applied (e.g. ["notch:50", "bandpass:1,40"]).
    extra : dict or None
        Any additional key-value pairs.

    Returns
    -------
    dict — metadata ready to pass into build_batch() or save_pkl().
    """
    meta = {}

    if subject_id is not None:
        meta["subject_id"] = subject_id
    if session is not None:
        meta["session"] = session
    if paradigm is not None:
        meta["paradigm"] = paradigm
    if pipeline_steps is not None:
        meta["pipeline"] = list(pipeline_steps)

    meta["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    if extra:
        meta.update(extra)

    return meta


def meta_from_mne_info(info) -> Dict[str, Any]:
    """Extract metadata from an MNE Info object.

    Parameters
    ----------
    info : mne.Info
        MNE measurement info (from raw.info).

    Returns
    -------
    dict with channel names, sampling rate, bad channels, etc.
    """
    meta = {
        "sfreq": info["sfreq"],
        "ch_names": list(info["ch_names"]),
        "n_channels": len(info["ch_names"]),
        "bads": list(info.get("bads", [])),
    }

    # Channel types — prefer the public ``get_channel_types()`` API so we get
    # human-readable strings ("eeg", "ecog", ...) rather than numeric kinds.
    try:
        ch_types = list(info.get_channel_types())
        if ch_types:
            meta["ch_types"] = ch_types
    except Exception:
        # Fall back to the raw `kind` integer counts for diagnostics only.
        from collections import Counter
        kinds = [ch["kind"] for ch in info["chs"]] if "chs" in info else []
        if kinds:
            meta["ch_type_counts"] = dict(Counter(kinds))

    # Highpass/lowpass if set
    if info.get("highpass"):
        meta["highpass"] = info["highpass"]
    if info.get("lowpass"):
        meta["lowpass"] = info["lowpass"]

    # Measurement date — propagate as-is. The NWB writer normalises this
    # (datetime / tuple / None) via ``_coerce_session_start``.
    meas_date = info.get("meas_date")
    if meas_date is not None:
        meta["meas_date"] = meas_date

    # Subject info — pull a stable subject identifier when available.
    subject_info = info.get("subject_info")
    if subject_info:
        meta["subject_info"] = dict(subject_info)
        his_id = subject_info.get("his_id") or subject_info.get("id")
        if his_id:
            meta["subject_id"] = str(his_id)

    return meta


def meta_from_nwb(nwb_file) -> Dict[str, Any]:
    """Extract metadata from an NWB file object.

    Parameters
    ----------
    nwb_file : pynwb.NWBFile
        Open NWB file.

    Returns
    -------
    dict with session info, units, electrodes, etc.
    """
    meta = {}

    if hasattr(nwb_file, "session_description"):
        meta["session_description"] = nwb_file.session_description
    if hasattr(nwb_file, "identifier"):
        meta["identifier"] = nwb_file.identifier
    if hasattr(nwb_file, "session_start_time"):
        meta["session_start_time"] = str(nwb_file.session_start_time)

    # Units (spike sorting results)
    if hasattr(nwb_file, "units") and nwb_file.units is not None:
        units = nwb_file.units
        meta["n_units"] = len(units)
        if "electrode_group" in units.colnames:
            groups = set()
            for i in range(len(units)):
                eg = units["electrode_group"][i]
                if hasattr(eg, "name"):
                    groups.add(eg.name)
            meta["electrode_groups"] = sorted(groups)

    # Electrodes
    if hasattr(nwb_file, "electrodes") and nwb_file.electrodes is not None:
        meta["n_electrodes"] = len(nwb_file.electrodes)
        if "location" in nwb_file.electrodes.colnames:
            locations = set()
            for i in range(len(nwb_file.electrodes)):
                loc = nwb_file.electrodes["location"][i]
                if loc:
                    locations.add(str(loc))
            meta["electrode_locations"] = sorted(locations)

    return meta
