"""Unified neural data loader.

Single entry point: load_neural(filepath) → dict with numpy data.
Handles MNE formats (EEG/MEG/iEEG) and HDF5/NWB (spikes).

What we borrow from neuralset:
- Auto-detection of format from file extension
- allow_maxshield for Elekta FIF files
- clean_names for CTF .ds files
- NWB spike_times structure traversal

What we DON'T borrow:
- The Event class hierarchy (too heavy for a loader)
- exca caching (replaced by simple file I/O)
- pydantic validation on file load (unnecessary overhead)
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from easybci_lib.tools.neural_processing.io.validators import validate_loaded_data

logger = logging.getLogger(__name__)

# Extension → backend mapping
_FORMAT_MAP = {
    # MNE-supported continuous formats
    ".fif": "mne", ".edf": "mne", ".bdf": "mne", ".set": "mne",
    ".ds": "mne", ".cnt": "mne", ".gdf": "mne",
    ".vhdr": "mne", ".vmrk": "mne", ".eeg": "mne",  # BrainVision
    ".cdt": "mne",  # Neuroscan CURRY
    ".mff": "mne",  # EGI/Philips
    ".sqd": "mne", ".con": "mne",  # KIT/Yokogawa MEG
    # HDF5/NWB
    ".nwb": "hdf5", ".h5": "hdf5", ".hdf5": "hdf5",
    # MATLAB
    ".mat": "mat",
    # Tabular
    ".csv": "csv", ".tsv": "csv", ".parquet": "csv",
    # NumPy
    ".npz": "npz", ".npy": "npz",
    # Multi-stream (LSL)
    ".xdf": "xdf", ".xdfz": "xdf",
    # Pickle (EasyBCI native output format)
    ".pkl": "pkl", ".pickle": "pkl",
}


def load_neural(
    filepath: str,
    modality: str = "auto",
    preload: bool = True,
    max_duration: Optional[float] = None,
    inspect_only: bool = False,
) -> Dict[str, Any]:
    """Load neural recording into a standardized dict.

    This is the single entry point the agent calls.

    Parameters
    ----------
    filepath : str
        Path to recording file.
    modality : str
        "eeg", "meg", "ieeg", "spike", or "auto" (detect from format).
    preload : bool
        Load data into memory immediately. If False, MNE uses memory-mapped I/O.
    max_duration : float or None
        If set, only load the first N seconds (useful for large files).
    inspect_only : bool
        If True, load only metadata and a small sample (first 1 second) for
        stats computation. Avoids full data load for large files. The returned
        dict includes 'n_channels', 'n_samples', 'data_nbytes_estimate' in meta,
        and 'data' contains only the sample.

    Returns
    -------
    dict with keys:
        data : np.ndarray — shape (n_channels, n_samples) or small sample if inspect_only
        frequency : float — sampling rate in Hz
        channels : list[str] — channel/unit names
        duration : float — total duration in seconds
        meta : dict — format-specific metadata
    """
    filepath = str(filepath)
    path = Path(filepath)

    # Resolve companion files to their main data file
    filepath = _resolve_companion_to_main(filepath)
    path = Path(filepath)

    # Verify the file actually exists before routing
    if not path.exists():
        logger.warning("Data file not found: '%s'. Checked path: %s", path.name, filepath)
        return {
            "data": np.zeros((1, 0), dtype=np.float32),
            "frequency": 1.0,
            "channels": ["Ch0"],
            "duration": 0.0,
            "meta": {
                "format": "unknown",
                "source_file": filepath,
                "load_error": f"File not found: {filepath}",
            },
        }

    backend = _detect_backend(path)

    if backend == "mne":
        result = _load_mne(filepath, preload=preload, max_duration=max_duration,
                           inspect_only=inspect_only, modality=modality)
    elif backend == "hdf5":
        result = _load_nwb(filepath, inspect_only=inspect_only)
    elif backend == "mat":
        result = _load_mat(filepath, inspect_only=inspect_only)
    elif backend == "csv":
        result = _load_csv(filepath, inspect_only=inspect_only)
    elif backend == "npz":
        result = _load_npz(filepath, inspect_only=inspect_only)
    elif backend == "xdf":
        result = _load_xdf(filepath, inspect_only=inspect_only)
    elif backend == "pkl":
        result = _load_pkl(filepath, inspect_only=inspect_only)
    elif backend == "unknown":
        result = _load_unknown_format(filepath, inspect_only=inspect_only)
    else:
        logger.warning(
            "Backend '%s' not implemented for %s — trying unknown format fallback.",
            backend, path.suffix,
        )
        result = _load_unknown_format(filepath, inspect_only=inspect_only)

    # Single seam: every loader's result must carry meta['data_unit'].
    # Loaders that know the unit (MNE → V/T, Curry → V after µV→V conversion)
    # set it explicitly; everything else defaults to "unknown" so
    # drop_bads:auto can route to its relative-only path instead of falling
    # back to a hidden EEG-in-V assumption. The values follow a small fixed
    # vocabulary: "V" | "uV" | "T" | "fT" | "unknown".
    if isinstance(result, dict):
        _meta = result.setdefault("meta", {})
        if not _meta.get("data_unit"):
            _meta["data_unit"] = "unknown"

    validate_loaded_data(result)
    return result


def _detect_backend(path: Path) -> str:
    """Detect reader backend from file extension."""
    for suffix in reversed(path.suffixes):
        if suffix.lower() in _FORMAT_MAP:
            return _FORMAT_MAP[suffix.lower()]
    if path.is_dir() and path.suffix == ".ds":
        return "mne"
    return "unknown"


# Companion/sidecar extensions that should NOT be loaded directly.
# If a user passes one of these, we resolve to the main data file instead.
_COMPANION_EXTENSIONS = {
    ".hpi", ".ceo", ".dpo", ".dpa",  # Curry companions
    ".vmrk", ".vhdr",  # BrainVision (though .vhdr is also entry point)
    ".fdt",  # EEGLAB companion data
    ".elp", ".hsp", ".mrk",  # Polhemus / marker companions
}


def _resolve_companion_to_main(filepath: str) -> str:
    """If the user passed a companion file, resolve to the main data file.

    E.g., 'data.cdt.hpi' → 'data.cdt', 'data.cdt.dpo' → 'data.cdt'
    """
    path = Path(filepath)

    # Handle compound extensions: if last suffix is a known companion,
    # strip it and check if the parent path exists
    if path.suffix.lower() in _COMPANION_EXTENSIONS:
        main_path = path.with_suffix("")
        if main_path.exists():
            logger.info("Resolved companion file '%s' → main data file '%s'",
                        path.name, main_path.name)
            return str(main_path)

    # Handle .fdt → .set
    if path.suffix.lower() == ".fdt":
        set_path = path.with_suffix(".set")
        if set_path.exists():
            return str(set_path)

    return filepath


def _load_mne(filepath: str, preload: bool = True, max_duration: Optional[float] = None,
              inspect_only: bool = False, modality: str = "auto") -> Dict[str, Any]:
    """Load via MNE — covers EEG, MEG, sEEG, ECoG, EMG."""
    import mne

    path = Path(filepath)
    kwargs: Dict[str, Any] = {"preload": False if inspect_only else preload}

    # EDF/BDF reading requires pyedflib as MNE's backend — lazy-install it.
    if path.suffix.lower() in (".edf", ".bdf"):
        from easybci_lib.tools.lazy_deps import ensure
        ensure("neural.edflib")

    # Format-specific quirks (from neuralset)
    if ".fif" in path.suffixes:
        kwargs["allow_maxshield"] = True
    if ".ds" in path.suffixes:
        kwargs["clean_names"] = True

    try:
        raw = mne.io.read_raw(filepath, **kwargs)
    except FileNotFoundError as e:
        error_msg = str(e)
        if path.suffix.lower() == ".cdt":
            logger.info("MNE Curry reader failed (missing companions), trying raw binary fallback")
            return _load_curry_raw(filepath, inspect_only=inspect_only, modality=modality)
        logger.warning(
            "MNE cannot read '%s': companion files are missing. Details: %s",
            path.name, error_msg,
        )
        return {
            "data": np.zeros((1, 0), dtype=np.float32),
            "frequency": 1.0,
            "channels": ["Ch0"],
            "duration": 0.0,
            "meta": {
                "format": "mne_failed",
                "source_file": filepath,
                "load_error": f"Missing companion files: {error_msg}",
            },
        }
    except (ImportError, RuntimeError) as e:
        error_msg = str(e)
        if "module" in error_msg.lower() or "import" in error_msg.lower():
            if path.suffix.lower() == ".cdt":
                logger.info("curryreader not installed, using raw binary fallback for .cdt")
                return _load_curry_raw(filepath, inspect_only=inspect_only, modality=modality)
            logger.warning(
                "MNE recognized '%s' but the required reader plugin is not installed. %s",
                path.name, error_msg,
            )
            return {
                "data": np.zeros((1, 0), dtype=np.float32),
                "frequency": 1.0,
                "channels": ["Ch0"],
                "duration": 0.0,
                "meta": {
                    "format": "mne_failed",
                    "source_file": filepath,
                    "load_error": f"Missing plugin: {error_msg}",
                },
            }
        logger.warning("MNE RuntimeError reading '%s': %s", path.name, error_msg)
        return {
            "data": np.zeros((1, 0), dtype=np.float32),
            "frequency": 1.0,
            "channels": ["Ch0"],
            "duration": 0.0,
            "meta": {
                "format": "mne_failed",
                "source_file": filepath,
                "load_error": str(e),
            },
        }
    except OSError as e:
        logger.warning("MNE cannot open '%s': %s", path.name, e)
        return {
            "data": np.zeros((1, 0), dtype=np.float32),
            "frequency": 1.0,
            "channels": ["Ch0"],
            "duration": 0.0,
            "meta": {
                "format": "mne_failed",
                "source_file": filepath,
                "load_error": f"OS error: {e}",
            },
        }

    n_channels = len(raw.ch_names)
    n_samples = raw.n_times
    frequency = float(raw.info["sfreq"])
    duration = float(raw.times[-1] - raw.times[0] + 1.0 / raw.info["sfreq"])
    ch_types = raw.get_channel_types()

    # Neuroscan Curry .cdt (and similar formats) may report EOG / stim /
    # ECG channels as 'misc' when MNE cannot infer the type from the
    # channel name convention alone.  Correct them by name so downstream
    # steps (drop_nondata_channels, ICA, CAR) see the real types.
    _EOG_NAMES = {"VEOG", "HEOG", "EOG", "VEO", "HEO", "LO1", "LO2", "IO1", "IO2"}
    _STIM_NAMES = {"Trigger", "TRIGGER", "STI", "STIM", "Status", "Event", "Mark"}
    _ECG_NAMES = {"ECG", "EKG"}
    for _i, (_ch, _ct) in enumerate(zip(raw.ch_names, ch_types)):
        if _ct == "misc":
            if _ch in _EOG_NAMES:
                ch_types[_i] = "eog"
            elif _ch in _STIM_NAMES:
                ch_types[_i] = "stim"
            elif _ch in _ECG_NAMES:
                ch_types[_i] = "ecg"

    meta = {
        "format": "mne",
        "highpass": raw.info["highpass"],
        "lowpass": raw.info["lowpass"],
        "n_samples": n_samples,
        "ch_types": ch_types,
        "bad_channels": list(raw.info["bads"]),
        "first_samp": raw.first_samp,
    }

    # MNE returns data in SI base units — V for electrical channels (EEG /
    # EOG / ECG / sEEG / ECoG / EMG / bio) and T for magnetic (MEG mag) /
    # T/m for gradiometers. drop_bads:auto reads this to scale its
    # absolute-threshold fallback; QC report renders it as well.
    _ELECTRICAL = {"eeg", "eog", "ecg", "seeg", "ecog", "emg", "bio", "stim", "misc"}
    _MAGNETIC = {"mag", "grad"}  # MEG magnetometer / gradiometer
    _ct_set = {str(t).lower() for t in ch_types}
    if _ct_set & _ELECTRICAL:
        meta["data_unit"] = "V"
    elif _ct_set & _MAGNETIC:
        meta["data_unit"] = "T"
    else:
        meta["data_unit"] = "unknown"

    # Preserve annotations (event markers from EDF/FIF)
    if raw.annotations and len(raw.annotations) > 0:
        meta["annotations"] = {
            "onset": raw.annotations.onset.tolist(),
            "duration": raw.annotations.duration.tolist(),
            "description": list(raw.annotations.description),
        }

    # Extract events from annotations (for segment_data)
    try:
        events, event_id = mne.events_from_annotations(raw, verbose=False)
        if len(events) > 0:
            meta["events"] = events.tolist()
            meta["event_id"] = event_id
    except Exception:
        pass

    # Extract positions if available
    ch_locs = np.array([ch["loc"][:3] for ch in raw.info["chs"]])
    if not np.all(ch_locs == 0):
        meta["positions_3d"] = ch_locs

    if inspect_only:
        sample_duration = min(1.0, duration)
        raw_sample = raw.copy().crop(tmax=sample_duration)
        raw_sample.load_data()
        data = raw_sample.get_data().astype(np.float32)
        meta["n_channels"] = n_channels
        meta["n_samples_total"] = n_samples
        meta["data_nbytes_estimate"] = n_channels * n_samples * 4
        meta["inspect_only"] = True
    else:
        # Crop to max_duration if specified (for large files)
        if max_duration is not None and raw.times[-1] > max_duration:
            raw = raw.crop(tmax=max_duration)
            if not preload:
                raw.load_data()

        if not preload:
            raw.load_data()
        data = raw.get_data().astype(np.float32)

    return {
        "data": data,
        "frequency": frequency,
        "channels": list(raw.ch_names),
        "duration": duration,
        "meta": meta,
    }


def _load_curry_raw(
    filepath: str,
    inspect_only: bool = False,
    modality: str = "auto",
) -> Dict[str, Any]:
    """Fallback loader for Neuroscan Curry .cdt files using raw binary parsing.

    Reads metadata from the .cdt.dpo companion (DATA_PARAMETERS section) and
    loads raw float32 data directly from the .cdt file.
    """
    path = Path(filepath)

    # Find the .dpo companion (contains metadata)
    dpo_path = Path(str(path) + ".dpo")
    if not dpo_path.exists():
        # Try sibling pattern
        dpo_candidates = list(path.parent.glob(path.stem + "*.dpo"))
        if dpo_candidates:
            dpo_path = dpo_candidates[0]

    # Parse metadata from .dpo
    n_channels = 0
    n_samples = 0
    frequency = 0.0
    data_format = 6  # float32 by default
    byte_order = "little"
    samp_order = "SAMP"
    channels: List[str] = []
    data_unit = "uV"

    if dpo_path.exists():
        with open(dpo_path, "r", encoding="utf-8", errors="replace") as f:
            dpo_content = f.read()

        import re as _re

        # Extract key parameters
        m = _re.search(r"NumChannels\s*=\s*(\d+)", dpo_content)
        if m:
            n_channels = int(m.group(1))
        m = _re.search(r"NumSamples\s*=\s*(\d+)", dpo_content)
        if m:
            n_samples = int(m.group(1))
        m = _re.search(r"SampleFreqHz\s*=\s*([\d.]+)", dpo_content)
        if m:
            frequency = float(m.group(1))
        else:
            m = _re.search(r"SampleTimeUsec\s*=\s*([\d.]+)", dpo_content)
            if m:
                frequency = 1_000_000.0 / float(m.group(1))
        m = _re.search(r"DataFormat\s*=\s*(\d+)", dpo_content)
        if m:
            data_format = int(m.group(1))
        m = _re.search(r"DataByteOrder\s*=\s*(\w+)", dpo_content)
        if m:
            byte_order = "little" if "INTEL" in m.group(1).upper() else "big"
        m = _re.search(r"DataSampOrder\s*=\s*(\w+)", dpo_content)
        if m:
            samp_order = m.group(1).upper()
        m = _re.search(r"DataUnit\s*=\s*(\w+)", dpo_content)
        if m:
            data_unit = m.group(1)

        # Extract channel labels (main + auxiliary groups)
        in_labels = False
        current_section = None
        for line in dpo_content.split("\n"):
            if "LABELS" in line and "START_LIST" in line:
                in_labels = True
                continue
            if in_labels and "END_LIST" in line:
                in_labels = False
                continue
            if in_labels:
                stripped = line.strip()
                if stripped:
                    channels.append(stripped)
    else:
        # No .dpo — try to infer from file size
        file_size = path.stat().st_size
        logger.warning("No .dpo metadata file found for %s, inferring parameters", path.name)
        # Common Curry configs: 64ch, 128ch at 1000Hz or 2000Hz
        for n_ch in (64, 128, 32, 256, 67):
            if file_size % (n_ch * 4) == 0:
                n_samples = file_size // (n_ch * 4)
                n_channels = n_ch
                frequency = 1000.0
                break
        if n_channels == 0:
            # Last resort: try loading as single-channel float32
            logger.warning(
                "Cannot determine data layout for '%s': no .dpo metadata and file size "
                "(%d bytes) doesn't match common channel counts. "
                "Attempting single-channel float32 interpretation.",
                path.name, file_size,
            )
            n_channels = 1
            n_samples = file_size // 4
            frequency = 1000.0

    if not channels:
        channels = [f"Ch{i+1}" for i in range(n_channels)]

    # Determine dtype from DataFormat
    dtype_map = {1: np.dtype(np.int16), 2: np.dtype(np.int32), 3: np.dtype(np.float32),
                 4: np.dtype(np.float64), 5: np.dtype(np.int16), 6: np.dtype(np.float32)}
    dtype = dtype_map.get(data_format, np.dtype(np.float32))
    if byte_order == "big":
        dtype = dtype.newbyteorder(">")

    # Validate file size
    expected_size = n_channels * n_samples * dtype.itemsize
    actual_size = path.stat().st_size
    if actual_size != expected_size and n_samples > 0:
        # Try recalculating n_samples from actual file size
        n_samples_from_file = actual_size // (n_channels * dtype.itemsize)
        if n_samples_from_file * n_channels * dtype.itemsize == actual_size:
            logger.info("Adjusted n_samples from %d to %d based on file size", n_samples, n_samples_from_file)
            n_samples = n_samples_from_file

    duration = n_samples / max(frequency, 1.0)

    if inspect_only:
        # Load just the first second
        sample_n = min(int(frequency), n_samples)
        raw_bytes = sample_n * n_channels * dtype.itemsize
        with open(filepath, "rb") as f:
            buf = f.read(raw_bytes)
        flat = np.frombuffer(buf, dtype=dtype)

        if samp_order == "SAMP":
            data = flat.reshape(sample_n, n_channels).T.astype(np.float32)
        else:
            data = flat.reshape(n_channels, sample_n).astype(np.float32)

        # Convert to SI (V) when the .dpo declared µV so downstream steps
        # (drop_bads:auto, scale, ICA, QC visualisation) see a unit-consistent
        # signal regardless of acquisition vendor.
        _unit_raw = str(data_unit).strip().lower()
        if _unit_raw in ("uv", "µv", "microvolts"):
            data = data * 1e-6
            data_unit_norm = "V"
        else:
            data_unit_norm = data_unit if data_unit else "unknown"

        meta = {
            "format": "curry_raw",
            "n_channels": n_channels,
            "n_samples_total": n_samples,
            "data_nbytes_estimate": n_channels * n_samples * 4,
            "inspect_only": True,
            "data_unit": data_unit_norm,
            "data_unit_source": "dpo_metadata" if dpo_path.exists() else "default_assumption",
            "frequency_source": "dpo_metadata" if dpo_path.exists() else "inferred",
        }
    else:
        with open(filepath, "rb") as f:
            flat = np.fromfile(f, dtype=dtype)

        if samp_order == "SAMP":
            data = flat.reshape(n_samples, n_channels).T.astype(np.float32)
        else:
            data = flat.reshape(n_channels, n_samples).astype(np.float32)

        _unit_raw = str(data_unit).strip().lower()
        if _unit_raw in ("uv", "µv", "microvolts"):
            data = data * 1e-6
            data_unit_norm = "V"
        else:
            data_unit_norm = data_unit if data_unit else "unknown"

        meta = {
            "format": "curry_raw",
            "n_samples": n_samples,
            "data_unit": data_unit_norm,
            "data_unit_source": "dpo_metadata" if dpo_path.exists() else "default_assumption",
            "byte_order": byte_order,
            "samp_order": samp_order,
            "data_format_code": data_format,
        }

    # Parse events from .ceo if available
    ceo_path = Path(str(path) + ".ceo")
    if not ceo_path.exists():
        ceo_candidates = list(path.parent.glob(path.stem + "*.ceo"))
        if ceo_candidates:
            ceo_path = ceo_candidates[0]

    if ceo_path.exists():
        events = _parse_curry_events(ceo_path, frequency)
        if events:
            meta["events_raw"] = events
            meta["n_events"] = len(events)

    # Heuristic ch_types from channel names — without this the pipeline's
    # _get_mne_info silently re-types VEOG / HEOG / Trigger as EEG (it
    # falls back to "all eeg" when meta.ch_types is empty), and CAR /
    # set_eeg_reference then operates on non-data channels. The MNE-based
    # path takes ch_types from raw.get_channel_types(); this fallback has
    # to do it manually.
    #
    # The default for unknown channel names depends on the caller-declared
    # modality so generic names like Ch1 / A01 (no MNE convention to lean
    # on) don't get blindly typed as EEG when the user knows it's sEEG /
    # ECoG / MEG. EOG / STIM / ECG / EMG names are always recognised by
    # name regardless of the declared modality — those conventions are
    # universal across recording rigs.
    eog_set = {"VEOG", "HEOG", "EOG", "VEO", "HEO", "LO1", "LO2", "IO1", "IO2"}
    stim_set = {"Trigger", "TRIGGER", "STI", "STIM", "Status", "Event", "Mark"}
    ecg_set = {"ECG", "EKG"}
    emg_set = {"EMG", "Chin", "Leg"}
    _MOD_TO_DEFAULT_CH_TYPE = {
        "eeg": "eeg",
        "ecog": "ecog",
        "seeg": "seeg",
        "meg": "mag",       # MNE has separate "mag" / "grad" — default to mag, MEG-aware code overrides
        "auto": "eeg",      # Curry is overwhelmingly EEG
        "": "eeg",
    }
    default_ch_type = _MOD_TO_DEFAULT_CH_TYPE.get((modality or "").lower(), "misc")
    ch_types: List[str] = []
    for ch in channels[:n_channels]:
        if ch in eog_set:
            ch_types.append("eog")
        elif ch in stim_set:
            ch_types.append("stim")
        elif ch in ecg_set:
            ch_types.append("ecg")
        elif ch in emg_set:
            ch_types.append("emg")
        else:
            ch_types.append(default_ch_type)
    meta["ch_types"] = ch_types

    return {
        "data": data,
        "frequency": frequency,
        "channels": channels[:n_channels],
        "duration": duration,
        "meta": meta,
    }


def _parse_curry_events(ceo_path: Path, frequency: float) -> List[Dict[str, Any]]:
    """Parse event markers from a Curry .ceo file."""
    import re as _re

    with open(ceo_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    events: List[Dict[str, Any]] = []

    # Curry events are in the LOCATION_LIST section — rows of timestamp data
    # Format varies but typically: columns include sample index, type code, etc.
    in_list = False
    n_cols = 0
    for line in content.split("\n"):
        if "LOCATION_LIST" in line and "START_LIST" in line:
            in_list = True
            continue
        if in_list and "END_LIST" in line:
            break
        if "ListNrColumns" in line:
            m = _re.search(r"(\d+)", line)
            if m:
                n_cols = int(m.group(1))
        if in_list:
            parts = line.strip().split()
            if parts and len(parts) >= 2:
                try:
                    # First column is typically sample index, second might be annotation
                    sample_idx = int(float(parts[0]))
                    event_type = parts[1] if len(parts) > 1 else "event"
                    onset_sec = sample_idx / max(frequency, 1.0)
                    events.append({
                        "onset": onset_sec,
                        "sample": sample_idx,
                        "type": str(event_type),
                    })
                except (ValueError, IndexError):
                    continue

    return events


def _load_hdf5_spikes(filepath: str) -> Dict[str, Any]:
    """Load spike data from NWB/HDF5 structure.

    NWB convention: units/spike_times + units/spike_times_index
    Also supports: root-level spike_times, per-unit groups, and generic
    HDF5 layouts where spike time arrays are stored at various paths.

    Returns raw spike times per unit (NOT binned — binning is a preprocess step).
    """
    import h5py

    spike_trains: List[np.ndarray] = []
    unit_names: List[str] = []

    with h5py.File(filepath, "r") as f:
        # Strategy 1: NWB-style units group with spike_times + spike_times_index
        units_group = None
        for prefix in [
            "units",
            "processing/ecephys/units",
            "processing/ecephys/sorted_units",
            "acquisition/units",
        ]:
            if prefix in f:
                units_group = f[prefix]
                break

        if units_group is not None and "spike_times" in units_group:
            all_spikes = units_group["spike_times"][:]
            index = units_group["spike_times_index"][:]

            prev = 0
            for i, end in enumerate(index):
                spike_trains.append(all_spikes[prev:int(end)])
                prev = int(end)

            if "id" in units_group:
                ids = units_group["id"][:]
                unit_names = [f"unit_{uid}" for uid in ids]
            else:
                unit_names = [f"unit_{i}" for i in range(len(index))]

            if "electrodes" in units_group and "general/extracellular_ephys/electrodes" in f:
                try:
                    elec_idx = units_group["electrodes"][:]
                    elec_ids = f["general/extracellular_ephys/electrodes/id"][:]
                    ids = units_group["id"][:]
                    unit_names = [
                        f"unit_{ids[i]}_{elec_ids[elec_idx[i]]}"
                        for i in range(len(ids))
                    ]
                except Exception:
                    pass

        # Strategy 2: Root-level spike_times dataset (simple flat HDF5)
        elif "spike_times" in f:
            ds = f["spike_times"]
            if hasattr(ds, "shape"):
                arr = ds[:]
                if arr.ndim == 1:
                    spike_trains = [arr]
                    unit_names = ["unit_0"]
                elif arr.ndim == 2:
                    for i in range(arr.shape[0]):
                        spike_trains.append(arr[i][arr[i] > 0])
                        unit_names.append(f"unit_{i}")

        # Strategy 3: Per-unit groups or datasets at root or under a group
        else:
            spike_trains, unit_names = _find_spike_datasets_recursive(f)

        if not spike_trains:
            logger.warning("No spike_times found in %s — returning empty spike data.", filepath)
            spike_trains = [np.array([], dtype=np.float64)]
            unit_names = ["unit_0"]

        total_duration = float(max(
            (np.max(t) for t in spike_trains if len(t) > 0), default=0.0
        ))

    meta = {
        "format": "nwb_spikes",
        "n_units": len(spike_trains),
        "total_spikes": sum(len(t) for t in spike_trains),
        "source_file": filepath,
    }

    return {
        "data": spike_trains,
        "spike_trains": spike_trains,
        "frequency": 0.0,
        "channels": unit_names,
        "duration": total_duration,
        "meta": meta,
    }


def _find_spike_datasets_recursive(group) -> tuple:
    """Walk HDF5 groups to find spike time arrays.

    Looks for datasets whose name contains 'spike_times', 'spikes', or
    'times' within groups that look like unit containers.
    """
    import h5py

    spike_trains: List[np.ndarray] = []
    unit_names: List[str] = []

    spike_keywords = ("spike_times", "spikes", "times")

    for key in group:
        item = group[key]
        if isinstance(item, h5py.Dataset):
            if any(kw in key.lower() for kw in spike_keywords):
                arr = item[:]
                if arr.ndim == 1 and arr.dtype.kind == "f":
                    spike_trains.append(arr)
                    unit_names.append(f"unit_{len(spike_trains)-1}")
        elif isinstance(item, h5py.Group):
            for subkey in item:
                subitem = item[subkey]
                if isinstance(subitem, h5py.Dataset) and any(
                    kw in subkey.lower() for kw in spike_keywords
                ):
                    arr = subitem[:]
                    if arr.ndim == 1 and arr.dtype.kind == "f":
                        spike_trains.append(arr)
                        unit_names.append(key)
                        break

    return spike_trains, unit_names


def _load_nwb(filepath: str, inspect_only: bool = False) -> Dict[str, Any]:
    """Load NWB/HDF5 file — detects and extracts full structure.

    Distinguishes between:
    1. Spike-only NWB: only /units with spike_times
    2. Full NWB: contains acquisition (continuous signals), intervals (trials),
       processing (behavior), and/or units (spikes)

    Extracts trials table as events and behavioral timeseries as auxiliary data.
    """
    import h5py

    with h5py.File(filepath, "r") as f:
        has_acquisition = "acquisition" in f
        has_units = "units" in f or any(
            p in f for p in ["processing/ecephys/units", "processing/ecephys/sorted_units"]
        )
        has_intervals = "intervals" in f
        has_processing = "processing" in f

        # Detect continuous signal data in acquisition
        continuous_acquisition = None
        if has_acquisition:
            for key in f["acquisition"]:
                acq = f["acquisition"][key]
                if isinstance(acq, h5py.Group) and "data" in acq:
                    ds = acq["data"]
                    if ds.ndim == 2 and ds.shape[0] > 100:
                        continuous_acquisition = key
                        break

        # If we have continuous data → full NWB path
        if continuous_acquisition:
            return _load_nwb_continuous(f, filepath, continuous_acquisition,
                                       has_intervals, has_processing, inspect_only)

        # Otherwise → spike-only path (original behavior)
        # But still extract trials and behavioral data if present

    # Fall back to spike loading
    result = _load_hdf5_spikes(filepath)

    # Enhance with trials/behavior if available
    with h5py.File(filepath, "r") as f:
        if has_intervals:
            trials = _extract_nwb_trials(f)
            if trials:
                result["meta"]["trials"] = trials
                result["meta"]["annotations"] = {
                    "onset": [t["onset"] for t in trials],
                    "duration": [t["duration"] for t in trials],
                    "description": [t.get("type", "trial") for t in trials],
                }

        if has_processing:
            behavior = _extract_nwb_behavior(f)
            if behavior:
                result["meta"]["behavioral_timeseries"] = behavior

        # Build content index
        result["meta"]["nwb_content"] = _build_nwb_content_index(f)

    return result


def _load_nwb_continuous(
    f,  # h5py.File
    filepath: str,
    acquisition_key: str,
    has_intervals: bool,
    has_processing: bool,
    inspect_only: bool,
) -> Dict[str, Any]:
    """Load continuous signal data from a full NWB file."""
    acq = f["acquisition"][acquisition_key]
    ds = acq["data"]

    n_samples, n_channels = ds.shape
    # NWB stores as (n_samples, n_channels)

    # Get sampling rate
    frequency = 1.0
    if "starting_time" in acq and hasattr(acq["starting_time"], "attrs"):
        rate = acq["starting_time"].attrs.get("rate")
        if rate:
            frequency = float(rate)
    if frequency <= 1.0 and "timestamps" in acq:
        ts = acq["timestamps"]
        if ts.shape[0] > 1:
            dt = float(ts[1] - ts[0])
            if dt > 0:
                frequency = 1.0 / dt

    duration = float(n_samples / frequency) if frequency > 0 else 0.0

    # Channel names
    channels = [f"Ch{i}" for i in range(n_channels)]
    if "electrodes" in acq:
        try:
            elec_table = acq["electrodes"]
            if "table" in elec_table.attrs:
                elec_path = elec_table.attrs["table"]
                if isinstance(elec_path, bytes):
                    elec_path = elec_path.decode()
                if elec_path in f:
                    elec_group = f[elec_path]
                    if "label" in elec_group:
                        channels = [
                            s.decode() if isinstance(s, bytes) else str(s)
                            for s in elec_group["label"][:]
                        ][:n_channels]
        except Exception:
            pass

    # Load data
    if inspect_only:
        sample_len = min(int(frequency), n_samples)
        data = ds[:sample_len, :].T.astype(np.float32)
    else:
        data = ds[:].T.astype(np.float32)  # Transpose to (n_channels, n_samples)

    meta: Dict[str, Any] = {
        "format": "nwb_continuous",
        "source_file": filepath,
        "acquisition_key": acquisition_key,
        "n_samples": n_samples,
        "n_channels": n_channels,
    }

    if inspect_only:
        meta["inspect_only"] = True
        meta["n_samples_total"] = n_samples
        meta["data_nbytes_estimate"] = n_channels * n_samples * 4

    # Extract trials
    if has_intervals:
        trials = _extract_nwb_trials(f)
        if trials:
            meta["trials"] = trials
            meta["annotations"] = {
                "onset": [t["onset"] for t in trials],
                "duration": [t["duration"] for t in trials],
                "description": [t.get("type", "trial") for t in trials],
            }
            meta["events"] = [
                [int(t["onset"] * frequency), 0, hash(t.get("type", "trial")) % 1000]
                for t in trials
            ]
            # Build event_id mapping
            types = sorted(set(t.get("type", "trial") for t in trials))
            meta["event_id"] = {t: i + 1 for i, t in enumerate(types)}

    # Extract behavioral timeseries
    if has_processing:
        behavior = _extract_nwb_behavior(f)
        if behavior:
            meta["behavioral_timeseries"] = behavior

    # Content index
    meta["nwb_content"] = _build_nwb_content_index(f)

    return {
        "data": data,
        "frequency": frequency,
        "channels": channels[:n_channels],
        "duration": duration,
        "meta": meta,
    }


def _extract_nwb_trials(f) -> List[Dict[str, Any]]:
    """Extract trials table from NWB /intervals/trials."""
    import h5py

    trials = []
    trials_group = None

    for path in ["intervals/trials", "intervals/epochs"]:
        if path in f:
            trials_group = f[path]
            break

    if trials_group is None:
        return []

    try:
        start_times = trials_group["start_time"][:] if "start_time" in trials_group else None
        stop_times = trials_group["stop_time"][:] if "stop_time" in trials_group else None

        if start_times is None:
            return []

        n_trials = len(start_times)

        # Get trial type/condition if available
        type_data = None
        for type_key in ("trial_type", "type", "condition", "stimulus_type", "stim_type"):
            if type_key in trials_group:
                try:
                    raw = trials_group[type_key][:]
                    type_data = [
                        s.decode() if isinstance(s, bytes) else str(s)
                        for s in raw
                    ]
                except Exception:
                    pass
                break

        for i in range(n_trials):
            onset = float(start_times[i])
            duration = float(stop_times[i] - start_times[i]) if stop_times is not None else 0.0
            trial_type = type_data[i] if type_data and i < len(type_data) else "trial"

            trial = {
                "onset": onset,
                "duration": duration,
                "type": trial_type,
                "metadata": {"trial_index": i},
            }

            # Extract additional columns
            for key in trials_group:
                if key in ("start_time", "stop_time", "id") or key == type_key if type_data else False:
                    continue
                if isinstance(trials_group[key], h5py.Dataset):
                    try:
                        val = trials_group[key][i]
                        if isinstance(val, bytes):
                            val = val.decode()
                        trial["metadata"][key] = val
                    except Exception:
                        pass

            trials.append(trial)

    except Exception as exc:
        logger.debug("Failed to extract NWB trials: %s", exc)

    return trials


def _extract_nwb_behavior(f) -> List[Dict[str, Any]]:
    """Extract behavioral timeseries from NWB /processing/behavior/."""
    import h5py

    behavior_list = []

    behavior_group = None
    for path in ["processing/behavior", "processing/Behavior"]:
        if path in f:
            behavior_group = f[path]
            break

    if behavior_group is None:
        return []

    for key in behavior_group:
        item = behavior_group[key]
        if not isinstance(item, h5py.Group):
            continue

        # Look for data datasets inside behavioral containers
        for subkey in item:
            subitem = item[subkey]
            if isinstance(subitem, h5py.Group) and "data" in subitem:
                ds = subitem["data"]
                info: Dict[str, Any] = {
                    "name": f"{key}/{subkey}",
                    "shape": list(ds.shape),
                    "dtype": str(ds.dtype),
                }
                # Get rate if available
                if "timestamps" in subitem:
                    ts = subitem["timestamps"]
                    if ts.shape[0] > 1:
                        dt = float(ts[1] - ts[0])
                        info["frequency"] = round(1.0 / dt, 1) if dt > 0 else 0
                        info["duration_s"] = round(float(ts[-1] - ts[0]), 2)
                elif "starting_time" in subitem:
                    rate = subitem["starting_time"].attrs.get("rate")
                    if rate:
                        info["frequency"] = float(rate)

                behavior_list.append(info)
            elif isinstance(subitem, h5py.Dataset) and subitem.ndim >= 1:
                info = {
                    "name": f"{key}/{subkey}",
                    "shape": list(subitem.shape),
                    "dtype": str(subitem.dtype),
                }
                behavior_list.append(info)

    return behavior_list


def _build_nwb_content_index(f) -> Dict[str, Any]:
    """Build a content summary of what's in the NWB file."""
    import h5py

    index: Dict[str, Any] = {}

    if "acquisition" in f:
        acqs = []
        for key in f["acquisition"]:
            item = f["acquisition"][key]
            if isinstance(item, h5py.Group) and "data" in item:
                ds = item["data"]
                acqs.append({
                    "name": key,
                    "shape": list(ds.shape),
                    "dtype": str(ds.dtype),
                })
        if acqs:
            index["acquisition"] = acqs

    if "intervals" in f:
        intervals = {}
        for key in f["intervals"]:
            item = f["intervals"][key]
            if isinstance(item, h5py.Group) and "start_time" in item:
                intervals[key] = {"n_entries": len(item["start_time"])}
        if intervals:
            index["intervals"] = intervals

    if "units" in f:
        units_group = f["units"]
        if "spike_times_index" in units_group:
            index["units"] = {"n_units": len(units_group["spike_times_index"])}
        elif "id" in units_group:
            index["units"] = {"n_units": len(units_group["id"])}

    if "processing" in f:
        proc_modules = list(f["processing"].keys())
        index["processing_modules"] = proc_modules

    return index


def _load_mat(filepath: str, inspect_only: bool = False) -> Dict[str, Any]:
    """Load from MATLAB .mat file (FieldTrip or EEGLAB structure)."""
    import scipy.io

    try:
        mat = scipy.io.loadmat(filepath, squeeze_me=True)
    except NotImplementedError:
        return _load_mat_v73(filepath)

    # Try FieldTrip structure: data.trial, data.fsample, data.label
    if "data" in mat and hasattr(mat["data"], "dtype") and mat["data"].dtype.names:
        ft = mat["data"]
        fields = ft.dtype.names
        trial = ft["trial"].item() if "trial" in fields else None
        fsample = float(ft["fsample"]) if "fsample" in fields else None
        label = ft["label"].item() if "label" in fields else None

        if trial is not None and fsample is not None:
            data_arr = np.array(trial, dtype=np.float32)
            if data_arr.ndim == 1 and hasattr(data_arr[0], 'shape'):
                data_arr = data_arr[0]
            channels = [str(x) for x in label] if label is not None else [f"Ch{i}" for i in range(data_arr.shape[0])]

            meta = {"format": "fieldtrip_mat"}

            # Extract FieldTrip trial definition (cfg.trl)
            events = _extract_fieldtrip_events(ft, mat, fsample)
            if events:
                meta["embedded_events"] = events
                meta["annotations"] = {
                    "onset": [e["onset"] for e in events],
                    "duration": [e.get("duration", 0.0) for e in events],
                    "description": [e.get("type", "trial") for e in events],
                }

            return {
                "data": data_arr.astype(np.float32),
                "frequency": fsample,
                "channels": channels,
                "duration": float(data_arr.shape[-1] / fsample),
                "meta": meta,
            }

    # Try EEGLAB structure: EEG.data, EEG.srate, EEG.chanlocs
    if "EEG" in mat and hasattr(mat["EEG"], "dtype") and mat["EEG"].dtype.names:
        eeg = mat["EEG"]
        fields = eeg.dtype.names
        data_arr = np.array(eeg["data"].item(), dtype=np.float32) if "data" in fields else None
        srate = float(eeg["srate"]) if "srate" in fields else None

        if data_arr is not None and srate is not None:
            channels = [f"Ch{i}" for i in range(data_arr.shape[0])]
            if "chanlocs" in fields:
                try:
                    locs = eeg["chanlocs"]
                    if hasattr(locs, "dtype") and "labels" in locs.dtype.names:
                        channels = [str(x) for x in locs["labels"]]
                except Exception:
                    pass

            meta = {"format": "eeglab_mat"}

            # Extract EEGLAB EEG.event structure
            events = _extract_eeglab_events(eeg, srate)
            if events:
                meta["embedded_events"] = events
                meta["annotations"] = {
                    "onset": [e["onset"] for e in events],
                    "duration": [e.get("duration", 0.0) for e in events],
                    "description": [e.get("type", "event") for e in events],
                }

            return {
                "data": data_arr,
                "frequency": srate,
                "channels": channels,
                "duration": float(data_arr.shape[-1] / srate),
                "meta": meta,
            }

    # Generic: look for largest 2D array + srate/fs/frequency key
    data_arr = None
    frequency = None
    for key, val in mat.items():
        if key.startswith("_"):
            continue
        if isinstance(val, np.ndarray) and val.ndim == 2:
            if data_arr is None or val.size > data_arr.size:
                data_arr = val
        if key.lower() in ("srate", "fs", "fsample", "frequency", "sfreq", "sampling_rate"):
            frequency = float(val)

    if data_arr is None:
        # Try any array (1D, 3D) as a last resort
        for key, val in mat.items():
            if key.startswith("_"):
                continue
            if isinstance(val, np.ndarray) and val.size > 0:
                if val.ndim == 1:
                    data_arr = val.reshape(1, -1)
                elif val.ndim == 3:
                    data_arr = val.reshape(-1, val.shape[-1])
                break
        if data_arr is None:
            logger.warning("No suitable data array found in %s — returning empty.", filepath)
            data_arr = np.zeros((1, 0), dtype=np.float32)
    if frequency is None:
        logger.warning(
            "No sampling rate found in %s (expected key: srate/fs/fsample/frequency). "
            "Defaulting to 1.0 Hz — set correct frequency before processing.",
            filepath,
        )
        frequency = 1.0

    if data_arr.shape[0] > data_arr.shape[1]:
        data_arr = data_arr.T

    channels = [f"Ch{i}" for i in range(data_arr.shape[0])]
    return {
        "data": data_arr.astype(np.float32),
        "frequency": frequency,
        "channels": channels,
        "duration": float(data_arr.shape[-1] / frequency),
        "meta": {"format": "generic_mat"},
    }


def _extract_eeglab_events(eeg_struct, srate: float) -> List[Dict[str, Any]]:
    """Extract events from EEGLAB EEG.event structure.

    EEG.event is a struct array with fields: type, latency, duration, urevent, etc.
    latency is in samples (1-indexed in MATLAB).
    """
    events: List[Dict[str, Any]] = []
    fields = eeg_struct.dtype.names if hasattr(eeg_struct, "dtype") else ()
    if "event" not in fields:
        return events

    try:
        event_data = eeg_struct["event"].item()
    except (AttributeError, ValueError):
        return events

    # event_data can be a structured array or a scalar struct
    if event_data is None or (hasattr(event_data, "size") and event_data.size == 0):
        return events

    # Handle structured numpy array (most common)
    if hasattr(event_data, "dtype") and event_data.dtype.names:
        event_fields = event_data.dtype.names
        n_events = event_data.shape[0] if event_data.ndim > 0 else 1

        for i in range(n_events):
            ev = event_data[i] if event_data.ndim > 0 else event_data

            # Extract type
            ev_type = "event"
            if "type" in event_fields:
                try:
                    ev_type = str(ev["type"])
                except (TypeError, ValueError):
                    ev_type = "event"

            # Extract latency (samples, 1-indexed) → onset (seconds)
            onset = 0.0
            if "latency" in event_fields:
                try:
                    latency_samples = float(ev["latency"])
                    onset = (latency_samples - 1) / srate  # convert to 0-indexed seconds
                except (TypeError, ValueError):
                    pass

            # Extract duration (samples → seconds)
            duration = 0.0
            if "duration" in event_fields:
                try:
                    dur_val = ev["duration"]
                    if dur_val is not None and not (hasattr(dur_val, "__len__") and len(dur_val) == 0):
                        duration = float(dur_val) / srate
                except (TypeError, ValueError):
                    pass

            events.append({
                "onset": onset,
                "duration": duration,
                "type": ev_type,
                "latency_samples": int(round(onset * srate)),
            })

    # Handle non-structured (list of dicts or array of objects)
    elif hasattr(event_data, "__iter__") and not isinstance(event_data, str):
        for item in event_data:
            try:
                if hasattr(item, "dtype") and item.dtype.names:
                    ev_type = str(item["type"]) if "type" in item.dtype.names else "event"
                    latency = float(item["latency"]) if "latency" in item.dtype.names else 0.0
                    duration = float(item["duration"]) if "duration" in item.dtype.names else 0.0
                elif isinstance(item, dict):
                    ev_type = str(item.get("type", "event"))
                    latency = float(item.get("latency", 0))
                    duration = float(item.get("duration", 0))
                else:
                    continue

                onset = (latency - 1) / srate if srate > 0 else 0.0
                events.append({
                    "onset": onset,
                    "duration": duration / srate if srate > 0 else 0.0,
                    "type": ev_type,
                    "latency_samples": int(round(onset * srate)),
                })
            except (TypeError, ValueError, AttributeError):
                continue

    return events


def _extract_fieldtrip_events(
    ft_struct, mat_dict: dict, fsample: float
) -> List[Dict[str, Any]]:
    """Extract events from FieldTrip data.cfg.trl or top-level trl matrix.

    FieldTrip trl matrix format: [begin_sample, end_sample, offset, ...]
    Additional columns are condition codes. Samples are 1-indexed.
    """
    events: List[Dict[str, Any]] = []
    trl = None

    # Check for trl at top level
    if "trl" in mat_dict:
        trl = np.asarray(mat_dict["trl"])
    else:
        # Check inside data.cfg.trl
        fields = ft_struct.dtype.names if hasattr(ft_struct, "dtype") else ()
        if "cfg" in fields:
            try:
                cfg = ft_struct["cfg"].item()
                if hasattr(cfg, "dtype") and cfg.dtype.names and "trl" in cfg.dtype.names:
                    trl = np.asarray(cfg["trl"].item())
            except (AttributeError, ValueError, TypeError):
                pass

    if trl is None or trl.size == 0:
        return events

    if trl.ndim == 1:
        trl = trl.reshape(1, -1)

    n_trials = trl.shape[0]
    n_cols = trl.shape[1]

    for i in range(n_trials):
        begin_sample = int(trl[i, 0]) - 1  # convert to 0-indexed
        end_sample = int(trl[i, 1]) - 1
        offset = int(trl[i, 2]) if n_cols > 2 else 0

        onset = begin_sample / fsample if fsample > 0 else 0.0
        duration = (end_sample - begin_sample) / fsample if fsample > 0 else 0.0

        # Condition code from 4th column if present
        condition = str(int(trl[i, 3])) if n_cols > 3 else "trial"

        events.append({
            "onset": onset,
            "duration": duration,
            "type": condition,
            "begin_sample": begin_sample,
            "end_sample": end_sample,
            "offset_samples": offset,
        })

    return events


def _load_mat_v73(filepath: str) -> Dict[str, Any]:
    """Load MATLAB v7.3 .mat file (HDF5 format)."""
    import h5py

    with h5py.File(filepath, "r") as f:
        data_arr = None
        frequency = None

        for key in f.keys():
            if key.startswith("#"):
                continue
            val = f[key]
            if isinstance(val, h5py.Dataset) and val.ndim == 2:
                if data_arr is None or val.size > (data_arr.size if data_arr is not None else 0):
                    data_arr = val[:]
            elif isinstance(val, h5py.Dataset) and val.ndim == 0:
                if key.lower() in ("srate", "fs", "fsample", "frequency", "sfreq"):
                    frequency = float(val[()])

        if data_arr is None:
            logger.warning("No suitable data array in v7.3 mat: %s — returning empty.", filepath)
            data_arr = np.zeros((1, 0), dtype=np.float32)
        if frequency is None:
            logger.warning(
                "No sampling rate in v7.3 mat: %s. Defaulting to 1.0 Hz.",
                filepath,
            )
            frequency = 1.0

    if data_arr.shape[0] > data_arr.shape[1]:
        data_arr = data_arr.T

    return {
        "data": data_arr.astype(np.float32),
        "frequency": frequency,
        "channels": [f"Ch{i}" for i in range(data_arr.shape[0])],
        "duration": float(data_arr.shape[-1] / frequency),
        "meta": {"format": "mat_v73"},
    }


def _load_csv(filepath: str, inspect_only: bool = False) -> Dict[str, Any]:
    """Load from CSV/TSV file.

    Expected format:
    - First row: channel names (if non-numeric) or data
    - Remaining rows: data samples (one row per timepoint, columns = channels)
    - Sampling rate: from comment line '# srate: 256' or defaults to 256 Hz

    Supports both channels-as-columns (n_samples x n_channels) format.
    Handles NaN values in data.
    """
    path = Path(filepath)
    delimiter = "\t" if path.suffix == ".tsv" else ","

    frequency = 256.0  # default
    header_channels = None
    data_start_line = 0

    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                data_start_line = i + 1
                continue
            if line.startswith("#"):
                # Parse metadata comments
                lower = line.lower()
                for key in ("srate:", "srate=", "fs:", "fs=", "frequency:", "frequency=", "sampling_rate:", "sampling_rate="):
                    if key in lower:
                        try:
                            frequency = float(lower.split(key)[1].strip().split()[0])
                        except (ValueError, IndexError):
                            pass
                data_start_line = i + 1
                continue
            # Check if first data line is header (non-numeric)
            parts = line.split(delimiter)
            try:
                float(parts[0])
                break  # numeric, no header
            except ValueError:
                header_channels = [p.strip() for p in parts]
                data_start_line = i + 1
                break

    if inspect_only:
        max_rows = int(frequency)  # ~1 second of data
    else:
        max_rows = None

    data_arr = np.genfromtxt(
        filepath, delimiter=delimiter, skip_header=data_start_line,
        dtype=np.float32, comments="#", filling_values=np.nan,
        max_rows=max_rows,
    )

    if data_arr.ndim == 1:
        data_arr = data_arr[np.newaxis, :]
    elif data_arr.ndim == 2:
        # CSV typically stores as (n_samples x n_channels), we need (n_channels x n_samples)
        if data_arr.shape[0] > data_arr.shape[1]:
            data_arr = data_arr.T

    n_channels = data_arr.shape[0]
    if header_channels and len(header_channels) == n_channels:
        channels = header_channels
    else:
        channels = [f"Ch{i}" for i in range(n_channels)]

    if inspect_only:
        # Count total lines for duration estimate
        with open(filepath, "r", encoding="utf-8") as f:
            total_lines = sum(1 for line in f
                              if line.strip() and not line.startswith("#"))
        total_data_lines = total_lines - (1 if header_channels else 0)
        total_duration = float(total_data_lines / frequency)
        meta = {
            "format": "csv", "source_file": filepath,
            "inspect_only": True,
            "n_channels": n_channels,
            "n_samples_total": total_data_lines,
            "data_nbytes_estimate": n_channels * total_data_lines * 4,
        }
    else:
        total_duration = float(data_arr.shape[-1] / frequency)
        meta = {"format": "csv", "source_file": filepath}

    return {
        "data": data_arr,
        "frequency": frequency,
        "channels": channels,
        "duration": total_duration,
        "meta": meta,
    }


def _load_npz(filepath: str, inspect_only: bool = False) -> Dict[str, Any]:
    """Load from NumPy .npz or .npy file.

    For .npz: looks for the largest 2D array and a frequency key
    (frequency, srate, fs, sfreq, sampling_rate).
    For .npy: loads directly as a 2D array.
    """
    path = Path(filepath)

    if path.suffix.lower() == ".npy":
        data_arr = np.load(filepath, mmap_mode="r" if inspect_only else None)
        if data_arr.ndim == 1:
            data_arr = data_arr[np.newaxis, :]
        if data_arr.ndim == 3:
            logger.warning(
                "3D array in .npy file (shape %s) — reshaping to 2D by collapsing first two dims.",
                data_arr.shape,
            )
            data_arr = data_arr.reshape(-1, data_arr.shape[-1])
        elif data_arr.ndim > 3:
            logger.warning(
                "%dD array in .npy file (shape %s) — reshaping to 2D (channels x samples).",
                data_arr.ndim, data_arr.shape,
            )
            data_arr = data_arr.reshape(-1, data_arr.shape[-1])
        if data_arr.shape[0] > data_arr.shape[1]:
            data_arr = data_arr.T
        frequency = 256.0
        n_channels = data_arr.shape[0]
        n_samples = data_arr.shape[1]
        channels = [f"Ch{i}" for i in range(n_channels)]

        if inspect_only:
            sample_len = min(int(frequency), n_samples)
            sample = np.array(data_arr[:, :sample_len], dtype=np.float32)
            return {
                "data": sample,
                "frequency": frequency,
                "channels": channels,
                "duration": float(n_samples / frequency),
                "meta": {
                    "format": "npy", "source_file": filepath,
                    "inspect_only": True,
                    "n_channels": n_channels,
                    "n_samples_total": n_samples,
                    "data_nbytes_estimate": n_channels * n_samples * 4,
                },
            }

        return {
            "data": np.asarray(data_arr, dtype=np.float32),
            "frequency": frequency,
            "channels": channels,
            "duration": float(n_samples / frequency),
            "meta": {"format": "npy", "source_file": filepath},
        }

    # .npz file
    npz = np.load(filepath, allow_pickle=False, mmap_mode="r" if inspect_only else None)

    # Find frequency from stored keys
    frequency = None
    freq_keys = ("frequency", "srate", "fs", "sfreq", "fsample", "sampling_rate")
    for key in freq_keys:
        if key in npz:
            val = npz[key]
            frequency = float(val.item() if val.ndim == 0 else val.flat[0])
            break

    # Find the largest 2D array as data
    data_arr = None
    data_key = None
    for key in npz.files:
        if key.lower() in freq_keys:
            continue
        arr = npz[key]
        if arr.ndim == 2:
            if data_arr is None or arr.size > data_arr.size:
                data_arr = arr
                data_key = key
        elif arr.ndim == 1 and data_arr is None:
            data_arr = arr[np.newaxis, :]
            data_key = key

    if data_arr is None:
        # Try any array as last resort
        for key in npz.files:
            arr = npz[key]
            if arr.size > 0:
                if arr.ndim == 3:
                    data_arr = arr.reshape(-1, arr.shape[-1])
                elif arr.ndim >= 1:
                    data_arr = arr.reshape(1, -1) if arr.ndim == 1 else arr.reshape(-1, arr.shape[-1])
                data_key = key
                logger.warning(
                    "No 2D array in %s — using '%s' (shape %s) reshaped to %s.",
                    filepath, key, arr.shape, data_arr.shape,
                )
                break
        if data_arr is None:
            logger.warning("No suitable data array in %s (keys: %s) — returning empty.", filepath, npz.files)
            data_arr = np.zeros((1, 0), dtype=np.float32)
            data_key = "empty"

    if frequency is None:
        frequency = 256.0
        logger.warning("No frequency key found in %s, defaulting to 256 Hz", filepath)

    if data_arr.ndim == 2 and data_arr.shape[0] > data_arr.shape[1]:
        data_arr = data_arr.T

    n_channels = data_arr.shape[0]
    n_samples = data_arr.shape[1] if data_arr.ndim == 2 else data_arr.shape[0]

    # Look for channel names
    channels = None
    for key in ("channels", "ch_names", "channel_names"):
        if key in npz:
            try:
                channels = [str(ch) for ch in npz[key]]
            except Exception:
                pass
            break
    if channels is None or len(channels) != n_channels:
        channels = [f"Ch{i}" for i in range(n_channels)]

    if inspect_only:
        sample_len = min(int(frequency), n_samples)
        sample = np.array(data_arr[:, :sample_len], dtype=np.float32)
        return {
            "data": sample,
            "frequency": frequency,
            "channels": channels,
            "duration": float(n_samples / frequency),
            "meta": {
                "format": "npz", "source_file": filepath,
                "data_key": data_key,
                "inspect_only": True,
                "n_channels": n_channels,
                "n_samples_total": n_samples,
                "data_nbytes_estimate": n_channels * n_samples * 4,
            },
        }

    return {
        "data": np.asarray(data_arr, dtype=np.float32),
        "frequency": frequency,
        "channels": channels,
        "duration": float(n_samples / frequency),
        "meta": {"format": "npz", "source_file": filepath, "data_key": data_key},
    }


def _load_xdf(filepath: str, inspect_only: bool = False) -> Dict[str, Any]:
    """Load XDF (LSL multi-stream) file.

    Identifies the primary EEG/signal stream (highest channel count continuous stream)
    and extracts marker streams as events.
    """
    from easybci_lib.tools.lazy_deps import ensure
    ensure("neural.pyxdf")
    import pyxdf

    streams, header = pyxdf.load_xdf(filepath)

    # Classify streams by type
    continuous_streams = []
    marker_streams = []

    for stream in streams:
        info = stream["info"]
        stream_type = info["type"][0].lower() if info.get("type") else ""
        nominal_srate = float(info["nominal_srate"][0]) if info.get("nominal_srate") else 0
        n_channels = int(info["channel_count"][0]) if info.get("channel_count") else 0
        stream_name = info["name"][0] if info.get("name") else "unnamed"

        entry = {
            "name": stream_name,
            "type": stream_type,
            "n_channels": n_channels,
            "nominal_srate": nominal_srate,
            "data": stream["time_series"],
            "timestamps": stream["time_stamps"],
        }

        if nominal_srate > 0 and n_channels > 0:
            continuous_streams.append(entry)
        else:
            marker_streams.append(entry)

    if not continuous_streams:
        # Try treating any stream with data as usable
        if marker_streams:
            logger.warning(
                "No continuous data streams in XDF: %s. "
                "Found %d marker/irregular streams — using largest as data.",
                filepath, len(marker_streams),
            )
            primary_candidate = max(marker_streams, key=lambda s: len(s["data"]) if hasattr(s["data"], '__len__') else 0)
            continuous_streams = [primary_candidate]
            if primary_candidate["nominal_srate"] == 0:
                primary_candidate["nominal_srate"] = 1.0
        else:
            logger.warning("No streams found in XDF file: %s — returning empty.", filepath)
            return {
                "data": np.zeros((1, 0), dtype=np.float32),
                "frequency": 1.0,
                "channels": ["Ch0"],
                "duration": 0.0,
                "meta": {"format": "xdf", "source_file": filepath, "error": "no streams"},
            }

    # Primary stream: highest channel count (typically EEG)
    primary = max(continuous_streams, key=lambda s: s["n_channels"])
    data_arr = np.array(primary["data"], dtype=np.float32)
    timestamps = np.array(primary["timestamps"], dtype=np.float64)

    # XDF stores as (n_samples, n_channels), we need (n_channels, n_samples)
    if data_arr.ndim == 2 and data_arr.shape[0] > data_arr.shape[1]:
        data_arr = data_arr.T
    elif data_arr.ndim == 2 and data_arr.shape[1] >= data_arr.shape[0]:
        data_arr = data_arr.T

    # Compute effective sampling rate from timestamps
    if len(timestamps) > 1:
        dt = np.median(np.diff(timestamps))
        frequency = 1.0 / dt if dt > 0 else primary["nominal_srate"]
    else:
        frequency = primary["nominal_srate"]

    n_channels = data_arr.shape[0]
    n_samples = data_arr.shape[1] if data_arr.ndim == 2 else len(data_arr)
    duration = float(timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0.0

    # Channel names from stream info
    channels = [f"Ch{i}" for i in range(n_channels)]
    try:
        ch_info = primary.get("info", {})
        if isinstance(ch_info, dict) and "desc" in ch_info:
            desc = ch_info["desc"]
            if isinstance(desc, list) and len(desc) > 0:
                desc = desc[0]
            if isinstance(desc, dict) and "channels" in desc:
                ch_list = desc["channels"]
                if isinstance(ch_list, list) and len(ch_list) > 0:
                    ch_list = ch_list[0]
                if isinstance(ch_list, dict) and "channel" in ch_list:
                    ch_entries = ch_list["channel"]
                    if isinstance(ch_entries, list):
                        channels = [
                            ch.get("label", [ch.get("name", [f"Ch{i}"])])[0]
                            if isinstance(ch, dict) else f"Ch{i}"
                            for i, ch in enumerate(ch_entries)
                        ][:n_channels]
    except Exception:
        pass

    # Extract events from marker streams
    events_list = []
    t0 = timestamps[0] if len(timestamps) > 0 else 0.0
    for ms in marker_streams:
        ms_timestamps = ms["timestamps"]
        ms_data = ms["data"]
        for i, ts in enumerate(ms_timestamps):
            marker_val = ms_data[i][0] if isinstance(ms_data[i], (list, np.ndarray)) else str(ms_data[i])
            events_list.append({
                "onset": float(ts - t0),
                "duration": 0.0,
                "type": str(marker_val).strip(),
                "source_stream": ms["name"],
            })

    events_list.sort(key=lambda e: e["onset"])

    meta: Dict[str, Any] = {
        "format": "xdf",
        "source_file": filepath,
        "primary_stream": primary["name"],
        "primary_stream_type": primary["type"],
        "n_streams_total": len(streams),
        "n_continuous_streams": len(continuous_streams),
        "n_marker_streams": len(marker_streams),
        "stream_index": [
            {"name": s["name"], "type": s["type"], "n_channels": s["n_channels"],
             "srate": s["nominal_srate"]}
            for s in continuous_streams + marker_streams
        ],
    }

    if events_list:
        meta["annotations"] = {
            "onset": [e["onset"] for e in events_list],
            "duration": [e["duration"] for e in events_list],
            "description": [e["type"] for e in events_list],
        }
        meta["events_from_markers"] = True

    # Auxiliary continuous streams (not primary)
    aux_streams = [s for s in continuous_streams if s["name"] != primary["name"]]
    if aux_streams:
        meta["auxiliary_streams"] = [
            {"name": s["name"], "type": s["type"], "n_channels": s["n_channels"],
             "srate": s["nominal_srate"]}
            for s in aux_streams
        ]

    if inspect_only:
        sample_len = min(int(frequency), n_samples)
        sample = data_arr[:, :sample_len] if data_arr.ndim == 2 else data_arr[:sample_len]
        meta["inspect_only"] = True
        meta["n_channels"] = n_channels
        meta["n_samples_total"] = n_samples
        meta["data_nbytes_estimate"] = n_channels * n_samples * 4
        return {
            "data": sample,
            "frequency": float(frequency),
            "channels": channels,
            "duration": duration,
            "meta": meta,
        }

    return {
        "data": data_arr,
        "frequency": float(frequency),
        "channels": channels,
        "duration": duration,
        "meta": meta,
    }


def _load_pkl(filepath: str, inspect_only: bool = False) -> Dict[str, Any]:
    """Load EasyBCI native pkl format (output of save_pkl).

    The pkl payload has structure: {"data": {modality: ndarray}, "labels": {...}, "meta": {...}}
    Also handles legacy/third-party pickles where data is a raw ndarray or
    stored under alternate keys (eeg, signals).
    """
    import pickle

    path = Path(filepath)
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
    except Exception as e:
        logger.warning("Failed to unpickle '%s': %s", path.name, e)
        return {
            "data": np.zeros((1, 0), dtype=np.float32),
            "frequency": 1.0,
            "channels": ["Ch0"],
            "duration": 0.0,
            "meta": {"format": "pkl", "source_file": filepath, "load_error": str(e)},
        }

    if not isinstance(obj, dict):
        # Raw ndarray or other object
        if hasattr(obj, "shape"):
            data_arr = np.asarray(obj, dtype=np.float32)
            if data_arr.ndim == 1:
                data_arr = data_arr.reshape(1, -1)
            elif data_arr.ndim == 2 and data_arr.shape[0] > data_arr.shape[1]:
                data_arr = data_arr.T
            n_ch = data_arr.shape[0]
            return {
                "data": data_arr,
                "frequency": 256.0,
                "channels": [f"Ch{i}" for i in range(n_ch)],
                "duration": float(data_arr.shape[-1] / 256.0),
                "meta": {"format": "pkl", "source_file": filepath},
            }
        logger.warning("Pkl '%s' contains non-dict, non-array type: %s", path.name, type(obj).__name__)
        return {
            "data": np.zeros((1, 0), dtype=np.float32),
            "frequency": 1.0,
            "channels": ["Ch0"],
            "duration": 0.0,
            "meta": {"format": "pkl", "source_file": filepath, "load_error": "Unsupported pkl content type"},
        }

    # EasyBCI native format: {"data": {modality: ndarray}, "labels": {...}, "meta": {...}}
    raw_data = obj.get("data")
    meta = obj.get("meta", {})
    freq = meta.get("frequency") or obj.get("frequency") or obj.get("srate") or obj.get("fs", 256.0)
    channels = meta.get("channels") or obj.get("channels")

    if isinstance(raw_data, dict):
        # Multi-modality dict — pick first modality's array
        first_key = next(iter(raw_data), None)
        if first_key is not None and hasattr(raw_data[first_key], "shape"):
            data_arr = np.asarray(raw_data[first_key], dtype=np.float32)
        else:
            data_arr = np.zeros((1, 0), dtype=np.float32)
    elif hasattr(raw_data, "shape"):
        data_arr = np.asarray(raw_data, dtype=np.float32)
    elif raw_data is None:
        # Try alternate keys
        raw_data = obj.get("eeg") or obj.get("signals") or obj.get("neural")
        if raw_data is not None and hasattr(raw_data, "shape"):
            data_arr = np.asarray(raw_data, dtype=np.float32)
        else:
            data_arr = np.zeros((1, 0), dtype=np.float32)
    else:
        data_arr = np.zeros((1, 0), dtype=np.float32)

    if data_arr.ndim == 1:
        data_arr = data_arr.reshape(1, -1)
    elif data_arr.ndim == 2 and data_arr.shape[0] > data_arr.shape[1]:
        data_arr = data_arr.T
    elif data_arr.ndim >= 3:
        # Segmented data (n_seg, n_ch, n_t) — flatten to continuous for QC
        orig_shape = data_arr.shape
        data_arr = data_arr.reshape(data_arr.shape[-2], -1)
        meta["original_shape"] = list(orig_shape)

    freq = float(freq)
    n_ch = data_arr.shape[0] if data_arr.ndim >= 2 else 1
    n_samples = data_arr.shape[-1] if data_arr.size > 0 else 0
    if channels is None:
        channels = [f"Ch{i}" for i in range(n_ch)]

    return {
        "data": data_arr,
        "frequency": freq,
        "channels": list(channels),
        "duration": float(n_samples / freq) if freq > 0 else 0.0,
        "meta": {
            "format": "pkl",
            "source_file": filepath,
            **(meta if isinstance(meta, dict) else {}),
        },
    }


def _load_unknown_format(filepath: str, inspect_only: bool = False) -> Dict[str, Any]:
    """Attempt to load an unrecognized format via MNE's generic reader.

    Falls back gracefully with a descriptive error if all attempts fail.
    """
    path = Path(filepath)

    # Attempt 1: MNE generic read_raw (handles many formats via plugins)
    try:
        import mne
        raw = mne.io.read_raw(filepath, preload=not inspect_only, verbose=False)
        # Success — delegate to standard MNE path
        return _load_mne(filepath, preload=True, inspect_only=inspect_only)
    except Exception as mne_err:
        logger.debug("MNE generic reader failed for %s: %s", path.name, mne_err)

    # Attempt 2: Try pickle (common in BCI research)
    if path.suffix.lower() in (".pkl", ".pickle"):
        try:
            import pickle
            with open(filepath, "rb") as f:
                obj = pickle.load(f)
            if isinstance(obj, dict):
                data = obj.get("data") or obj.get("eeg") or obj.get("signals")
                freq = obj.get("frequency") or obj.get("srate") or obj.get("fs", 256.0)
                if data is not None:
                    data_arr = np.asarray(data, dtype=np.float32)
                    if data_arr.ndim == 2 and data_arr.shape[0] > data_arr.shape[1]:
                        data_arr = data_arr.T
                    n_ch = data_arr.shape[0] if data_arr.ndim == 2 else 1
                    channels = obj.get("channels", [f"Ch{i}" for i in range(n_ch)])
                    return {
                        "data": data_arr,
                        "frequency": float(freq),
                        "channels": list(channels),
                        "duration": float(data_arr.shape[-1] / float(freq)),
                        "meta": {"format": "pickle", "source_file": filepath},
                    }
        except Exception as pkl_err:
            logger.debug("Pickle reader failed for %s: %s", path.name, pkl_err)

    logger.warning(
        "Cannot load '%s': format not recognized and all fallback readers failed. "
        "Supported formats: %s. "
        "The agent should generate custom loading code for this file.",
        path.name, sorted(_FORMAT_MAP.keys()),
    )
    return {
        "data": np.zeros((1, 0), dtype=np.float32),
        "frequency": 1.0,
        "channels": ["Ch0"],
        "duration": 0.0,
        "meta": {
            "format": "unknown",
            "source_file": filepath,
            "load_error": f"Format not recognized: {path.suffix}",
            "supported_formats": sorted(_FORMAT_MAP.keys()),
        },
    }

