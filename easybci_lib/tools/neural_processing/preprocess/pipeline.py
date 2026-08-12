"""Configurable preprocessing pipeline.

Design philosophy:
- Single function `preprocess()` that takes data + a list of step descriptors
- Each step is a pure function: ndarray in → ndarray out (or mne.Raw → mne.Raw)
- Steps are ordered and composable — the agent picks which ones to apply
- No hidden state, no class hierarchies, no caching layer

What we borrow from neuralset:
- The 10-step ordering (proven correct in production)
- Notch filter with automatic harmonics
- Nyquist guard on bandpass
- Bipolar reference derivation logic for sEEG

What we DON'T borrow:
- MapInfra/exca caching
- BaseExtractor class hierarchy
- torch anything
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from easybci_lib.tools.neural_processing._seed import EASYBCI_SEED
from easybci_lib.tools.neural_processing.preprocess.operator_vocab import (
    engine_operators as _engine_operators,
)

logger = logging.getLogger(__name__)

# Derived from the single source of truth (operator_vocab.OPERATOR_EXECUTORS),
# NOT hand-maintained — this is what keeps the engine vocabulary from drifting
# away from the codegen bundle's. Add/rename operators in operator_vocab only.
AVAILABLE_STEPS = _engine_operators()

STEP_FULL_NAMES = {
    "pick_channels": "Pick Channels (Channel Selection)",
    "drop_bads": "Drop Bad Channels",
    "drop_nondata_channels": "Drop Non-Data Channels (markers/misc/physio)",
    "interpolate_bads": "Interpolate Bad Channels (Spherical Spline)",
    "car": "Common Average Reference (CAR)",
    "bipolar_ref": "Bipolar Re-referencing",
    "notch": "Notch Filter (Power Line Removal)",
    "bandpass": "Bandpass Filter",
    "highpass": "Highpass Filter",
    "lowpass": "Lowpass Filter",
    "hilbert": "Hilbert Transform (Envelope Extraction)",
    "ica": "Independent Component Analysis (ICA Artifact Removal)",
    "resample": "Resample (Downsampling)",
    "scale": "Amplitude Scaling (Normalization)",
    "clip": "Amplitude Clipping",
    "fill_nan": "Fill NaN/Inf Values",
    "reject_by_labels": "Reject Labelled Time Segments (seizure/stim/IID)",
    "spike_sorting": "Spike Detection and Sorting",
    "extract_psd_bands": "PSD Band Power Feature Extraction",
    "extract_csp": "Common Spatial Pattern (CSP) Feature Extraction",
    "extract_tfr": "Time-Frequency Representation Feature Extraction",
    "extract_connectivity": "Functional Connectivity Feature Extraction",
}

# Steps that require MNE Raw objects
_MNE_STEPS = {"notch", "bandpass", "hilbert", "resample", "bipolar_ref", "car", "ica", "interpolate_bads"}


def _get_mne_info(d: Dict) -> Any:
    """Get or create MNE Info object, preserving channel types across steps."""
    import mne

    if "_mne_info" in d:
        info = d["_mne_info"]
        if len(info.ch_names) == len(d["channels"]):
            return info

    ch_types = d.get("meta", {}).get("ch_types", [])
    n_channels = len(d["channels"])

    if ch_types and len(ch_types) == n_channels:
        valid_types = {"eeg", "meg", "seeg", "ecog", "emg", "eog", "ecg", "misc", "stim", "bio"}
        safe_types = [t if t in valid_types else "misc" for t in ch_types]
        info = mne.create_info(d["channels"], d["frequency"], ch_types=safe_types)
    else:
        info = mne.create_info(n_channels, d["frequency"], ch_types="eeg")

    d["_mne_info"] = info
    return info


def _update_mne_info(d: Dict, raw: Any):
    """Update stored MNE info after a step that changes channels or sampling."""
    d["_mne_info"] = raw.info.copy()


def preprocess(
    data_dict: Dict[str, Any],
    steps: Optional[List[str]] = None,
    record_states: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Apply preprocessing steps to loaded neural data.

    Works directly on the dict returned by `skills.io.load_neural()`.
    Steps are specified as strings: "step_name" or "step_name:param".

    Optimization: consecutive MNE steps share a single Raw object to avoid
    repeated ndarray→Raw→ndarray copies.

    Parameters
    ----------
    data_dict : dict
        Output of load_neural(). Must have "data" (ndarray) and "frequency".
    steps : list of str
        Ordered steps to apply. Format: "step_name:param1,param2"
        Examples: ["notch:50", "bandpass:1,40", "resample:256", "scale:robust"]
    record_states : bool
        If True, record data shape/frequency/channels before and after each step.
        Results stored in data_dict["meta"]["step_states"].
    **kwargs : additional parameters passed to individual steps.

    Returns
    -------
    dict : same structure, with "data" transformed and "meta" updated.

    Step reference
    --------------
    - "pick_channels:eeg" or "pick_channels:Fp1,Fp2,F3"
    - "drop_bads"
    - "interpolate_bads" (spherical spline interpolation of bad channels)
    - "car" (common average reference)
    - "bipolar_ref" or "bipolar_ref:auto"
    - "notch:50" (notch at 50Hz + harmonics) or "notch:50,60"
    - "bandpass:1,40" (1-40 Hz) or "bandpass:,40" (lowpass only)
    - "hilbert"
    - "ica" or "ica:eog" or "ica:eog,ecg" (auto-detect artifact components)
    - "resample:256"
    - "scale:robust" or "scale:standard" or "scale:1e6" (factor)
    - "clip:5" (clamp to ±5)
    - "fill_nan:0"
    """
    if steps is None:
        return data_dict

    # Track what was applied for meta
    applied: List[str] = []
    step_states: List[Dict[str, Any]] = [] if record_states else None

    for step_str in steps:
        # Normalize operator name before dispatch: known synonyms
        # (highpass/lowpass/bad_channels/...) are rewritten to canonical form so
        # they are never silently skipped; truly unknown names fail loud below.
        try:
            from easybci_lib.tools.neural_processing.preprocess.operator_vocab import (
                normalize_step as _normalize_step,
                UnknownOperatorError as _UnknownOperatorError,
            )
            step_str, _ = _normalize_step(step_str)
        except _UnknownOperatorError:
            raise
        except Exception:
            # operator_vocab unavailable — fall back to raw name (legacy behaviour).
            pass

        parts = step_str.split(":", 1)
        step_name = parts[0].strip()
        step_param = parts[1].strip() if len(parts) > 1 else ""

        # Snapshot state before this step
        if record_states:
            before_state = _snapshot_state(data_dict)

        # Non-MNE steps need the raw flushed first
        if step_name not in _MNE_STEPS and "_raw" in data_dict:
            _flush_raw(data_dict)

        if step_name == "notch":
            data_dict = _step_notch(data_dict, step_param)
        elif step_name == "bandpass":
            data_dict = _step_bandpass(data_dict, step_param)
        elif step_name == "hilbert":
            data_dict = _step_hilbert(data_dict)
        elif step_name == "resample":
            data_dict = _step_resample(data_dict, step_param)
        elif step_name == "car":
            data_dict = _step_car(data_dict)
        elif step_name == "ica":
            data_dict = _step_ica(data_dict, step_param)
        elif step_name == "interpolate_bads":
            data_dict = _step_interpolate_bads(data_dict)
        elif step_name == "scale":
            data_dict = _step_scale(data_dict, step_param)
        elif step_name == "clip":
            data_dict = _step_clip(data_dict, step_param)
        elif step_name == "fill_nan":
            data_dict = _step_fill_nan(data_dict, step_param)
        elif step_name == "reject_by_labels":
            data_dict = _step_reject_by_labels(data_dict, step_param, **kwargs)
        elif step_name == "pick_channels":
            data_dict = _step_pick_channels(data_dict, step_param, **kwargs)
        elif step_name == "drop_bads":
            data_dict = _step_drop_bads(data_dict, step_param)
        elif step_name == "drop_nondata_channels":
            data_dict = _step_drop_nondata_channels(data_dict, step_param)
        elif step_name == "bipolar_ref":
            data_dict = _step_bipolar_ref(data_dict, step_param)
        elif step_name.startswith("extract_"):
            data_dict = _step_feature_extract(data_dict, step_name, step_param, **kwargs)
        else:
            # Defense-in-depth backstop: names are normalized above, so
            # reaching here means a canonical operator this engine does not
            # implement (or a name that bypassed normalization). Fail loud —
            # NEVER silently drop a step, which is the bug this replaces.
            raise ValueError(
                f"Unknown/unsupported step {step_name!r} in the runtime engine. "
                f"Engine-supported: {AVAILABLE_STEPS}"
            )

        applied.append(step_str)

        # Snapshot state after this step
        if record_states:
            after_state = _snapshot_state(data_dict)
            # Surface non-data channels removed by drop_nondata_channels so the UI
            # can render a per-step disclosure of what was dropped.
            dropped = (data_dict.get("meta") or {}).get("dropped_channels")
            if dropped and step_str.startswith("drop_nondata_channels"):
                # Only attach the channels removed *by this step* (not the cumulative meta list).
                step_states.append({
                    "step": step_str,
                    "before": before_state,
                    "after": after_state,
                    "dropped_channels": list(dropped),
                })
            else:
                step_states.append({
                    "step": step_str,
                    "before": before_state,
                    "after": after_state,
                })

    # Flush any remaining Raw object
    if "_raw" in data_dict:
        _flush_raw(data_dict)

    # Clean up internal MNE info before returning
    data_dict.pop("_mne_info", None)
    data_dict.setdefault("meta", {})["preprocessing"] = applied
    if record_states and step_states:
        data_dict["meta"]["step_states"] = step_states
    return data_dict


def _snapshot_state(d: Dict) -> Dict[str, Any]:
    """Capture a lightweight state snapshot of the data dict for reasoning records."""
    # If MNE Raw is active, use it for accurate state (data ndarray may be stale)
    if "_raw" in d:
        raw = d["_raw"]
        n_channels = len(raw.ch_names)
        n_samples = raw.n_times
        frequency = raw.info["sfreq"]
        duration_s = round(n_samples / frequency, 2) if frequency else 0
        # Get a small data slice for stats without copying all data
        snippet = raw.get_data(start=0, stop=min(n_samples, int(frequency * 5)))
        out = {
            "n_channels": n_channels,
            "n_samples": n_samples,
            "frequency": float(frequency),
            "duration_s": duration_s,
            "dtype": str(snippet.dtype),
            # Channel names enable downstream "Removed channels" annotation
            # without re-reading the data.
            "channels": list(raw.ch_names),
        }
        if getattr(snippet, "size", 0) > 0:
            out["value_range"] = [round(float(np.min(snippet)), 4), round(float(np.max(snippet)), 4)]
            out["mean_std"] = [round(float(np.mean(snippet)), 4), round(float(np.std(snippet)), 4)]
        else:
            out["value_range"] = [0.0, 0.0]
            out["mean_std"] = [0.0, 0.0]
            out["empty"] = True
        return out

    data = d.get("data")
    snapshot = {
        "n_channels": int(data.shape[0]) if data is not None else 0,
        "n_samples": int(data.shape[1]) if data is not None and data.ndim >= 2 else 0,
        "frequency": float(d.get("frequency", 0)),
        "duration_s": round(float(data.shape[1]) / float(d["frequency"]), 2) if data is not None and data.ndim >= 2 and d.get("frequency") else 0,
    }
    _chs = d.get("channels")
    if _chs:
        snapshot["channels"] = list(_chs)
    if data is not None:
        snapshot["dtype"] = str(data.dtype)
        # Guard against zero-size arrays (e.g. loader sentinel for missing
        # files). np.min/np.max raise "zero-size array to reduction operation"
        # which masks the real cause and sends the agent into a retry loop.
        if getattr(data, "size", 0) > 0:
            snapshot["value_range"] = [round(float(np.min(data)), 4), round(float(np.max(data)), 4)]
            snapshot["mean_std"] = [round(float(np.mean(data)), 4), round(float(np.std(data)), 4)]
        else:
            snapshot["value_range"] = [0.0, 0.0]
            snapshot["mean_std"] = [0.0, 0.0]
            snapshot["empty"] = True
    return snapshot


def _ensure_raw(d: Dict) -> Any:
    """Get or create Raw object for MNE steps. Reuses existing _raw if available."""
    if "_raw" in d:
        return d["_raw"]
    info = _get_mne_info(d)
    raw = _create_raw_with_annotations(d, info)
    d["_raw"] = raw
    return raw


def _flush_raw(d: Dict) -> None:
    """Extract data from live Raw back to ndarray and remove _raw."""
    raw = d.pop("_raw", None)
    if raw is None:
        return
    d["data"] = raw.get_data().astype(np.float32)
    d["channels"] = list(raw.ch_names)
    d["frequency"] = float(raw.info["sfreq"])
    d["duration"] = float(raw.times[-1] - raw.times[0] + 1.0 / raw.info["sfreq"])
    _update_mne_info(d, raw)


def _prune_meta_to_channels(d: Dict) -> None:
    """Drop names from meta['bads']/meta['bad_channels'] that no longer exist in
    d['channels']. Called by every step that mutates the channel list — keeps
    downstream consumers (interpolate_bads, drop_bads, codegen) from referencing
    channels that have already been removed by an earlier step. Modality-agnostic:
    works on whatever channel naming the input uses (10-20 EEG, sEEG L/R prefix,
    MEG MAG/GRAD, …).
    """
    meta = d.get("meta")
    if not isinstance(meta, dict):
        return
    current = set(d.get("channels") or [])
    for key in ("bads", "bad_channels"):
        old = meta.get(key)
        if not isinstance(old, list) or not old:
            continue
        new = [ch for ch in old if ch in current]
        if len(new) != len(old):
            removed = [ch for ch in old if ch not in current]
            logger.info(
                "Pruned %d stale name(s) from meta[%r] after channel mutation: %s",
                len(removed), key, removed,
            )
            meta[key] = new


# --- Step implementations ---


def _create_raw_with_annotations(d: Dict, info: Any) -> Any:
    """Create RawArray and restore annotations if available."""
    import mne
    raw = mne.io.RawArray(d["data"], info, verbose=False)
    annotations = d.get("meta", {}).get("annotations")
    if annotations:
        try:
            raw.set_annotations(mne.Annotations(
                onset=annotations["onset"],
                duration=annotations["duration"],
                description=annotations["description"],
            ))
        except Exception:
            pass
    return raw


def _step_notch(d: Dict, param: str) -> Dict:
    """Notch filter with harmonics up to Nyquist."""
    freqs = [float(x) for x in param.split(",") if x]
    if not freqs:
        freqs = [50.0]  # default: powerline

    sfreq = d["frequency"]
    nyquist = sfreq / 2.0
    all_freqs: List[float] = []
    for f in freqs:
        all_freqs.extend(np.arange(f, min(nyquist, 301), f).tolist())

    if not all_freqs:
        logger.info("No valid notch frequencies (all above Nyquist).")
        return d

    raw = _ensure_raw(d)
    raw.notch_filter(sorted(set(all_freqs)), phase="zero", verbose=False)
    return d


def _step_bandpass(d: Dict, param: str) -> Dict:
    """Band-pass filter with Nyquist guard."""
    parts = param.split(",")
    l_freq = float(parts[0]) if parts[0] else None
    h_freq = float(parts[1]) if len(parts) > 1 and parts[1] else None

    sfreq = d["frequency"]
    if h_freq is not None and h_freq >= sfreq / 2:
        logger.warning("h_freq >= Nyquist (%.1f). Disabling lowpass.", sfreq / 2)
        h_freq = None

    if l_freq is None and h_freq is None:
        return d

    raw = _ensure_raw(d)
    raw.filter(l_freq, h_freq, verbose=False)
    return d


def _step_hilbert(d: Dict) -> Dict:
    """Hilbert transform → envelope."""
    raw = _ensure_raw(d)
    raw.apply_hilbert(envelope=True)
    return d


def _step_resample(d: Dict, param: str) -> Dict:
    """Resample to target frequency."""
    target_freq = float(param)
    if target_freq == d["frequency"]:
        return d

    raw = _ensure_raw(d)
    raw.resample(target_freq, verbose=False)
    d["frequency"] = target_freq
    return d


def _step_scale(d: Dict, param: str) -> Dict:
    """Scale data: 'robust', 'standard', or a numeric factor."""
    data = d["data"]

    try:
        factor = float(param)
        d["data"] = data * factor
        return d
    except ValueError:
        pass

    import sklearn.preprocessing
    if param == "robust":
        scaler = sklearn.preprocessing.RobustScaler()
    elif param == "standard":
        scaler = sklearn.preprocessing.StandardScaler()
    else:
        logger.warning(
            "Unknown scaler '%s', falling back to 'robust'. Available: robust, standard, or a number.",
            param,
        )
        scaler = sklearn.preprocessing.RobustScaler()

    # Scale per channel (fit across time)
    d["data"] = scaler.fit_transform(data.T).T.astype(np.float32)
    return d


def _step_clip(d: Dict, param: str) -> Dict:
    """Clamp values to ±max_abs."""
    max_abs = float(param)
    d["data"] = np.clip(d["data"], -max_abs, max_abs)
    return d


def _step_fill_nan(d: Dict, param: str) -> Dict:
    """Replace non-finite values."""
    fill_val = float(param) if param else 0.0
    data = d["data"]
    mask = ~np.isfinite(data)
    n_bad = np.count_nonzero(mask)
    if n_bad > 0:
        logger.warning("Replacing %d non-finite values with %s", n_bad, fill_val)
        d["data"] = np.nan_to_num(data, nan=fill_val, posinf=fill_val, neginf=fill_val)
    return d


def _step_reject_by_labels(d: Dict, param: str, **kwargs) -> Dict:
    """Excise event-labelled time windows (seizure/stim/IID) from the data.

    Unlike the gold workflow, which discards the ENTIRE recording when any
    label matches, this keeps the clean remainder. Keywords are the UNION of
    (a) the ``param`` comma-list, (b) an ``extra_reject_keywords`` kwarg the
    agent supplies per environment, and (c) a multilingual built-in floor —
    the floor is a starting point, never authoritative. A ``reject_pad_s``
    kwarg (default 1.0 s) widens each excised window.

    Observability: labels the union did NOT match are written to
    ``meta['unmatched_labels']``, and the clinically suspicious subset (by
    multilingual seizure/stim word-roots) to ``meta['suspicious_labels']`` —
    so an unknown environment surfaces for review instead of silently passing.
    """
    from easybci_lib.tools.neural_processing.preprocess.label_reject import (
        DEFAULT_REJECT_KEYWORDS,
        merge_keywords,
        label_reject_mask,
        collect_unmatched_labels,
        flag_suspicious_labels,
    )
    param_kws = [k.strip() for k in (param or "").split(",") if k.strip()]
    extra_kws = kwargs.get("extra_reject_keywords") or []
    keywords = merge_keywords(param_kws, extra_kws, DEFAULT_REJECT_KEYWORDS)
    if not keywords:
        return d
    data = d["data"]
    if data is None or data.ndim < 2:
        return d
    sfreq = float(d.get("frequency") or 0.0)
    n_samples = int(data.shape[1])
    annotations = (d.get("meta") or {}).get("annotations")
    pad_s = float(kwargs.get("reject_pad_s", 1.0))

    keep = label_reject_mask(annotations, keywords, sfreq, n_samples, pad_s=pad_s)
    n_reject = int(n_samples - int(keep.sum()))
    meta = d.setdefault("meta", {})

    # Observability: surface labels the union missed + the suspicious subset.
    descriptions = (annotations or {}).get("description") or []
    unmatched = collect_unmatched_labels(descriptions, keywords)
    suspicious = flag_suspicious_labels(unmatched)
    meta["unmatched_labels"] = unmatched
    meta["suspicious_labels"] = suspicious
    if suspicious:
        logger.warning(
            "reject_by_labels: %d unmatched label(s) look clinically suspicious "
            "(review + extend keywords?): %s", len(suspicious), suspicious,
        )

    if n_reject > 0:
        d["data"] = data[:, keep]
        meta["rejected_samples"] = n_reject
        meta["rejected_seconds"] = round(n_reject / sfreq, 3) if sfreq else 0
        # INFO, not WARNING: excising labelled windows is the operator working
        # as intended (not an anomaly), and it fires per-file — a WARNING here
        # floods errors.log during batch runs.
        logger.info(
            "reject_by_labels: excised %d samples (%.1f s) matching keywords",
            n_reject, (n_reject / sfreq if sfreq else 0),
        )
    else:
        meta.setdefault("rejected_samples", 0)
    return d


def _step_pick_channels(d: Dict, param: str, **kwargs) -> Dict:
    """Pick channels by name, type, or count."""
    if not param:
        return d

    names = [x.strip() for x in param.split(",")]
    channels = d["channels"]

    # If param is a single integer, interpret as "keep first N channels"
    if len(names) == 1 and names[0].isdigit():
        n = int(names[0])
        if n >= len(channels):
            logger.info("pick_channels:%s — requested %d but only %d available. Keeping all.", param, n, len(channels))
            return d
        d["data"] = d["data"][:n]
        d["channels"] = channels[:n]
        if "ch_types" in d.get("meta", {}):
            d["meta"]["ch_types"] = d["meta"]["ch_types"][:n]
        d.pop("_mne_info", None)
        _prune_meta_to_channels(d)
        logger.info("pick_channels:%s — selected first %d of %d channels.", param, n, len(channels))
        return d

    # Check if it's a type selection (handled by caller via kwargs)
    if len(names) == 1 and names[0] in ("eeg", "meg", "seeg", "ecog", "emg", "data"):
        ch_types = d.get("meta", {}).get("ch_types", [])
        if ch_types:
            target = names[0]
            indices = [i for i, t in enumerate(ch_types) if t == target]
            if indices:
                d["data"] = d["data"][indices]
                d["channels"] = [channels[i] for i in indices]
                if "ch_types" in d.get("meta", {}):
                    d["meta"]["ch_types"] = [ch_types[i] for i in indices]
                d.pop("_mne_info", None)
                _prune_meta_to_channels(d)
            else:
                logger.warning(
                    "pick_channels:%s — no channels with type '%s' found (available types: %s). "
                    "Keeping all %d channels.",
                    param, target, sorted(set(ch_types))[:5], len(channels),
                )
            return d
        else:
            # No ch_types metadata — try matching channel names by common patterns
            target = names[0]
            pattern_indices = _match_channels_by_type_heuristic(channels, target)
            if pattern_indices:
                d["data"] = d["data"][pattern_indices]
                d["channels"] = [channels[i] for i in pattern_indices]
                d.pop("_mne_info", None)
                _prune_meta_to_channels(d)
                logger.info("pick_channels:%s — selected %d channels by name pattern.", param, len(pattern_indices))
            else:
                logger.warning(
                    "pick_channels:%s — no channel type metadata and no name patterns matched. "
                    "Keeping all %d channels.",
                    param, len(channels),
                )
            return d

    # Channel name selection
    indices = [i for i, ch in enumerate(channels) if ch in names]
    if not indices:
        # Try case-insensitive match
        names_lower = [n.lower() for n in names]
        indices = [i for i, ch in enumerate(channels) if ch.lower() in names_lower]
    if not indices:
        # Try prefix/substring match
        indices = [i for i, ch in enumerate(channels) if any(n in ch for n in names)]
    if not indices:
        logger.warning(
            "pick_channels:%s — no channels found matching %s out of %d available. "
            "Keeping all channels. Available: %s",
            param, names, len(channels), channels[:10],
        )
        return d
    d["data"] = d["data"][indices]
    d["channels"] = [channels[i] for i in indices]
    d.pop("_mne_info", None)
    _prune_meta_to_channels(d)
    return d


def _match_channels_by_type_heuristic(channels: List[str], target_type: str) -> List[int]:
    """Match channels by name patterns when ch_types metadata is missing."""
    import re
    eeg_patterns = re.compile(
        r"^(Fp[12z]|AF[34578z]|F[1-8pz]|FC[1-6z]|C[1-6pz]|CP[1-6z]|"
        r"P[1-8Oz]|PO[34789z]|O[12z]|Fz|Cz|Pz|Oz|T[34578]|TP[789]|"
        r"FT[789]|A[12]|M[12]|EEG[_\s]?\d+|[A-Z]{1,3}\d{1,2})$",
        re.IGNORECASE,
    )
    eog_patterns = re.compile(r"(EOG|VEOG|HEOG|vEOG|hEOG)", re.IGNORECASE)
    emg_patterns = re.compile(r"(EMG|Chin|Leg)", re.IGNORECASE)
    stim_patterns = re.compile(r"(STI|Trigger|Status|Event|Mark)", re.IGNORECASE)

    if target_type == "eeg":
        return [
            i for i, ch in enumerate(channels)
            if eeg_patterns.match(ch)
            and not eog_patterns.search(ch)
            and not emg_patterns.search(ch)
            and not stim_patterns.search(ch)
        ]
    elif target_type == "emg":
        return [i for i, ch in enumerate(channels) if emg_patterns.search(ch)]
    return []


def _step_drop_bads(d: Dict, param: str = "") -> Dict:
    """Drop channels marked as bad, or auto-detect when ``param == 'auto'``.

    Auto-detection is unit-aware and relative-first so the same step works on
    EEG-in-volts (MNE default), µV-scale NPZ exports, MEG-in-tesla, and
    z-scored / dimensionless data without retuning thresholds per modality.

    Detection criteria (auto):
      * flat (relative)   — ``std_i < max(median_std * 1e-4, atol)`` where
                             ``atol`` derives from ``meta['data_unit']`` when
                             known (1e-9 V, 1e-15 T) and falls back to
                             ``median_std * 1e-6`` otherwise.
      * high-σ (relative) — ``std_i > 5 × median(non-zero σ)``. Loose multiplier
                             so frontal EEG channels (Fp1 / Fp2 / AF3-4) — which
                             carry legitimate eye-blink signal at 2-4× median
                             std before ICA cleans them — are not flagged.
                             Truly noisy channels (e.g. cerebellar Cb1 / Cb2
                             with broken contact) typically run 50-100× median
                             and are still caught.
      * spike (relative)  — per-channel ``max(|x|) > 5 × global_99p_abs`` where
                             ``global_99p_abs`` is the 99-th percentile of
                             |x| over the whole array. Only fires when the
                             global 99p is itself non-zero (no information ⇒
                             no judgement).

    When ``meta['data_unit']`` is one of ``"V" | "uV" | "T" | "fT"`` an optional
    second-pass absolute spike check is run, scaled to that unit. For
    ``"unknown"`` (and the legacy default) NO absolute threshold is applied —
    relative criteria carry the load.

    If the resulting drop ratio exceeds 50% of channels, a one-line warning
    is appended to ``meta['warnings']`` so the QC report and grading layer
    can surface the over-aggressive run. This is output validation, not a
    hard guard — the drop still happens (the caller is responsible for
    reacting).

    Detected names are appended to ``meta['bad_channels']`` and dropped.
    The dropped name list is also written to ``meta['dropped_channels']``
    along with ``meta['dropped_ratio']`` for the QC report.

    A no-op when no bads are marked AND ``param != 'auto'``.
    """
    import numpy as np

    mode = (param or "").strip().lower()
    meta = d.setdefault("meta", {})
    bads = list(meta.get("bad_channels", []))

    if mode == "auto":
        data = d["data"]
        if data.size > 0 and data.ndim == 2:
            stds = data.std(axis=1)
            nz = stds[stds > 0.0]
            median_std = float(np.median(nz)) if nz.size else 0.0
            global_p99 = float(np.percentile(np.abs(data), 99)) if data.size else 0.0

            data_unit = str(meta.get("data_unit", "")).strip().lower()
            # Unit-aware absolute "dead-electrode" floor — used only as a
            # *lower bound* on the relative flat threshold so we never flag
            # channels whose tiny but legitimate signal happens to look
            # smaller than 1e-4 × median. None ⇒ no absolute floor.
            _abs_flat_atol = {
                "v":  1e-9,    # 1 nV  (deep below realistic EEG noise floor)
                "uv": 1e-3,    # 1e-3 µV
                "t":  1e-15,   # 1 fT
                "ft": 1e-3,    # 1e-3 fT
            }.get(data_unit)
            # Relative flat threshold; fall back to a permissive
            # median_std * 1e-6 sentinel when unit is unknown so we don't
            # accidentally re-introduce a hardcoded EEG-in-V assumption.
            if _abs_flat_atol is None:
                flat_threshold = max(median_std * 1e-4, median_std * 1e-6)
            else:
                flat_threshold = max(median_std * 1e-4, _abs_flat_atol)

            for i, ch in enumerate(d["channels"]):
                if ch in bads:
                    continue
                if median_std > 0 and stds[i] < flat_threshold:
                    bads.append(ch)
                    continue
                if median_std > 0 and stds[i] > 5.0 * median_std:
                    bads.append(ch)
                    continue
                if global_p99 > 0 and float(np.abs(data[i]).max()) > 5.0 * global_p99:
                    bads.append(ch)

        if bads:
            meta["bad_channels"] = bads
            meta["dropped_channels"] = list(bads)
            n_total = len(d.get("channels") or [])
            ratio = (len(bads) / n_total) if n_total else 0.0
            meta["dropped_ratio"] = round(ratio, 3)
            if ratio > 0.5:
                warn = (
                    f"drop_bads:auto removed {len(bads)} of {n_total} channels "
                    f"({ratio:.0%}) — check data_unit ({meta.get('data_unit', 'unknown')}) "
                    "and consider a scale step before drop_bads"
                )
                warnings_list = meta.setdefault("warnings", [])
                if isinstance(warnings_list, list):
                    warnings_list.append(warn)
                logger.warning("%s", warn)

    if not bads:
        return d

    channels = d["channels"]
    indices = [i for i, ch in enumerate(channels) if ch not in bads]
    d["data"] = d["data"][indices]
    d["channels"] = [channels[i] for i in indices]
    # Keep meta.ch_types in lock-step with channels — without this, downstream
    # _get_mne_info sees len(ch_types) != len(channels), falls through to the
    # "all eeg" default, and silently re-types whatever EOG / misc channels
    # survived the drop. drop_nondata_channels and pick_channels do the same.
    if isinstance(meta.get("ch_types"), list) and len(meta["ch_types"]) == len(channels):
        meta["ch_types"] = [meta["ch_types"][i] for i in indices]
    d.pop("_mne_info", None)
    _prune_meta_to_channels(d)
    return d


_NONDATA_MODES = ("markers_only", "markers_and_misc", "data_only")


def _step_drop_nondata_channels(d: Dict, param: str) -> Dict:
    """Remove non-data channels (markers / misc / physio) by graded mode.

    mode:
      markers_only      — drop marker channels only (default)
      markers_and_misc  — drop marker + misc, keep physio (EOG/ECG)
      data_only         — drop marker + misc + physio, keep data only
    """
    from easybci_lib.tools.neural_processing.io.channel_classifier import classify_channels

    mode = (param or "markers_only").strip()
    if mode not in _NONDATA_MODES:
        logger.warning(
            "drop_nondata_channels: unknown mode '%s'; using 'markers_only'.", mode,
        )
        mode = "markers_only"

    channels = d["channels"]
    meta = d.get("meta", {}) or {}
    modality = meta.get("modality", "") or d.get("modality", "")

    cls = classify_channels(
        channels,
        ch_types=meta.get("ch_types"),
        modality=modality,
        bad_channels=meta.get("bad_channels"),
    )
    if not cls["applicable"]:
        logger.info("drop_nondata_channels: not applicable for modality '%s'; no-op.", modality)
        return d

    cats = cls["categories"]
    if mode == "markers_only":
        drop_cats = {"marker"}
    elif mode == "markers_and_misc":
        drop_cats = {"marker", "misc"}
    else:  # data_only
        drop_cats = {"marker", "misc", "physio"}

    keep_idx = [i for i, ch in enumerate(channels) if cats.get(ch) not in drop_cats]
    dropped = [ch for ch in channels if cats.get(ch) in drop_cats]

    if not dropped:
        logger.info("drop_nondata_channels:%s — nothing to drop.", mode)
        return d

    d["data"] = d["data"][keep_idx]
    d["channels"] = [channels[i] for i in keep_idx]
    if isinstance(meta.get("ch_types"), list) and len(meta["ch_types"]) == len(channels):
        d["meta"]["ch_types"] = [meta["ch_types"][i] for i in keep_idx]
    # Record what was dropped for the mini-repo evidence / UI report
    d.setdefault("meta", {}).setdefault("dropped_channels", []).extend(dropped)
    d.pop("_mne_info", None)
    _prune_meta_to_channels(d)
    logger.info("drop_nondata_channels:%s — dropped %d: %s", mode, len(dropped), dropped)
    return d


def _step_bipolar_ref(d: Dict, param: str) -> Dict:
    """Bipolar referencing for sEEG.

    "auto" → derive pairs from neighboring electrodes on same probe.
    """
    channels = d["channels"]

    if param == "auto" or not param:
        anodes, cathodes = _derive_bipolar_pairs(channels)
    else:
        # Explicit: "A1-A2,B1-B2,..." format
        pairs = [p.split("-") for p in param.split(",")]
        anodes = [p[0] for p in pairs]
        cathodes = [p[1] for p in pairs]

    if not anodes:
        logger.warning("No valid bipolar pairs found.")
        return d

    import mne
    # Bipolar ref changes channels — flush any existing Raw first
    if "_raw" in d:
        _flush_raw(d)
    info = mne.create_info(channels, d["frequency"], ch_types="seeg")
    raw = _create_raw_with_annotations(d, info)
    raw = mne.set_bipolar_reference(raw, anodes, cathodes, verbose="WARNING")
    d["data"] = raw.get_data().astype(np.float32)
    d["channels"] = list(raw.ch_names)
    _update_mne_info(d, raw)
    _prune_meta_to_channels(d)
    return d


def _step_car(d: Dict) -> Dict:
    """Common Average Reference — subtract the mean across all channels at each time point.

    This is the standard EEG re-referencing scheme: each sample is referenced
    to the average of all channels, reducing shared noise while preserving
    local neural activity.
    """
    raw = _ensure_raw(d)
    raw.set_eeg_reference("average", projection=False, verbose=False)
    return d


def _step_ica(d: Dict, param: str) -> Dict:
    """ICA decomposition with automatic artifact component detection.

    Param format: "eog" or "eog,ecg" (which reference channels to use for
    correlation-based detection). If empty, defaults to EOG detection only.

    Uses MNE's FastICA with automatic component count selection (0.999 variance).
    Components correlated with EOG/ECG channels are excluded.
    """
    import mne
    from mne.preprocessing import ICA

    raw = _ensure_raw(d)

    artifact_types = [t.strip() for t in param.split(",") if t.strip()] if param else ["eog"]

    n_components = min(raw.info["nchan"] - 1, 25)
    if n_components < 2:
        logger.warning("Too few channels for ICA (need >= 3). Skipping.")
        return d

    ica = ICA(n_components=n_components, method="fastica", random_state=EASYBCI_SEED, max_iter=500)
    ica.fit(raw, verbose=False)

    exclude_indices = []
    for atype in artifact_types:
        try:
            if atype == "eog":
                indices, _ = ica.find_bads_eog(raw, verbose=False)
            elif atype == "ecg":
                indices, _ = ica.find_bads_ecg(raw, verbose=False)
            else:
                continue
            exclude_indices.extend(indices)
        except Exception as e:
            logger.info("ICA %s detection failed: %s", atype, e)

    if exclude_indices:
        ica.exclude = list(set(exclude_indices))
        logger.info("ICA excluding %d component(s): %s", len(ica.exclude), ica.exclude)
        ica.apply(raw, verbose=False)
    else:
        logger.info("ICA found no artifact components to exclude.")

    return d


def _step_interpolate_bads(d: Dict) -> Dict:
    """Interpolate bad channels using spherical spline interpolation.

    Bad channels are identified by the 'bads' key in data_dict meta,
    or by the MNE Raw info. After interpolation, the channels remain
    in the data but with interpolated values replacing the bad data.
    """
    import mne

    raw = _ensure_raw(d)

    bads = d.get("meta", {}).get("bads", [])
    if not bads:
        bads = raw.info.get("bads", [])

    if not bads:
        logger.info("No bad channels marked — skipping interpolation.")
        return d

    # Defensive filter — bads may have been computed against the original
    # channel list, but earlier steps (drop_nondata_channels, pick_channels,
    # bipolar_ref, …) can mutate raw.info["ch_names"]. MNE's raw.info["bads"]
    # setter rejects names not present in info, so trim here. Modality-
    # agnostic: the bad-channel detector is variance-based, so the names that
    # disappear could be anything (10-20 EEG, sEEG/MEG/ECoG/fNIRS).
    present = set(raw.info["ch_names"])
    valid_bads = [b for b in bads if b in present]
    dropped = [b for b in bads if b not in present]
    if dropped:
        logger.warning(
            "interpolate_bads: %d bad channel name(s) no longer in data, skipped: %s. "
            "Likely an upstream step (drop_nondata_channels / pick_channels / bipolar_ref) "
            "removed them after the bad-channel scan ran.",
            len(dropped), dropped,
        )
    if not valid_bads:
        logger.info("interpolate_bads: nothing left to interpolate after filtering.")
        return d

    raw.info["bads"] = valid_bads
    raw.interpolate_bads(reset_bads=True, verbose=False)
    logger.info("Interpolated %d bad channel(s): %s", len(valid_bads), valid_bads)
    return d


def _derive_bipolar_pairs(ch_names: List[str]) -> Tuple[List[str], List[str]]:
    """Auto-derive bipolar pairs from sEEG channel naming convention.

    Assumes channels ordered by probe, ascending contact number.
    E.g.: ['OF1', 'OF2', ..., 'H1', 'H2', ...] → OF1-OF2, OF2-OF3, ...
    Cross-probe pairs are skipped.
    """
    anodes = ch_names[:-1]
    cathodes = ch_names[1:]

    valid_a, valid_c = [], []
    for a, c in zip(anodes, cathodes):
        prefix_a = "".join(ch for ch in a if not ch.isdigit())
        prefix_c = "".join(ch for ch in c if not ch.isdigit())
        if prefix_a == prefix_c:
            valid_a.append(a)
            valid_c.append(c)

    return valid_a, valid_c


def _step_feature_extract(d: Dict, step_name: str, param: str, **kwargs) -> Dict:
    """Dispatch feature extraction steps.

    Feature extraction operates on the data array directly and stores
    the result in d["features"] while preserving the original data.
    """
    from easybci_lib.tools.neural_processing.features.extractors import (
        extract_psd_bands, extract_csp, extract_tfr, extract_connectivity,
    )

    data = d["data"]
    frequency = d["frequency"]
    channels = d.get("channels", [])
    labels = d.get("labels") if d.get("labels") is not None else kwargs.get("labels")

    if step_name == "extract_psd_bands":
        bands = param.split(",") if param else None
        result = extract_psd_bands(data, frequency, bands=bands, labels=labels, channels=channels)
    elif step_name == "extract_csp":
        n_comp = 6
        if param:
            for part in param.split(","):
                if part.startswith("n_components="):
                    n_comp = int(part.split("=")[1])
                elif part.isdigit():
                    n_comp = int(part)
        if labels is None:
            logger.warning(
                "extract_csp requires labels (epoched data with class labels) — skipping. "
                "Provide labels via data_dict['labels'] or segment the data first."
            )
            return d
        result = extract_csp(data, labels, n_components=n_comp, frequency=frequency, channels=channels)
    elif step_name == "extract_tfr":
        method = "morlet"
        freq_range = (4.0, 40.0)
        n_freqs = 20
        if param:
            for part in param.split(","):
                if part.startswith("method="):
                    method = part.split("=")[1]
                elif part.startswith("freqs="):
                    rng = part.split("=")[1]
                    if "-" in rng:
                        lo, hi = rng.split("-")
                        freq_range = (float(lo), float(hi))
                elif part.startswith("n_freqs="):
                    n_freqs = int(part.split("=")[1])
        result = extract_tfr(data, frequency, method=method, freq_range=freq_range,
                             n_freqs=n_freqs, labels=labels, channels=channels)
    elif step_name == "extract_connectivity":
        method = "plv"
        bands = None
        if param:
            parts_list = param.split(",")
            for part in parts_list:
                if part.startswith("method="):
                    method = part.split("=")[1]
                elif part.startswith("bands="):
                    bands = part.split("=")[1].split("+")
            if bands is None:
                bands = [p for p in parts_list if p in ("delta", "theta", "alpha", "beta", "gamma")]
        result = extract_connectivity(data, frequency, method=method, bands=bands or None,
                                      labels=labels, channels=channels)
    else:
        logger.warning("Unknown feature extraction step '%s' — skipping.", step_name)
        return d

    d.setdefault("features", {})[step_name] = {
        "X": result.X,
        "y": result.y,
        "feature_names": result.feature_names,
        "metadata": result.metadata,
    }
    d.setdefault("meta", {})["feature_extraction"] = step_name

    return d
