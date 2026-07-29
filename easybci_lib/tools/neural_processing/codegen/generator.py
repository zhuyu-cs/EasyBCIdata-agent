"""Pipeline code generator — transforms pipeline records into executable Python.

Generates deterministic, readable Python code specific to the processed dataset.
Comments explain WHY each step was chosen (from reasoning collector), not what it does.

Reproducibility contract
------------------------
Every generated script opens by locking all randomness to EASYBCI_SEED (42)
via an inline lock block placed right after the numpy import. Any operator with
a random component (ICA, train/test split, sampling) MUST receive EASYBCI_SEED.
"""

import logging
import time
from typing import Any, Dict, List, Optional

from easybci_lib.tools.neural_processing._seed import EASYBCI_SEED
from easybci_lib.tools.neural_processing.output.format_policy import is_invasive

try:
    from easybci_lib.tools.neural_processing.preprocess.analysis_goals import (
        REGISTRY as _GOAL_REGISTRY,
    )
except Exception:  # noqa: BLE001
    _GOAL_REGISTRY = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output-cleanup enforcement
# ---------------------------------------------------------------------------


_CLEANUP_OP = "drop_nondata_channels"
_CLEANUP_DATA_ONLY = f"{_CLEANUP_OP}:data_only"

# Single-source-of-truth for whether the
# post-ICA "clean output" trimming fires. Goals that consume EOG / physio
# downstream (source modelling, exploratory feature checks) opt OUT.
# Generic and unknown goals opt IN: defaulting to a clean output is the
# conservative, "won't surprise the next pipeline" choice.
_GOAL_TRIGGERS_CLEANUP = {
    "classification": True,
    "feature_extraction": True,
    "clinical_screening": True,
    "generic": True,
    "source_localization": False,
    "exploratory": False,
}

# Drop_bads goal-conditional auto-injection. Goals that need
# pristine data downstream (classification / feature_extraction / clinical /
# generic) auto-detect and DROP bad channels before reference / ICA so the
# pollution never propagates. source_localization and exploratory keep
# channels — the user / downstream step decides whether to interpolate.
_GOAL_TRIGGERS_DROP_BADS = {
    "classification": True,
    "feature_extraction": True,
    "clinical_screening": True,
    "generic": True,
    "source_localization": False,
    "exploratory": False,
}

_DROP_BADS_AUTO = "drop_bads:auto"


def _step_op(step) -> str:
    """Return the operator name for a string-form OR object-form step."""
    if isinstance(step, str):
        return step.split(":", 1)[0].strip()
    if isinstance(step, dict):
        return str(step.get("operator", "")).strip()
    return ""


def _step_param(step) -> str:
    """Return the post-colon param string for a string-form step (else "")."""
    if isinstance(step, str):
        if ":" in step:
            return step.split(":", 1)[1].strip()
        return ""
    return ""


def _enforce_clean_output(
    steps: List[Any],
    *,
    analysis_goal: Optional[str] = None,
) -> List[Any]:
    """Append/replace a ``drop_nondata_channels:data_only`` step so the final
    pipeline output is free of marker/physio (EOG/ECG/Trigger) channels —
    **conditioned on `analysis_goal`.

    Goals that opt IN (default behaviour, append data_only):
      classification, feature_extraction, clinical_screening, generic.
    Goals that opt OUT (skip the trim — caller wants EOG/physio retained):
      source_localization, exploratory.
    Unknown goal strings are treated as ``generic`` (conservative — clean
    output won't break source modelling-style downstream runs the way leaving
    EOG in would break a classifier).

    Rules within an opted-in goal (idempotent):
      1. If the last drop_nondata_channels:data_only step already sits AFTER
         the last ICA step (or there is no ICA), no-op.
      2. Otherwise, if any ICA step is present, insert
         ``drop_nondata_channels:data_only`` immediately after the LAST ICA
         step. ICA's EOG-aware ability stays intact; the cleanup runs after.
      3. Otherwise (no ICA in pipeline): if a
         ``drop_nondata_channels:markers_only`` step exists, replace its
         param with ``data_only``. If neither variant exists, prepend a new
         ``drop_nondata_channels:data_only`` step.

    Raises
    ------
    RuntimeError
        If ``analysis_goal`` is missing / empty. The PLAN_PIPELINE_SCHEMA
        marks the field required, so callers reaching this function without
        a goal indicate a code-path bug — fail loudly rather than silently
        defaulting.
    """
    if not analysis_goal or not isinstance(analysis_goal, str) or not analysis_goal.strip():
        raise RuntimeError(
            "_enforce_clean_output: analysis_goal is required. "
            "PLAN_PIPELINE_SCHEMA marks it required; reaching this function without "
            "a goal means an upstream caller skipped validation."
        )
    goal_key = analysis_goal.strip()
    # Read goal flags from the analysis_goals REGISTRY when available; fall
    # back to the legacy hardcoded dicts so existing 6-goal behaviour stays
    # intact for any callers still on that path.
    spec = _GOAL_REGISTRY.get(goal_key) if _GOAL_REGISTRY is not None else None
    if spec is not None:
        triggers_data_only = spec.inject_drop_nondata
        triggers_drop_bads = spec.inject_drop_bads
    else:
        triggers_data_only = _GOAL_TRIGGERS_CLEANUP.get(goal_key, True)
        triggers_drop_bads = _GOAL_TRIGGERS_DROP_BADS.get(goal_key, True)

    out = list(steps)

    # --- Injection 1: drop_bads:auto ----------------------------------------
    if triggers_drop_bads:
        has_drop_bads = any(_step_op(s) == "drop_bads" for s in out)
        if not has_drop_bads:
            insert_at = None
            for i, s in enumerate(out):
                if _step_op(s) in ("ica", "car", "bipolar_ref"):
                    insert_at = i
                    break
            if insert_at is None:
                last_bp = -1
                for i, s in enumerate(out):
                    if _step_op(s) == "bandpass":
                        last_bp = i
                insert_at = last_bp + 1 if last_bp >= 0 else 0
            else:
                last_bp = -1
                for i in range(insert_at):
                    if _step_op(out[i]) == "bandpass":
                        last_bp = i
                if last_bp >= 0:
                    insert_at = last_bp + 1
            out.insert(insert_at, _DROP_BADS_AUTO)

    # --- Injection 2: drop_nondata_channels:data_only -----------------------
    if not triggers_data_only:
        return out

    if not out:
        return [_CLEANUP_DATA_ONLY]

    ica_idx = -1
    data_only_idx = -1
    for i, s in enumerate(out):
        op = _step_op(s)
        if op == "ica":
            ica_idx = i
        if op == _CLEANUP_OP and _step_param(s) == "data_only":
            data_only_idx = i

    # Rule 1 — already clean.
    if data_only_idx > ica_idx:
        return out

    # Rule 2 — has ICA, insert data_only right after the LAST ica step.
    if ica_idx >= 0:
        out.insert(ica_idx + 1, _CLEANUP_DATA_ONLY)
        return out

    # Rule 3 — no ICA: replace existing markers_only or prepend new step.
    for i, s in enumerate(out):
        if _step_op(s) == _CLEANUP_OP and _step_param(s) == "markers_only":
            if isinstance(s, str):
                out[i] = _CLEANUP_DATA_ONLY
            elif isinstance(s, dict):
                s = dict(s)
                params = dict(s.get("params") or {})
                params["mode"] = "data_only"
                s["params"] = params
                out[i] = s
            return out
    # Insert data_only right after a leading drop_bads if present so the
    # canonical order (drop_bads → data_only → ...) holds; else prepend.
    insert_at = 1 if (out and _step_op(out[0]) == "drop_bads") else 0
    out.insert(insert_at, _CLEANUP_DATA_ONLY)
    return out


# ---------------------------------------------------------------------------
# Idempotent-skip helper — embedded into every generated script.
#
# Each per-file function (`_process_one` / `_qc_one` / `_vis_one` /
# `_ai_ready_one`) calls _already_done(target, __file__) at entry. When the
# target artefact already exists AND its mtime is >= the script's own mtime,
# the file is treated as already processed (LLM did not edit the plan since
# the artefact was written, so re-processing would produce the same output).
# Regenerating a stage script bumps its mtime above the old artefact's, so
# the skip disengages automatically when the plan changes.
#
# EASYBCI_FORCE_REPROCESS=1 disables the check everywhere for a clean rerun.
# ---------------------------------------------------------------------------
_ALREADY_DONE_HELPER_SRC = '''
def _already_done(target_path, script_path):
    """True when `target_path` was produced by the current version of `script_path`.

    Skips only when EASYBCI_FORCE_REPROCESS is unset AND the target exists
    AND its mtime is not older than the script's. Corrupt filesystem state
    (OSError on stat) treats the file as not-done so processing still runs.
    """
    import os as _os
    from pathlib import Path as _Path
    if _os.environ.get("EASYBCI_FORCE_REPROCESS"):
        return False
    try:
        tp = _Path(target_path)
        sp = _Path(script_path)
        return tp.is_file() and tp.stat().st_mtime >= sp.stat().st_mtime
    except OSError:
        return False
'''


# ---------------------------------------------------------------------------
# codegen bundle — pipeline.py / qc.py / build_ai_ready.py / run.py
# ---------------------------------------------------------------------------
#
# Each script's docstring carries machine-readable header markers so the
# adapted handlers (`_handle_preprocess_neural` / `_handle_save_processed` /
# `_handle_quality_check`) can decide whether to regenerate or preserve an
# agent-applied repair edit. The handlers compare the EASYBCI_STEPS / GOAL
# fields against the call's args; a match means the script is up-to-date.
#
# Bundle contract:
#   - pipeline.py        : preprocessing only → preprocessed_output/preprocessed/
#   - qc.py              : figures + QC report → preprocessed_output/{figures,QC_out}/
#   - build_ai_ready.py  : optional epoching → preprocessed_output/AI_ready/
#                          (only generated when events_present or label_config)
#   - run.py             : one-click chain (fail-fast on first non-zero retcode)


_PIPELINE_SCRIPT_TEMPLATE = '''"""Auto-generated preprocessing pipeline.

EASYBCI_STEPS: {steps_repr}
EASYBCI_GOAL: {analysis_goal}
EASYBCI_MODALITY: {modality}
EASYBCI_VERSION: 3
EASYBCI_CODE_STANDARD: 0.0.1

Standalone script — runs on a plain `pip install mne numpy scipy scikit-learn`
without any easybci_* dependency. See CODE_STANDARD.md Rule 15.

Run: python pipeline.py <input_path> <work_dir>
"""

import json
import os as _os
import pickle
import random as _random
import re as _re
import sys
from pathlib import Path

import numpy as np

EASYBCI_SEED = 42
_os.environ.setdefault("PYTHONHASHSEED", str(EASYBCI_SEED))
_random.seed(EASYBCI_SEED)
np.random.seed(EASYBCI_SEED)


# --------------------------------------------------------------------------
# Loader — wraps mne.io.read_raw for MNE-supported formats; falls back to
# numpy / scipy / pickle for CSV / NPZ / MAT / PKL.
# --------------------------------------------------------------------------
_MNE_EXTS = {{
    ".fif", ".edf", ".bdf", ".set", ".ds", ".cnt", ".gdf",
    ".vhdr", ".vmrk", ".eeg", ".cdt", ".mff", ".sqd", ".con",
}}


def _load_input(path):
    """Return a dict with keys: data (n_ch, n_samples float32), frequency, channels, meta."""
    p = Path(path)
    ext = p.suffix.lower()
    if ext in _MNE_EXTS or (p.is_dir() and p.suffix == ".ds"):
        import mne
        raw = mne.io.read_raw(str(p), preload=True, verbose="ERROR")
        return {{
            "data": raw.get_data().astype(np.float32),
            "frequency": float(raw.info["sfreq"]),
            "channels": list(raw.ch_names),
            "meta": {{"format": "mne", "source_file": str(p)}},
            "_mne_info": raw.info,
        }}
    if ext in (".npz", ".npy"):
        npz = np.load(str(p), allow_pickle=True)
        if ext == ".npy":
            arr = np.asarray(npz, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr[None, :]
            fs = 1.0
            ch = ["Ch{{}}".format(i) for i in range(arr.shape[0])]
        else:
            arr = np.asarray(npz["data"] if "data" in npz.files else npz[npz.files[0]], dtype=np.float32)
            fs = float(npz["frequency"]) if "frequency" in npz.files else 1.0
            ch = list(npz["channels"]) if "channels" in npz.files else ["Ch{{}}".format(i) for i in range(arr.shape[0])]
        return {{"data": arr, "frequency": fs, "channels": ch, "meta": {{"format": "npz", "source_file": str(p)}}}}
    if ext in (".csv", ".tsv"):
        import csv as _csv
        delim = "\\t" if ext == ".tsv" else ","
        with open(str(p), "r", encoding="utf-8", errors="replace") as f:
            reader = _csv.reader(f, delimiter=delim)
            rows = list(reader)
        if not rows:
            raise ValueError("CSV empty: {{}}".format(p))
        header = rows[0]
        try:
            [float(x) for x in header]
            data_rows = rows
            channels = ["Ch{{}}".format(i) for i in range(len(header))]
        except ValueError:
            channels = [h.strip() for h in header]
            data_rows = rows[1:]
        arr = np.array(
            [[float(x) if x.strip() not in ("", "nan", "NaN") else np.nan for x in r] for r in data_rows],
            dtype=np.float32,
        )
        # CSV layout: rows=samples, cols=channels → transpose to (n_ch, n_samples)
        arr = arr.T if arr.ndim == 2 else arr[None, :]
        return {{"data": arr, "frequency": 1.0, "channels": channels[: arr.shape[0]], "meta": {{"format": "csv", "source_file": str(p)}}}}
    if ext in (".pkl", ".pickle"):
        with open(str(p), "rb") as f:
            d = pickle.load(f)
        return {{
            "data": np.asarray(d["data"], dtype=np.float32),
            "frequency": float(d.get("frequency", 1.0)),
            "channels": list(d.get("channels", ["Ch{{}}".format(i) for i in range(np.asarray(d["data"]).shape[0])])),
            "meta": d.get("meta", {{"format": "pkl", "source_file": str(p)}}),
        }}
    if ext == ".nwb":
        try:
            from pynwb import NWBHDF5IO
        except ImportError:
            import subprocess as _sp_nwb, sys as _sys_nwb, os as _os_nwb
            if _os_nwb.environ.get("EASYBCI_DISABLE_LAZY_INSTALLS") == "1":
                raise
            _sp_nwb.check_call([_sys_nwb.executable, "-m", "pip", "install", "pynwb==3.1.3", "hdmf==4.3.1"])
            from pynwb import NWBHDF5IO
        with NWBHDF5IO(str(p), "r") as _io_nwb:
            _nwb = _io_nwb.read()
            _es_name = "preprocessed" if "preprocessed" in _nwb.acquisition else next(iter(_nwb.acquisition))
            _es = _nwb.acquisition[_es_name]
            _data_arr = np.asarray(_es.data[:]).T.astype(np.float32)
            _fs = float(_es.rate)
            try:
                _df = _nwb.electrodes.to_dataframe()
                _ch_names_v = list(_df["channel_name"]) if "channel_name" in _df.columns else ["Ch{{}}".format(i) for i in range(_data_arr.shape[0])]
            except Exception:
                _ch_names_v = ["Ch{{}}".format(i) for i in range(_data_arr.shape[0])]
        return {{"data": _data_arr, "frequency": _fs, "channels": _ch_names_v, "meta": {{"format": "nwb", "source_file": str(p)}}}}
    raise ValueError("Unsupported input format: {{}}".format(p))


# --------------------------------------------------------------------------
# Operator implementations — pure mne/scipy/numpy. Each takes a data_dict and
# returns a NEW data_dict (no in-place mutation). Operators are intentionally
# minimal — they match the behaviour of the corresponding easybci operator
# closely enough for routine preprocessing, but you can edit any of them
# in-place without touching the rest of the script.
# --------------------------------------------------------------------------

def _to_mne_raw(d):
    """Build an in-memory mne.io.RawArray from a data_dict."""
    import mne
    info = d.get("_mne_info")
    if info is None:
        info = mne.create_info(ch_names=list(d["channels"]), sfreq=float(d["frequency"]), ch_types="eeg")
    raw = mne.io.RawArray(np.asarray(d["data"], dtype=np.float64), info, verbose="ERROR")
    return raw


def _from_mne_raw(raw, prev_meta):
    return {{
        "data": raw.get_data().astype(np.float32),
        "frequency": float(raw.info["sfreq"]),
        "channels": list(raw.ch_names),
        "meta": dict(prev_meta),
        "_mne_info": raw.info,
    }}


def op_notch(d, param):
    freq = float(param) if param else 50.0
    raw = _to_mne_raw(d)
    raw.notch_filter(freqs=[freq], verbose="ERROR")
    return _from_mne_raw(raw, d.get("meta", {{}}))


def op_bandpass(d, param):
    parts = (param or "").split(",")
    lo = float(parts[0]) if len(parts) >= 1 and parts[0] else 1.0
    hi = float(parts[1]) if len(parts) >= 2 and parts[1] else 40.0
    raw = _to_mne_raw(d)
    raw.filter(l_freq=lo, h_freq=hi, verbose="ERROR")
    return _from_mne_raw(raw, d.get("meta", {{}}))


def op_highpass(d, param):
    lo = float(param) if param else 1.0
    raw = _to_mne_raw(d)
    raw.filter(l_freq=lo, h_freq=None, verbose="ERROR")
    return _from_mne_raw(raw, d.get("meta", {{}}))


def op_lowpass(d, param):
    hi = float(param) if param else 40.0
    raw = _to_mne_raw(d)
    raw.filter(l_freq=None, h_freq=hi, verbose="ERROR")
    return _from_mne_raw(raw, d.get("meta", {{}}))


def op_resample(d, param):
    target = float(param) if param else 256.0
    raw = _to_mne_raw(d)
    raw.resample(sfreq=target, verbose="ERROR")
    return _from_mne_raw(raw, d.get("meta", {{}}))


def op_car(d):
    raw = _to_mne_raw(d)
    raw.set_eeg_reference(ref_channels="average", projection=False, verbose="ERROR")
    return _from_mne_raw(raw, d.get("meta", {{}}))


def op_ica(d, param):
    import mne
    target_labels = [s.strip().lower() for s in (param or "").split(",") if s.strip()]
    if not target_labels:
        target_labels = ["eog", "ecg"]
    raw = _to_mne_raw(d)
    n_components = min(raw.info["nchan"], 20)
    ica = mne.preprocessing.ICA(
        n_components=n_components, method="fastica", random_state=EASYBCI_SEED, verbose="ERROR",
    )
    ica.fit(raw, verbose="ERROR")
    bad = []
    if "eog" in target_labels:
        try:
            eog_idx, _ = ica.find_bads_eog(raw, verbose="ERROR")
            bad.extend(eog_idx)
        except (ValueError, RuntimeError):
            pass
    if "ecg" in target_labels:
        try:
            ecg_idx, _ = ica.find_bads_ecg(raw, verbose="ERROR")
            bad.extend(ecg_idx)
        except (ValueError, RuntimeError):
            pass
    ica.exclude = sorted(set(bad))
    raw_clean = ica.apply(raw.copy(), verbose="ERROR")
    out = _from_mne_raw(raw_clean, d.get("meta", {{}}))
    out["meta"]["ica_excluded"] = ica.exclude
    return out


def op_drop_bads(d, param):
    """Auto-detect bad channels by amplitude variance and drop them.

    Heuristic: any channel whose std falls outside [median * 0.1, median * 10]
    or contains >50% NaN is flagged. Matches the spirit of easybci's
    drop_bads:auto without depending on its implementation.
    """
    data = np.asarray(d["data"], dtype=np.float64)
    channels = list(d["channels"])
    if data.ndim != 2 or data.shape[0] == 0:
        return dict(d)
    std = np.nanstd(data, axis=1)
    nan_frac = np.isnan(data).mean(axis=1)
    finite_std = std[np.isfinite(std) & (std > 0)]
    median_std = float(np.median(finite_std)) if finite_std.size else 0.0
    keep = []
    dropped = []
    for i, ch in enumerate(channels):
        if nan_frac[i] > 0.5 or not np.isfinite(std[i]) or std[i] == 0:
            dropped.append(ch); continue
        if median_std > 0 and (std[i] < median_std * 0.1 or std[i] > median_std * 10):
            dropped.append(ch); continue
        keep.append(i)
    if not keep:
        return dict(d)
    out = dict(d)
    out["data"] = data[keep, :].astype(np.float32)
    out["channels"] = [channels[i] for i in keep]
    out["meta"] = dict(d.get("meta", {{}}))
    out["meta"]["dropped_channels"] = list(out["meta"].get("dropped_channels", [])) + dropped
    out.pop("_mne_info", None)  # channel set changed; rebuild on next op
    return out


def op_drop_nondata_channels(d, param):
    """Drop marker/physio channels by name pattern.

    param=markers_only → drop trigger/marker/status/stim channels only.
    param=data_only    → also drop EOG/ECG/EMG/temp/resp/physio.
    """
    mode = (param or "data_only").strip().lower()
    channels = list(d["channels"])
    data = np.asarray(d["data"])
    marker_pat = _re.compile(r"^(trig|trigger|marker|markers|status|stim|stimulus|event|events)$", _re.IGNORECASE)
    physio_pat = _re.compile(r"^(eog|veog|heog|ecg|ekg|emg|emg\\d*|gsr|edr|temp|temperature|resp|respiration|pulse|spo2)\\b", _re.IGNORECASE)
    keep = []
    dropped = []
    for i, ch in enumerate(channels):
        if marker_pat.match(ch):
            dropped.append(ch); continue
        if mode == "data_only" and physio_pat.match(ch):
            dropped.append(ch); continue
        keep.append(i)
    if not keep:
        return dict(d)
    out = dict(d)
    out["data"] = data[keep, :]
    out["channels"] = [channels[i] for i in keep]
    out["meta"] = dict(d.get("meta", {{}}))
    out["meta"]["dropped_channels"] = list(out["meta"].get("dropped_channels", [])) + dropped
    out.pop("_mne_info", None)
    return out


def op_scale(d, param):
    """Per-channel amplitude normalisation. param=robust|zscore|minmax."""
    method = (param or "robust").strip().lower()
    data = np.asarray(d["data"], dtype=np.float64)
    if method == "zscore":
        mu = np.nanmean(data, axis=1, keepdims=True)
        sd = np.nanstd(data, axis=1, keepdims=True)
        sd[sd == 0] = 1.0
        scaled = (data - mu) / sd
    elif method == "minmax":
        lo = np.nanmin(data, axis=1, keepdims=True)
        hi = np.nanmax(data, axis=1, keepdims=True)
        rng = hi - lo
        rng[rng == 0] = 1.0
        scaled = (data - lo) / rng
    else:  # robust (median / IQR)
        med = np.nanmedian(data, axis=1, keepdims=True)
        q75 = np.nanpercentile(data, 75, axis=1, keepdims=True)
        q25 = np.nanpercentile(data, 25, axis=1, keepdims=True)
        iqr = q75 - q25
        iqr[iqr == 0] = 1.0
        scaled = (data - med) / iqr
    out = dict(d)
    out["data"] = scaled.astype(np.float32)
    return out


def op_clip(d, param):
    """Clip amplitudes to +/- N standard deviations. param defaults to 5."""
    n_std = float(param) if param else 5.0
    data = np.asarray(d["data"], dtype=np.float64)
    sd = np.nanstd(data, axis=1, keepdims=True)
    mu = np.nanmean(data, axis=1, keepdims=True)
    lo = mu - n_std * sd
    hi = mu + n_std * sd
    out = dict(d)
    out["data"] = np.clip(data, lo, hi).astype(np.float32)
    return out


def op_fill_nan(d, param):
    """Replace NaN/Inf with 0 (or the value in param)."""
    fill = float(param) if param else 0.0
    data = np.asarray(d["data"], dtype=np.float64)
    data = np.where(np.isfinite(data), data, fill)
    out = dict(d)
    out["data"] = data.astype(np.float32)
    return out


def op_threshold_spike(d, param):
    """MAD-based extracellular spike detection.

    param: <mad_multiplier>, default 5.0. Threshold per-channel =
    mad_multiplier * sigma_hat, with sigma_hat = median(|x|) / 0.6745
    (Quiroga 2004 robust noise estimate). Refractory: 1 ms.

    Writes meta["spike_times"]: list[ndarray] of sample indices per channel.
    Continuous data array is NOT modified (operator Rule 5).
    """
    mult = float(param) if param else 5.0
    data = np.asarray(d["data"], dtype=np.float32)
    if data.ndim != 2 or data.shape[0] == 0:
        return dict(d)
    abs_x = np.abs(data)
    med = np.median(abs_x, axis=1)
    sigma = np.where(med > 0, med / 0.6745, 1e-9)
    thresholds = (mult * sigma).astype(np.float32)
    fs = float(d.get("frequency", 30000.0))
    refrac = max(1, int(0.001 * fs))
    spike_times = []
    for ch in range(data.shape[0]):
        crosses = np.where(abs_x[ch] > thresholds[ch])[0]
        if crosses.size == 0:
            spike_times.append(np.asarray([], dtype=np.int64))
            continue
        keep = [int(crosses[0])]
        for t in crosses[1:]:
            if int(t) - keep[-1] >= refrac:
                keep.append(int(t))
        spike_times.append(np.asarray(keep, dtype=np.int64))
    out = dict(d)
    meta = dict(out.get("meta") or {{}})
    meta["spike_times"] = spike_times
    meta["thresholds"] = thresholds
    meta["spike_detector"] = "mad_threshold"
    meta["mad_multiplier"] = mult
    out["meta"] = meta
    return out


def op_mua_binning(d, param):
    """Bin per-channel spike times into firing-rate matrix.

    param: <bin_ms>, default 25. Reads meta["spike_times"] (from threshold_spike)
    and writes meta["mua_train"] (n_channels, n_bins) in spikes-per-second +
    meta["bin_centers"] (seconds). data array unchanged.
    """
    bin_ms = float(param) if param else 25.0
    bin_s = bin_ms / 1000.0
    fs = float(d.get("frequency", 30000.0))
    meta = dict(d.get("meta") or {{}})
    spike_times = meta.get("spike_times")
    if not spike_times:
        return dict(d)
    n_samples = int(np.asarray(d["data"]).shape[1])
    duration = n_samples / fs if fs > 0 else 0.0
    n_bins = max(1, int(duration / bin_s)) if duration > 0 else 1
    edges = np.arange(n_bins + 1, dtype=np.float64) * bin_s
    n_ch = len(spike_times)
    mua = np.zeros((n_ch, n_bins), dtype=np.float32)
    for ch in range(n_ch):
        times_s = (np.asarray(spike_times[ch], dtype=np.float64) / fs
                   if fs > 0 else np.asarray([]))
        counts, _ = np.histogram(times_s, bins=edges)
        mua[ch] = counts.astype(np.float32) / bin_s
    centers = ((edges[:-1] + edges[1:]) / 2.0).astype(np.float32)
    meta["mua_train"] = mua
    meta["bin_centers"] = centers
    meta["bin_width_ms"] = bin_ms
    out = dict(d)
    out["meta"] = meta
    return out


_OPS = {{
    "notch": op_notch,
    "bandpass": op_bandpass,
    "highpass": op_highpass,
    "lowpass": op_lowpass,
    "resample": op_resample,
    "car": op_car,
    "ica": op_ica,
    "drop_bads": op_drop_bads,
    "drop_nondata_channels": op_drop_nondata_channels,
    "scale": op_scale,
    "clip": op_clip,
    "fill_nan": op_fill_nan,
    "threshold_spike": op_threshold_spike,
    "mua_binning": op_mua_binning,
}}

_NO_PARAM_OPS = {{"car"}}


def _apply_step(d, step):
    if ":" in step:
        op_name, param = step.split(":", 1)
    else:
        op_name, param = step, ""
    op_name = op_name.strip()
    param = param.strip()
    fn = _OPS.get(op_name)
    if fn is None:
        print("[pipeline] WARNING: unknown step {{!r}} — skipping".format(step), file=sys.stderr)
        return d
    if op_name in _NO_PARAM_OPS:
        return fn(d)
    return fn(d, param)


def _session_id_from_path(p):
    """Legacy fallback only — derive ``ses-YYYYMMDDTHHMM`` from a filename
    timestamp when the routing table is absent (single-file legacy runs).
    Identity in multi-input mode comes from the routing table, NEVER from
    the file stem.
    """
    stem = Path(p).stem
    m = _re.search(r"(\\d{{4}})[_-]?(\\d{{2}})[_-]?(\\d{{2}})[_T]?(\\d{{2}})(\\d{{2}})", stem)
    if m:
        y, mo, d, h, mi = m.groups()
        return "{{}}{{}}{{}}T{{}}{{}}".format(y, mo, d, h, mi)
    return "001"


def _load_inputs(work_dir, argv):
    """Resolve the list of inputs to process for this work_dir.

    Multi-input mode: ``<work_dir>/middle_process/inputs_routing.json`` exists
    → return its ``inputs`` array verbatim. Each entry carries
    ``(data_path, stem_safe, subject_id, session_id, file_id,
    override_script)`` — downstream code uses these directly, NEVER re-deriving
    identity from the file stem.

    Legacy single-input mode: routing table absent → build one entry from the
    ``argv`` raw_path; ``subject_id`` defaults to the file stem, ``session_id``
    to the timestamp parsed from the stem.
    """
    routing = work_dir / "middle_process" / "inputs_routing.json"
    if routing.is_file():
        try:
            data = json.loads(routing.read_text(encoding="utf-8"))
            inputs = data.get("inputs") or []
            if inputs:
                return inputs
            print("[pipeline] routing table exists but is empty - falling back to argv", file=sys.stderr)
        except Exception as exc:
            print("[pipeline] routing table unreadable ({{}}); falling back to argv".format(exc), file=sys.stderr)

    if len(argv) < 2:
        print("Usage: python pipeline.py <work_dir>  OR  python pipeline.py <input_path> <work_dir>", file=sys.stderr)
        sys.exit(2)

    legacy_input = argv[1]
    stem = Path(legacy_input).stem.replace(" ", "_")
    return [{{
        "data_path": legacy_input,
        "stem_safe": stem,
        "subject_id": stem,
        "session_id": _session_id_from_path(legacy_input),
        "file_id": "legacy",
        "override_script": None,
    }}]


def _process_one(work_dir, inp, steps):
    """Process a single input, writing all artefacts under its (sub, ses) bucket."""
    data_path = inp["data_path"]
    stem = inp["stem_safe"]
    sub_id = inp["subject_id"]
    ses = inp["session_id"]
    file_id = inp.get("file_id") or "legacy"

    # Idempotent skip: target already produced by the current pipeline.py.
    out_dir = (
        work_dir / "preprocessed_output" / "preprocessed"
        / "sub-{{}}".format(sub_id) / "ses-{{}}".format(ses)
    )
    out_file = out_dir / "{{}}_preprocessed.nwb".format(stem)
    if _already_done(out_file, __file__):
        print("[pipeline] skip file_id={{}} (already processed)".format(file_id))
        mp = work_dir / "middle_process"
        prev_status_path = mp / "pipeline_status__{{}}.json".format(file_id)
        if prev_status_path.is_file():
            try:
                prev = json.loads(prev_status_path.read_text(encoding="utf-8"))
                if isinstance(prev, dict):
                    prev["skipped"] = True
                    return prev
            except Exception:
                pass
        return {{
            "file_id": file_id, "subject_id": sub_id, "session_id": ses,
            "output_file": str(out_file), "success": True, "skipped": True,
        }}

    print("Loading: {{}}  (sub={{}} ses={{}} file_id={{}})".format(data_path, sub_id, ses, file_id))
    data_dict = _load_input(data_path)
    n_ch_in = len(data_dict.get("channels", []))
    fs_in = float(data_dict.get("frequency", 0.0))
    print("  Channels in: {{}}, fs: {{}} Hz".format(n_ch_in, fs_in))

    print("Steps: {{}}".format(steps))
    for step in steps:
        data_dict = _apply_step(data_dict, step)

    n_ch_out = len(data_dict.get("channels", []))
    fs_out = float(data_dict.get("frequency", 0.0))
    print("  Channels out: {{}}, fs: {{}} Hz".format(n_ch_out, fs_out))

    out_dir.mkdir(parents=True, exist_ok=True)

    # NOTE: `_mne_info` is non-picklable but the NWB writer wants it for
    # meas_date / channel-type fallback, so we keep it on the dict until
    # after the writer runs.

    EASYBCI_MODALITY = "{modality}"
    EASYBCI_GOAL = "{analysis_goal}"

    _save_nwb_inline(
        data_dict=data_dict,
        subject_id=sub_id,
        out_path=out_file,
        analysis_goal=EASYBCI_GOAL,
        modality=EASYBCI_MODALITY,
    )
    print("Wrote {{}}".format(out_file))

    status = {{
        "file_id": file_id,
        "subject_id": sub_id,
        "session_id": ses,
        "n_channels": n_ch_out,
        "frequency_hz": fs_out,
        "n_samples": int(data_dict["data"].shape[-1]) if data_dict["data"].ndim else 0,
        "dropped_channels": list(data_dict.get("meta", {{}}).get("dropped_channels", [])),
        "output_file": str(out_file),
        "success": True,
    }}
    mp = work_dir / "middle_process"
    mp.mkdir(parents=True, exist_ok=True)
    (mp / "pipeline_status__{{}}.json".format(file_id)).write_text(
        json.dumps(status, indent=2), encoding="utf-8"
    )
    # Legacy single-file root sidecar for back-compat.
    (mp / "pipeline_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    return status


{nwb_helpers_block}
{already_done_helper}
def main():
    if len(sys.argv) >= 3:
        # Legacy form: pipeline.py <input_path> <work_dir>
        work_dir = Path(sys.argv[2])
    elif len(sys.argv) == 2:
        # Multi-input form: pipeline.py <work_dir>
        work_dir = Path(sys.argv[1])
    else:
        print("Usage: python pipeline.py <work_dir>  OR  python pipeline.py <input_path> <work_dir>", file=sys.stderr)
        sys.exit(2)
    work_dir.mkdir(parents=True, exist_ok=True)

    inputs = _load_inputs(work_dir, sys.argv)
    steps = {steps_repr}

    aggregate = {{"inputs": [], "n_success": 0, "n_failed": 0}}
    for inp in inputs:
        try:
            status = _process_one(work_dir, inp, steps)
            aggregate["inputs"].append(status)
            aggregate["n_success"] += 1
        except Exception as exc:
            file_id = inp.get("file_id") or "legacy"
            err_payload = {{
                "file_id": file_id,
                "subject_id": inp.get("subject_id"),
                "session_id": inp.get("session_id"),
                "success": False,
                "error": "{{}}: {{}}".format(type(exc).__name__, exc),
            }}
            aggregate["inputs"].append(err_payload)
            aggregate["n_failed"] += 1
            print("[pipeline] FAILED for file_id={{}}: {{}}".format(file_id, exc), file=sys.stderr)
            mp = work_dir / "middle_process"
            mp.mkdir(parents=True, exist_ok=True)
            (mp / "pipeline_status__{{}}.json".format(file_id)).write_text(
                json.dumps(err_payload, indent=2), encoding="utf-8"
            )

    # Aggregate sidecar so the dispatcher can read one file regardless of how
    # many inputs ran.
    mp = work_dir / "middle_process"
    mp.mkdir(parents=True, exist_ok=True)
    (mp / "pipeline_status_aggregate.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )

    if aggregate["n_failed"] > 0 and aggregate["n_success"] == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
'''


def _override_notch_freq(steps, line_freq: float):
    """If steps contain a notch op, rewrite its frequency to line_freq."""
    out = []
    for s in steps:
        if isinstance(s, str) and s.startswith("notch:"):
            out.append(f"notch:{int(round(line_freq))}")
        elif isinstance(s, dict) and s.get("operator") == "notch_filter":
            new = dict(s)
            new["params"] = dict(new.get("params") or {})
            new["params"]["freq"] = float(line_freq)
            out.append(new)
        else:
            out.append(s)
    return out


def _format_inspection_hints(report) -> str:
    """Render inspection_report as a comment block prepended to pipeline.py."""
    if not report:
        return ""
    lines = ["# === Inspection-driven hints (from deep_inspect) ==="]
    fp = report.get("fingerprint") or {}
    lines.append(
        f"# Modality: {fp.get('modality')} | Channels: {fp.get('n_channels')} "
        f"| fs: {fp.get('sampling_freq_hz')} Hz"
    )
    cs = report.get("channel_summary") or {}
    if cs.get("bad_candidates_high_variance"):
        lines.append(
            f"# Bad candidates (high-variance): {cs['bad_candidates_high_variance']}"
        )
    if cs.get("bad_candidates_flat"):
        lines.append(f"# Bad candidates (flat): {cs['bad_candidates_flat']}")
    if cs.get("must_drop"):
        lines.append(f"# Must drop (marker/trigger): {cs['must_drop']}")
    psd = report.get("psd_summary") or {}
    if psd.get("power_line_peak_hz"):
        lines.append(
            f"# Line-frequency peak: {psd['power_line_peak_hz']} Hz "
            f"({psd.get('power_line_peak_db_above_floor')} dB above floor)"
        )
    if psd.get("low_freq_drift_below_1hz_present"):
        lines.append("# Low-frequency drift detected (<1 Hz)")
    if psd.get("high_freq_noise_above_40hz_present"):
        lines.append("# High-frequency noise detected (>40 Hz)")
    if report.get("degraded"):
        lines.append(
            f"# DEGRADED inspection (reason: {report.get('degraded_reason')!r}) — "
            "operator/parameter choices are best-effort."
        )
    return "\n".join(lines) + "\n"


# Inline NWB writer source — injected into pipeline.py when chosen_format == "nwb".
# Kept in sync with easybci_lib/tools/neural_processing/output/nwb_writer.py:save_nwb.
# Must remain self-contained (no easybci_lib imports) — mini-repo runs on a plain
# `pip install mne pynwb hdmf` venv. Inlined as a single string so `.format()` on
# the surrounding template substitutes this whole block verbatim (no `{}` clashes).
_NWB_HELPERS_BLOCK = '''
# --- NWB output helpers (inlined; mirrors easybci_lib/.../nwb_writer.py) ---
try:
    from pynwb import NWBFile, NWBHDF5IO
    from pynwb.ecephys import ElectricalSeries
    from pynwb.file import Subject
    _PYNWB_OK = True
except ImportError:
    _PYNWB_OK = False
    try:
        import subprocess as _sp_nwb, sys as _sys_nwb, os as _os_nwb
        if _os_nwb.environ.get("EASYBCI_DISABLE_LAZY_INSTALLS") != "1":
            _sp_nwb.check_call([_sys_nwb.executable, "-m", "pip", "install", "pynwb==3.1.3", "hdmf==4.3.1"])
            from pynwb import NWBFile, NWBHDF5IO
            from pynwb.ecephys import ElectricalSeries
            from pynwb.file import Subject
            _PYNWB_OK = True
    except Exception:
        _PYNWB_OK = False

from datetime import datetime as _dt_nwb, timezone as _tz_nwb
from collections.abc import Mapping as _Mapping_nwb
import uuid as _uuid_nwb

# Conservative alias lists — first non-None/non-empty match wins. Mirror of
# easybci_lib/tools/neural_processing/output/nwb_writer.py.
_NWB_SFREQ_KEYS = ("sfreq", "frequency", "sampling_rate", "fs", "srate", "rate")
_NWB_CH_NAMES_KEYS = ("ch_names", "channels", "channel_names")
_NWB_CH_TYPES_KEYS = ("ch_types", "channel_types", "types")
_NWB_MEAS_DATE_KEYS = ("meas_date", "measurement_date", "date")
_NWB_SUBJECT_ID_KEYS = ("subject_id", "subject", "sub_id")


def _nwb_pick(d, keys, default=None):
    if not isinstance(d, _Mapping_nwb):
        return default
    for k in keys:
        v = d.get(k)
        if v is None:
            continue
        if isinstance(v, str) and not v:
            continue
        return v
    return default


def _nwb_info_get(info, name):
    if info is None:
        return None
    try:
        return info[name]
    except (KeyError, TypeError, ValueError):
        try:
            return getattr(info, name, None)
        except Exception:
            return None


def _nwb_info_channel_types(info):
    if info is None:
        return None
    fn = getattr(info, "get_channel_types", None)
    if callable(fn):
        try:
            return list(fn())
        except Exception:
            return None
    return None


def _nwb_info_subject_id(info):
    si = _nwb_info_get(info, "subject_info")
    if not si:
        return None
    if isinstance(si, _Mapping_nwb):
        return si.get("his_id") or si.get("id") or si.get("subject_id")
    return getattr(si, "his_id", None) or getattr(si, "id", None)


def _nwb_coerce_session_start(value):
    if isinstance(value, _dt_nwb):
        return value if value.tzinfo else value.replace(tzinfo=_tz_nwb.utc)
    if isinstance(value, (tuple, list)) and len(value) == 2:
        try:
            secs, usecs = value
            return _dt_nwb.fromtimestamp(float(secs) + float(usecs) * 1e-6, tz=_tz_nwb.utc)
        except Exception:
            return _dt_nwb(1970, 1, 1, tzinfo=_tz_nwb.utc)
    return _dt_nwb(1970, 1, 1, tzinfo=_tz_nwb.utc)


def _save_nwb_inline(*, data_dict, subject_id, out_path,
                     analysis_goal="", modality=None):
    """Write data_dict (loader-shape) to an NWB file.

    Metadata is resolved with multi-alias lookup:
        data_dict / data_dict["meta"] / data_dict["_mne_info"]
    Order of resolution per field: meta-aliases -> top-level-aliases -> mne_info.
    """
    if not _PYNWB_OK:
        raise ImportError("pynwb not available (lazy install disabled or failed)")
    data_arr = data_dict["data"]
    if data_arr.ndim != 2:
        raise ValueError("NWB writer expects 2-D (n_ch, n_samp) data; got shape " + str(data_arr.shape))
    n_ch = data_arr.shape[0]

    meta_d = data_dict.get("meta") if isinstance(data_dict, _Mapping_nwb) else None
    info = data_dict.get("_mne_info") if isinstance(data_dict, _Mapping_nwb) else None

    # ch_names: meta -> top-level data_dict -> mne_info["ch_names"]
    ch_names = _nwb_pick(meta_d, _NWB_CH_NAMES_KEYS)
    if ch_names is None:
        ch_names = _nwb_pick(data_dict, _NWB_CH_NAMES_KEYS)
    if ch_names is None:
        ch_names = _nwb_info_get(info, "ch_names")
    if ch_names is None:
        raise ValueError("NWB: ch_names not found in data_dict or _mne_info")
    ch_names = list(ch_names)

    # ch_types: meta -> mne_info.get_channel_types() -> ["unknown"] * n_ch
    ch_types = _nwb_pick(meta_d, _NWB_CH_TYPES_KEYS)
    if ch_types is None:
        ch_types = _nwb_info_channel_types(info)
    if ch_types is None:
        ch_types = ["unknown"] * n_ch
    else:
        ch_types = list(ch_types)

    if len(ch_names) != n_ch:
        raise ValueError("ch_names length mismatch: " + str(len(ch_names)) + " vs " + str(n_ch))
    if len(ch_types) != n_ch:
        raise ValueError("ch_types length mismatch: " + str(len(ch_types)) + " vs " + str(n_ch))

    # sfreq: meta -> top-level -> mne_info["sfreq"]
    sfreq_raw = _nwb_pick(meta_d, _NWB_SFREQ_KEYS)
    if sfreq_raw is None:
        sfreq_raw = _nwb_pick(data_dict, _NWB_SFREQ_KEYS)
    if sfreq_raw is None:
        sfreq_raw = _nwb_info_get(info, "sfreq")
    if sfreq_raw is None:
        raise ValueError("NWB: sampling rate not found in data_dict or _mne_info")
    sfreq = float(sfreq_raw)

    # subject_id: caller arg wins, then meta, then mne_info.subject_info.his_id.
    if not subject_id:
        subject_id = _nwb_pick(meta_d, _NWB_SUBJECT_ID_KEYS)
    if not subject_id:
        subject_id = _nwb_info_subject_id(info)

    # meas_date: meta -> mne_info["meas_date"] -> epoch fallback.
    meas_date = _nwb_pick(meta_d, _NWB_MEAS_DATE_KEYS)
    if meas_date is None:
        meas_date = _nwb_info_get(info, "meas_date")
    session_start = _nwb_coerce_session_start(meas_date)

    session_desc = (analysis_goal + " \\u2014 preprocessed by EasyBCIdata") if analysis_goal else "preprocessed by EasyBCIdata"
    identifier = (str(subject_id) + "/" + out_path.stem) if subject_id else _uuid_nwb.uuid4().hex
    nwb = NWBFile(
        session_description=session_desc,
        identifier=identifier,
        session_start_time=session_start,
        subject=Subject(subject_id=str(subject_id) if subject_id else "unknown", species="unspecified"),
    )
    device = nwb.create_device(name="EasyBCIdata-recording-device")
    eg = nwb.create_electrode_group(name="preprocessed-group", description="EasyBCIdata preprocessing channels.", location="unspecified", device=device)
    nwb.add_electrode_column(name="channel_name", description="MNE channel name.")
    nwb.add_electrode_column(name="channel_type", description="MNE channel type.")
    for _name, _ctype in zip(ch_names, ch_types):
        nwb.add_electrode(location="unspecified", group=eg, channel_name=str(_name), channel_type=str(_ctype))
    region = nwb.create_electrode_table_region(region=list(range(n_ch)), description="All preprocessed channels")
    es = ElectricalSeries(name="preprocessed", data=data_arr.T, electrodes=region, rate=sfreq, starting_time=0.0, description="Continuous post-pipeline signal")
    nwb.add_acquisition(es)
    _mod_l = (modality or "").strip().lower()
    spike_times = data_dict.get("spike_times") if isinstance(data_dict, _Mapping_nwb) else None
    if _mod_l in ("spike", "spikes", "unit", "units") and spike_times:
        for _i, _st in enumerate(spike_times):
            nwb.add_unit(spike_times=list(_st), id=_i)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with NWBHDF5IO(str(out_path), "w") as io:
        io.write(nwb)
    return out_path
# --- end NWB helpers ---
'''


# Inline NWB reader source — injected into qc.py / vis.py / build_ai_ready.py
# so those scripts can read the preprocessed NWB file without depending on the
# easybci_lib codebase. pynwb is lazy-installed on first import. The shape of
# the returned tuple matches what the pickle-era code expected
# (data, fs, meta, channels) so downstream usage stays mechanical.
_NWB_PREPROCESSED_LOADER_BLOCK = '''
# --- Preprocessed NWB loader (inlined; preprocessed/ is NWB-only) ---
def _load_preprocessed_nwb(pre_nwb):
    """Read a preprocessed NWB file written by pipeline.py.

    Returns (data, fs, meta, channels) where:
      data:      np.ndarray of shape (n_channels, n_samples), float32
      fs:        float, sampling rate in Hz
      meta:      dict (always at least ``{"format": "nwb", "source_file": ...}``)
      channels:  list[str], channel names from the electrode table
    """
    try:
        from pynwb import NWBHDF5IO
    except ImportError:
        import subprocess as _sp_nwb, sys as _sys_nwb, os as _os_nwb
        if _os_nwb.environ.get("EASYBCI_DISABLE_LAZY_INSTALLS") == "1":
            raise
        _sp_nwb.check_call([_sys_nwb.executable, "-m", "pip", "install",
                            "pynwb==3.1.3", "hdmf==4.3.1"])
        from pynwb import NWBHDF5IO
    with NWBHDF5IO(str(pre_nwb), "r") as _io_nwb:
        _nwb = _io_nwb.read()
        _acq = _nwb.acquisition
        _es_name = "preprocessed" if "preprocessed" in _acq else next(iter(_acq))
        _es = _acq[_es_name]
        _data = np.asarray(_es.data[:]).T.astype(np.float32)
        _fs = float(_es.rate)
        try:
            _channels = [str(_n) for _n in _nwb.electrodes["channel_name"][:]]
        except Exception:
            _channels = ["Ch{}".format(_i) for _i in range(_data.shape[0])]
        return _data, _fs, {"format": "nwb", "source_file": str(pre_nwb)}, _channels
# --- end Preprocessed NWB loader ---
'''


def generate_pipeline_script(
    *,
    steps: List[str],
    data_info: Dict[str, Any],
    modality: str = "eeg",
    analysis_goal: str = "generic",
    inspection_report: Dict[str, Any] | None = None,
) -> str:
    """Generate the canonical pipeline.py script.

    The script is standalone — it inlines mne / scipy / numpy operator
    logic instead of importing from ``easybci_lib`` (see CODE_STANDARD.md
    Rule 15). The mini-repo under ``code/`` must run on a machine with
    only ``pip install mne numpy scipy scikit-learn pynwb hdmf`` — pynwb
    is lazy-installed by the inlined helpers on first use unless
    ``EASYBCI_DISABLE_LAZY_INSTALLS=1``.

    The preprocessed/ layer is NWB-only — see :mod:`format_policy`. The
    legacy ``output_format`` parameter (``"auto"`` | ``"pkl"`` | ``"nwb"``)
    has been removed; pkl is no longer a legal preprocessed format.

    When ``inspection_report`` (from deep_inspect) is provided, the generator:
    - Overrides notch_filter frequency to the detected power-line peak
    - Prepends an "Inspection-driven hints" comment block so a human
      reading pipeline.py can see why these parameters were chosen
    """
    steps_list = list(steps)
    if inspection_report:
        line_freq = (inspection_report.get("psd_summary") or {}).get("power_line_peak_hz")
        if line_freq:
            steps_list = _override_notch_freq(steps_list, line_freq)
    enforced = _enforce_clean_output(steps_list, analysis_goal=analysis_goal)
    body = _PIPELINE_SCRIPT_TEMPLATE.format(
        steps_repr=repr(enforced),
        analysis_goal=analysis_goal,
        modality=modality,
        nwb_helpers_block=_NWB_HELPERS_BLOCK,
        already_done_helper=_ALREADY_DONE_HELPER_SRC,
    )
    hints = _format_inspection_hints(inspection_report)
    if hints:
        body = hints + body
    return body


_QC_SCRIPT_TEMPLATE = '''"""Auto-generated QC report.

EASYBCI_STEPS: {steps_repr}
EASYBCI_GOAL: {analysis_goal}
EASYBCI_VERSION: 3
EASYBCI_CODE_STANDARD: 1.1.0

Standalone script — runs on a plain `pip install mne numpy scipy`
without any easybci_* dependency. See CODE_STANDARD.md Rule 15.

Run: python qc.py <input_path> <work_dir>
  - reads raw data from <input_path>
  - reads preprocessed.nwb from <work_dir>/preprocessed_output/preprocessed/sub-<id>/...
  - writes QC report to <work_dir>/preprocessed_output/QC_out/sub-<id>/<ses>/

Figures are produced by code/vis.py (run as part of the run.py chain).
"""

import json
import os as _os
import pickle
import random as _random
import sys
from pathlib import Path

import numpy as np

EASYBCI_SEED = 42
_os.environ.setdefault("PYTHONHASHSEED", str(EASYBCI_SEED))
_random.seed(EASYBCI_SEED)
np.random.seed(EASYBCI_SEED)

_MNE_EXTS = {{
    ".fif", ".edf", ".bdf", ".set", ".ds", ".cnt", ".gdf",
    ".vhdr", ".vmrk", ".eeg", ".cdt", ".mff", ".sqd", ".con",
}}


def _load_input(path):
    p = Path(path)
    ext = p.suffix.lower()
    if ext in _MNE_EXTS or (p.is_dir() and p.suffix == ".ds"):
        import mne
        raw = mne.io.read_raw(str(p), preload=True, verbose="ERROR")
        return {{
            "data": raw.get_data().astype(np.float32),
            "frequency": float(raw.info["sfreq"]),
            "channels": list(raw.ch_names),
        }}
    if ext in (".npz", ".npy"):
        npz = np.load(str(p), allow_pickle=True)
        if ext == ".npy":
            arr = np.asarray(npz, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr[None, :]
            return {{"data": arr, "frequency": 1.0, "channels": ["Ch{{}}".format(i) for i in range(arr.shape[0])]}}
        arr = np.asarray(npz["data"] if "data" in npz.files else npz[npz.files[0]], dtype=np.float32)
        fs = float(npz["frequency"]) if "frequency" in npz.files else 1.0
        ch = list(npz["channels"]) if "channels" in npz.files else ["Ch{{}}".format(i) for i in range(arr.shape[0])]
        return {{"data": arr, "frequency": fs, "channels": ch}}
    if ext in (".csv", ".tsv"):
        import csv as _csv
        delim = "\\t" if ext == ".tsv" else ","
        with open(str(p), "r", encoding="utf-8", errors="replace") as f:
            rows = list(_csv.reader(f, delimiter=delim))
        header = rows[0]
        try:
            [float(x) for x in header]
            data_rows = rows
            channels = ["Ch{{}}".format(i) for i in range(len(header))]
        except ValueError:
            channels = [h.strip() for h in header]
            data_rows = rows[1:]
        arr = np.array(
            [[float(x) if x.strip() not in ("", "nan", "NaN") else np.nan for x in r] for r in data_rows],
            dtype=np.float32,
        )
        arr = arr.T if arr.ndim == 2 else arr[None, :]
        return {{"data": arr, "frequency": 1.0, "channels": channels[: arr.shape[0]]}}
    if ext in (".pkl", ".pickle"):
        with open(str(p), "rb") as f:
            d = pickle.load(f)
        return {{
            "data": np.asarray(d["data"], dtype=np.float32),
            "frequency": float(d.get("frequency", 1.0)),
            "channels": list(d.get("channels", [])),
        }}
    raise ValueError("Unsupported input format: {{}}".format(p))


def _read_identity(work_dir):
    """Deprecated — kept for legacy work_dirs that have no routing table.

    Reads a single ``identity`` field from ``middle_process/inspection_report.json``.
    Multi-input runs ignore this and route via ``inputs_routing.json``; this
    helper survives so old single-file mini-repos keep running.
    """
    report_path = work_dir / "middle_process" / "inspection_report.json"
    if not report_path.is_file():
        return (None, None)
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
    except Exception:
        return (None, None)
    identity = report.get("identity") or {{}}
    sub = identity.get("subject_id")
    ses = identity.get("session_id")
    if sub and ses:
        ses_prefixed = ses if ses.startswith("ses-") else ("ses-" + ses)
        return (sub, ses_prefixed)
    return (None, None)


def _psd_welch(data, fs, fmax=80.0):
    """Per-channel Welch PSD via scipy.signal — falls back to numpy FFT if needed."""
    try:
        from scipy.signal import welch
        nperseg = int(min(data.shape[-1], max(256, fs * 2)))
        freqs, psd = welch(data, fs=fs, nperseg=nperseg, axis=-1)
    except ImportError:
        # Pure numpy fallback
        n = data.shape[-1]
        win = np.hanning(n)
        spec = np.fft.rfft(data * win, axis=-1)
        psd = (np.abs(spec) ** 2) / (fs * (win ** 2).sum())
        freqs = np.fft.rfftfreq(n, 1.0 / fs)
    mask = freqs <= fmax
    return freqs[mask], psd[..., mask]


def _compute_metrics(raw_d, proc):
    rb = np.asarray(raw_d["data"], dtype=np.float64)
    ra = np.asarray(proc["data"], dtype=np.float64)

    def _stats(arr):
        finite = np.isfinite(arr)
        if not finite.any():
            return {{"mean": None, "std": None, "min": None, "max": None, "nan_frac": 1.0}}
        return {{
            "mean": float(np.nanmean(arr)),
            "std": float(np.nanstd(arr)),
            "min": float(np.nanmin(arr)),
            "max": float(np.nanmax(arr)),
            "nan_frac": float((~finite).mean()),
        }}

    fs_b = float(raw_d["frequency"])
    fs_a = float(proc["frequency"])
    _, pb = _psd_welch(rb, fs_b)
    _, pa = _psd_welch(ra, fs_a)
    snr_before = float(pb.mean()) / (float(pb.std()) + 1e-30)
    snr_after = float(pa.mean()) / (float(pa.std()) + 1e-30)

    overall_grade = "Pass" if (
        np.isfinite(ra).mean() > 0.99 and ra.shape[0] > 0 and ra.shape[1] > 0
    ) else "Fail"

    return {{
        "before": {{
            "n_channels": int(rb.shape[0]),
            "n_samples": int(rb.shape[-1]) if rb.ndim else 0,
            "frequency_hz": fs_b,
            "stats": _stats(rb),
            "psd_snr_estimate": snr_before,
        }},
        "after": {{
            "n_channels": int(ra.shape[0]),
            "n_samples": int(ra.shape[-1]) if ra.ndim else 0,
            "frequency_hz": fs_a,
            "stats": _stats(ra),
            "psd_snr_estimate": snr_after,
        }},
        "overall": {{"grade": overall_grade}},
    }}


def _write_report(qc_dir, session_id, subject_id, data_path, steps, metrics):
    payload = {{
        "session_id": session_id,
        "subject_id": subject_id,
        "data_path": str(data_path),
        "steps": list(steps),
        "metrics": metrics,
    }}
    (qc_dir / "qc_report.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    md = [
        "# QC Report",
        "",
        "- Subject: `{{}}`".format(subject_id),
        "- Session: `{{}}`".format(session_id),
        "- Data: `{{}}`".format(data_path),
        "- Steps: `{{}}`".format(steps),
        "- Overall: **{{}}**".format(metrics["overall"]["grade"]),
        "",
        "## Before",
        "```json",
        json.dumps(metrics["before"], indent=2, default=str),
        "```",
        "",
        "## After",
        "```json",
        json.dumps(metrics["after"], indent=2, default=str),
        "```",
        "",
    ]
    (qc_dir / "qc_report.md").write_text("\\n".join(md), encoding="utf-8")


def _load_inputs(work_dir, argv):
    """Resolve the list of inputs to process for this work_dir.

    Multi-input mode: ``<work_dir>/middle_process/inputs_routing.json`` exists
    → return its ``inputs`` array verbatim. Each entry carries
    ``(data_path, stem_safe, subject_id, session_id, file_id)`` — qc.py uses
    these directly and NEVER re-derives identity from the raw file stem.

    Legacy single-input mode: routing table absent → build one entry from the
    ``argv`` raw_path; identity falls back to the first sub-*/ses-*/ bucket
    that contains a matching ``_preprocessed.nwb``.
    """
    routing = work_dir / "middle_process" / "inputs_routing.json"
    if routing.is_file():
        try:
            data = json.loads(routing.read_text(encoding="utf-8"))
            inputs = data.get("inputs") or []
            if inputs:
                return inputs
            print("[qc] routing table exists but is empty - falling back to argv", file=sys.stderr)
        except Exception as exc:
            print("[qc] routing table unreadable ({{}}); falling back to argv".format(exc), file=sys.stderr)

    if len(argv) < 2:
        print("Usage: python qc.py <work_dir>  OR  python qc.py <input_path> <work_dir>", file=sys.stderr)
        sys.exit(2)

    legacy_input = argv[1]
    stem = Path(legacy_input).stem.replace(" ", "_")
    # Legacy: find the first sub-X/ses-Y/ that holds a matching preprocessed.nwb.
    base = work_dir / "preprocessed_output" / "preprocessed"
    sub_id, ses, _stem_safe = stem, "001", stem
    if base.is_dir():
        for sd in sorted(base.glob("sub-*")):
            for sess_dir in sorted(sd.iterdir()):
                if sess_dir.is_dir() and any(sess_dir.glob(stem + "_preprocessed.nwb")):
                    sub_id = sd.name.replace("sub-", "")
                    ses = sess_dir.name.replace("ses-", "")
                    break
    return [{{
        "data_path": legacy_input,
        "stem_safe": stem,
        "subject_id": sub_id,
        "session_id": ses,
        "file_id": "legacy",
    }}]


def _qc_one(work_dir, inp, steps):
    """Run QC for one routing entry: load preprocessed.nwb, write figures + report."""
    raw_path = inp["data_path"]
    stem = inp["stem_safe"]
    sub_id = inp["subject_id"]
    ses = inp["session_id"]
    file_id = inp.get("file_id") or "legacy"

    qc_report_path = (
        work_dir / "preprocessed_output" / "QC_out"
        / "sub-{{}}".format(sub_id) / "ses-{{}}".format(ses) / "qc_report.json"
    )
    if _already_done(qc_report_path, __file__):
        print("[qc] skip file_id={{}} (already processed)".format(file_id))
        return {{
            "file_id": file_id, "subject_id": sub_id, "session_id": ses,
            "grade": "skipped", "score": None, "skipped": True,
        }}

    pre_nwb = (
        work_dir / "preprocessed_output" / "preprocessed"
        / "sub-{{}}".format(sub_id) / "ses-{{}}".format(ses)
        / "{{}}_preprocessed.nwb".format(stem)
    )
    if not pre_nwb.is_file():
        print("[qc] preprocessed nwb missing for file_id={{}}: {{}}".format(file_id, pre_nwb), file=sys.stderr)
        raise FileNotFoundError(pre_nwb)

    raw_d = _load_input(raw_path)
    _proc_data, _proc_fs, _proc_meta, _proc_channels = _load_preprocessed_nwb(pre_nwb)
    proc = {{
        "data": _proc_data,
        "frequency": _proc_fs,
        "channels": _proc_channels,
        "meta": _proc_meta,
    }}

    qc_dir = work_dir / "preprocessed_output" / "QC_out" / "sub-{{}}".format(sub_id) / "ses-{{}}".format(ses)
    qc_dir.mkdir(parents=True, exist_ok=True)

    metrics = _compute_metrics(raw_d, proc)
    _write_report(qc_dir, "ses-" + ses, sub_id, raw_path, steps, metrics)

    return {{
        "file_id": file_id,
        "subject_id": sub_id,
        "session_id": ses,
        "grade": metrics["overall"]["grade"],
        "score": None,
    }}


def main():
    if len(sys.argv) >= 3:
        # Legacy form: qc.py <input_path> <work_dir>
        work_dir = Path(sys.argv[2])
    elif len(sys.argv) == 2:
        # Multi-input form: qc.py <work_dir>
        work_dir = Path(sys.argv[1])
    else:
        print("Usage: python qc.py <work_dir>  OR  python qc.py <input_path> <work_dir>", file=sys.stderr)
        sys.exit(2)

    inputs = _load_inputs(work_dir, sys.argv)
    steps = {steps_repr}

    aggregate = {{"inputs": [], "n_success": 0, "n_failed": 0}}
    for inp in inputs:
        try:
            status = _qc_one(work_dir, inp, steps)
            aggregate["inputs"].append(status)
            aggregate["n_success"] += 1
            mp = work_dir / "middle_process"
            mp.mkdir(parents=True, exist_ok=True)
            (mp / "qc_status__{{}}.json".format(status["file_id"])).write_text(
                json.dumps(status, indent=2), encoding="utf-8"
            )
        except Exception as exc:
            file_id = inp.get("file_id") or "legacy"
            err = {{
                "file_id": file_id,
                "subject_id": inp.get("subject_id"),
                "session_id": inp.get("session_id"),
                "grade": "Fail",
                "error": "{{}}: {{}}".format(type(exc).__name__, exc),
            }}
            aggregate["inputs"].append(err)
            aggregate["n_failed"] += 1
            print("[qc] FAILED for file_id={{}}: {{}}".format(file_id, exc), file=sys.stderr)

    mp = work_dir / "middle_process"
    mp.mkdir(parents=True, exist_ok=True)
    (mp / "qc_status.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")

    if aggregate["n_failed"] > 0 and aggregate["n_success"] == 0:
        sys.exit(1)


{nwb_loader_block}
{already_done_helper}
if __name__ == "__main__":
    main()
'''


def generate_qc_script_v2(
    *,
    steps: List[str],
    data_info: Dict[str, Any],
    modality: str = "eeg",
    analysis_goal: str = "generic",
    inspection_report: Dict[str, Any] | None = None,
) -> str:
    """Generate the canonical qc.py script for figures + QC report.

    When ``inspection_report`` is provided, a comment block is prepended so
    the QC reader can see why the pipeline used certain parameters.
    """
    steps_list = list(steps)
    if inspection_report:
        line_freq = (inspection_report.get("psd_summary") or {}).get("power_line_peak_hz")
        if line_freq:
            steps_list = _override_notch_freq(steps_list, line_freq)
    body = _QC_SCRIPT_TEMPLATE.format(
        steps_repr=repr(steps_list),
        analysis_goal=analysis_goal,
        modality=modality,
        nwb_loader_block=_NWB_PREPROCESSED_LOADER_BLOCK,
        already_done_helper=_ALREADY_DONE_HELPER_SRC,
    )
    hints = _format_inspection_hints(inspection_report)
    if hints:
        body = hints + body
    return body


# ── vis.py templates ───────────────────────────────────────────────────────
# Two flavours, gated by easybci_lib.tools.neural_processing.output.format_policy.is_invasive:
#   invasive (seeg/ecog/ieeg/dbs/spike/spikes/unit/units) → 4 single-state figures
#     (PSD / channel variance / amplitude distribution / timeseries). No raw reload.
#   non-invasive (eeg/meg/fnirs/...)                       → 4 single-state figures
#     + 1 before/after timeseries panel (loads raw via MNE / pickle / npz).
# Spec: improved_docs/plans/2026-06-22-nwb-visualization-rule-design.md


_VIS_INVASIVE_TEMPLATE = '''"""Auto-generated multi-figure visualization (invasive modality — 4 single-state figs).

EASYBCI_GOAL: {analysis_goal}
EASYBCI_MODALITY: {modality}
EASYBCI_VERSION: 2
EASYBCI_CODE_STANDARD: 1.1.0

Standalone script — runs on `pip install numpy scipy matplotlib pynwb hdmf`.

Run: python vis.py <work_dir>
  - reads preprocessed.nwb from <work_dir>/preprocessed_output/preprocessed/sub-<id>/ses-<ses>/
  - writes 4 figures to <work_dir>/preprocessed_output/figures/sub-<id>/ses-<ses>/
    (psd, channel_variance, amplitude_distribution, timeseries)

Per-figure failures do NOT abort the run — errors accumulate into
middle_process/vis_status.json and the script exits 0 so the chain continues.
"""

import json
import os as _os
import random as _random
import sys
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

EASYBCI_SEED = 42
_os.environ.setdefault("PYTHONHASHSEED", str(EASYBCI_SEED))
_random.seed(EASYBCI_SEED)
np.random.seed(EASYBCI_SEED)


def _plot_psd(data, fs, channels, out_path, *, n_ch_cap=8):
    n_ch = min(data.shape[0], n_ch_cap)
    if n_ch == 0:
        raise ValueError("PSD requires at least one channel")
    fig, ax = plt.subplots(figsize=(8, 4))
    nper = min(1024, data.shape[1])
    freqs, psd = welch(data[:n_ch], fs=fs, nperseg=nper, axis=1)
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, n_ch))
    for i in range(n_ch):
        label = channels[i] if i < len(channels) else "Ch{{}}".format(i)
        ax.semilogy(freqs, psd[i], color=colors[i], linewidth=1.2, label=label)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD")
    ax.set_title("Power Spectral Density (processed)")
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=120, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def _plot_channel_variance(data, channels, out_path):
    if data.shape[0] == 0:
        raise ValueError("Channel-variance plot needs at least one channel")
    var = np.var(data, axis=1)
    fig, ax = plt.subplots(figsize=(max(6.0, data.shape[0] * 0.3), 3))
    xs = np.arange(data.shape[0])
    labels = [channels[i] if i < len(channels) else "Ch{{}}".format(i)
              for i in range(data.shape[0])]
    ax.bar(xs, var, color="#3b82f6", alpha=0.85)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=75, fontsize=7)
    ax.set_ylabel("Variance")
    ax.set_title("Per-channel Variance (processed)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=120, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def _plot_amplitude_dist(data, out_path):
    flat = data.flatten()
    if flat.size == 0:
        raise ValueError("Amplitude distribution plot needs samples")
    if flat.size > 100000:
        flat = np.random.default_rng(EASYBCI_SEED).choice(flat, 100000, replace=False)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(flat, bins=80, color="#3b82f6", alpha=0.85, edgecolor="none")
    ax.set_xlabel("Amplitude")
    ax.set_ylabel("Count")
    ax.set_title("Amplitude Distribution (processed)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=120, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def _plot_timeseries(data, fs, channels, out_path, *, n_ch_cap=8, secs=5.0):
    n_ch = min(data.shape[0], n_ch_cap)
    if n_ch == 0:
        raise ValueError("Timeseries plot needs at least one channel")
    n_samp = min(data.shape[1], int(fs * secs))
    if n_samp == 0:
        raise ValueError("Not enough samples to render timeseries")
    sub = data[:n_ch, :n_samp]
    centered = sub - np.mean(sub, axis=1, keepdims=True)
    ptp = np.ptp(centered, axis=1)
    scale = float(np.median(ptp)) * 1.2 if np.median(ptp) > 0 else 1.0
    t = np.arange(n_samp) / fs
    fig, ax = plt.subplots(figsize=(10, max(3.0, n_ch * 0.4)))
    for i in range(n_ch):
        label = channels[i] if i < len(channels) else "Ch{{}}".format(i)
        color = plt.cm.viridis(i / max(n_ch, 1) * 0.8 + 0.1)
        ax.plot(t, centered[i] + i * scale, color=color, linewidth=0.5, label=label)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Channels (offset)")
    ax.set_title("Processed Signal - first {{:.1f}}s".format(secs))
    ax.legend(loc="upper right", fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=120, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def _vis_one(work_dir, inp):
    stem = inp["stem_safe"]
    sub_id = inp["subject_id"]
    ses = inp["session_id"]
    file_id = inp.get("file_id") or "legacy"

    fig_dir_check = (
        work_dir / "preprocessed_output" / "figures"
        / "sub-{{}}".format(sub_id) / "ses-{{}}".format(ses)
    )
    if fig_dir_check.is_dir():
        candidates = sorted(fig_dir_check.glob("{{}}_*.png".format(stem)))
        if candidates and _already_done(candidates[0], __file__):
            print("[vis] skip file_id={{}} (already processed)".format(file_id))
            return {{
                "file_id": file_id, "subject_id": sub_id, "session_id": ses,
                "ok": True, "figures": [p.name for p in candidates], "errors": [],
                "skipped": True,
            }}

    pre_nwb = (
        work_dir / "preprocessed_output" / "preprocessed"
        / "sub-{{}}".format(sub_id) / "ses-{{}}".format(ses)
        / "{{}}_preprocessed.nwb".format(stem)
    )
    if not pre_nwb.is_file():
        return {{
            "file_id": file_id, "ok": False,
            "errors": ["preprocessed nwb missing: {{}}".format(pre_nwb)],
            "figures": [],
        }}

    data, fs, _meta, channels = _load_preprocessed_nwb(pre_nwb)
    data = np.asarray(data, dtype=np.float32)

    fig_dir = (
        work_dir / "preprocessed_output" / "figures"
        / "sub-{{}}".format(sub_id) / "ses-{{}}".format(ses)
    )
    fig_dir.mkdir(parents=True, exist_ok=True)

    plots = [
        ("psd",       lambda p: _plot_psd(data, fs, channels, p)),
        ("variance",  lambda p: _plot_channel_variance(data, channels, p)),
        ("amplitude", lambda p: _plot_amplitude_dist(data, p)),
        ("timeseries", lambda p: _plot_timeseries(data, fs, channels, p)),
    ]
    saved, errors = [], []
    for name, fn in plots:
        out_path = fig_dir / "{{}}_{{}}.png".format(stem, name)
        try:
            fn(out_path)
            saved.append(out_path.name)
        except Exception as exc:
            errors.append("{{}}: {{}}".format(name, exc))
            print("[vis] {{}}: {{}} for file_id={{}}".format(name, exc, file_id), file=sys.stderr)
            traceback.print_exc()

    return {{
        "file_id": file_id, "subject_id": sub_id, "session_id": ses,
        "ok": bool(saved), "figures": saved, "errors": errors,
    }}


def _load_inputs(work_dir, argv):
    routing = work_dir / "middle_process" / "inputs_routing.json"
    if routing.is_file():
        try:
            data = json.loads(routing.read_text(encoding="utf-8"))
            inputs = data.get("inputs") or []
            if inputs:
                return inputs
        except Exception as exc:
            print("[vis] routing table unreadable ({{}}); falling back to argv".format(exc),
                  file=sys.stderr)
    if len(argv) < 2:
        print("Usage: python vis.py <work_dir>", file=sys.stderr)
        sys.exit(2)
    legacy_raw = argv[1]
    stem = Path(legacy_raw).stem.replace(" ", "_")
    base = work_dir / "preprocessed_output" / "preprocessed"
    sub_id, ses = stem, "001"
    if base.is_dir():
        for sd in sorted(base.glob("sub-*")):
            for sess_dir in sorted(sd.iterdir()):
                if sess_dir.is_dir() and any(sess_dir.glob(stem + "_preprocessed.nwb")):
                    sub_id = sd.name.replace("sub-", "")
                    ses = sess_dir.name.replace("ses-", "")
                    break
    return [{{
        "data_path": legacy_raw, "stem_safe": stem,
        "subject_id": sub_id, "session_id": ses, "file_id": "legacy",
    }}]


def main():
    argv = sys.argv
    if len(argv) < 2:
        print("Usage: python vis.py <work_dir>", file=sys.stderr)
        sys.exit(2)
    candidate = Path(argv[1])
    work_dir = candidate.resolve()

    inputs = _load_inputs(work_dir, argv)
    statuses = []
    for inp in inputs:
        try:
            statuses.append(_vis_one(work_dir, inp))
        except Exception as exc:
            print("[vis] FAILED for entry {{}}: {{}}".format(inp.get("file_id"), exc),
                  file=sys.stderr)
            traceback.print_exc()
            statuses.append({{"file_id": inp.get("file_id"), "ok": False,
                              "errors": [str(exc)], "figures": []}})

    mp = work_dir / "middle_process"
    mp.mkdir(parents=True, exist_ok=True)
    aggregate = {{"stage": "vis", "results": statuses,
                  "any_ok": any(s.get("ok") for s in statuses)}}
    (mp / "vis_status.json").write_text(
        json.dumps(aggregate, indent=2, default=str), encoding="utf-8")
    sys.exit(0)


{nwb_loader_block}
{already_done_helper}
if __name__ == "__main__":
    main()
'''


_VIS_NON_INVASIVE_TEMPLATE = '''"""Auto-generated multi-figure visualization (non-invasive modality — 4 single-state + before/after).

EASYBCI_GOAL: {analysis_goal}
EASYBCI_MODALITY: {modality}
EASYBCI_VERSION: 2
EASYBCI_CODE_STANDARD: 1.1.0

Standalone script — runs on `pip install numpy scipy matplotlib mne pynwb hdmf`.

Run: python vis.py <work_dir>
  - reads preprocessed.nwb from <work_dir>/preprocessed_output/preprocessed/sub-<id>/ses-<ses>/
  - reads the raw input file via routing entry's `data_path` (MNE / npz / pkl)
  - writes 5 figures to <work_dir>/preprocessed_output/figures/sub-<id>/ses-<ses>/
    (psd, channel_variance, amplitude_distribution, timeseries, before_after_timeseries)

Per-figure failures do NOT abort the run — errors accumulate into
middle_process/vis_status.json and the script exits 0 so the chain continues.
If the raw file cannot be loaded, only the before_after panel is skipped;
the other 4 single-state figs still save.
"""

import json
import os as _os
import pickle
import random as _random
import sys
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

EASYBCI_SEED = 42
_os.environ.setdefault("PYTHONHASHSEED", str(EASYBCI_SEED))
_random.seed(EASYBCI_SEED)
np.random.seed(EASYBCI_SEED)


_MNE_EXTS = {{
    ".fif", ".edf", ".bdf", ".set", ".ds", ".cnt", ".gdf",
    ".vhdr", ".vmrk", ".eeg", ".cdt", ".mff", ".sqd", ".con",
}}


def _load_raw(path):
    p = Path(path)
    ext = p.suffix.lower()
    if ext in _MNE_EXTS or (p.is_dir() and p.suffix == ".ds"):
        import mne
        raw = mne.io.read_raw(str(p), preload=True, verbose="ERROR")
        return {{"data": raw.get_data().astype(np.float32),
                 "frequency": float(raw.info["sfreq"]),
                 "channels": list(raw.ch_names)}}
    if ext in (".npz", ".npy"):
        if ext == ".npy":
            arr = np.asarray(np.load(str(p)), dtype=np.float32)
            if arr.ndim == 1:
                arr = arr[None, :]
            return {{"data": arr, "frequency": 1.0,
                     "channels": ["Ch{{}}".format(i) for i in range(arr.shape[0])]}}
        npz = np.load(str(p), allow_pickle=True)
        arr = np.asarray(npz["data"] if "data" in npz.files else npz[npz.files[0]], dtype=np.float32)
        fs = float(npz["frequency"]) if "frequency" in npz.files else 1.0
        ch = list(npz["channels"]) if "channels" in npz.files else ["Ch{{}}".format(i) for i in range(arr.shape[0])]
        return {{"data": arr, "frequency": fs, "channels": ch}}
    if ext in (".pkl", ".pickle"):
        with open(str(p), "rb") as f:
            d = pickle.load(f)
        return {{"data": np.asarray(d["data"], dtype=np.float32),
                 "frequency": float(d.get("frequency", 1.0)),
                 "channels": list(d.get("channels", []))}}
    raise ValueError("Unsupported raw format: {{}}".format(p))


def _plot_psd(data, fs, channels, out_path, *, n_ch_cap=8):
    n_ch = min(data.shape[0], n_ch_cap)
    if n_ch == 0:
        raise ValueError("PSD requires at least one channel")
    fig, ax = plt.subplots(figsize=(8, 4))
    nper = min(1024, data.shape[1])
    freqs, psd = welch(data[:n_ch], fs=fs, nperseg=nper, axis=1)
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, n_ch))
    for i in range(n_ch):
        label = channels[i] if i < len(channels) else "Ch{{}}".format(i)
        ax.semilogy(freqs, psd[i], color=colors[i], linewidth=1.2, label=label)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD")
    ax.set_title("Power Spectral Density (processed)")
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=120, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def _plot_channel_variance(data, channels, out_path):
    if data.shape[0] == 0:
        raise ValueError("Channel-variance plot needs at least one channel")
    var = np.var(data, axis=1)
    fig, ax = plt.subplots(figsize=(max(6.0, data.shape[0] * 0.3), 3))
    xs = np.arange(data.shape[0])
    labels = [channels[i] if i < len(channels) else "Ch{{}}".format(i)
              for i in range(data.shape[0])]
    ax.bar(xs, var, color="#3b82f6", alpha=0.85)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=75, fontsize=7)
    ax.set_ylabel("Variance")
    ax.set_title("Per-channel Variance (processed)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=120, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def _plot_amplitude_dist(data, out_path):
    flat = data.flatten()
    if flat.size == 0:
        raise ValueError("Amplitude distribution plot needs samples")
    if flat.size > 100000:
        flat = np.random.default_rng(EASYBCI_SEED).choice(flat, 100000, replace=False)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(flat, bins=80, color="#3b82f6", alpha=0.85, edgecolor="none")
    ax.set_xlabel("Amplitude")
    ax.set_ylabel("Count")
    ax.set_title("Amplitude Distribution (processed)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=120, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def _plot_timeseries(data, fs, channels, out_path, *, n_ch_cap=8, secs=5.0):
    n_ch = min(data.shape[0], n_ch_cap)
    if n_ch == 0:
        raise ValueError("Timeseries plot needs at least one channel")
    n_samp = min(data.shape[1], int(fs * secs))
    if n_samp == 0:
        raise ValueError("Not enough samples to render timeseries")
    sub = data[:n_ch, :n_samp]
    centered = sub - np.mean(sub, axis=1, keepdims=True)
    ptp = np.ptp(centered, axis=1)
    scale = float(np.median(ptp)) * 1.2 if np.median(ptp) > 0 else 1.0
    t = np.arange(n_samp) / fs
    fig, ax = plt.subplots(figsize=(10, max(3.0, n_ch * 0.4)))
    for i in range(n_ch):
        label = channels[i] if i < len(channels) else "Ch{{}}".format(i)
        color = plt.cm.viridis(i / max(n_ch, 1) * 0.8 + 0.1)
        ax.plot(t, centered[i] + i * scale, color=color, linewidth=0.5, label=label)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Channels (offset)")
    ax.set_title("Processed Signal - first {{:.1f}}s".format(secs))
    ax.legend(loc="upper right", fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=120, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def _plot_timeseries_before_after(raw_d, proc_d, out_path, *, n_ch_cap=8, secs=2.0):
    raw_data = np.asarray(raw_d["data"], dtype=np.float32)
    proc_data = np.asarray(proc_d["data"], dtype=np.float32)
    raw_fs = float(raw_d["frequency"])
    proc_fs = float(proc_d["frequency"])
    channels = list(raw_d.get("channels", []) or proc_d.get("channels", []))

    n_ch = min(raw_data.shape[0], proc_data.shape[0], n_ch_cap)
    if n_ch == 0:
        raise ValueError("before/after needs >=1 channel in common")

    b_samp = min(raw_data.shape[1], int(raw_fs * secs))
    a_samp = min(proc_data.shape[1], int(proc_fs * secs))
    if b_samp == 0 or a_samp == 0:
        raise ValueError("Not enough samples for before/after window")

    before = raw_data[:n_ch, :b_samp]
    after = proc_data[:n_ch, :a_samp]
    centered_b = before - np.mean(before, axis=1, keepdims=True)
    centered_a = after - np.mean(after, axis=1, keepdims=True)
    scale_b = float(np.median(np.ptp(centered_b, axis=1))) * 1.2 or 1.0
    scale_a = float(np.median(np.ptp(centered_a, axis=1))) * 1.2 or 1.0
    t_b = np.arange(b_samp) / raw_fs
    t_a = np.arange(a_samp) / proc_fs

    panel_h = max(3.0, n_ch * 0.35)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, panel_h * 2 + 1.5))
    for i in range(n_ch):
        ax1.plot(t_b, centered_b[i] + i * scale_b, color="#3b82f6",
                 linewidth=0.4, alpha=0.85)
        ax2.plot(t_a, centered_a[i] + i * scale_a, color="#000000",
                 linewidth=0.4, alpha=0.85)
    ax1.set_title("BEFORE preprocessing (raw)", fontweight="bold", color="#1e40af")
    ax1.set_ylabel("Channels")
    ax1.set_xlim(0, secs)
    ax2.set_title("AFTER preprocessing", fontweight="bold", color="#000000")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Channels")
    ax2.set_xlim(0, secs)
    fig.suptitle("Before vs After: first {{:.1f}}s, first {{}} channels".format(secs, n_ch),
                 fontweight="bold", y=1.005)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=120, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def _vis_one(work_dir, inp):
    stem = inp["stem_safe"]
    sub_id = inp["subject_id"]
    ses = inp["session_id"]
    file_id = inp.get("file_id") or "legacy"
    raw_path = inp.get("data_path")

    fig_dir_check = (
        work_dir / "preprocessed_output" / "figures"
        / "sub-{{}}".format(sub_id) / "ses-{{}}".format(ses)
    )
    if fig_dir_check.is_dir():
        candidates = sorted(fig_dir_check.glob("{{}}_*.png".format(stem)))
        if candidates and _already_done(candidates[0], __file__):
            print("[vis] skip file_id={{}} (already processed)".format(file_id))
            return {{
                "file_id": file_id, "subject_id": sub_id, "session_id": ses,
                "ok": True, "figures": [p.name for p in candidates], "errors": [],
                "skipped": True,
            }}

    pre_nwb = (
        work_dir / "preprocessed_output" / "preprocessed"
        / "sub-{{}}".format(sub_id) / "ses-{{}}".format(ses)
        / "{{}}_preprocessed.nwb".format(stem)
    )
    if not pre_nwb.is_file():
        return {{
            "file_id": file_id, "ok": False,
            "errors": ["preprocessed nwb missing: {{}}".format(pre_nwb)],
            "figures": [],
        }}

    data, fs, _meta, channels = _load_preprocessed_nwb(pre_nwb)
    data = np.asarray(data, dtype=np.float32)

    fig_dir = (
        work_dir / "preprocessed_output" / "figures"
        / "sub-{{}}".format(sub_id) / "ses-{{}}".format(ses)
    )
    fig_dir.mkdir(parents=True, exist_ok=True)

    plots = [
        ("psd",       lambda p: _plot_psd(data, fs, channels, p)),
        ("variance",  lambda p: _plot_channel_variance(data, channels, p)),
        ("amplitude", lambda p: _plot_amplitude_dist(data, p)),
        ("timeseries", lambda p: _plot_timeseries(data, fs, channels, p)),
    ]
    saved, errors = [], []
    for name, fn in plots:
        out_path = fig_dir / "{{}}_{{}}.png".format(stem, name)
        try:
            fn(out_path)
            saved.append(out_path.name)
        except Exception as exc:
            errors.append("{{}}: {{}}".format(name, exc))
            print("[vis] {{}}: {{}} for file_id={{}}".format(name, exc, file_id), file=sys.stderr)
            traceback.print_exc()

    # 5th plot: before/after timeseries — needs raw load. Failure here only
    # drops this one figure; the 4 above are already saved.
    ba_out = fig_dir / "{{}}_before_after_timeseries.png".format(stem)
    try:
        if not raw_path:
            raise ValueError("routing entry has no data_path; before/after skipped")
        raw_d = _load_raw(raw_path)
        proc_d = {{"data": data, "frequency": fs, "channels": channels}}
        _plot_timeseries_before_after(raw_d, proc_d, ba_out)
        saved.append(ba_out.name)
    except Exception as exc:
        errors.append("before_after_timeseries: {{}}".format(exc))
        print("[vis] before_after_timeseries: {{}} for file_id={{}}".format(exc, file_id), file=sys.stderr)
        traceback.print_exc()

    return {{
        "file_id": file_id, "subject_id": sub_id, "session_id": ses,
        "ok": bool(saved), "figures": saved, "errors": errors,
    }}


def _load_inputs(work_dir, argv):
    routing = work_dir / "middle_process" / "inputs_routing.json"
    if routing.is_file():
        try:
            data = json.loads(routing.read_text(encoding="utf-8"))
            inputs = data.get("inputs") or []
            if inputs:
                return inputs
        except Exception as exc:
            print("[vis] routing table unreadable ({{}}); falling back to argv".format(exc),
                  file=sys.stderr)
    if len(argv) < 2:
        print("Usage: python vis.py <work_dir>", file=sys.stderr)
        sys.exit(2)
    legacy_raw = argv[1]
    stem = Path(legacy_raw).stem.replace(" ", "_")
    base = work_dir / "preprocessed_output" / "preprocessed"
    sub_id, ses = stem, "001"
    if base.is_dir():
        for sd in sorted(base.glob("sub-*")):
            for sess_dir in sorted(sd.iterdir()):
                if sess_dir.is_dir() and any(sess_dir.glob(stem + "_preprocessed.nwb")):
                    sub_id = sd.name.replace("sub-", "")
                    ses = sess_dir.name.replace("ses-", "")
                    break
    return [{{
        "data_path": legacy_raw, "stem_safe": stem,
        "subject_id": sub_id, "session_id": ses, "file_id": "legacy",
    }}]


def main():
    argv = sys.argv
    if len(argv) < 2:
        print("Usage: python vis.py <work_dir>", file=sys.stderr)
        sys.exit(2)
    candidate = Path(argv[1])
    work_dir = candidate.resolve()

    inputs = _load_inputs(work_dir, argv)
    statuses = []
    for inp in inputs:
        try:
            statuses.append(_vis_one(work_dir, inp))
        except Exception as exc:
            print("[vis] FAILED for entry {{}}: {{}}".format(inp.get("file_id"), exc),
                  file=sys.stderr)
            traceback.print_exc()
            statuses.append({{"file_id": inp.get("file_id"), "ok": False,
                              "errors": [str(exc)], "figures": []}})

    mp = work_dir / "middle_process"
    mp.mkdir(parents=True, exist_ok=True)
    aggregate = {{"stage": "vis", "results": statuses,
                  "any_ok": any(s.get("ok") for s in statuses)}}
    (mp / "vis_status.json").write_text(
        json.dumps(aggregate, indent=2, default=str), encoding="utf-8")
    sys.exit(0)


{nwb_loader_block}
{already_done_helper}
if __name__ == "__main__":
    main()
'''


def generate_vis_script(
    *,
    modality: str = "eeg",
    analysis_goal: str = "generic",
    inspection_report: Dict[str, Any] | None = None,
) -> str:
    """Generate the canonical vis.py script.

    Selection rule (single source of truth: format_policy.is_invasive):
      - invasive modalities (seeg/ecog/ieeg/dbs/spike/spikes/unit/units) →
        _VIS_INVASIVE_TEMPLATE (4 single-state figures, no raw reload).
      - non-invasive modalities (eeg/meg/fnirs/...) →
        _VIS_NON_INVASIVE_TEMPLATE (the same 4 figs + a 5th before/after
        timeseries panel that loads raw via MNE / pickle / npz).

    Empty / unknown modality → non-invasive (matches the safe-default in
    format_policy.is_invasive — "uncertain → not invasive").
    """
    template = _VIS_INVASIVE_TEMPLATE if is_invasive(modality) else _VIS_NON_INVASIVE_TEMPLATE
    body = template.format(
        analysis_goal=analysis_goal,
        modality=modality,
        nwb_loader_block=_NWB_PREPROCESSED_LOADER_BLOCK,
        already_done_helper=_ALREADY_DONE_HELPER_SRC,
    )
    hints = _format_inspection_hints(inspection_report) if inspection_report else ""
    if hints:
        body = hints + body
    return body


_AI_READY_TEMPLATE = '''"""Auto-generated AI_ready epoching.

EASYBCI_GOAL: {analysis_goal}
EASYBCI_VERSION: 2

Run: python build_ai_ready.py <preprocessed_nwb> <work_dir>

Reads the preprocessed NWB file written by pipeline.py and writes per-input
epoch tensors into ``AI_ready/<sub>/ses-<ses>/<stem>_epochs.pkl``. AI_ready
output stays pickle for downstream ML loader compatibility — the input side
(preprocessed/) is NWB-only since the format unification.
"""

import json
import os as _os
import pickle
import random as _random
import sys
from pathlib import Path

import numpy as np

EASYBCI_SEED = 42
_os.environ.setdefault("PYTHONHASHSEED", str(EASYBCI_SEED))
_random.seed(EASYBCI_SEED)
np.random.seed(EASYBCI_SEED)

SEGMENT_DURATION = {segment_duration}
STRIDE = {stride}
EVENTS_PRESENT = {events_present}
LABEL_CONFIG = {label_config_repr}


def _sliding(data, fs, dur, stride):
    win = int(dur * fs)
    step = max(int(stride * fs), 1)
    n_ch, n = data.shape
    segs = []
    start = 0
    while start + win <= n:
        segs.append(data[:, start:start + win])
        start += step
    if not segs:
        segs.append(data[:, :n])
    return np.asarray(segs, dtype=np.float32)


def _epoch_from_events(data, fs, events, tmin, tmax):
    win_samples = int((tmax - tmin) * fs)
    pre = int(-tmin * fs)
    epochs = []
    for ev in events:
        s = int(ev["start"]) - pre
        e = s + win_samples
        if 0 <= s and e <= data.shape[1]:
            epochs.append(data[:, s:e])
    return np.asarray(epochs, dtype=np.float32)


def _load_inputs(work_dir, argv):
    """Multi-input: read inputs_routing.json. Legacy: single argv-driven entry."""
    routing = work_dir / "middle_process" / "inputs_routing.json"
    if routing.is_file():
        try:
            data = json.loads(routing.read_text(encoding="utf-8"))
            inputs = data.get("inputs") or []
            if inputs:
                return inputs
        except Exception as exc:
            print("[ai_ready] routing table unreadable ({{}}); falling back to argv".format(exc), file=sys.stderr)

    # Legacy: argv[1] is a preprocessed_nwb path. Recover (sub, ses) from the
    # path itself — this is the only path where stem-derivation is allowed
    # (single-file, no routing table).
    if len(argv) < 2:
        print("Usage: python build_ai_ready.py <work_dir>  OR  python build_ai_ready.py <preprocessed_nwb> <work_dir>", file=sys.stderr)
        sys.exit(2)
    pre_nwb = Path(argv[1])
    if not pre_nwb.is_file():
        # auto-discover under work_dir/preprocessed_output/preprocessed/
        base = work_dir / "preprocessed_output" / "preprocessed"
        if base.is_dir():
            for sub_dir in sorted(base.glob("sub-*")):
                for sess_dir in sorted(sub_dir.iterdir()):
                    cands = sorted(sess_dir.glob("*_preprocessed.nwb"))
                    if cands:
                        pre_nwb = cands[0]
                        break
                if pre_nwb.is_file():
                    break
    if not pre_nwb.is_file():
        print("[ai_ready] no preprocessed nwb found", file=sys.stderr)
        sys.exit(3)

    parts = pre_nwb.parts
    sub_part = next((p for p in parts if p.startswith("sub-")), "sub-unknown")
    ses_part = parts[parts.index(sub_part) + 1] if sub_part in parts else "ses-001"
    sub_id = sub_part.replace("sub-", "")
    ses = ses_part.replace("ses-", "")
    stem = pre_nwb.stem.replace("_preprocessed", "")

    return [{{
        "data_path": str(pre_nwb),     # legacy path uses the .nwb directly
        "stem_safe": stem,
        "subject_id": sub_id,
        "session_id": ses,
        "file_id": "legacy",
        "events_path": None,
    }}]


def _load_preprocessed(pre_nwb):
    """Read the preprocessed NWB file written by pipeline.py.

    Returns (data, fs, meta, channels) — same shape the pickle-era code
    expected. preprocessed/ is NWB-only since the format unification.
    """
    pre_nwb = Path(pre_nwb)
    return _load_preprocessed_nwb(pre_nwb)


def _ai_ready_one(work_dir, inp):
    sub_id = inp["subject_id"]
    ses = inp["session_id"]
    stem = inp["stem_safe"]
    file_id = inp.get("file_id") or "legacy"

    # AI_ready uses bare subject_id (no "sub-" prefix) — ML convention.
    epochs_path = (
        work_dir / "preprocessed_output" / "AI_ready"
        / sub_id / "ses-{{}}".format(ses) / "{{}}_epochs.pkl".format(stem)
    )
    if _already_done(epochs_path, __file__):
        print("[ai_ready] skip file_id={{}} (already processed)".format(file_id))
        return {{
            "file_id": file_id, "subject_id": sub_id, "session_id": ses,
            "n_epochs": 0, "n_classes": None, "output_file": str(epochs_path),
            "skipped": True,
        }}

    pre_nwb = (
        work_dir / "preprocessed_output" / "preprocessed"
        / "sub-{{}}".format(sub_id) / "ses-{{}}".format(ses)
        / "{{}}_preprocessed.nwb".format(stem)
    )
    if not pre_nwb.is_file():
        # Legacy entry may have provided the nwb path directly via data_path
        legacy = Path(inp.get("data_path") or "")
        if legacy.is_file() and legacy.suffix.lower() == ".nwb":
            pre_nwb = legacy
        else:
            print("[ai_ready] preprocessed missing for file_id={{}}: {{}}".format(file_id, pre_nwb), file=sys.stderr)
            raise FileNotFoundError(pre_nwb)

    data, fs, meta, channels = _load_preprocessed(pre_nwb)

    # Events: prefer routing entry's events_path, fall back to meta.events.
    events = None
    events_path = inp.get("events_path")
    if events_path and Path(events_path).is_file():
        events = _load_events_csv(events_path)
    if not events and isinstance(meta, dict):
        events = meta.get("events") or None

    if EVENTS_PRESENT and events:
        epochs = _epoch_from_events(data, fs, events, tmin=-0.2, tmax=SEGMENT_DURATION)
        labels = np.asarray([e.get("label", 0) for e in events])
    else:
        epochs = _sliding(data, fs, SEGMENT_DURATION, STRIDE)
        labels = None
        if LABEL_CONFIG and LABEL_CONFIG.get("label_type") == "L3_continuous":
            lp = LABEL_CONFIG.get("label_path", "labels.npy")
            try:
                labels = np.load(lp)
            except OSError:
                labels = None

    # AI_ready uses bare subject_id (no "sub-" prefix) — ML convention.
    out_dir = work_dir / "preprocessed_output" / "AI_ready" / sub_id / "ses-{{}}".format(ses)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "{{}}_epochs.pkl".format(stem)

    payload = {{
        "epochs": epochs,
        "labels": labels,
        "frequency": fs,
        "channels": channels,
    }}
    with open(out_file, "wb") as f:
        pickle.dump(payload, f)

    n_classes = None
    if labels is not None and getattr(labels, "ndim", 0) >= 1:
        try:
            n_classes = int(len(set(labels.tolist())))
        except (AttributeError, TypeError):
            n_classes = None

    return {{
        "file_id": file_id,
        "subject_id": sub_id,
        "session_id": ses,
        "n_epochs": int(epochs.shape[0]),
        "n_classes": n_classes,
        "output_file": str(out_file),
    }}


def _load_events_csv(csv_path):
    import csv as _csv
    out = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            try:
                start = int(float(row.get("sample_index") or 0))
            except (TypeError, ValueError):
                start = 0
            code = row.get("event_code", "").strip()
            out.append({{
                "start": start,
                "code": int(code) if code.isdigit() else code,
                "label": int(code) if code.isdigit() else code,
            }})
    return out


def main():
    if len(sys.argv) >= 3:
        work_dir = Path(sys.argv[2])
    elif len(sys.argv) == 2:
        work_dir = Path(sys.argv[1])
    else:
        print("Usage: python build_ai_ready.py <work_dir>  OR  python build_ai_ready.py <preprocessed_pkl> <work_dir>", file=sys.stderr)
        sys.exit(2)

    inputs = _load_inputs(work_dir, sys.argv)
    aggregate = {{"inputs": [], "n_success": 0, "n_failed": 0}}
    for inp in inputs:
        try:
            status = _ai_ready_one(work_dir, inp)
            aggregate["inputs"].append(status)
            aggregate["n_success"] += 1
            mp = work_dir / "middle_process"
            mp.mkdir(parents=True, exist_ok=True)
            (mp / "build_ai_ready_status__{{}}.json".format(status["file_id"])).write_text(
                json.dumps(status, indent=2, default=str), encoding="utf-8"
            )
        except Exception as exc:
            file_id = inp.get("file_id") or "legacy"
            err = {{
                "file_id": file_id,
                "subject_id": inp.get("subject_id"),
                "session_id": inp.get("session_id"),
                "error": "{{}}: {{}}".format(type(exc).__name__, exc),
            }}
            aggregate["inputs"].append(err)
            aggregate["n_failed"] += 1
            print("[ai_ready] FAILED for file_id={{}}: {{}}".format(file_id, exc), file=sys.stderr)

    mp = work_dir / "middle_process"
    mp.mkdir(parents=True, exist_ok=True)
    (mp / "build_ai_ready_status.json").write_text(json.dumps(aggregate, indent=2, default=str), encoding="utf-8")
    # Compatibility alias — older readers look for ai_ready_status.json
    (mp / "ai_ready_status.json").write_text(json.dumps(aggregate, indent=2, default=str), encoding="utf-8")

    if aggregate["n_failed"] > 0 and aggregate["n_success"] == 0:
        sys.exit(1)


{nwb_loader_block}
{already_done_helper}
if __name__ == "__main__":
    main()
'''


def generate_build_ai_ready_script(
    *,
    modality: str = "eeg",
    analysis_goal: str = "generic",
    events_present: bool = False,
    label_config: Optional[Dict[str, Any]] = None,
    segment_duration: float = 2.0,
    stride: float = 1.0,
) -> str:
    """Generate the canonical build_ai_ready.py script."""
    return _AI_READY_TEMPLATE.format(
        analysis_goal=analysis_goal,
        segment_duration=segment_duration,
        stride=stride,
        events_present=repr(bool(events_present)),
        label_config_repr=repr(label_config or {}),
        nwb_loader_block=_NWB_PREPROCESSED_LOADER_BLOCK,
        already_done_helper=_ALREADY_DONE_HELPER_SRC,
    )


def generate_run_script_v2(*, has_build_ai_ready: bool, has_vis: bool = True) -> str:
    """One-click wrapper that chains pipeline → [build_ai_ready] → qc → [vis].

    Each stage runs as subprocess. Fail-fast on non-zero retcode. vis is the
    final, non-critical step — visualization failure is non-fatal at the
    vis.py level (the script exits 0 unless ALL figures failed), but if it
    does return non-zero we still surface the retcode.

    Usage: python run.py [<work_dir>]   — defaults to parent of code/.

    Multi-input runs: each stage reads ``middle_process/inputs_routing.json``
    and loops internally; this wrapper invokes each stage script ONCE with
    only the work_dir argument.

    Legacy single-input runs: pass ``python run.py <raw_path> [<work_dir>]``
    — the wrapper forwards the raw path to pipeline.py / qc.py legacy mode.
    """
    stages = ["pipeline"]
    if has_build_ai_ready:
        stages.append("build_ai_ready")
    stages.append("qc")
    if has_vis:
        stages.append("vis")
    stages_repr = repr(stages)
    scripts_repr = repr([f"{s}.py" for s in stages])
    chain_desc = " -> ".join(stages)
    return f'''"""One-click wrapper. Runs {chain_desc}.

EASYBCI_CODE_STANDARD: 1.1.0
Conformance verified by easybci_lib/tools/neural_processing/codegen/code_standard_check.py
at codegen time.
"""

import shlex
import subprocess
import sys
from pathlib import Path

STAGES = {stages_repr}
SCRIPTS = {scripts_repr}


def main():
    args = sys.argv[1:]
    raw = None
    work_dir = None

    # Detect form: `run.py <work_dir>` (multi) vs `run.py <raw> [<work_dir>]` (legacy)
    if len(args) == 0:
        # Default: parent of code/
        work_dir = str(Path(__file__).resolve().parent.parent)
    elif len(args) == 1:
        # Either work_dir (multi) or raw_path (legacy with implicit work_dir).
        candidate = Path(args[0])
        if candidate.is_dir() and (candidate / "code").is_dir():
            work_dir = str(candidate)
        else:
            raw = args[0]
            work_dir = str(Path(__file__).resolve().parent.parent)
    else:
        raw = args[0]
        work_dir = args[1]

    code_dir = Path(__file__).resolve().parent
    routing = Path(work_dir) / "middle_process" / "inputs_routing.json"
    multi = routing.is_file() and raw is None

    for stage, script_name in zip(STAGES, SCRIPTS):
        script = code_dir / script_name
        if multi:
            cmd = [sys.executable, str(script), work_dir]
        else:
            # Legacy: pass raw path. build_ai_ready / vis legacy only need work_dir.
            if stage in ("build_ai_ready", "vis"):
                cmd = [sys.executable, str(script), work_dir]
            else:
                cmd = [sys.executable, str(script), raw, work_dir]
        print("$", shlex.join(cmd))
        rc = subprocess.call(cmd)
        if rc != 0:
            print("[run] {{}} failed with retcode {{}}".format(stage, rc), file=sys.stderr)
            sys.exit(rc)


if __name__ == "__main__":
    main()
'''



def cleanup_was_appended(before: List[Any], after: List[Any]) -> bool:
    """True iff ``_enforce_clean_output(before) == after`` modified the list.

    Used by reasoning_writer / repo_builder to render the "output cleanup"
    note ONLY when this hook actually changed the pipeline.
    """
    if len(before) != len(after):
        return True
    for a, b in zip(before, after):
        if isinstance(a, str) and isinstance(b, str):
            if a != b:
                return True
        else:
            if a != b:
                return True
    return False


def generate_split_code(
    split_config: Dict[str, Any],
    n_segments_estimate: int = 0,
) -> str:
    """Generate a standalone split.py file for the mini-repo.

    Parameters
    ----------
    split_config : dict
        Split configuration with keys:
        - method: str (random, temporal, group_kfold, loso, hash, sequential)
        - ratios: dict (e.g., {"train": 0.7, "val": 0.15, "test": 0.15})
        - params: dict (method-specific parameters)
        - rationale: str (explanation for the choice)
    n_segments_estimate : int
        Estimated number of segments (for documentation).

    Returns
    -------
    str : Complete Python source code for split.py
    """
    method = split_config.get("method", "random")
    ratios = split_config.get("ratios", {"train": 0.7, "val": 0.15, "test": 0.15})
    params = split_config.get("params", {})
    rationale = split_config.get("rationale", "")

    ratios_repr = repr(ratios)
    temporal_gap = params.get("temporal_gap", 0)
    n_folds = params.get("n_folds", 5)
    seed = params.get("seed", 42)
    use_stratify = params.get("stratify", False)

    code = f'''"""Data split assignment — train/val/test partitioning.

Method: {method}
Ratios: {ratios}
Rationale: {rationale}

This script loads processed segments and assigns each to a split.
Outputs: splits/ directory with train.npz, val.npz, test.npz (or per-fold).

Usage:
    python split.py <processed_data.pkl> [output_dir]
"""

import sys
import pickle
import json
from pathlib import Path

import numpy as np
import os as _os
import random as _random

# --- Reproducibility lock (EasyBCI) ------------------------------------------
EASYBCI_SEED = 42
_os.environ.setdefault("PYTHONHASHSEED", str(EASYBCI_SEED))
_random.seed(EASYBCI_SEED)
np.random.seed(EASYBCI_SEED)
# -----------------------------------------------------------------------------


def split_data(
    n_items: int,
    method: str = "{method}",
    ratios: dict = None,
    groups: list = None,
    stratify: list = None,
    seed: int = {seed},
    temporal_gap: int = {temporal_gap},
    n_folds: int = {n_folds},
) -> np.ndarray:
    """Assign items to splits.

    Returns array of split labels, one per item.
    """
    if ratios is None:
        ratios = {ratios_repr}

    n = n_items

    if method == "random":
        return _random_split(n, ratios, groups, stratify, seed)
    elif method == "sequential":
        return _sequential_split(n, ratios)
    elif method == "temporal":
        return _temporal_split(n, ratios, temporal_gap)
    elif method == "group_kfold":
        return _group_kfold_split(n, groups, n_folds, seed)
    elif method == "loso":
        return _loso_split(n, groups)
    elif method == "hash":
        return _hash_split(n, ratios, groups, seed)
    else:
        print(f"  Unknown method '{{method}}' — using random")
        return _random_split(n, ratios, groups, stratify, seed)


def _random_split(n, ratios, groups, stratify, seed):
    from sklearn.model_selection import train_test_split
    result = np.empty(n, dtype=object)
    split_names = list(ratios.keys())
    indices = np.arange(n)

    if len(split_names) == 1:
        result[:] = split_names[0]
        return result

    test_ratio = ratios[split_names[-1]]
    strat_arr = np.array(stratify) if stratify else None

    idx_rest, idx_test = train_test_split(
        indices, test_size=test_ratio, random_state=seed,
        stratify=strat_arr,
    )
    result[idx_test] = split_names[-1]

    if len(split_names) == 2:
        result[idx_rest] = split_names[0]
        return result

    val_ratio = ratios[split_names[1]] / (1.0 - test_ratio)
    strat_rest = strat_arr[idx_rest] if strat_arr is not None else None
    idx_train, idx_val = train_test_split(
        idx_rest, test_size=val_ratio, random_state=seed,
        stratify=strat_rest,
    )
    result[idx_train] = split_names[0]
    result[idx_val] = split_names[1]
    return result


def _sequential_split(n, ratios):
    result = np.empty(n, dtype=object)
    cumulative = np.cumsum(list(ratios.values()))
    names = list(ratios.keys())
    for i in range(n):
        pos = i / n
        for j, threshold in enumerate(cumulative):
            if pos < threshold:
                result[i] = names[j]
                break
        else:
            result[i] = names[-1]
    return result


def _temporal_split(n, ratios, gap):
    result = np.empty(n, dtype=object)
    split_names = list(ratios.keys())
    n_gaps = len(split_names) - 1
    total_gap = gap * n_gaps
    usable = max(n - total_gap, n)

    cumulative = np.cumsum([ratios[s] for s in split_names])
    boundaries = [int(r * usable) for r in cumulative]
    boundaries[-1] = usable

    pos = 0
    for split_idx, name in enumerate(split_names):
        n_in_split = boundaries[split_idx] - (boundaries[split_idx - 1] if split_idx > 0 else 0)
        for _ in range(n_in_split):
            if pos < n:
                result[pos] = name
                pos += 1
        if split_idx < len(split_names) - 1 and gap > 0:
            for _ in range(gap):
                if pos < n:
                    result[pos] = "gap"
                    pos += 1
    while pos < n:
        result[pos] = split_names[-1]
        pos += 1
    return result


def _group_kfold_split(n, groups, n_folds, seed):
    result = np.empty(n, dtype=object)
    if groups is None:
        rng = np.random.RandomState(seed)
        for i in range(n):
            result[i] = f"fold_{{rng.randint(0, n_folds)}}"
        return result

    unique_groups = sorted(set(groups))
    rng = np.random.RandomState(seed)
    order = list(range(len(unique_groups)))
    rng.shuffle(order)
    group_to_fold = {{unique_groups[g]: idx % n_folds for idx, g in enumerate(order)}}
    for i, g in enumerate(groups):
        result[i] = f"fold_{{group_to_fold[g]}}"
    return result


def _loso_split(n, groups):
    result = np.empty(n, dtype=object)
    if groups is None:
        for i in range(n):
            result[i] = f"fold_{{i}}"
        return result
    for i, g in enumerate(groups):
        result[i] = f"fold_{{g}}"
    return result


def _hash_split(n, ratios, groups, seed):
    import hashlib
    if groups is None:
        groups = [str(i) for i in range(n)]
    split_names = list(ratios.keys())
    cumulative = np.cumsum([ratios[s] for s in split_names])
    result = np.empty(n, dtype=object)
    for i, group in enumerate(groups):
        h = hashlib.sha256(f"{{seed}}:{{group}}".encode()).hexdigest()
        val = int(h[:8], 16) / 0xFFFFFFFF
        for j, threshold in enumerate(cumulative):
            if val <= threshold:
                result[i] = split_names[j]
                break
        else:
            result[i] = split_names[-1]
    return result


def save_splits(segments, split_assignments, output_dir, labels=None):
    """Save split data to separate files."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    unique_splits = sorted(set(split_assignments))
    manifest = {{"n_total": len(segments), "splits": {{}}}}

    for split_name in unique_splits:
        if split_name == "gap":
            continue
        mask = split_assignments == split_name
        split_data = segments[mask]
        split_path = out / f"{{split_name}}.npz"

        save_dict = {{"data": split_data}}
        if labels is not None:
            split_labels = labels[mask] if hasattr(labels, "__getitem__") else None
            if split_labels is not None:
                save_dict["labels"] = split_labels

        np.savez_compressed(str(split_path), **save_dict)
        manifest["splits"][split_name] = {{
            "n_items": int(mask.sum()),
            "shape": list(split_data.shape),
            "path": str(split_path.name),
        }}
        print(f"  {{split_name}}: {{mask.sum()}} items → {{split_path.name}}")

    # Save manifest
    manifest_path = out / "split_manifest.json"
    with open(str(manifest_path), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def main():
    if len(sys.argv) < 2:
        print("Usage: python split.py <processed_data.pkl> [output_dir]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "splits"

    print(f"Loading: {{input_path}}")
    ext = Path(input_path).suffix.lower()

    if ext in (".pkl",):
        with open(input_path, "rb") as f:
            loaded = pickle.load(f)
        if isinstance(loaded, dict):
            # Find the data array
            for key in ("data", "segments"):
                if key in loaded and isinstance(loaded[key], (np.ndarray, dict)):
                    if isinstance(loaded[key], dict):
                        segments = list(loaded[key].values())[0]
                    else:
                        segments = loaded[key]
                    break
            else:
                segments = np.array([])
            labels = loaded.get("labels")
            if isinstance(labels, dict):
                labels = list(labels.values())[0] if labels else None
        else:
            segments = loaded
            labels = None
    elif ext == ".npz":
        npz = np.load(input_path, allow_pickle=True)
        segments = npz[npz.files[0]]
        labels = npz[npz.files[1]] if len(npz.files) > 1 else None
    else:
        print(f"  Unsupported format: {{ext}}")
        sys.exit(1)

    n_items = segments.shape[0] if hasattr(segments, "shape") else len(segments)
    print(f"  Items: {{n_items}}, Shape: {{segments.shape if hasattr(segments, 'shape') else '?'}}")

    print(f"Splitting (method={method})...")
    assignments = split_data(n_items)

    unique, counts = np.unique(assignments, return_counts=True)
    for name, count in zip(unique, counts):
        print(f"  {{name}}: {{count}} items ({{count/n_items*100:.1f}}%)")

    print(f"Saving to {{output_dir}}/...")
    save_splits(segments, assignments, output_dir, labels=labels)
    print("Done.")


if __name__ == "__main__":
    main()
'''
    return code


def generate_config_yaml(
    steps: List[str],
    modality: str = "eeg",
    segment_duration: float = 2.0,
    stride: float = 1.0,
    subject_id: str = "",
    paradigm: str = "",
    output_format: str = "pkl",
    split_config: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate config.yaml content.

    Pure hyperparameter speed-dial: ``preprocessing`` / ``segmentation`` /
    ``modality`` / ``subject_id`` / ``paradigm`` / ``output_format``. Path
    routing (input file + per-bucket output destination) is NOT carried here
    — it lives in ``<work_dir>/middle_process/inputs_routing.json``, which is
    the single source of truth read by ``pipeline.py`` / ``qc.py`` /
    ``build_ai_ready.py`` / ``run.py``. Carrying paths in config.yaml was
    the documented root cause of the multi-session "input=1623 ↔
    output=ses-1842" collision; removing them eliminates the trap.
    """
    from easybci_lib.tools.neural_processing.preprocess.pipeline import STEP_FULL_NAMES

    step_lines = []
    for s in steps:
        step_name = s.split(":")[0]
        full_name = STEP_FULL_NAMES.get(step_name, "")
        if full_name:
            step_lines.append(f"  - {s}  # {full_name}")
        else:
            step_lines.append(f"  - {s}")
    steps_yaml = "\n".join(step_lines)

    split_section = ""
    if split_config:
        method = split_config.get("method", "random")
        ratios = split_config.get("ratios", {})
        params = split_config.get("params", {})
        ratios_lines = "\n".join(f"    {k}: {v}" for k, v in ratios.items())
        params_lines = "\n".join(f"    {k}: {v}" for k, v in params.items())
        split_section = f"""
split:
  method: {method}
  ratios:
{ratios_lines}"""
        if params_lines:
            split_section += f"""
  params:
{params_lines}"""
        split_section += "\n"

    return f"""# EasyBCIdata pipeline configuration
# Generated: {time.strftime("%Y-%m-%dT%H:%M:%S")}
#
# Input/output routing is NOT stored here. The single source of truth is
# <work_dir>/middle_process/inputs_routing.json, written by deep_inspect and
# read by pipeline.py / qc.py / build_ai_ready.py / run.py. This file holds
# hyperparameters only — edit them and re-run `python run.py` to re-execute
# every input in the routing table with the new settings.

output_format: {output_format}  # pkl | nwb

modality: {modality}
subject_id: "{subject_id}"
paradigm: "{paradigm}"

preprocessing:
{steps_yaml}

segmentation:
  duration: {segment_duration}
  stride: {stride}
{split_section}
# Modify parameters above and re-run: python run.py
"""


def generate_visualize_script() -> str:
    """REMOVED — superseded by generate_vis_script (vis.py refactor).

    This function previously emitted a before/after visualize.py that was
    never written to disk (no caller in repo_builder or _handle_generate_code).
    Kept as a stub raising NotImplementedError so any latent external caller
    surfaces immediately instead of getting empty figures.
    """
    raise NotImplementedError(
        "generate_visualize_script was removed; use generate_vis_script "
        "(emits code/vis.py with the 4-figure continuous template or the "
        "single-figure spike template)."
    )


def generate_requirements() -> str:
    """Generate requirements.txt with pinned versions."""
    return """# Auto-generated by EasyBCIdata
numpy>=1.24
scipy>=1.10
mne>=1.5
scikit-learn>=1.3
matplotlib>=3.7
pyyaml>=6.0
"""


def _classify_steps(steps: List[str]) -> tuple:
    """Classify steps into MNE-chainable vs numpy-only operations.

    Returns (mne_steps, numpy_before, numpy_after).
    MNE steps are chained on a single Raw object.
    numpy_before runs before MNE (fill_nan).
    numpy_after runs after MNE (scale, clip).
    """
    mne_ops = {"notch", "bandpass", "hilbert", "resample", "bipolar_ref", "drop_bads", "pick_channels"}
    numpy_before_ops = {"fill_nan"}
    numpy_after_ops = {"scale", "clip"}

    mne_steps = []
    numpy_before = []
    numpy_after = []

    for step_str in steps:
        step_name = step_str.split(":", 1)[0].strip()
        if step_name in mne_ops:
            mne_steps.append(step_str)
        elif step_name in numpy_before_ops:
            numpy_before.append(step_str)
        elif step_name in numpy_after_ops:
            numpy_after.append(step_str)
        else:
            numpy_after.append(step_str)

    return mne_steps, numpy_before, numpy_after


def _generate_mne_chain_step(step_name: str, param: str, modality: str) -> List[str]:
    """Generate a single MNE operation line (chained on existing Raw object)."""
    if step_name == "notch":
        freq_list = [f.strip() for f in (param or "50").split(",")]
        if len(freq_list) == 1:
            return [f"raw.notch_filter({freq_list[0]}.0, verbose=False)"]
        freqs_str = ", ".join(f"{f}.0" for f in freq_list)
        return [f"raw.notch_filter([{freqs_str}], verbose=False)"]

    elif step_name == "bandpass":
        parts = param.split(",")
        l_freq = parts[0] if parts[0] else "None"
        h_freq = parts[1] if len(parts) > 1 and parts[1] else "None"
        return [f"raw.filter(l_freq={l_freq}, h_freq={h_freq}, verbose=False)"]

    elif step_name == "hilbert":
        return ["raw.apply_hilbert(envelope=True)"]

    elif step_name == "resample":
        target = param or "256"
        return [f"raw.resample({target}.0, verbose=False)"]

    elif step_name == "bipolar_ref":
        return [
            "anodes = list(raw.ch_names[:-1])",
            "cathodes = list(raw.ch_names[1:])",
            "raw = mne.set_bipolar_reference(raw, anodes, cathodes, verbose=False)",
        ]

    elif step_name == "drop_bads":
        if param:
            bads = [f'"{ch.strip()}"' for ch in param.split(",")]
            bads_str = ", ".join(bads)
            return [f"raw.drop_channels([{bads_str}])"]
        return []

    elif step_name == "pick_channels":
        if param:
            names = [f'"{ch.strip()}"' for ch in param.split(",")]
            names_str = ", ".join(names)
            return [f"raw.pick_channels([{names_str}])"]
        return []

    return [f"# Unknown MNE step: {step_name}"]


def _generate_numpy_step(step_name: str, param: str) -> List[str]:
    """Generate numpy-only preprocessing step code."""
    if step_name == "scale":
        if param == "robust":
            return [
                "from sklearn.preprocessing import RobustScaler",
                "scaler = RobustScaler()",
                "data = scaler.fit_transform(data.T).T.astype(np.float32)",
            ]
        elif param == "standard":
            return [
                "from sklearn.preprocessing import StandardScaler",
                "scaler = StandardScaler()",
                "data = scaler.fit_transform(data.T).T.astype(np.float32)",
            ]
        else:
            return [f"data = data * {param}"]

    elif step_name == "clip":
        return [f"data = np.clip(data, -{param}, {param})"]

    elif step_name == "fill_nan":
        val = param or "0"
        return [f"data = np.nan_to_num(data, nan={val}, posinf={val}, neginf={val})"]

    return [f"# Unknown numpy step: {step_name}:{param}"]


def _generate_step_code(step_name: str, param: str, modality: str) -> List[str]:
    """Generate Python code lines for a single preprocessing step (legacy fallback)."""
    generators = {
        "notch": _gen_notch,
        "bandpass": _gen_bandpass,
        "hilbert": _gen_hilbert,
        "resample": _gen_resample,
        "scale": _gen_scale,
        "clip": _gen_clip,
        "fill_nan": _gen_fill_nan,
        "bipolar_ref": _gen_bipolar_ref,
        "drop_bads": _gen_drop_bads,
        "pick_channels": _gen_pick_channels,
    }
    gen = generators.get(step_name)
    if gen:
        return gen(param, modality)
    return [f"# Unknown step: {step_name}:{param}"]


def _gen_notch(param: str, modality: str) -> List[str]:
    freqs = param or "50"
    freq_list = [f.strip() for f in freqs.split(",")]
    if len(freq_list) == 1:
        return [
            f"info = mne.create_info(len(channels), sfreq, ch_types='{modality}')",
            "raw = mne.io.RawArray(data, info, verbose=False)",
            f"raw.notch_filter({freq_list[0]}.0, verbose=False)",
            "data = raw.get_data().astype(np.float32)",
        ]
    freqs_str = ", ".join(f"{f}.0" for f in freq_list)
    return [
        f"info = mne.create_info(len(channels), sfreq, ch_types='{modality}')",
        "raw = mne.io.RawArray(data, info, verbose=False)",
        f"raw.notch_filter([{freqs_str}], verbose=False)",
        "data = raw.get_data().astype(np.float32)",
    ]


def _gen_bandpass(param: str, modality: str) -> List[str]:
    parts = param.split(",")
    l_freq = parts[0] if parts[0] else "None"
    h_freq = parts[1] if len(parts) > 1 and parts[1] else "None"
    return [
        f"info = mne.create_info(len(channels), sfreq, ch_types='{modality}')",
        "raw = mne.io.RawArray(data, info, verbose=False)",
        f"raw.filter(l_freq={l_freq}, h_freq={h_freq}, verbose=False)",
        "data = raw.get_data().astype(np.float32)",
    ]


def _gen_hilbert(param: str, modality: str) -> List[str]:
    return [
        f"info = mne.create_info(len(channels), sfreq, ch_types='{modality}')",
        "raw = mne.io.RawArray(data, info, verbose=False)",
        "raw.apply_hilbert(envelope=True)",
        "data = raw.get_data().astype(np.float32)",
    ]


def _gen_resample(param: str, modality: str) -> List[str]:
    target = param or "256"
    return [
        f"info = mne.create_info(len(channels), sfreq, ch_types='{modality}')",
        "raw = mne.io.RawArray(data, info, verbose=False)",
        f"raw.resample({target}.0, verbose=False)",
        "data = raw.get_data().astype(np.float32)",
        f"sfreq = {target}.0",
    ]


def _gen_scale(param: str, modality: str) -> List[str]:
    if param == "robust":
        return [
            "from sklearn.preprocessing import RobustScaler",
            "scaler = RobustScaler()",
            "data = scaler.fit_transform(data.T).T.astype(np.float32)",
        ]
    elif param == "standard":
        return [
            "from sklearn.preprocessing import StandardScaler",
            "scaler = StandardScaler()",
            "data = scaler.fit_transform(data.T).T.astype(np.float32)",
        ]
    else:
        return [f"data = data * {param}"]


def _gen_clip(param: str, modality: str) -> List[str]:
    return [f"data = np.clip(data, -{param}, {param})"]


def _gen_fill_nan(param: str, modality: str) -> List[str]:
    val = param or "0"
    return [f"data = np.nan_to_num(data, nan={val}, posinf={val}, neginf={val})"]


def _gen_bipolar_ref(param: str, modality: str) -> List[str]:
    return [
        f"info = mne.create_info(channels, sfreq, ch_types='seeg')",
        "raw = mne.io.RawArray(data, info, verbose=False)",
        "# Auto-derive bipolar pairs from neighboring contacts",
        "anodes = channels[:-1]",
        "cathodes = channels[1:]",
        "raw = mne.set_bipolar_reference(raw, anodes, cathodes, verbose=False)",
        "data = raw.get_data().astype(np.float32)",
        "channels = list(raw.ch_names)",
    ]


def _gen_drop_bads(param: str, modality: str) -> List[str]:
    if param:
        bads = [f'"{ch.strip()}"' for ch in param.split(",")]
        bads_str = ", ".join(bads)
        return [
            f"bad_channels = [{bads_str}]",
            "keep_idx = [i for i, ch in enumerate(channels) if ch not in bad_channels]",
            "data = data[keep_idx]",
            "channels = [channels[i] for i in keep_idx]",
        ]
    # No explicit list → reproduce the runtime variance-outlier auto-detection
    # (mirrors DataProfile._analyze_channels) so the standalone script drops the
    # same channels the pipeline did, instead of emitting a no-op comment.
    return [
        "# Auto-detect bad channels by variance outliers (flat / extreme).",
        "_stds = data.std(axis=1)",
        "_median_std = float(np.median(_stds))",
        "if _median_std > 0:",
        "    _bad_mask = (_stds < _median_std * 0.01) | (_stds > _median_std * 5) | (_stds < _median_std * 0.1)",
        "else:",
        "    _bad_mask = np.ones(data.shape[0], dtype=bool)",
        "bad_channels = [channels[i] for i in np.where(_bad_mask)[0]]",
        "keep_idx = [i for i, ch in enumerate(channels) if ch not in bad_channels]",
        "data = data[keep_idx]",
        "channels = [channels[i] for i in keep_idx]",
    ]


def _gen_pick_channels(param: str, modality: str) -> List[str]:
    if not param:
        return ["# No channel selection specified"]
    names = [f'"{ch.strip()}"' for ch in param.split(",")]
    names_str = ", ".join(names)
    return [
        f"pick_names = [{names_str}]",
        "pick_idx = [i for i, ch in enumerate(channels) if ch in pick_names]",
        "data = data[pick_idx]",
        "channels = [channels[i] for i in pick_idx]",
    ]
