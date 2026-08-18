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
from easybci_lib.tools.neural_processing.preprocess.operator_vocab import (
    normalize_steps as _normalize_steps,
)

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
EASYBCI_VERSION: 5
EASYBCI_CODE_STANDARD: 0.0.1

Standalone script — runs on a plain `pip install mne numpy scipy scikit-learn`
without any easybci_* dependency. See CODE_STANDARD.md Rule 15.

Run: python pipeline.py <input_path> <work_dir>
"""

import gc
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


def _easybci_home():
    """Locate EASYBCI_HOME the same way the app does, WITHOUT importing easybci
    (CODE_STANDARD Rule 15 — generated scripts stay self-contained). Env var
    wins (set by the batch orchestrator / CLI); falls back to ~/.easybci."""
    import os
    h = os.environ.get("EASYBCI_HOME")
    return Path(h) if h else (Path.home() / ".easybci")


def _discover_io_plugin(path):
    """Return (load_callable, name) for the first registered io_loader plugin
    whose matches(path) is True, else (None, None).

    Mirrors easybci's loader_registry.find(): the agent may have written a
    loader for a format the built-ins can't read (e.g. Nihon Kohden .EEG). The
    generated pipeline MUST prefer it over a partial built-in reader, so the
    reproducible repo reads data the same way the interactive session did.
    A repo-local ``code/io_loaders/`` (bundled at export) is scanned FIRST so
    the repo is portable, then the machine-global ``~/.easybci/io_loaders/``.
    Each plugin file is imported in isolation; a broken one is skipped."""
    import importlib.util as _ilu
    _dirs = []
    _self = globals().get("__file__")
    if _self:
        _dirs.append(Path(_self).resolve().parent / "io_loaders")
    _dirs.append(_easybci_home() / "io_loaders")
    for d in _dirs:
        if not d.is_dir():
            continue
        for py in sorted(d.glob("*.py")):
            if py.name.startswith("_"):
                continue
            _mod_name = "_ebci_io_" + py.stem
            try:
                spec = _ilu.spec_from_file_location(_mod_name, str(py))
                mod = _ilu.module_from_spec(spec)
                # Register BEFORE exec so decorators that resolve the owning
                # module (e.g. @dataclass -> sys.modules[cls.__module__]) work.
                sys.modules[_mod_name] = mod
                try:
                    spec.loader.exec_module(mod)
                    matches = getattr(mod, "matches", None)
                    load = getattr(mod, "load", None)
                    if callable(matches) and callable(load) and bool(matches(str(path))):
                        return load, py.stem
                finally:
                    sys.modules.pop(_mod_name, None)
            except Exception:
                continue
    return None, None


def _call_plugin_load(load, path, target_hz):
    """Call a plugin's load(), passing target_hz only if its signature accepts
    it (v1 plugins are load(path, inspect_only=False))."""
    import inspect as _insp
    kwargs = {{}}
    if target_hz is not None:
        try:
            params = _insp.signature(load).parameters
            if "target_hz" in params or any(
                pp.kind == _insp.Parameter.VAR_KEYWORD for pp in params.values()):
                kwargs["target_hz"] = target_hz
        except (ValueError, TypeError):
            pass
    return load(str(path), **kwargs)


def _load_input(path, target_hz=None):
    """Return a dict with keys: data (n_ch, n_samples float32), frequency, channels, meta.

    A registered io_loader plugin that matches `path` is tried FIRST (so custom
    formats read correctly and load-time decimation to `target_hz` applies);
    built-in MNE/npz/csv/pkl/nwb branches are the fallback."""
    p = Path(path)
    ext = p.suffix.lower()

    _plugin_load, _plugin_name = _discover_io_plugin(p)
    if _plugin_load is not None:
        _res = _call_plugin_load(_plugin_load, p, target_hz)
        if isinstance(_res, dict) and "data" in _res:
            _res.setdefault("meta", {{}}).setdefault("loaded_by_plugin", _plugin_name)
            return _res
        # Plugin returned something unusable — fall through to built-ins.

    if ext in _MNE_EXTS or (p.is_dir() and p.suffix == ".ds"):
        import mne
        raw = mne.io.read_raw(str(p), preload=True, verbose="ERROR")
        _meta = {{"format": "mne", "source_file": str(p)}}
        # Carry channel types (drop_bads scaling / QC) and annotations
        # (event markers) so per-file operators like reject_by_labels can
        # act on them at runtime — mirrors io/loader.py.
        try:
            _meta["ch_types"] = list(raw.get_channel_types())
        except Exception:
            pass
        if raw.annotations is not None and len(raw.annotations) > 0:
            _meta["annotations"] = {{
                "onset": raw.annotations.onset.tolist(),
                "duration": raw.annotations.duration.tolist(),
                "description": list(raw.annotations.description),
            }}
        return {{
            "data": raw.get_data().astype(np.float32),
            "frequency": float(raw.info["sfreq"]),
            "channels": list(raw.ch_names),
            "meta": _meta,
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
            _dset_v = _es.data
            _fs = float(_es.rate)
            # Zero-copy memmap onto the contiguous HDF5 region (chunks/compression
            # None, float32); falls back to a full read for older chunked files.
            _mm_v = None
            try:
                _off_v = _dset_v.id.get_offset()
                if (_off_v is not None and _dset_v.chunks is None
                        and _dset_v.compression is None
                        and _dset_v.dtype == np.dtype("float32")):
                    _mm_v = np.memmap(str(p), dtype=np.float32, mode="r",
                                      offset=int(_off_v), shape=tuple(_dset_v.shape))
            except Exception:
                _mm_v = None
            _data_arr = _mm_v.T if _mm_v is not None else np.asarray(_dset_v[:]).T.astype(np.float32)
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
    meta = dict(prev_meta)
    # Refresh ch_types from the raw so it always matches raw.ch_names — an MNE
    # op that drops/reorders channels would otherwise leave a stale ch_types
    # and trip the NWB writer's length check.
    try:
        meta["ch_types"] = list(raw.get_channel_types())
    except Exception:
        meta.pop("ch_types", None)
    return {{
        "data": raw.get_data().astype(np.float32),
        "frequency": float(raw.info["sfreq"]),
        "channels": list(raw.ch_names),
        "meta": meta,
        "_mne_info": raw.info,
    }}


def _detect_powerline_hz(data, fs):
    """Detect mains frequency + harmonics from this file's PSD.

    Mirrors easybci deep_inspect._psd_summary: 48-62 Hz band, peak must be
    >6 dB above the median floor; harmonics ×2/3/4 under Nyquist. Returns a
    list of notch frequencies (base + harmonics) or [] when none detected.
    """
    from scipy.signal import welch
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 2 or fs <= 0 or arr.shape[1] < int(fs * 4):
        return []
    nperseg = max(64, int(fs / 0.5))
    avg = arr.mean(axis=0)
    freqs, pxx = welch(avg, fs=fs, nperseg=min(nperseg, len(avg)))
    pxx_db = 10 * np.log10(pxx + 1e-20)
    median_db = float(np.median(pxx_db))
    band = (freqs >= 48) & (freqs <= 62)
    if not band.any():
        return []
    idx = int(np.argmax(pxx_db[band]))
    peak_hz = float(freqs[band][idx])
    if float(pxx_db[band][idx] - median_db) <= 6.0:
        return []
    out = [peak_hz]
    for k in (2, 3, 4):
        h = peak_hz * k
        if h >= freqs[-1]:
            break
        i = int(np.argmin(np.abs(freqs - h)))
        if pxx_db[i] - median_db > 6.0:
            out.append(round(float(freqs[i]), 1))
    return out


def op_notch(d, param):
    """Notch filter. param='auto' detects this file's mains freq + harmonics
    at runtime (per-file power-line detection); a numeric param notches that
    fixed frequency. 'auto' with no detectable line noise falls back to 50 Hz.
    """
    if (param or "").strip().lower() == "auto":
        freqs = _detect_powerline_hz(d["data"], float(d.get("frequency") or 0.0))
        if not freqs:
            freqs = [50.0]
    else:
        freqs = [float(param) if param else 50.0]
    raw = _to_mne_raw(d)
    raw.notch_filter(freqs=freqs, verbose="ERROR")
    return _from_mne_raw(raw, d.get("meta", {{}}))


def op_bandpass(d, param):
    # Single-sided aware, matching the runtime engine: an empty side means
    # "no bound on that side" (None), NOT a hard-coded default. This is what
    # lets ``highpass:X`` / ``lowpass:X`` normalize safely to ``bandpass:X,`` /
    # ``bandpass:,X`` without silently turning into a 1-40 Hz band-pass.
    parts = (param or "").split(",")
    lo = float(parts[0]) if len(parts) >= 1 and parts[0] else None
    hi = float(parts[1]) if len(parts) >= 2 and parts[1] else None
    if lo is None and hi is None:
        return d
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


_RESAMPLE_COMMON_TARGETS = [1000.0, 500.0, 512.0, 256.0, 250.0, 200.0, 128.0, 100.0]


def _largest_safe_target(src_sfreq):
    """Largest common resample target strictly below src (Nyquist-safe)."""
    below = [t for t in _RESAMPLE_COMMON_TARGETS if t < src_sfreq]
    return max(below) if below else float(int(src_sfreq))


def op_resample(d, param):
    """Resample. param='auto' clamps to the largest common target strictly
    below this file's source rate (Nyquist-safe, per-file). A numeric target
    that is >= this file's source rate is likewise clamped down rather than
    upsampled. Mirrors easybci proven_adapt nyquist_bounded.
    """
    src = float(d.get("frequency") or 0.0)
    if (param or "").strip().lower() == "auto":
        target = _largest_safe_target(src) if src > 0 else 256.0
    else:
        target = float(param) if param else 256.0
        # Only clamp DOWN when the requested target strictly exceeds source
        # (can't upsample). target == src is a genuine no-op — critical when
        # the loader already decimated to this exact rate at load time, so we
        # don't accidentally drop it a second time (500==500 must NOT → 256).
        if src > 0 and target > src:
            target = _largest_safe_target(src)
    if src > 0 and abs(target - src) < 1e-6:
        return dict(d)  # already at target — skip the resample entirely
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
    kept_data = data[keep, :].astype(np.float32)
    kept_channels = [channels[i] for i in keep]
    # Surviving channels may still carry NaN (<=50% NaN passed the drop
    # threshold). Interpolate those in place so NaN never propagates into
    # later MNE-based ops. The 50% DROP threshold above is unchanged; this only
    # cleans channels we chose to keep.
    nan_cleaned = []
    for r in range(kept_data.shape[0]):
        row = kept_data[r]
        bad = np.isnan(row)
        if not bad.any():
            continue
        good = ~bad
        if good.any():
            xs = np.flatnonzero(good)
            row[bad] = np.interp(np.flatnonzero(bad), xs, row[good])
        else:
            row[bad] = 0.0
        kept_data[r] = row
        nan_cleaned.append(kept_channels[r])
    out = dict(d)
    out["data"] = kept_data
    out["channels"] = kept_channels
    out["meta"] = dict(d.get("meta", {{}}))
    out["meta"]["dropped_channels"] = list(out["meta"].get("dropped_channels", [])) + dropped
    _ct = out["meta"].get("ch_types")
    if isinstance(_ct, list) and len(_ct) == len(channels):
        out["meta"]["ch_types"] = [_ct[i] for i in keep]
    if nan_cleaned:
        out["meta"]["nan_interpolated_channels"] = list(
            out["meta"].get("nan_interpolated_channels", [])
        ) + nan_cleaned
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
    _ct = out["meta"].get("ch_types")
    if isinstance(_ct, list) and len(_ct) == len(channels):
        out["meta"]["ch_types"] = [_ct[i] for i in keep]
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


# Multilingual reject-keyword floor — frozen copy of easybci
# DEFAULT_REJECT_KEYWORDS so the operator is self-contained. Always union'd
# with keywords baked into the step param.
_DEFAULT_REJECT_KEYWORDS = [
    "Seiz", "Seizure", "Ictal", "SZ\\\\b", "pre-ictal", "post-ictal",
    "epilep", "convuls",
    "IID", "spike", "sharp wave", "polyspike", "discharge",
    "Stim", "stimulat", "electrical stim",
    "发作", "癫", "痫", "刺激", "电刺激", "痉挛",
]


def _reject_compile(keywords):
    """Compile keywords into one word-START-boundary case-insensitive regex."""
    parts = []
    for kw in keywords:
        kw = str(kw).strip()
        if not kw:
            continue
        if "\\\\b" in kw:
            parts.append(kw)
        else:
            parts.append(r"(?<![0-9A-Za-z])" + _re.escape(kw))
    if not parts:
        return None
    return _re.compile("|".join(parts), _re.IGNORECASE)


def op_reject_by_labels(d, param):
    """Excise labelled time windows (seizure / stim / discharge) ± 1 s pad.

    param is a comma-joined keyword list; it is union'd with the built-in
    multilingual floor. Reads meta['annotations'] (onset/duration/description),
    carried from the raw file by _load_input. No-op when a recording has no
    annotations or no matching labels. Records rejected_seconds +
    unmatched/suspicious labels into meta for batch diagnostics.
    """
    meta = dict(d.get("meta") or {{}})
    annotations = meta.get("annotations")
    param_kws = [k.strip() for k in (param or "").split(",") if k.strip()]
    seen = set()
    keywords = []
    for kw in list(param_kws) + list(_DEFAULT_REJECT_KEYWORDS):
        kw = str(kw).strip()
        if kw and kw.lower() not in seen:
            seen.add(kw.lower())
            keywords.append(kw)
    data = np.asarray(d["data"])
    if not annotations or data.ndim < 2 or not keywords:
        out = dict(d)
        out["meta"] = meta
        meta.setdefault("rejected_samples", 0)
        return out
    sfreq = float(d.get("frequency") or 0.0)
    n_samples = int(data.shape[1])
    pad_s = 1.0
    rx = _reject_compile(keywords)
    keep = np.ones(n_samples, dtype=bool)
    # Excised windows in ORIGINAL SECONDS (time-base invariant). Downstream
    # epoching (build_ai_ready) must remap event onsets across these gaps;
    # seconds survive a later resample where sample indices would not. See
    # _remap_events_after_reject.
    rejected_intervals_s = []
    if rx is not None and sfreq > 0:
        onsets = annotations.get("onset") or []
        durations = annotations.get("duration") or []
        descriptions = annotations.get("description") or []
        for i, desc in enumerate(descriptions):
            if rx.search(str(desc)) is None:
                continue
            onset = float(onsets[i]) if i < len(onsets) else 0.0
            dur = float(durations[i]) if i < len(durations) else 0.0
            start = max(0, int(np.floor((onset - pad_s) * sfreq)))
            stop = min(n_samples, int(np.ceil((onset + dur + pad_s) * sfreq)))
            if stop > start:
                keep[start:stop] = False
    # Derive merged excised intervals from the final keep-mask (adjacent /
    # overlapping windows collapse into one), expressed in seconds so the map
    # is correct even after a subsequent resample changes the sample rate.
    if sfreq > 0 and not keep.all():
        flip = np.diff(np.concatenate(([1], keep.astype(np.int64), [1])))
        starts = np.where(flip == -1)[0]
        stops = np.where(flip == 1)[0]
        rejected_intervals_s = [
            [round(float(s) / sfreq, 6), round(float(e) / sfreq, 6)]
            for s, e in zip(starts, stops)
        ]
    # Observability: labels the keyword union missed.
    descriptions = (annotations or {{}}).get("description") or []
    unmatched = sorted({{str(l) for l in descriptions
                        if rx is None or rx.search(str(l)) is None}})
    meta["unmatched_labels"] = unmatched
    n_reject = int(n_samples - int(keep.sum()))
    out = dict(d)
    if n_reject > 0:
        out["data"] = data[:, keep]
        meta["rejected_samples"] = n_reject
        meta["rejected_seconds"] = round(n_reject / sfreq, 3) if sfreq else 0
        meta["rejected_intervals"] = rejected_intervals_s
        out.pop("_mne_info", None)  # sample count changed; rebuild on next op
    else:
        meta.setdefault("rejected_samples", 0)
    out["meta"] = meta
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
    "resample": op_resample,
    "car": op_car,
    "ica": op_ica,
    "drop_bads": op_drop_bads,
    "drop_nondata_channels": op_drop_nondata_channels,
    "scale": op_scale,
    "clip": op_clip,
    "fill_nan": op_fill_nan,
    "reject_by_labels": op_reject_by_labels,
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
        # Fail loud — steps are normalized to canonical names before this script
        # is generated, so an unknown op here means a hand-edit introduced a name
        # this bundle cannot run. NEVER silently skip a core preprocessing step.
        raise ValueError(
            "[pipeline] unknown/unsupported step {{!r}} — refusing to silently "
            "skip. Supported: {{}}".format(step, sorted(_OPS))
        )
    if op_name in _NO_PARAM_OPS:
        return fn(d)
    return fn(d, param)


def _resample_target_from_steps(steps):
    """Return the concrete resample target (Hz) from the step list, or None.
    ``resample:auto`` has no fixed target → None (loader reads native)."""
    for s in steps:
        if not isinstance(s, str):
            continue
        kind, _, param = s.partition(":")
        if kind.strip() == "resample":
            param = param.strip().lower()
            if param and param != "auto":
                try:
                    return float(param)
                except ValueError:
                    return None
    return None


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


# --------------------------------------------------------------------------
# Cross-instance memory gate (stdlib-only inline copy of
# batch/global_gate.py — Rule 15: generated code is self-contained). Reserves a
# file's estimated peak footprint in a machine-wide ledger BEFORE loading, so
# two independent easybci instances on one host cannot both load a ~30 GB
# recording and OOM the machine. Keep in sync with global_gate.py.
# --------------------------------------------------------------------------
def _gate_ledger_path():
    return _easybci_home() / "batch" / "mem_ledger.json"


def _gate_budget_mb():
    import os
    env = os.environ.get("EASYBCI_MEMORY_BUDGET_MB")
    if env:
        try:
            v = float(env)
            if v > 0:
                return v * 0.7
        except ValueError:
            pass
    total = None
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    total = int(line.split()[1]) / 1024  # kB -> MB
                    break
    except (OSError, ValueError, IndexError):
        pass
    if total is None:
        total = 8000.0
    return total * 0.7


def _gate_stale_s():
    import os
    env = os.environ.get("EASYBCI_GATE_STALE_S")
    if env:
        try:
            v = float(env)
            if v > 0:
                return v
        except ValueError:
            pass
    return 3600.0


def _gate_pid_alive(pid):
    import os
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _gate_peak_mb(n_channels, frequency, duration_s, has_ica, target_hz=None):
    n_ch = int(n_channels or 0)
    fs = float(frequency or 0.0)
    dur = float(duration_s or 0.0)
    if n_ch <= 0 or fs <= 0 or dur <= 0:
        return 0.0
    eff_fs = fs
    if target_hz and target_hz > 0 and target_hz < fs:
        eff_fs = float(target_hz)
    overhead = 8.0 if has_ica else 3.0
    return (n_ch * eff_fs * dur * 4 * overhead) / (1024 * 1024)  # float32


def _gate_try_reserve(peak_mb, file_id, token, now):
    import fcntl
    import os
    p = _gate_ledger_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    lock_path = p.with_suffix(".json.lock")
    fd = open(lock_path, "a+", encoding="utf-8")
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        ledger = {{}}
        if p.is_file():
            try:
                _d = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(_d, dict):
                    ledger = _d
            except Exception:
                ledger = {{}}
        stale = _gate_stale_s()
        kept = {{}}
        for _tk, _r in ledger.items():
            if not isinstance(_r, dict):
                continue
            _rpid = int(_r.get("pid", 0) or 0)
            _rts = float(_r.get("ts", 0.0) or 0.0)
            if _rpid and not _gate_pid_alive(_rpid):
                continue
            if now - _rts > stale:
                continue
            kept[_tk] = _r
        ledger = kept
        budget = _gate_budget_mb()
        reserved = sum(float(_r.get("reserved_mb", 0.0) or 0.0)
                       for _r in ledger.values() if isinstance(_r, dict))
        admitted = False
        # Admit if it fits, OR if nothing else is reserved (anti-starvation: a
        # file bigger than the whole budget was already vetted upstream).
        if reserved + peak_mb <= budget or not ledger:
            ledger[token] = {{"pid": os.getpid(),
                             "reserved_mb": round(float(peak_mb), 1),
                             "ts": now, "file_id": file_id}}
            admitted = True
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
        tmp.replace(p)
        return admitted
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        fd.close()


def _gate_acquire(peak_mb, file_id="", timeout=None):
    """Block until peak_mb fits the machine budget; return an opaque token.

    Fail-open: if the gate infra errors or the wait exceeds ``timeout``, return
    "" and let processing proceed rather than deadlock the batch. The lock is
    held only during each brief sweep+admit attempt; the wait is lock-free."""
    import os
    import time
    if peak_mb <= 0:
        return ""
    token = "{{}}-{{}}-{{}}".format(os.getpid(), int(time.time() * 1000), file_id)
    start = time.time()
    while True:
        try:
            if _gate_try_reserve(peak_mb, file_id, token, time.time()):
                return token
        except Exception:
            return ""  # gate unavailable — fail open
        if timeout is not None and (time.time() - start) >= timeout:
            print("[pipeline] memory gate: waited {{:.0f}}s for ~{{:.0f}} MB "
                  "(file_id={{}}); proceeding".format(timeout, peak_mb, file_id),
                  file=sys.stderr)
            return ""
        print("[pipeline] memory gate: ~{{:.0f}} MB not yet available for "
              "file_id={{}} — waiting for another instance to finish".format(
                  peak_mb, file_id), file=sys.stderr)
        time.sleep(2.0)


def _gate_release(token):
    """Drop the reservation for ``token``. Idempotent; never raises."""
    import fcntl
    if not token:
        return
    p = _gate_ledger_path()
    lock_path = p.with_suffix(".json.lock")
    try:
        fd = open(lock_path, "a+", encoding="utf-8")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            ledger = {{}}
            if p.is_file():
                try:
                    _d = json.loads(p.read_text(encoding="utf-8"))
                    if isinstance(_d, dict):
                        ledger = _d
                except Exception:
                    ledger = {{}}
            if token in ledger:
                del ledger[token]
                tmp = p.with_suffix(".json.tmp")
                tmp.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
                tmp.replace(p)
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            fd.close()
    except Exception:
        pass


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

    # Load-time decimation hint: a concrete resample:<N> step lets the loader
    # decimate on the fly (huge sEEG never materializes at native rate). Any
    # subsequent resample:<N> step is then a no-op (src == target).
    _load_target = _resample_target_from_steps(steps)

    # Read the deep_inspect fingerprint ONCE up front — used both for the
    # cross-instance memory gate (peak estimate, before loading) and the
    # post-load channel-count sanity check. inspection_report_path is relative
    # to work_dir.
    _exp_nch, _exp_fs, _exp_dur = 0, 0.0, 0.0
    _irp = inp.get("inspection_report_path")
    if _irp:
        try:
            _rp = Path(_irp)
            if not _rp.is_absolute():
                _rp = work_dir / _irp
            _fp = json.loads(_rp.read_text(encoding="utf-8")).get("fingerprint", {{}})
            _exp_nch = int(_fp.get("n_channels") or 0)
            _exp_fs = float(_fp.get("sampling_freq_hz") or 0.0)
            _exp_dur = float(_fp.get("duration_s") or 0.0)
        except Exception:
            _exp_nch, _exp_fs, _exp_dur = 0, 0.0, 0.0

    # Cross-instance memory gate: reserve this file's estimated peak footprint
    # in the machine-wide ledger BEFORE loading. If another easybci instance (a
    # second tmux session) is already holding a large recording, we block here
    # until it releases, rather than both loading ~30 GB and OOM-killing the
    # host. Fail-open: an empty token means the gate was unavailable/timed out.
    _has_ica = any(isinstance(_s, str) and _s.split(":", 1)[0].strip() == "ica"
                   for _s in steps)
    _peak_mb = _gate_peak_mb(_exp_nch, _exp_fs, _exp_dur, _has_ica,
                             target_hz=_load_target)
    _gate_token = _gate_acquire(_peak_mb, file_id=file_id)
    try:
        print("Loading: {{}}  (sub={{}} ses={{}} file_id={{}}, target_hz={{}})".format(
            data_path, sub_id, ses, file_id, _load_target))
        data_dict = _load_input(data_path, target_hz=_load_target)
        n_ch_in = len(data_dict.get("channels", []))
        fs_in = float(data_dict.get("frequency", 0.0))
        print("  Channels in: {{}}, fs: {{}} Hz".format(n_ch_in, fs_in))

        # Defense-in-depth against a silent misread (e.g. a 15GB Nihon Kohden
        # .EEG read as 1-channel BrainVision garbage that still "passes" QC).
        # Compare the loaded channel count against what deep_inspect recorded; a
        # gross mismatch means the wrong reader was used — fail LOUD, never emit
        # a Pass NWB.
        if (_exp_nch > 0 and n_ch_in > 0 and
                ((n_ch_in == 1 and _exp_nch > 4)
                 or n_ch_in * 4 < _exp_nch or n_ch_in > _exp_nch * 4)):
            raise ValueError(
                "channel-count mismatch: loader returned {{}} channel(s) but "
                "deep_inspect recorded {{}} — wrong reader for {{}} (refusing to "
                "write a misread output).".format(n_ch_in, _exp_nch, data_path))

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
            steps=steps,
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
            "rejected_seconds": data_dict.get("meta", {{}}).get("rejected_seconds"),
            "unmatched_labels": list(data_dict.get("meta", {{}}).get("unmatched_labels", [])),
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
        # Release the large per-file working set (decimated data array ~7GB, plus
        # MNE _mne_info / intermediate copies) BEFORE returning so it never
        # accumulates across a serial multi-file batch. `status` holds only scalars
        # and small lists, so this is safe. Without this, MNE/numpy circular refs
        # keep each file's ~14GB peak alive into the next iteration and a
        # multi-hundred-file sEEG batch OOMs a 62GB host mid-run.
        data_dict.clear()
        del data_dict
        gc.collect()
        return status
    finally:
        # Release the ledger reservation AFTER the working set is freed (gc
        # above) so a waiting instance only sees capacity once this file's peak
        # is genuinely gone. Idempotent for an empty/fail-open token.
        _gate_release(_gate_token)


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
        finally:
            # Defense-in-depth: force a collection after every file (success OR
            # failure) so a mid-load exception can't leave a partial ~14GB
            # working set alive into the next serial iteration.
            gc.collect()

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
    """If steps contain a notch op, rewrite its frequency to line_freq.

    Steps already set to ``notch:auto`` (adaptive template — per-file
    runtime detection) are left untouched: baking a single representative
    frequency would defeat the point of auto-detection.
    """
    out = []
    for s in steps:
        if isinstance(s, str) and s.startswith("notch:"):
            if s.split(":", 1)[1].strip().lower() == "auto":
                out.append(s)
            else:
                out.append(f"notch:{int(round(line_freq))}")
        elif isinstance(s, dict) and s.get("operator") == "notch_filter":
            new = dict(s)
            new["params"] = dict(new.get("params") or {})
            new["params"]["freq"] = float(line_freq)
            out.append(new)
        else:
            out.append(s)
    return out


def build_adaptive_steps(skill, *, reject_keywords=None) -> List[str]:
    """Build a self-adapting step list from a proven skill's step KINDS.

    Codegen analogue of ``proven_adapt.adapt_pipeline``: instead of baking
    per-file numbers, it emits ``:auto`` markers that the generated operators
    resolve per-input at runtime, so ONE step list reproduces correctly across
    a heterogeneous batch. Transforms (mirrors the adaptation slots):

    - ``notch:<n>`` → ``notch:auto`` (runtime power-line detection)
    - ``resample:<n>`` → ``resample:auto`` (runtime Nyquist clamp)
    - prepend ``reject_by_labels:<kw>`` when keywords exist (runtime excision)
    - ``drop_bads`` already resolves per-file at runtime — kept as-is

    Only KINDS/order from the skill are honoured; numeric params are dropped in
    favour of ``:auto`` so no single file's numbers leak into the shared repo.
    """
    slots = {s.get("param") for s in (getattr(skill, "adaptation_slots", []) or [])}
    kws = [str(k).strip() for k in
           (reject_keywords if reject_keywords is not None
            else getattr(skill, "reject_keywords", []) or []) if str(k).strip()]
    src_steps = list(getattr(skill, "steps", []) or [])

    out: List[str] = []
    for s in src_steps:
        if not isinstance(s, str):
            out.append(s); continue
        kind = s.split(":", 1)[0].strip()
        if kind == "notch" and "notch_freqs" in slots:
            if "notch:auto" not in out:
                out.append("notch:auto")
        elif kind == "resample" and "resample_target_hz" in slots:
            # Keep the CONCRETE gold target (e.g. resample:500), NOT resample:auto.
            # op_resample already Nyquist-guards a concrete target per file
            # (clamps down only if it exceeds that file's source), so `auto`
            # was strictly worse: _largest_safe_target(2000)=1000 deviates from
            # the proven recipe AND doubles memory. The concrete target also
            # becomes the load-time decimation hint (see resample_target_hz()).
            out.append(s)
        elif kind == "reject_by_labels":
            continue  # re-added (deduped) below at position 0
        else:
            out.append(s)
    # Prepend reject_by_labels (excise BEFORE filtering) when keywords exist.
    if kws and "reject_time_segments" in slots:
        out = ["reject_by_labels:" + ",".join(kws)] + out
    return out


def resample_target_hz(steps) -> Optional[float]:
    """Return the concrete resample target (Hz) from a step list, or None.

    This is the load-time decimation hint: when a recipe contains
    ``resample:500`` the loader can decimate on the fly to 500 Hz so a
    multi-hour high-channel recording never materializes at native rate (261ch/
    2000Hz float32 ≈ 56 GB peak — OOMs a 62 GB host). ``resample:auto`` returns
    None (no fixed target known until per-file runtime), so such recipes load
    native and rely on the OOM guard / crop instead.
    """
    for s in steps or []:
        if not isinstance(s, str):
            continue
        kind, _, param = s.partition(":")
        if kind.strip() == "resample":
            p = param.strip().lower()
            if p and p != "auto":
                try:
                    return float(p)
                except ValueError:
                    return None
    return None


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
                     analysis_goal="", modality=None, steps=None):
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
    # Persist reject_by_labels excised windows (seconds) so build_ai_ready can
    # remap event onsets across the gaps. Stored as a (n,2) scratch array;
    # absent when no excision happened.
    _rej_iv = meta_d.get("rejected_intervals") if isinstance(meta_d, _Mapping_nwb) else None
    if _rej_iv:
        try:
            _rej_arr = np.asarray(_rej_iv, dtype=np.float64)
            if _rej_arr.ndim == 2 and _rej_arr.shape[1] == 2 and _rej_arr.shape[0] > 0:
                nwb.add_scratch(
                    _rej_arr,
                    name="easybci_rejected_intervals",
                    description="reject_by_labels excised [onset,offset] windows in seconds",
                )
        except Exception:
            pass
    _mod_l = (modality or "").strip().lower()
    # Self-contained provenance: the processing chain, modality and dropped
    # channels are NOT otherwise recoverable from the NWB (channels/rate/shape
    # already live in the electrode table + ElectricalSeries). Stored as a JSON
    # scratch string so a reader gets full provenance from the NWB alone, with
    # no external sidecar. Reading it does not touch the signal array.
    try:
        _prov = {
            "producer": "EasyBCI Data Agent",
            "analysis_goal": analysis_goal or "",
            "modality": modality or "",
            "steps": list(steps) if steps else [],
            "dropped_channels": list(meta_d.get("dropped_channels", []))
                if isinstance(meta_d, _Mapping_nwb) else [],
        }
        nwb.add_scratch(
            json.dumps(_prov, ensure_ascii=False),
            name="easybci_provenance",
            description="EasyBCIdata preprocessing provenance (JSON): analysis_goal, modality, ordered steps, dropped_channels",
        )
    except Exception:
        pass
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
#
# Memory: the ElectricalSeries data is written contiguously and uncompressed
# (chunks=None, compression=None — see nwb_writer), so instead of ``data[:]``
# (which eagerly reads the whole file into RAM — ~7 GB for a 259ch/7.18M-sample
# sEEG recording, the batch-summary OOM culprit) we memory-map the on-disk
# region and return a lazy ``(n_channels, n_samples)`` view. Resident memory
# for the load itself is ~0; the OS pages in only the slices callers touch and
# can evict them under pressure. Falls back to a full read for chunked /
# compressed / non-float32 datasets (older files) so the contract is unchanged.
_NWB_PREPROCESSED_LOADER_BLOCK = '''
# --- Preprocessed NWB loader (inlined; preprocessed/ is NWB-only) ---
def _load_preprocessed_nwb(pre_nwb):
    """Read a preprocessed NWB file written by pipeline.py.

    Returns (data, fs, meta, channels) where:
      data:      np.ndarray of shape (n_channels, n_samples), float32.
                 A memory-mapped lazy view when the on-disk dataset is
                 contiguous+uncompressed+float32 (the common case); otherwise
                 a materialized array (fallback for chunked/compressed files).
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
    _pre = str(pre_nwb)
    _mm = None
    with NWBHDF5IO(_pre, "r") as _io_nwb:
        _nwb = _io_nwb.read()
        _acq = _nwb.acquisition
        _es_name = "preprocessed" if "preprocessed" in _acq else next(iter(_acq))
        _es = _acq[_es_name]
        _dset = _es.data
        _fs = float(_es.rate)
        try:
            _channels = [str(_n) for _n in _nwb.electrodes["channel_name"][:]]
        except Exception:
            _channels = None
        # reject_by_labels excised windows (seconds), if the pipeline stashed
        # them — build_ai_ready remaps event onsets across these gaps.
        _rej_iv = None
        try:
            _sc = _nwb.get_scratch("easybci_rejected_intervals")
            _rej_iv = np.asarray(_sc).tolist()
        except Exception:
            _rej_iv = None
        # Zero-copy memmap onto the contiguous HDF5 data region. get_offset()
        # returns None for chunked/compressed datasets → fall back to full read.
        try:
            _off = _dset.id.get_offset()
            if (_off is not None and _dset.chunks is None
                    and _dset.compression is None
                    and _dset.dtype == np.dtype("float32")):
                _mm = np.memmap(_pre, dtype=np.float32, mode="r",
                                offset=int(_off), shape=tuple(_dset.shape))
        except Exception:
            _mm = None
        if _mm is None:
            _data = np.asarray(_dset[:]).T.astype(np.float32)
        else:
            # on-disk layout is (n_samples, n_channels); .T is a lazy view
            _data = _mm.T
        _n_ch = int(_data.shape[0])
    if _channels is None:
        _channels = ["Ch{}".format(_i) for _i in range(_n_ch)]
    _meta = {"format": "nwb", "source_file": _pre}
    if _rej_iv:
        _meta["rejected_intervals"] = _rej_iv
    return _data, _fs, _meta, _channels
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
    # Normalize synonyms to canonical names BEFORE embedding into the generated
    # script, so the standalone bundle only ever sees canonical operators
    # (highpass→bandpass:X, etc.). Unknown names fail loud here at generation
    # time rather than being silently skipped at run time.
    steps_list, _norm_notes = _normalize_steps(steps_list)
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
EASYBCI_SCENARIO: {scenario}
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

_SCENARIO = "{scenario}"

# Scenario-specific QC thresholds:
# research — lenient (preserving data is more important than rejecting artifacts)
# clinical — strict (safety: flag anything that could mask clinically-relevant morphology)
# deployment — moderate (balance between reliability and throughput)
_QC_THRESHOLDS = {{
    "research":   {{"finite_floor": 0.95, "snr_warn_drop_db": -6.0}},
    "clinical":   {{"finite_floor": 0.995, "snr_warn_drop_db": -2.0}},
    "deployment": {{"finite_floor": 0.99, "snr_warn_drop_db": -3.0}},
}}
_THR = _QC_THRESHOLDS.get(_SCENARIO, _QC_THRESHOLDS["research"])

_MNE_EXTS = {{
    ".fif", ".edf", ".bdf", ".set", ".ds", ".cnt", ".gdf",
    ".vhdr", ".vmrk", ".eeg", ".cdt", ".mff", ".sqd", ".con",
}}


def _qc_easybci_home():
    import os
    h = os.environ.get("EASYBCI_HOME")
    return Path(h) if h else (Path.home() / ".easybci")


def _qc_discover_io_plugin(path):
    """First registered io_loader plugin whose matches(path) is True (see
    pipeline.py _discover_io_plugin). Ensures qc.py reads the raw side the same
    way pipeline.py did — same custom loader, same load-time decimation.
    Scans repo-local ``code/io_loaders/`` first, then the machine-global dir."""
    import importlib.util as _ilu
    _dirs = []
    _self = globals().get("__file__")
    if _self:
        _dirs.append(Path(_self).resolve().parent / "io_loaders")
    _dirs.append(_qc_easybci_home() / "io_loaders")
    for d in _dirs:
        if not d.is_dir():
            continue
        for py in sorted(d.glob("*.py")):
            if py.name.startswith("_"):
                continue
            _mod_name = "_ebci_qcio_" + py.stem
            try:
                spec = _ilu.spec_from_file_location(_mod_name, str(py))
                mod = _ilu.module_from_spec(spec)
                sys.modules[_mod_name] = mod
                try:
                    spec.loader.exec_module(mod)
                    matches = getattr(mod, "matches", None)
                    load = getattr(mod, "load", None)
                    if callable(matches) and callable(load) and bool(matches(str(path))):
                        return load, py.stem
                finally:
                    sys.modules.pop(_mod_name, None)
            except Exception:
                continue
    return None, None


def _load_input(path, target_hz=None):
    p = Path(path)
    ext = p.suffix.lower()

    _pl, _pn = _qc_discover_io_plugin(p)
    if _pl is not None:
        import inspect as _insp
        _kw = {{}}
        if target_hz is not None:
            try:
                _pp = _insp.signature(_pl).parameters
                if "target_hz" in _pp or any(
                    x.kind == _insp.Parameter.VAR_KEYWORD for x in _pp.values()):
                    _kw["target_hz"] = target_hz
            except (ValueError, TypeError):
                pass
        _res = _pl(str(p), **_kw)
        if isinstance(_res, dict) and "data" in _res:
            return _res

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


def _stat_window_rows(arr, budget_bytes=128 * 1024 * 1024):
    """Time-window width (samples) so one (n_ch, w) float64 block fits budget."""
    n_ch = int(arr.shape[0]) if getattr(arr, "ndim", 0) else 1
    n_samp = int(arr.shape[-1]) if getattr(arr, "ndim", 0) else int(arr.shape[0])
    per_col = max(1, n_ch * 8)
    return max(1, min(n_samp, int(budget_bytes // per_col)))


def _chunked_stats(arr):
    """Exact mean/std/min/max/nan_frac without materializing the whole array.

    Reads the (n_channels, n_samples) source in time windows (only one window
    is float64 in RAM at a time — a memmap view stays on disk otherwise) and
    accumulates finite sum / sumsq / count and running min/max. The results are
    bit-equivalent to ``np.nanmean/np.nanstd/np.nanmin/np.nanmax`` over the full
    array and ``(~np.isfinite(arr)).mean()`` — population std (ddof=0), same as
    the original full-array path.
    """
    a2 = arr if getattr(arr, "ndim", 1) >= 1 else np.asarray(arr)
    n_samp = int(a2.shape[-1]) if getattr(a2, "ndim", 0) else 0
    total = float(a2.size) if hasattr(a2, "size") else float(np.asarray(a2).size)
    if n_samp == 0 or total == 0:
        return {{"mean": None, "std": None, "min": None, "max": None, "nan_frac": 1.0}}
    s = 0.0
    ss = 0.0
    cnt = 0.0
    vmin = np.inf
    vmax = -np.inf
    w = _stat_window_rows(a2)
    for t0 in range(0, n_samp, w):
        blk = np.asarray(a2[..., t0:min(n_samp, t0 + w)], dtype=np.float64)
        fin = np.isfinite(blk)
        if fin.any():
            vals = blk[fin]
            s += float(vals.sum())
            ss += float(np.square(vals).sum())
            cnt += float(vals.size)
            vmin = min(vmin, float(vals.min()))
            vmax = max(vmax, float(vals.max()))
    if cnt == 0.0:
        return {{"mean": None, "std": None, "min": None, "max": None, "nan_frac": 1.0}}
    mean = s / cnt
    var = max(ss / cnt - mean * mean, 0.0)
    return {{
        "mean": float(mean),
        "std": float(np.sqrt(var)),
        "min": float(vmin),
        "max": float(vmax),
        "nan_frac": float((total - cnt) / total),
    }}


def _finite_fraction(arr):
    """Fraction of finite samples — chunked equivalent of np.isfinite(arr).mean()."""
    a2 = arr if getattr(arr, "ndim", 1) >= 1 else np.asarray(arr)
    n_samp = int(a2.shape[-1]) if getattr(a2, "ndim", 0) else 0
    total = float(a2.size) if hasattr(a2, "size") else float(np.asarray(a2).size)
    if total == 0:
        return 0.0
    fin = 0.0
    w = _stat_window_rows(a2)
    for t0 in range(0, n_samp, w):
        blk = a2[..., t0:min(n_samp, t0 + w)]
        fin += float(np.count_nonzero(np.isfinite(np.asarray(blk))))
    return fin / total


def _bounded_psd_snr(arr, fs, *, max_seconds=120.0, fmax=80.0):
    """PSD-based SNR proxy on a bounded leading window (memory-safe).

    The original computed Welch PSD over the ENTIRE recording (float64 copy of
    the whole array) then took mean/std of the PSD matrix. For a 7 GB memmap
    that reintroduces the OOM we are removing, so we estimate the same proxy on
    the first ``max_seconds`` of signal — enough spectral content for a QC
    sanity score. Only this bounded slice is materialized.
    """
    n_samp = int(arr.shape[-1]) if getattr(arr, "ndim", 0) else 0
    if n_samp == 0:
        return 0.0
    win = min(n_samp, max(1, int(fs * max_seconds)))
    block = np.asarray(arr[..., :win], dtype=np.float64)
    _, p = _psd_welch(block, fs, fmax=fmax)
    return float(p.mean()) / (float(p.std()) + 1e-30)


def _compute_metrics(raw_d, proc):
    rb = raw_d["data"]
    ra = proc["data"]

    fs_b = float(raw_d["frequency"])
    fs_a = float(proc["frequency"])
    snr_before = _bounded_psd_snr(rb, fs_b)
    snr_after = _bounded_psd_snr(ra, fs_a)

    ra_finite_frac = _finite_fraction(ra)
    ra_ndim = getattr(ra, "ndim", 0)

    # Scenario-aware grading: research is lenient (preserving data over
    # rejecting borderline artifacts); clinical is strict.
    grade_warnings = []
    finite_ok = ra_finite_frac > _THR["finite_floor"]
    shape_ok = ra_ndim >= 2 and int(ra.shape[0]) > 0 and int(ra.shape[-1]) > 0
    snr_drop = snr_after - snr_before
    if snr_drop < _THR["snr_warn_drop_db"]:
        grade_warnings.append(
            "SNR dropped by {{:.1f}} dB (threshold {{:.1f}} dB for {{}} scenario)".format(
                abs(snr_drop), _THR["snr_warn_drop_db"], _SCENARIO))

    if not finite_ok:
        grade_warnings.append(
            "Finite fraction {{:.4f}} below {{}} threshold {{}}".format(
                ra_finite_frac, _SCENARIO, _THR["finite_floor"]))

    overall_grade = "Pass" if (finite_ok and shape_ok) else "Fail"

    return {{
        "before": {{
            "n_channels": int(rb.shape[0]) if getattr(rb, "ndim", 0) else 0,
            "n_samples": int(rb.shape[-1]) if getattr(rb, "ndim", 0) else 0,
            "frequency_hz": fs_b,
            "stats": _chunked_stats(rb),
            "psd_snr_estimate": snr_before,
        }},
        "after": {{
            "n_channels": int(ra.shape[0]) if getattr(ra, "ndim", 0) else 0,
            "n_samples": int(ra.shape[-1]) if getattr(ra, "ndim", 0) else 0,
            "frequency_hz": fs_a,
            "stats": _chunked_stats(ra),
            "psd_snr_estimate": snr_after,
        }},
        "overall": {{
            "grade": overall_grade,
            "scenario": _SCENARIO,
            "warnings": grade_warnings,
            "snr_change_db": round(snr_drop, 2),
        }},
    }}


def _write_report(qc_dir, session_id, subject_id, data_path, steps, metrics, stem=""):
    payload = {{
        "session_id": session_id,
        "subject_id": subject_id,
        "data_path": str(data_path),
        "steps": list(steps),
        "metrics": metrics,
    }}
    prefix = "{{}}_".format(stem) if stem else ""
    (qc_dir / "{{}}qc_report.json".format(prefix)).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
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
    (qc_dir / "{{}}qc_report.md".format(prefix)).write_text("\\n".join(md), encoding="utf-8")


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
        / "sub-{{}}".format(sub_id) / "ses-{{}}".format(ses) / "{{}}_qc_report.json".format(stem)
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

    # Load raw at the SAME decimated rate the pipeline used (concrete resample
    # target), so the before/after comparison is fair and qc.py doesn't OOM
    # re-reading a huge recording at native rate.
    _qc_tgt = None
    for _s in steps:
        if isinstance(_s, str) and _s.partition(":")[0].strip() == "resample":
            _pv = _s.partition(":")[2].strip().lower()
            if _pv and _pv != "auto":
                try:
                    _qc_tgt = float(_pv)
                except ValueError:
                    _qc_tgt = None
            break
    raw_d = _load_input(raw_path, target_hz=_qc_tgt)
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
    _write_report(qc_dir, "ses-" + ses, sub_id, raw_path, steps, metrics, stem)

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
    scenario: str = "research",
    inspection_report: Dict[str, Any] | None = None,
) -> str:
    """Generate the canonical qc.py script for figures + QC report.

    When ``inspection_report`` is provided, a comment block is prepended so
    the QC reader can see why the pipeline used certain parameters.
    """
    steps_list = list(steps)
    # Same canonical normalization as generate_pipeline_script — keep the qc
    # script's EASYBCI_STEPS marker in the canonical vocabulary too.
    steps_list, _ = _normalize_steps(steps_list)
    if inspection_report:
        line_freq = (inspection_report.get("psd_summary") or {}).get("power_line_peak_hz")
        if line_freq:
            steps_list = _override_notch_freq(steps_list, line_freq)
    body = _QC_SCRIPT_TEMPLATE.format(
        steps_repr=repr(steps_list),
        analysis_goal=analysis_goal,
        modality=modality,
        scenario=scenario,
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


_VIS_INVASIVE_TEMPLATE = '''"""Auto-generated multi-figure visualization (invasive modality — 4 single-state figs + per-channel time-frequency).

EASYBCI_GOAL: {analysis_goal}
EASYBCI_MODALITY: {modality}
EASYBCI_VERSION: 2
EASYBCI_CODE_STANDARD: 1.1.0

Standalone script — runs on `pip install numpy scipy matplotlib pynwb hdmf`.

Run: python vis.py <work_dir>
  - reads preprocessed.nwb from <work_dir>/preprocessed_output/preprocessed/sub-<id>/ses-<ses>/
  - writes 4 single-state figures + one time-frequency spectrogram PER CHANNEL
    to <work_dir>/preprocessed_output/figures/sub-<id>/ses-<ses>/
    (psd, channel_variance, amplitude_distribution, timeseries, <stem>_tf_NNN_<ch>.png)

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


_QC_DPI = 300
plt.rcParams.update({{
    "font.size": 9, "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "figure.titlesize": 12,
}})


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
    ax.legend(loc="upper right", ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=_QC_DPI, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def _plot_channel_variance(data, channels, out_path):
    if data.shape[0] == 0:
        raise ValueError("Channel-variance plot needs at least one channel")
    # Per-channel variance by time-window accumulation (var = E[x^2] - E[x]^2)
    # so a memmap source is never fully materialized; bit-equivalent to
    # np.var(data, axis=1). Only one (n_ch, w) window is resident at a time.
    n_ch, n_samp = int(data.shape[0]), int(data.shape[1])
    w = max(1, min(n_samp, int((128 * 1024 * 1024) // max(1, n_ch * 8))))
    s = np.zeros(n_ch, dtype=np.float64)
    ss = np.zeros(n_ch, dtype=np.float64)
    for t0 in range(0, n_samp, w):
        blk = np.asarray(data[:, t0:min(n_samp, t0 + w)], dtype=np.float64)
        s += blk.sum(axis=1)
        ss += np.square(blk).sum(axis=1)
    mean = s / max(1, n_samp)
    var = np.maximum(ss / max(1, n_samp) - np.square(mean), 0.0)
    fig, ax = plt.subplots(figsize=(max(6.0, data.shape[0] * 0.3), 3))
    xs = np.arange(data.shape[0])
    labels = [channels[i] if i < len(channels) else "Ch{{}}".format(i)
              for i in range(data.shape[0])]
    ax.bar(xs, var, color="#3b82f6", alpha=0.85)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=75)
    ax.set_ylabel("Variance")
    ax.set_title("Per-channel Variance (processed)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=_QC_DPI, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def _plot_amplitude_dist(data, out_path):
    # Reservoir-free bounded sampling: draw up to 100k samples from evenly
    # spaced time windows instead of flattening the whole array (a memmap would
    # otherwise be fully paged in). The histogram shape is unchanged.
    n_ch = int(data.shape[0]) if getattr(data, "ndim", 0) else 0
    n_samp = int(data.shape[-1]) if getattr(data, "ndim", 0) else 0
    if n_ch == 0 or n_samp == 0:
        raise ValueError("Amplitude distribution plot needs samples")
    target = 100000
    rng = np.random.default_rng(EASYBCI_SEED)
    total = n_ch * n_samp
    if total <= target:
        flat = np.asarray(data, dtype=np.float32).reshape(-1)
    else:
        # sample whole time-columns until we have enough values
        per_col = n_ch
        n_cols = max(1, min(n_samp, (target // max(1, per_col)) + 1))
        cols = np.sort(rng.choice(n_samp, size=n_cols, replace=False))
        picks = [np.asarray(data[:, c], dtype=np.float32) for c in cols]
        flat = np.concatenate(picks) if picks else np.asarray([], dtype=np.float32)
        if flat.size > target:
            flat = rng.choice(flat, target, replace=False)
    if flat.size == 0:
        raise ValueError("Amplitude distribution plot needs samples")
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(flat, bins=80, color="#3b82f6", alpha=0.85, edgecolor="none")
    ax.set_xlabel("Amplitude")
    ax.set_ylabel("Count")
    ax.set_title("Amplitude Distribution (processed)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=_QC_DPI, facecolor="white", bbox_inches="tight")
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
    ax.legend(loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=_QC_DPI, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def _safe_name(name):
    keep = [c if (c.isalnum() or c in "-_") else "_" for c in str(name)]
    s = "".join(keep).strip("_")
    return s or "ch"


def _plot_timefreq_per_channel(data, fs, channels, fig_dir, stem):
    """Per-channel time-frequency spectrograms (invasive modality, no channel cap).

    Each channel is read as a single (n_samp,) vector, turned into a
    spectrogram, and its figure is closed immediately, so peak memory is one
    channel + one figure regardless of channel count. Returns written names.
    """
    from scipy.signal import spectrogram as _spectrogram
    from scipy.ndimage import gaussian_filter as _gaussian_filter
    n_ch, n_samp = int(data.shape[0]), int(data.shape[1])
    if n_ch == 0 or n_samp == 0:
        raise ValueError("time-frequency plot needs samples")
    nperseg = int(min(1024, n_samp))
    noverlap = int(nperseg * 0.75)
    high_cut = min(float(fs) / 2.0, 250.0)
    written = []
    for ch in range(n_ch):
        x = np.asarray(data[ch], dtype=np.float32)
        freqs, times, power = _spectrogram(
            x, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap,
            detrend="constant", scaling="density", mode="psd",
        )
        keep = freqs <= high_cut
        if not np.any(keep):
            keep = np.ones_like(freqs, dtype=bool)
        spec = 10.0 * np.log10(power[keep] + 1e-12)
        spec = _gaussian_filter(spec, sigma=(1.0, 1.0))
        vmin, vmax = np.nanpercentile(spec, [5, 95])
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
            vmin, vmax = float(np.nanmin(spec)), float(np.nanmax(spec) + 1.0)
        label = channels[ch] if ch < len(channels) else "Ch{{}}".format(ch)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        im = ax.imshow(
            spec, aspect="auto", origin="lower",
            extent=[times[0], times[-1], freqs[keep][0], freqs[keep][-1]],
            cmap="magma", vmin=vmin, vmax=vmax,
        )
        ax.set_title("{{}} time-frequency".format(label))
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        fig.colorbar(im, ax=ax, label="Power (dB)")
        fig.tight_layout()
        out_path = fig_dir / "{{}}_tf_{{:03d}}_{{}}.png".format(stem, ch, _safe_name(label))
        fig.savefig(str(out_path), dpi=_QC_DPI, facecolor="white", bbox_inches="tight")
        plt.close(fig)
        written.append(out_path.name)
    return written


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

    # Per-channel time-frequency spectrograms (invasive-only, no channel cap).
    try:
        saved.extend(_plot_timefreq_per_channel(data, fs, channels, fig_dir, stem))
    except Exception as exc:
        errors.append("timefreq: {{}}".format(exc))
        print("[vis] timefreq: {{}} for file_id={{}}".format(exc, file_id), file=sys.stderr)
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


_QC_DPI = 300
plt.rcParams.update({{
    "font.size": 9, "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "figure.titlesize": 12,
}})


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
    ax.legend(loc="upper right", ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=_QC_DPI, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def _plot_channel_variance(data, channels, out_path):
    if data.shape[0] == 0:
        raise ValueError("Channel-variance plot needs at least one channel")
    # Per-channel variance by time-window accumulation (var = E[x^2] - E[x]^2)
    # so a memmap source is never fully materialized; bit-equivalent to
    # np.var(data, axis=1). Only one (n_ch, w) window is resident at a time.
    n_ch, n_samp = int(data.shape[0]), int(data.shape[1])
    w = max(1, min(n_samp, int((128 * 1024 * 1024) // max(1, n_ch * 8))))
    s = np.zeros(n_ch, dtype=np.float64)
    ss = np.zeros(n_ch, dtype=np.float64)
    for t0 in range(0, n_samp, w):
        blk = np.asarray(data[:, t0:min(n_samp, t0 + w)], dtype=np.float64)
        s += blk.sum(axis=1)
        ss += np.square(blk).sum(axis=1)
    mean = s / max(1, n_samp)
    var = np.maximum(ss / max(1, n_samp) - np.square(mean), 0.0)
    fig, ax = plt.subplots(figsize=(max(6.0, data.shape[0] * 0.3), 3))
    xs = np.arange(data.shape[0])
    labels = [channels[i] if i < len(channels) else "Ch{{}}".format(i)
              for i in range(data.shape[0])]
    ax.bar(xs, var, color="#3b82f6", alpha=0.85)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=75)
    ax.set_ylabel("Variance")
    ax.set_title("Per-channel Variance (processed)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=_QC_DPI, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def _plot_amplitude_dist(data, out_path):
    # Reservoir-free bounded sampling: draw up to 100k samples from evenly
    # spaced time windows instead of flattening the whole array (a memmap would
    # otherwise be fully paged in). The histogram shape is unchanged.
    n_ch = int(data.shape[0]) if getattr(data, "ndim", 0) else 0
    n_samp = int(data.shape[-1]) if getattr(data, "ndim", 0) else 0
    if n_ch == 0 or n_samp == 0:
        raise ValueError("Amplitude distribution plot needs samples")
    target = 100000
    rng = np.random.default_rng(EASYBCI_SEED)
    total = n_ch * n_samp
    if total <= target:
        flat = np.asarray(data, dtype=np.float32).reshape(-1)
    else:
        # sample whole time-columns until we have enough values
        per_col = n_ch
        n_cols = max(1, min(n_samp, (target // max(1, per_col)) + 1))
        cols = np.sort(rng.choice(n_samp, size=n_cols, replace=False))
        picks = [np.asarray(data[:, c], dtype=np.float32) for c in cols]
        flat = np.concatenate(picks) if picks else np.asarray([], dtype=np.float32)
        if flat.size > target:
            flat = rng.choice(flat, target, replace=False)
    if flat.size == 0:
        raise ValueError("Amplitude distribution plot needs samples")
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(flat, bins=80, color="#3b82f6", alpha=0.85, edgecolor="none")
    ax.set_xlabel("Amplitude")
    ax.set_ylabel("Count")
    ax.set_title("Amplitude Distribution (processed)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=_QC_DPI, facecolor="white", bbox_inches="tight")
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
    ax.legend(loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=_QC_DPI, facecolor="white", bbox_inches="tight")
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
    fig.savefig(str(out_path), dpi=_QC_DPI, facecolor="white", bbox_inches="tight")
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
EASYBCI_VERSION: 3

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


def _align_continuous_labels(labels, n_samples, fs, dur, stride):
    """Reduce L3 continuous (per-sample) labels to one label per sliding window.

    L3_continuous labels carry one value PER SAMPLE (length == n_samples). The
    sliding-window epoching produces far fewer windows, so loading labels.npy
    verbatim mislabels every epoch. This maps each window to the label of its
    CENTRE sample — correct for both continuous regression and per-window
    classification — using the EXACT window geometry of ``_sliding``.

    * length already == n_windows  -> pass through unchanged,
    * length == n_samples          -> sample the centre of each window,
    * anything else                -> truncate to n_windows with a loud warning
      (never silently emit a mismatched-length label array).
    """
    labels = np.asarray(labels)
    win = int(dur * fs)
    step = max(int(stride * fs), 1)
    starts = list(range(0, n_samples - win + 1, step))
    if not starts:
        starts = [0]
        win = min(win, n_samples)
    n_windows = len(starts)

    if labels.shape[0] == n_windows:
        return labels
    if labels.shape[0] == n_samples:
        centers = [min(s + win // 2, n_samples - 1) for s in starts]
        return labels[np.asarray(centers, dtype=int)]
    # Neither per-window nor per-sample — do not guess silently.
    print("[ai_ready] WARNING: continuous label length {{}} matches neither "
          "n_windows ({{}}) nor n_samples ({{}}); truncating to {{}} to align."
          .format(labels.shape[0], n_windows, n_samples, n_windows),
          file=sys.stderr)
    if labels.shape[0] >= n_windows:
        return labels[:n_windows]
    # Fewer than n_windows: pad by repeating the last label.
    pad = np.full(n_windows - labels.shape[0], labels[-1] if labels.shape[0] else 0)
    return np.concatenate([labels, pad])


def _epoch_from_events(data, fs, events, tmin, tmax):
    win_samples = int((tmax - tmin) * fs)
    pre = int(-tmin * fs)
    epochs = []
    kept_labels = []
    for ev in events:
        s = int(ev["start"]) - pre
        e = s + win_samples
        if 0 <= s and e <= data.shape[1]:
            epochs.append(data[:, s:e])
            kept_labels.append(ev.get("label", 0))
    # Return labels aligned to the KEPT epochs. Out-of-bounds events are
    # dropped from both, so epochs.shape[0] == len(labels) always holds. A
    # caller that labels all `events` would silently misalign every epoch
    # after the first dropped event.
    return (
        np.asarray(epochs, dtype=np.float32),
        np.asarray(kept_labels),
    )


def _remap_events_after_reject(events, rejected_intervals, fs):
    """Remap event onsets across reject_by_labels excised windows.

    reject_by_labels (pipeline.py) deletes samples, so an onset measured on the
    ORIGINAL time axis no longer points at the right sample post-reject. The
    excised windows are carried as [onset_s, offset_s] pairs in seconds
    (time-base invariant — correct even after a later resample). For each event:
      * drop it if its onset falls inside an excised window,
      * else shift it left by the total excised time BEFORE it.
    ``onset_s`` (original seconds) is authoritative; ``start`` is recomputed
    against the post-reject axis at the current fs.
    """
    if not rejected_intervals:
        return list(events)
    ivs = sorted((float(a), float(b)) for a, b in rejected_intervals)
    out = []
    for ev in events:
        onset_s = ev.get("onset_s")
        if onset_s is None:
            onset_s = (float(ev.get("start", 0)) / fs) if fs else 0.0
        onset_s = float(onset_s)
        excised_before = 0.0
        dropped = False
        for a, b in ivs:
            if a <= onset_s < b:
                dropped = True
                break
            if b <= onset_s:
                excised_before += (b - a)
        if dropped:
            continue
        new_s = onset_s - excised_before
        ne = dict(ev)
        ne["onset_s"] = new_s
        ne["start"] = int(round(new_s * fs)) if fs else int(ev.get("start", 0))
        out.append(ne)
    return out


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
        # No explicit file → auto-discover EVERY preprocessed nwb under the
        # work_dir and build one input entry per file. Taking only the first
        # match silently processed a single arbitrary recording in a
        # multi-subject work_dir.
        base = work_dir / "preprocessed_output" / "preprocessed"
        discovered = []
        if base.is_dir():
            for sub_dir in sorted(base.glob("sub-*")):
                if not sub_dir.is_dir():
                    continue
                for sess_dir in sorted(sub_dir.iterdir()):
                    if not sess_dir.is_dir():
                        continue
                    for nwb in sorted(sess_dir.glob("*_preprocessed.nwb")):
                        discovered.append(nwb)
        if discovered:
            entries = []
            for nwb in discovered:
                parts = nwb.parts
                sub_part = next((p for p in parts if p.startswith("sub-")), "sub-unknown")
                idx = parts.index(sub_part) if sub_part in parts else -1
                ses_part = parts[idx + 1] if 0 <= idx < len(parts) - 1 else "ses-001"
                entries.append({{
                    "data_path": str(nwb),
                    "stem_safe": nwb.stem.replace("_preprocessed", ""),
                    "subject_id": sub_part.replace("sub-", ""),
                    "session_id": ses_part.replace("ses-", ""),
                    "file_id": "legacy",
                    "events_path": None,
                }})
            return entries
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
        events = _load_events_csv(events_path, fs)
    if not events and isinstance(meta, dict):
        events = meta.get("events") or None

    # reject_by_labels (pipeline.py) excised time windows before this NWB was
    # written; remap event onsets across those gaps (seconds domain) so epochs
    # land on the correct post-reject samples. Events inside excised windows
    # are dropped.
    if events:
        rejected_intervals = None
        if isinstance(meta, dict):
            rejected_intervals = meta.get("rejected_intervals")
        if rejected_intervals:
            n_before = len(events)
            events = _remap_events_after_reject(events, rejected_intervals, fs)
            print("[ai_ready] remapped events across {{}} reject_by_labels window(s): "
                  "{{}} -> {{}} events".format(len(rejected_intervals), n_before, len(events)),
                  file=sys.stderr)

    if EVENTS_PRESENT and events:
        epochs, labels = _epoch_from_events(data, fs, events, tmin=-0.2, tmax=SEGMENT_DURATION)
    else:
        epochs = _sliding(data, fs, SEGMENT_DURATION, STRIDE)
        labels = None
        if LABEL_CONFIG and LABEL_CONFIG.get("label_type") == "L3_continuous":
            lp = LABEL_CONFIG.get("label_path", "labels.npy")
            try:
                raw_labels = np.load(lp)
            except OSError:
                raw_labels = None
            if raw_labels is not None:
                # L3 labels are per-sample; reduce to one-per-window so
                # len(labels) == epochs.shape[0] (was silently mismatched).
                labels = _align_continuous_labels(
                    raw_labels, data.shape[1], fs, SEGMENT_DURATION, STRIDE
                )

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


def _load_events_csv(csv_path, fs):
    """Load a sidecar event table into a list of start/code/label dicts.

    Supports two schemas, auto-detected per row:

    * legacy CSV -- comma-separated sample_index / time_s / event_code.
      time_s (seconds) is the authoritative, rate-invariant time base and is
      mapped onto the current (resampled) fs; sample_index is a raw-rate offset
      used only as a fallback when time_s is absent.
    * BIDS TSV -- tab-separated onset (seconds) / duration / trial_type /
      value. onset is converted to a sample offset via fs; the label prefers
      trial_type then falls back to value.

    Separator is chosen by extension (.tsv -> tab, else comma).
    """
    import csv as _csv

    delimiter = "\t" if str(csv_path).lower().endswith((".tsv", ".tab")) else ","
    out = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = _csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            onset_s = None
            if "onset" in row and row.get("onset") not in (None, ""):
                # BIDS: onset is in seconds → convert to a sample offset.
                try:
                    onset_s = float(row["onset"])
                    start = int(round(onset_s * float(fs))) if fs else 0
                except (TypeError, ValueError):
                    start = 0
                    onset_s = None
                raw = row.get("trial_type")
                if raw is None or str(raw).strip() in ("", "n/a"):
                    raw = row.get("value")
            else:
                # Legacy. time_s (seconds) is the authoritative time base — it
                # survives the pipeline's resample, so map it onto the current
                # (resampled) fs. sample_index is captured at the RAW rate; using
                # it verbatim against a resampled signal offsets every epoch by
                # the resample ratio. Fall back to sample_index only when the
                # sidecar has no time_s column.
                _t = (row.get("time_s") or "").strip()
                if _t:
                    try:
                        onset_s = float(_t)
                        start = int(round(onset_s * float(fs))) if fs else 0
                    except (TypeError, ValueError):
                        start = 0
                        onset_s = None
                else:
                    try:
                        start = int(float(row.get("sample_index") or 0))
                        onset_s = (start / float(fs)) if fs else None
                    except (TypeError, ValueError):
                        start = 0
                        onset_s = None
                raw = row.get("event_code")

            code = str(raw).strip() if raw is not None else ""
            label = int(code) if code.lstrip("-").isdigit() else code
            # onset_s (original seconds) lets _remap_events_after_reject shift
            # onsets across reject_by_labels gaps in a time-base-invariant way.
            out.append({{"start": start, "code": label, "label": label,
                        "onset_s": onset_s}})
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

    # Normalize ratios to sum to 1.0 so the cumsum-threshold logic in the
    # sequential/temporal splitters partitions correctly even when the caller
    # passes raw weights (e.g. {{"train": 70, "test": 30}}). Without this the
    # tail split is silently starved (sum>1) or over-filled (sum<1).
    _total = float(sum(ratios.values()))
    if _total > 0:
        ratios = {{k: (v / _total) for k, v in ratios.items()}}
    else:
        # Degenerate all-zero weights: fall back to equal shares.
        _nk = len(ratios) or 1
        ratios = {{k: (1.0 / _nk) for k in ratios}}

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
    # Reserve gap samples so train/test are not temporally adjacent. Guard
    # against a gap larger than the data (keep at least 1 usable sample).
    # NOTE: must be `n - total_gap`, not `max(n - total_gap, n)` — the latter
    # is always n (total_gap >= 0), silently disabling the anti-leakage gap.
    usable = max(n - total_gap, 1)

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
