"""MATLAB struct-container ``.mat`` loader.

Many acquisition systems export one top-level MATLAB struct that bundles the
continuous signal with its metadata (sampling rate, channel names, trial
onsets, labels, impedance, ...). Field names vary by vendor, so this loader
resolves each role from a built-in alias list (the floor) plus shape/dtype
evidence from the data itself, and records which field fed which role in
``meta["mat_field_map"]``. Every field it could not map is listed in
``meta["mat_struct_fields"]`` so the agent can see the full structure and
register a dedicated io_loader when the heuristics are wrong.

Depends on numpy + scipy + stdlib only (bundled verbatim into generated repos
as an io_loader plugin). Source data is read-only.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

FORMAT_NAME = "mat_struct"

_FS_KEYS = (
    "srate", "fs", "fsample", "sfreq", "frequency", "sample_rate",
    "sampling_rate", "sample_rate_hz", "sampling_rate_hz", "samplerate",
)
_CH_KEYS = (
    "channels", "ch_names", "channel_names", "chan_names", "labels", "label",
    "electrodes", "chanlabels",
)
_ONSET_KEYS = (
    "trial", "trials", "trial_start", "trial_starts", "trial_onsets",
    "onsets", "onset", "event_onsets", "event_samples", "events", "latency",
    "latencies", "stim_onsets", "cue_onsets", "pos",
)
_LABEL_KEYS = (
    "y", "labels", "label", "trial_labels", "event_labels", "targets",
    "target", "class", "classes_idx", "condition", "conditions", "type",
    "event_type", "event_codes", "codes",
)
_CLASS_NAME_KEYS = (
    "classes", "class_names", "classnames", "label_names", "event_names",
    "condition_names", "classlabels",
)
_DUR_RX = re.compile(
    r"^(trial|epoch|event)s?_?(expected_)?(samples|length|len|duration|dur)$"
    r"|^(durations?|trial_len|window_samples)$",
    re.IGNORECASE,
)
_TRIAL_FLAG_RX = re.compile(r"(artifact|invalid|missing|bad|reject)", re.IGNORECASE)
_UNIT_KEYS = ("unit", "units", "data_unit", "signal_unit")

# Signal candidates must be clearly bigger than per-trial / per-channel tables.
_MIN_SIGNAL_SAMPLES = 1000

_CACHE: Dict[str, Any] = {}


# --------------------------------------------------------------------------
# struct helpers
# --------------------------------------------------------------------------
def _is_struct(v: Any) -> bool:
    return hasattr(v, "_fieldnames")


def _fields(s: Any) -> List[str]:
    return list(getattr(s, "_fieldnames", []) or [])


def _get(s: Any, name: str) -> Any:
    return getattr(s, name, None)


def _as_str_list(v: Any) -> Optional[List[str]]:
    if v is None:
        return None
    if isinstance(v, str):
        return [v.strip()]
    arr = np.asarray(v)
    if arr.dtype.kind in ("U", "S"):
        return [str(x).strip() for x in arr.ravel()]
    if arr.dtype == object and arr.size and all(isinstance(x, str) for x in arr.ravel()):
        return [str(x).strip() for x in arr.ravel()]
    return None


def _as_scalar(v: Any) -> Optional[float]:
    try:
        arr = np.asarray(v)
        if arr.dtype.kind in "biuf" and arr.size == 1:
            f = float(arr.ravel()[0])
            return f if np.isfinite(f) else None
    except Exception:
        pass
    return None


def _as_numeric_1d(v: Any) -> Optional[np.ndarray]:
    try:
        arr = np.asarray(v)
    except Exception:
        return None
    if arr.dtype.kind not in "biuf" or arr.ndim != 1 or arr.size == 0:
        return None
    return arr


def _describe(v: Any) -> str:
    if _is_struct(v):
        return "struct{" + ",".join(_fields(v)[:12]) + ("…" if len(_fields(v)) > 12 else "") + "}"
    if isinstance(v, str):
        return "str[{}]".format(v[:40])
    arr = np.asarray(v)
    if arr.dtype == object:
        return "cell{}".format(list(arr.shape))
    if arr.size == 1:
        return "{}={}".format(arr.dtype, arr.ravel()[0])
    return "{}{}".format(arr.dtype, list(arr.shape))


def _load_raw(path: str):
    import scipy.io

    p = Path(path)
    key = str(p.resolve())
    try:
        mtime = p.stat().st_mtime
    except OSError:
        return None
    hit = _CACHE.get(key)
    if hit is not None and hit[0] == mtime:
        return hit[1]
    try:
        mat = scipy.io.loadmat(str(p), squeeze_me=True, struct_as_record=False)
    except Exception:
        return None
    _CACHE.clear()
    _CACHE[key] = (mtime, mat)
    return mat


def _container(mat: Dict[str, Any]) -> Tuple[Optional[str], Any]:
    """Return the single top-level struct that holds the signal, if any."""
    structs = [(k, v) for k, v in mat.items() if not k.startswith("__") and _is_struct(v)]
    if len(structs) != 1:
        return None, None
    name, s = structs[0]
    fset = set(_fields(s))
    # EEGLAB / FieldTrip have dedicated readers — never hijack them.
    if name == "EEG" and {"data", "srate"} <= fset:
        return None, None
    if "fsample" in fset and "trial" in fset:
        tr = np.asarray(_get(s, "trial"))
        if tr.dtype == object or tr.ndim >= 2:
            return None, None
    return name, s


def _find_signal(s: Any) -> Tuple[Optional[str], Optional[np.ndarray]]:
    best_name, best = None, None
    for f in _fields(s):
        v = _get(s, f)
        if _is_struct(v):
            continue
        arr = np.asarray(v) if not isinstance(v, str) else None
        if arr is None or arr.dtype.kind not in "biuf" or arr.ndim != 2:
            continue
        if max(arr.shape) < _MIN_SIGNAL_SAMPLES or min(arr.shape) < 1:
            continue
        if best is None or arr.size > best.size:
            best_name, best = f, arr
    return best_name, best


def _find_fs(s: Any, depth: int = 0) -> Tuple[Optional[str], Optional[float]]:
    lower = {f.lower(): f for f in _fields(s)}
    for k in _FS_KEYS:
        if k in lower:
            val = _as_scalar(_get(s, lower[k]))
            if val and val > 0:
                return lower[k], val
    if depth < 2:
        for f in _fields(s):
            v = _get(s, f)
            if _is_struct(v):
                name, val = _find_fs(v, depth + 1)
                if val:
                    return f + "." + name, val
    return None, None


# --------------------------------------------------------------------------
# role resolvers
# --------------------------------------------------------------------------
def _resolve_channels(s: Any, sig: np.ndarray) -> Tuple[Optional[str], Optional[List[str]], bool]:
    """Return (field, names, transpose) — transpose=True when sig is samples×channels."""
    lower = {f.lower(): f for f in _fields(s)}
    candidates = [lower[k] for k in _CH_KEYS if k in lower]
    candidates += [f for f in _fields(s) if "chan" in f.lower() and f not in candidates]
    for f in candidates:
        names = _as_str_list(_get(s, f))
        if not names:
            continue
        if len(names) == sig.shape[1] and len(names) != sig.shape[0]:
            return f, names, True
        if len(names) == sig.shape[0]:
            return f, names, False
    return None, None, sig.shape[0] > sig.shape[1]


def _resolve_onsets(s: Any, n_samp: int, fs: float, used: set) -> Tuple[Optional[str], Optional[np.ndarray], Optional[str]]:
    """Return (field, onsets_in_samples_0based, convention)."""
    lower = {f.lower(): f for f in _fields(s)}
    ordered = [lower[k] for k in _ONSET_KEYS if k in lower]
    for f in ordered:
        if f in used:
            continue
        arr = _as_numeric_1d(_get(s, f))
        if arr is None or arr.size < 2 or arr.size >= n_samp // 10:
            continue
        a = arr.astype(np.float64)
        if not np.all(np.isfinite(a)) or np.any(np.diff(a) < 0):
            continue
        is_int = np.allclose(a, np.round(a))
        if is_int and a.min() >= 0 and a.max() <= n_samp:
            # MATLAB indices are 1-based; a 0 anywhere proves 0-based storage.
            if a.min() >= 1:
                return f, (a - 1).astype(np.int64), "matlab_1based_samples"
            return f, a.astype(np.int64), "0based_samples"
        if not is_int and fs > 0 and a.min() >= 0 and a.max() <= n_samp / fs:
            return f, np.round(a * fs).astype(np.int64), "seconds"
    return None, None, None


def _resolve_labels(s: Any, n_ev: int, used: set) -> Tuple[Optional[str], Optional[List[Any]]]:
    lower = {f.lower(): f for f in _fields(s)}
    for k in _LABEL_KEYS:
        f = lower.get(k)
        if f is None or f in used:
            continue
        v = _get(s, f)
        names = _as_str_list(v)
        if names is not None and len(names) == n_ev:
            return f, names
        arr = _as_numeric_1d(v)
        if arr is not None and arr.size == n_ev:
            return f, arr.tolist()
    return None, None


def _resolve_class_names(s: Any, used: set) -> Tuple[Optional[str], Optional[List[str]]]:
    lower = {f.lower(): f for f in _fields(s)}
    for k in _CLASS_NAME_KEYS:
        f = lower.get(k)
        if f is None or f in used:
            continue
        names = _as_str_list(_get(s, f))
        if names:
            return f, names
    return None, None


def _label_descriptions(labels: List[Any], class_names: Optional[List[str]]) -> Tuple[List[str], Optional[str]]:
    if not labels:
        return [], None
    if not class_names or not all(isinstance(x, (int, float, np.integer, np.floating)) for x in labels):
        return [str(x).strip() for x in labels], None
    vals = np.asarray(labels, dtype=np.float64)
    if not np.allclose(vals, np.round(vals)):
        return [str(x) for x in labels], None
    iv = vals.astype(int)
    n = len(class_names)
    if iv.min() >= 0 and iv.max() < n and (iv.min() == 0 or iv.max() < n - 1):
        return [class_names[i] for i in iv], "0based"
    if iv.min() >= 1 and iv.max() <= n:
        return [class_names[i - 1] for i in iv], "1based"
    return [str(int(x)) for x in iv], "unmapped"


def _resolve_durations(s: Any, n_ev: int, fs: float, used: set) -> Tuple[Optional[str], Optional[np.ndarray]]:
    for f in _fields(s):
        if f in used or not _DUR_RX.match(f):
            continue
        v = _get(s, f)
        in_samples = "sample" in f.lower() or "len" in f.lower()
        scalar = _as_scalar(v)
        if scalar is not None and scalar > 0:
            arr = np.full(n_ev, scalar, dtype=np.float64)
        else:
            arr = _as_numeric_1d(v)
            if arr is None or arr.size != n_ev:
                continue
            arr = arr.astype(np.float64)
        if in_samples:
            if not fs:
                continue
            arr = arr / fs
        if np.all(np.isfinite(arr)) and np.all(arr > 0):
            return f, arr
    return None, None


def _resolve_trial_flags(s: Any, n_ev: int, used: set) -> Tuple[Dict[str, List[float]], List[str]]:
    """Per-trial numeric quality fields (artifact / invalid / missing counts)."""
    flags: Dict[str, List[float]] = {}
    per_event: List[str] = [""] * n_ev
    for f in _fields(s):
        if f in used or not _TRIAL_FLAG_RX.search(f):
            continue
        arr = _as_numeric_1d(_get(s, f))
        if arr is None or arr.size != n_ev:
            continue
        a = np.nan_to_num(arr.astype(np.float64), nan=0.0)
        flags[f] = a.tolist()
        for i in np.flatnonzero(a != 0):
            per_event[i] = (per_event[i] + ";" if per_event[i] else "") + "{}={:g}".format(f, a[i])
    return flags, per_event


def _find_impedance(s: Any, depth: int = 0) -> Tuple[Optional[str], Any]:
    for f in _fields(s):
        v = _get(s, f)
        if "imped" in f.lower() or "kohm" in f.lower():
            return f, v
        if _is_struct(v) and depth < 1:
            name, sub = _find_impedance(v, depth + 1)
            if sub is not None:
                return f + "." + name, sub
    return None, None


def _impedance_bad_channels(imp: Any, channels: List[str]) -> Dict[str, Any]:
    """Derive bad channels from an impedance block (struct or per-channel array)."""
    out: Dict[str, Any] = {"bad": [], "reasons": {}, "values_kohm": {}, "warnings": []}
    ch_index = {c.strip().lower(): c for c in channels}
    poor_thr = None
    imp_ch: Optional[List[str]] = None
    values: Optional[np.ndarray] = None
    open_flags: Optional[np.ndarray] = None
    measured: Optional[np.ndarray] = None
    if _is_struct(imp):
        for f in _fields(imp):
            lf = f.lower()
            v = _get(imp, f)
            if "poor" in lf and "thresh" in lf:
                poor_thr = _as_scalar(v)
            elif poor_thr is None and lf in ("threshold_kohm", "max_kohm", "threshold"):
                poor_thr = _as_scalar(v)
            elif lf in ("channels", "labels", "ch_names", "channel_names"):
                imp_ch = _as_str_list(v)
            elif ("kohm" in lf or lf in ("values", "impedance", "impedances")) and "thresh" not in lf:
                arr = _as_numeric_1d(v)
                if arr is not None:
                    values = arr.astype(np.float64)
            elif "open" in lf:
                arr = _as_numeric_1d(v)
                if arr is not None:
                    open_flags = arr
            elif "measured" in lf:
                arr = _as_numeric_1d(v)
                if arr is not None:
                    measured = arr
    else:
        arr = _as_numeric_1d(imp)
        if arr is not None and arr.size == len(channels):
            values = arr.astype(np.float64)
    if imp_ch is None:
        imp_ch = list(channels) if values is not None and values.size == len(channels) else None
    if values is None or imp_ch is None or len(imp_ch) != values.size:
        return out
    thr = poor_thr if poor_thr and poor_thr > 0 else 20.0
    out["poor_threshold_kohm"] = thr
    out["threshold_source"] = "file" if poor_thr else "default"
    high: List[Tuple[str, float]] = []
    for i, name in enumerate(imp_ch):
        ch = ch_index.get(name.strip().lower())
        if ch is None:
            continue
        val = float(values[i])
        out["values_kohm"][ch] = None if not np.isfinite(val) else round(val, 2)
        is_open = open_flags is not None and i < open_flags.size and bool(open_flags[i])
        was_measured = measured is None or (i < measured.size and bool(measured[i]))
        if is_open or (not np.isfinite(val) and was_measured):
            out["bad"].append(ch)
            out["reasons"][ch] = "open_circuit" if is_open else "impedance_nan"
        elif np.isfinite(val) and val > thr:
            high.append((ch, val))
    # A default threshold flagging most of the cap means the system simply
    # runs at higher impedance — surface it, never mass-drop on a guess.
    n_meas = sum(1 for v in out["values_kohm"].values() if v is not None)
    if out["threshold_source"] == "default" and n_meas and len(high) > 0.5 * n_meas:
        out["warnings"].append(
            "{}/{} channels exceed the default {:g} kOhm impedance threshold; "
            "not marking them bad (system may be high-impedance by design)".format(
                len(high), n_meas, thr))
    else:
        for ch, val in high:
            out["bad"].append(ch)
            out["reasons"][ch] = "impedance_high:{:.1f}kOhm>{:g}".format(val, thr)
    return out


# --------------------------------------------------------------------------
# public io_loader API
# --------------------------------------------------------------------------
def matches(path: str) -> bool:
    p = Path(path)
    if p.suffix.lower() != ".mat" or not p.is_file():
        return False
    try:
        import scipy.io

        info = scipy.io.whosmat(str(p))
    except Exception:
        return False
    # whosmat reports some struct exports as 1x1 'cell'; only rule out files
    # that contain plain numeric/char arrays exclusively.
    if not any(cls in ("struct", "cell", "object") for _, _, cls in info):
        return False
    mat = _load_raw(str(p))
    if mat is None:
        return False
    name, s = _container(mat)
    if s is None:
        return False
    return _find_signal(s)[1] is not None


def load(path: str, inspect_only: bool = False, target_hz: Optional[float] = None) -> dict:
    mat = _load_raw(path)
    # The cache only bridges matches() -> load(); never pin a recording in memory.
    _CACHE.clear()
    if mat is None:
        raise ValueError("cannot read MATLAB file: {}".format(path))
    cname, s = _container(mat)
    if s is None:
        raise ValueError("no single struct container in {}".format(path))
    sig_field, sig = _find_signal(s)
    if sig is None:
        raise ValueError("no 2-D signal array in struct '{}'".format(cname))

    used = {sig_field}
    field_map: Dict[str, str] = {"container": cname, "signal": sig_field}
    degraded: List[str] = []

    fs_field, fs = _find_fs(s)
    if fs:
        field_map["fs"] = fs_field
        used.add(fs_field.split(".")[0])
    else:
        degraded.append("sampling_rate_not_found")
        fs = 1.0

    ch_field, channels, transpose = _resolve_channels(s, sig)
    data = np.asarray(sig.T if transpose else sig, dtype=np.float32)
    n_ch, n_samp = data.shape
    field_map["orientation"] = "samples_x_channels" if transpose else "channels_x_samples"
    if channels:
        field_map["channels"] = ch_field
        used.add(ch_field)
    else:
        channels = ["Ch{}".format(i) for i in range(n_ch)]

    meta: Dict[str, Any] = {
        "format": FORMAT_NAME,
        "source_file": str(path),
        "n_channels": int(n_ch),
        "n_samples_total": int(n_samp),
    }
    nan_mask = ~np.isfinite(data)
    if nan_mask.any():
        meta["nan_samples"] = int(nan_mask.any(axis=0).sum())

    for f in _fields(s):
        if f.lower() in _UNIT_KEYS:
            unit = _as_str_list(_get(s, f))
            if unit:
                u = unit[0].replace("µ", "u").replace("μ", "u")
                if u in ("V", "uV", "mV", "T", "fT"):
                    meta["data_unit"] = u
                used.add(f)

    on_field, onsets, convention = _resolve_onsets(s, n_samp, fs, used)
    if onsets is not None:
        used.add(on_field)
        field_map["event_onsets"] = on_field
        meta["event_index_convention"] = convention
        n_ev = int(onsets.size)
        lab_field, labels = _resolve_labels(s, n_ev, used)
        if lab_field:
            used.add(lab_field)
            field_map["event_labels"] = lab_field
        cn_field, class_names = _resolve_class_names(s, used)
        if cn_field:
            used.add(cn_field)
            field_map["class_names"] = cn_field
        if labels is not None:
            descriptions, base = _label_descriptions(labels, class_names)
            if base:
                meta["label_index_base"] = base
        else:
            descriptions = ["event"] * n_ev
        dur_field, durations = _resolve_durations(s, n_ev, fs, used)
        if dur_field:
            used.add(dur_field)
            field_map["event_durations"] = dur_field
        else:
            durations = np.zeros(n_ev)
        trial_flags, per_event_flags = _resolve_trial_flags(s, n_ev, used)
        used.update(trial_flags)
        meta["annotations"] = {
            "onset": [float(o) / fs for o in onsets],
            "duration": [float(d) for d in durations],
            "description": descriptions,
        }
        if any(per_event_flags):
            meta["annotations"]["source_flags"] = per_event_flags
        if trial_flags:
            meta["trial_quality_fields"] = sorted(trial_flags)
        types = sorted(set(descriptions))
        meta["event_id"] = {t: i + 1 for i, t in enumerate(types)}
        meta["events"] = [[int(o), 0, meta["event_id"][d]] for o, d in zip(onsets, descriptions)]

    imp_field, imp = _find_impedance(s)
    if imp is not None:
        field_map["impedance"] = imp_field
        used.add(imp_field.split(".")[0])
        res = _impedance_bad_channels(imp, channels)
        if res.get("values_kohm"):
            meta["impedance_kohm"] = res["values_kohm"]
            meta["impedance_poor_threshold_kohm"] = res.get("poor_threshold_kohm")
        if res["bad"]:
            meta["bad_channels"] = list(res["bad"])
            meta["bad_channel_reasons"] = dict(res["reasons"])
        if res["warnings"]:
            meta.setdefault("load_warnings", []).extend(res["warnings"])

    meta["mat_field_map"] = field_map
    unmapped = {f: _describe(_get(s, f)) for f in _fields(s) if f not in used}
    if unmapped:
        meta["mat_struct_fields"] = dict(list(unmapped.items())[:40])
    if degraded:
        meta["load_degraded"] = True
        meta["load_degraded_reasons"] = degraded

    if inspect_only:
        n_keep = int(min(n_samp, max(fs, 1.0)))
        data = data[:, :n_keep]
    return {
        "data": data,
        "frequency": float(fs),
        "channels": channels,
        "duration": float(n_samp / fs) if fs else 0.0,
        "meta": meta,
    }
