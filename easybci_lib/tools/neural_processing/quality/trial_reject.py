"""Trial-level quality control: event collection, epoch window, rejection.

Pure numpy + stdlib. The generated ``pipeline.py`` inlines this module's
source verbatim (between the ``# --- trial_reject:`` markers) so the
mini-repo stays self-contained; the agent imports the same functions.

Rejection is data-adaptive: every channel is normalised by its OWN
distribution across trials (median / MAD), so a channel that is noisy in
every trial (e.g. frontal electrodes with blinks) raises its own baseline
instead of condemning all trials. A trial is rejected when one channel is an
extreme outlier or several channels are clear outliers together. A cap on
the rejected fraction prevents over-deletion when the whole recording is
poor — the cap is reported, never silently applied.
"""
# --- trial_reject: begin ---
import re as _tr_re

import numpy as _tr_np

TRIAL_REJECT_DEFAULTS = {
    "z_single": 5.0,
    "z_multi": 3.5,
    "min_channels": 2,
    "max_frac": 0.2,
    "min_trials": 10,
}

# Annotation descriptions that mark artifacts / boundaries, not trials.
_TR_NON_TRIAL_RX = _tr_re.compile(
    r"^(bad|edge|boundary|artifact|artefact|reject|nan|gap|discontinuity|"
    r"new segment|segment|impedance|pause|resume|start|stop|end)\b",
    _tr_re.IGNORECASE,
)
_TR_KEY_ALIASES = {
    "z": "z_single", "z_single": "z_single", "single": "z_single",
    "z_thresh": "z_single", "z_threshold": "z_single", "threshold": "z_single",
    "multi": "z_multi", "z_multi": "z_multi",
    "min_ch": "min_channels", "min_channels": "min_channels",
    "max_frac": "max_frac", "cap": "max_frac",
    "min_trials": "min_trials",
}
_TR_WINDOW_KEYS = ("window", "epoch_window", "win", "tmin_tmax")


def _tr_parse_window(val, rest):
    """``a~b`` / ``a:b`` / ``[a b]`` in ``val``, or ``a`` with ``b`` as the next bare token."""
    v = val.strip().strip("[]()")
    for sep in ("~", ":", ";", " "):
        if sep in v:
            a, b = v.split(sep, 1)
            return (float(a), float(b.strip().strip("[]()"))), False
    if rest and "=" not in rest[0]:
        return (float(v), float(rest[0].strip().strip("[]()"))), True
    raise ValueError(val)


def parse_trial_reject_param(param, warnings=None):
    """``auto`` | ``off`` | ``z=5,multi=3.5,min_ch=2,max_frac=0.2[,window=a~b]`` -> dict or None.

    ``window=a,b`` / ``window=a~b`` sets ``cfg["epoch_window"]`` (seconds
    relative to the event onset). Unknown keys and unparsable values are
    appended to ``warnings`` (when a list is given) instead of vanishing.
    """
    p = (param or "").strip().lower()
    if p in ("off", "none", "false", "0", "disable", "disabled"):
        return None
    cfg: dict = dict(TRIAL_REJECT_DEFAULTS)
    if p in ("", "auto", "default", "on", "true"):
        return cfg
    toks = [t.strip() for t in p.split(",") if t.strip()]
    while toks:
        tok = toks.pop(0)
        if "=" not in tok:
            if warnings is not None:
                warnings.append("reject_epochs: ignored token {!r}".format(tok))
            continue
        k, v = [x.strip() for x in tok.split("=", 1)]
        if k in _TR_WINDOW_KEYS:
            try:
                win, consumed = _tr_parse_window(v, toks)
                if consumed:
                    toks.pop(0)
                if win[1] > win[0]:
                    cfg["epoch_window"] = [float(win[0]), float(win[1])]
                elif warnings is not None:
                    warnings.append("reject_epochs: window {} has tmax <= tmin".format(list(win)))
            except (ValueError, IndexError):
                if warnings is not None:
                    warnings.append("reject_epochs: cannot parse window {!r}".format(v))
            continue
        key = _TR_KEY_ALIASES.get(k)
        if key is None:
            if warnings is not None:
                warnings.append("reject_epochs: unknown parameter {!r} ignored".format(k))
            continue
        try:
            cfg[key] = int(float(v)) if key in ("min_channels", "min_trials") else float(v)
        except ValueError:
            if warnings is not None:
                warnings.append("reject_epochs: bad value {}={!r}".format(k, v))
    return cfg


def collect_trial_events(annotations, fs, n_samples):
    """Trial events from loader annotations (onset/duration in seconds).

    Returns (events, n_excluded). Each event: dict(onset_s, duration_s,
    label, source_flag, trial_index). Descriptions that look like artifact or
    boundary markers are excluded and counted.
    """
    if not isinstance(annotations, dict):
        return [], 0
    onsets = list(annotations.get("onset") or [])
    durations = list(annotations.get("duration") or [])
    descs = list(annotations.get("description") or [])
    flags = list(annotations.get("source_flags") or [])
    out, excluded = [], 0
    for i, on in enumerate(onsets):
        try:
            on = float(on)
        except (TypeError, ValueError):
            excluded += 1
            continue
        desc = str(descs[i]) if i < len(descs) else "event"
        if _TR_NON_TRIAL_RX.match(desc.strip()):
            excluded += 1
            continue
        dur = 0.0
        if i < len(durations):
            try:
                dur = max(float(durations[i]), 0.0)
            except (TypeError, ValueError):
                dur = 0.0
        out.append({
            "onset_s": on,
            "duration_s": dur,
            "label": desc,
            "source_flag": str(flags[i]) if i < len(flags) and flags[i] else "",
            "trial_index": len(out),
        })
    return out, excluded


def remap_after_excision(events, rejected_intervals):
    """Shift onsets across excised windows (seconds); drop events inside them."""
    if not rejected_intervals:
        return list(events), 0
    ivs = sorted((float(a), float(b)) for a, b in rejected_intervals)
    out, dropped = [], 0
    for ev in events:
        on = float(ev["onset_s"])
        shift, inside = 0.0, False
        for a, b in ivs:
            if a <= on < b:
                inside = True
                break
            if b <= on:
                shift += b - a
        if inside:
            dropped += 1
            continue
        ne = dict(ev)
        ne["onset_s"] = on - shift
        out.append(ne)
    return out, dropped


def resolve_epoch_window(events, explicit=None, fallback=(-0.2, 2.0)):
    """Return ((tmin, tmax), source).

    explicit (tmin, tmax) wins; otherwise a duration shared by (nearly) all
    events defines the trial window ``(0, duration)``; otherwise fallback.
    """
    if explicit is not None:
        try:
            tmin, tmax = float(explicit[0]), float(explicit[1])
            if tmax > tmin:
                return (tmin, tmax), "explicit"
        except (TypeError, ValueError, IndexError):
            pass
    durs = [round(float(e.get("duration_s") or 0.0), 3) for e in events]
    durs = [d for d in durs if d > 0]
    if durs and len(durs) >= 0.9 * len(events):
        vals, counts = _tr_np.unique(durs, return_counts=True)
        mode = float(vals[int(_tr_np.argmax(counts))])
        if counts.max() >= 0.9 * len(durs):
            return (0.0, mode), "event_duration"
    return (float(fallback[0]), float(fallback[1])), "default"


def epoch_bounds(events, fs, window, n_samples):
    """Sample bounds for every event; out-of-range events are reported."""
    tmin, tmax = window
    win = int(round((tmax - tmin) * fs))
    kept, oob = [], []
    for ev in events:
        s = int(round((float(ev["onset_s"]) + tmin) * fs))
        e = s + win
        if s < 0 or e > n_samples:
            oob.append(int(ev["trial_index"]))
            continue
        kept.append((ev, s, e))
    return kept, oob


def spans_to_current(spans_s, rejected_intervals):
    """Map original-timeline spans (seconds) onto the post-excision timeline.

    Parts inside an excised window vanish; the rest shift left by the excised
    time before them. Returns merged ``[[start, stop], ...]`` in seconds.
    """
    ivs = sorted((float(a), float(b)) for a, b in (rejected_intervals or []))

    def _map(t):
        shift = 0.0
        for a, b in ivs:
            if b <= t:
                shift += b - a
            elif a < t:
                shift += t - a
        return t - shift

    out = []
    for a, b in sorted((float(a), float(b)) for a, b in (spans_s or [])):
        ca, cb = _map(a), _map(b)
        if cb <= ca:
            continue
        if out and ca <= out[-1][1]:
            out[-1][1] = max(out[-1][1], cb)
        else:
            out.append([ca, cb])
    return out


def trials_in_spans(bounds, spans_s, fs):
    """Per trial: True when its sample range overlaps a bad span (seconds)."""
    spans = [(float(a) * fs, float(b) * fs) for a, b in (spans_s or [])]
    if not spans:
        return [False] * len(bounds)
    return [any(b > s and a < e for a, b in spans) for _ev, s, e in bounds]


def trial_features(data, bounds):
    """Per-trial x channel peak-to-peak and log-variance (streaming over trials)."""
    n_tr = len(bounds)
    n_ch = int(data.shape[0])
    ptp = _tr_np.zeros((n_tr, n_ch), dtype=_tr_np.float64)
    logvar = _tr_np.zeros((n_tr, n_ch), dtype=_tr_np.float64)
    hard = [""] * n_tr
    for k, (_ev, s, e) in enumerate(bounds):
        seg = _tr_np.asarray(data[:, s:e], dtype=_tr_np.float64)
        if not _tr_np.all(_tr_np.isfinite(seg)):
            hard[k] = "non_finite"
            ptp[k] = _tr_np.nan
            logvar[k] = _tr_np.nan
            continue
        p = seg.max(axis=1) - seg.min(axis=1)
        ptp[k] = p
        v = seg.var(axis=1)
        logvar[k] = _tr_np.log(_tr_np.maximum(v, 1e-30))
    return ptp, logvar, hard


def _robust_z(x):
    """Column-wise one-sided robust z (median / 1.4826*MAD), NaN-safe."""
    med = _tr_np.nanmedian(x, axis=0)
    mad = 1.4826 * _tr_np.nanmedian(_tr_np.abs(x - med), axis=0)
    mad = _tr_np.where((mad > 0) & _tr_np.isfinite(mad), mad, _tr_np.inf)
    return (x - med) / mad


def reject_trials(ptp, logvar, hard, *, channels, labels=None, source_flags=None,
                  z_single=5.0, z_multi=3.5, min_channels=2, max_frac=0.2,
                  min_trials=10):
    """Decide which trials to keep.

    Returns (keep_mask, report). Hard rejections (non-finite / flat channel /
    flagged by the recording itself) are unconditional and do not count
    against ``max_frac``; statistical rejections are capped.
    """
    n = int(ptp.shape[0])
    keep = _tr_np.ones(n, dtype=bool)
    reasons = {}
    hard_idx = []
    # A channel flat in most trials is a dead channel (a channel-level
    # problem), not evidence against individual trials.
    flat = _tr_np.nan_to_num(ptp, nan=1.0) == 0
    live = flat.mean(axis=0) <= 0.5 if n else _tr_np.ones(ptp.shape[1], dtype=bool)
    for k in range(n):
        why = hard[k] if k < len(hard) else ""
        if not why and _tr_np.any(flat[k] & live):
            why = "flat_channel:" + ",".join(
                str(channels[c]) for c in _tr_np.flatnonzero(flat[k] & live)[:4])
        if not why and source_flags is not None and k < len(source_flags) and source_flags[k]:
            why = "recording_flag:" + str(source_flags[k])
        if why:
            keep[k] = False
            reasons[k] = why
            hard_idx.append(k)
    report = {
        "n_before": n,
        "params": {"z_single": z_single, "z_multi": z_multi,
                   "min_channels": int(min_channels), "max_frac": max_frac,
                   "min_trials": int(min_trials)},
        "cap_hit": False,
        "skipped_reason": None,
    }
    scores = _tr_np.zeros(n)
    stat_idx = []
    n_good = int(keep.sum())
    if n_good < int(min_trials):
        report["skipped_reason"] = "too_few_trials"
    else:
        ok = _tr_np.flatnonzero(keep)
        z = _tr_np.full_like(ptp, _tr_np.nan)
        z[ok] = _tr_np.maximum(_robust_z(ptp[ok]), _robust_z(logvar[ok]))
        z[:, ~live] = _tr_np.nan
        zz = _tr_np.nan_to_num(z, nan=-_tr_np.inf)
        scores = zz.max(axis=1)
        n_multi = (zz > z_multi).sum(axis=1)
        cand = [k for k in ok if scores[k] > z_single or n_multi[k] >= min_channels]
        cap = int(_tr_np.floor(max_frac * n))
        if len(cand) > cap:
            report["cap_hit"] = True
            report["n_candidates"] = len(cand)
            cand = sorted(cand, key=lambda k: -scores[k])[:cap]
        for k in sorted(cand):
            keep[k] = False
            worst = int(_tr_np.argmax(zz[k]))
            reasons[k] = (
                "z_single:{}={:.1f}".format(channels[worst], scores[k])
                if scores[k] > z_single
                else "z_multi:{}ch>{:g}".format(int(n_multi[k]), z_multi)
            )
            stat_idx.append(k)
    report.update({
        "n_after": int(keep.sum()),
        "n_rejected": int((~keep).sum()),
        "n_rejected_hard": len(hard_idx),
        "n_rejected_statistical": len(stat_idx),
        "reasons": {str(k): v for k, v in sorted(reasons.items())},
        "score_max_z": [None if not _tr_np.isfinite(s) else round(float(s), 2) for s in scores],
    })
    if n and len(hard_idx) > 0.5 * n:
        report["hard_reject_warning"] = (
            "{}/{} trials failed hard checks (non-finite / flat / recording flag) — "
            "inspect the recording before trusting this output".format(len(hard_idx), n))
    if labels is not None:
        per = {}
        for k, lab in enumerate(labels):
            d = per.setdefault(str(lab), {"before": 0, "after": 0})
            d["before"] += 1
            d["after"] += int(bool(keep[k]))
        report["per_class"] = per
        report["class_imbalance_warning"] = [
            c for c, d in per.items() if d["before"] and d["after"] < 0.5 * d["before"]
        ]
    return keep, report
# --- trial_reject: end ---


def normalize_epoch_window(value):
    """``[tmin, tmax]`` as floats when valid (tmax > tmin), else ``None``."""
    try:
        tmin, tmax = float(value[0]), float(value[1])
    except (TypeError, ValueError, IndexError, KeyError):
        return None
    if not tmax > tmin:
        return None
    return [tmin, tmax]


def inline_source() -> str:
    """Module source between the inline markers (for codegen embedding)."""
    import inspect
    import sys

    src = inspect.getsource(sys.modules[__name__])
    begin = src.index("# --- trial_reject: begin ---")
    end = src.index("# --- trial_reject: end ---") + len("# --- trial_reject: end ---")
    return src[begin:end] + "\n"


def summarize_work_dir(work_dir) -> dict | None:
    """Aggregate the per-file trial decisions pipeline.py wrote.

    Reads ``middle_process/pipeline_status__<file_id>.json`` (the
    ``trial_rejection`` summary) and returns a tool-return block, or ``None``
    when no file carried trial events. ``must_present`` is set when the cap
    was hit, a class lost more than half its trials, or most trials failed
    hard checks — situations a user must decide on.
    """
    import json
    from pathlib import Path

    mp = Path(work_dir) / "middle_process"
    if not mp.is_dir():
        return None
    per_file = []
    for p in sorted(mp.glob("pipeline_status__*.json")):
        try:
            st = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        tr = st.get("trial_rejection") if isinstance(st, dict) else None
        if not tr:
            continue
        per_file.append({
            "file_id": st.get("file_id"),
            "subject_id": st.get("subject_id"),
            "session_id": st.get("session_id"),
            **{k: tr.get(k) for k in (
                "epoch_window", "epoch_window_source", "n_events_total",
                "n_out_of_bounds", "n_kept", "n_rejected", "rejected_trial_index",
                "cap_hit", "skipped_reason", "per_class", "class_imbalance_warning",
                "hard_reject_warning", "rejection_enabled",
            )},
        })
    if not per_file:
        return None
    warnings = []
    for f in per_file:
        tag = "sub-{}/ses-{}".format(f["subject_id"], f["session_id"])
        if f.get("cap_hit"):
            warnings.append(tag + ": rejection cap hit — recording quality is poor overall "
                            "or thresholds do not fit; check bad channels / reject_epochs params")
        if f.get("class_imbalance_warning"):
            warnings.append(tag + ": classes lost >50% of trials: "
                            + ", ".join(map(str, f["class_imbalance_warning"])))
        if f.get("hard_reject_warning"):
            warnings.append(tag + ": " + str(f["hard_reject_warning"]))
        if f.get("n_out_of_bounds"):
            warnings.append(tag + ": {} trial(s) outside the recording for window {}".format(
                f["n_out_of_bounds"], f.get("epoch_window")))
    n_before = sum(int(f.get("n_kept") or 0) + int(f.get("n_rejected") or 0) for f in per_file)
    n_rej = sum(int(f.get("n_rejected") or 0) for f in per_file)
    return {
        "n_files": len(per_file),
        "n_trials_before": n_before,
        "n_trials_rejected": n_rej,
        "n_trials_kept": n_before - n_rej,
        "per_file": per_file,
        "warnings": warnings,
        "must_present": bool(warnings),
        "where": "curated NWB: trials table = kept, invalid_times = rejected; "
                 "AI_ready epochs contain kept trials only (trial_index maps back)",
    }
