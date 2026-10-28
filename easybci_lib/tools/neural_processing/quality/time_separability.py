"""Time-resolved class separability of the kept trials.

Splits the epoch window into short segments and asks, per segment, whether any
channel's log-variance differs between classes more than under shuffled
labels. A long run of uninformative segments at the start or end of the window
usually means the window covers phases that are identical for every class
(fixation, cue, rest) — worth checking against the experiment's timeline.
Report-only: it never changes the window or the grade, and must not be used to
pick a window (that would leak labels into later cross-validation).

numpy only. The generated ``qc.py`` inlines the block between the
``# --- time_separability:`` markers; the agent imports the same code.
"""
# --- time_separability: begin ---
import numpy as _ts_np

TS_DEFAULTS = {
    "seg_s": 0.5, "n_perm": 200, "seed": 42,
    "min_per_class": 8, "min_window_s": 2.0,
    "p_null": 0.1, "p_sig": 0.05,
    "min_span_s": 1.5, "min_span_frac": 0.25, "min_sig_segments": 2,
}


def _ts_segment_logvar(data, kept, fs, seg_s):
    """(n_trials, n_seg, n_ch) log-variance per segment, plus labels."""
    n_seg_len = int(round(seg_s * fs))
    feats, labels = [], []
    for k in kept:
        s, e = int(k["start"]), int(k["stop"])
        n_seg = (e - s) // n_seg_len
        if n_seg < 1:
            continue
        x = _ts_np.asarray(data[:, s:s + n_seg * n_seg_len], dtype=_ts_np.float64)
        if not _ts_np.isfinite(x).all():
            continue
        x = x.reshape(x.shape[0], n_seg, n_seg_len)
        feats.append(_ts_np.log(x.var(axis=2).T + 1e-20))
        labels.append(str(k.get("label")))
    if not feats:
        return None, []
    n_seg = min(f.shape[0] for f in feats)
    return _ts_np.stack([f[:n_seg] for f in feats]), labels


def _ts_max_f(feats, onehot):
    """Max-over-channels one-way F per segment. feats (n, n_seg, n_ch)."""
    n, k = onehot.shape
    flat = feats.reshape(n, -1)
    counts = onehot.sum(0)
    means = (onehot.T @ flat) / counts[:, None]
    grand = flat.mean(0)
    ssb = (counts[:, None] * (means - grand) ** 2).sum(0)
    sst = ((flat - grand) ** 2).sum(0)
    f = (ssb / (k - 1)) / (_ts_np.maximum(sst - ssb, 1e-20) / (n - k))
    return f.reshape(feats.shape[1], feats.shape[2]).max(axis=1)


def assess_time_separability(data, fs, kept, window, **kw):
    """Per-segment permutation p-values + window flags.

    ``kept``: ``[{start, stop, label}, ...]`` sample bounds on ``data``
    (n_ch, n_samples); ``window``: ``[tmin, tmax]`` the bounds were cut with.
    Status: PASS / WARN / INFO (nothing separable anywhere) / SKIPPED.
    """
    cfg = dict(TS_DEFAULTS, **kw)
    tmin, tmax = float(window[0]), float(window[1])
    rep = {"status": "SKIPPED", "epoch_window": [tmin, tmax], "seg_s": cfg["seg_s"],
           "segments": [], "flags": []}
    if tmax - tmin < cfg["min_window_s"]:
        rep["reason"] = "window shorter than {} s".format(cfg["min_window_s"])
        return rep
    feats, labels = _ts_segment_logvar(data, kept, float(fs), cfg["seg_s"])
    classes, counts = _ts_np.unique(_ts_np.asarray(labels), return_counts=True)
    if feats is None or len(classes) < 2 or counts.min() < cfg["min_per_class"]:
        rep["reason"] = "needs >= 2 classes with >= {} kept trials each".format(cfg["min_per_class"])
        rep["class_counts"] = {str(c): int(n) for c, n in zip(classes, counts)}
        return rep
    y = _ts_np.searchsorted(classes, _ts_np.asarray(labels))
    onehot = _ts_np.eye(len(classes))[y]
    obs = _ts_max_f(feats, onehot)
    rng = _ts_np.random.default_rng(int(cfg["seed"]))
    ge = _ts_np.zeros_like(obs)
    for _ in range(int(cfg["n_perm"])):
        ge += _ts_max_f(feats, onehot[rng.permutation(len(y))]) >= obs
    p = (1.0 + ge) / (1.0 + cfg["n_perm"])
    seg = cfg["seg_s"]
    rep["segments"] = [{"start_s": round(tmin + i * seg, 3), "end_s": round(tmin + (i + 1) * seg, 3),
                        "max_f": round(float(o), 3), "p": round(float(q), 4)}
                       for i, (o, q) in enumerate(zip(obs, p))]
    rep["class_counts"] = {str(c): int(n) for c, n in zip(classes, counts)}
    sig = p < cfg["p_sig"]
    if int(sig.sum()) < cfg["min_sig_segments"]:
        rep["status"] = "INFO"
        rep["flags"].append({"kind": "no_time_resolved_separability"})
        return rep
    null = p >= cfg["p_null"]
    span_total = len(p) * seg
    for edge, run in (("start", _ts_run(null)), ("end", _ts_run(null[::-1]))):
        span = run * seg
        others = sig[run:] if edge == "start" else sig[:len(sig) - run]
        if span >= cfg["min_span_s"] and span / span_total >= cfg["min_span_frac"] \
                and int(others.sum()) >= cfg["min_sig_segments"]:
            a = tmin if edge == "start" else round(tmax - span, 3)
            rep["flags"].append({"kind": "uninformative_window_span", "edge": edge,
                                 "span_s": round(span, 3), "from_s": a, "to_s": round(a + span, 3),
                                 "fraction": round(span / span_total, 3)})
    rep["status"] = "WARN" if rep["flags"] else "PASS"
    if rep["flags"]:
        rep["hint"] = ("Part of the epoch window shows no class difference. Check the window "
                       "against the experiment's trial timeline / dataset documentation; do not "
                       "pick the window from this result (that leaks labels into evaluation).")
    return rep


def _ts_run(mask):
    n = 0
    for v in mask:
        if not v:
            break
        n += 1
    return n


def separability_markdown(rep):
    lines = ["## Time-resolved separability", "", "- Status: **{}**".format(rep.get("status"))]
    if rep.get("status") == "SKIPPED":
        lines.append("- Reason: {}".format(rep.get("reason")))
        return "\n".join(lines) + "\n"
    strip = "".join("█" if s["p"] < 0.05 else ("▒" if s["p"] < 0.1 else "·") for s in rep["segments"])
    lines.append("- Window {} s, {} s segments: `{}` (█ p<0.05, ▒ p<0.1, · n.s.)".format(
        rep["epoch_window"], rep["seg_s"], strip))
    for f in rep.get("flags") or []:
        if f["kind"] == "uninformative_window_span":
            lines.append("- WARN: {from_s}–{to_s} s ({span_s} s, {fraction:.0%} of the window) "
                         "shows no class difference".format(**f))
        else:
            lines.append("- INFO: no segment separates the classes (low SNR or few trials)")
    if rep.get("hint"):
        lines.append("- " + rep["hint"])
    return "\n".join(lines) + "\n"
# --- time_separability: end ---


def inline_source() -> str:
    """Module source between the inline markers (for codegen embedding)."""
    import inspect
    import sys

    src = inspect.getsource(sys.modules[__name__])
    begin = src.index("# --- time_separability: begin ---")
    end = src.index("# --- time_separability: end ---") + len("# --- time_separability: end ---")
    return src[begin:end] + "\n"


def summarize_work_dir(work_dir) -> dict | None:
    """Tool-return block from ``middle_process/time_separability__*.json``."""
    import json
    from pathlib import Path

    mp = Path(work_dir) / "middle_process"
    per_file = []
    for p in sorted(mp.glob("time_separability__*.json")) if mp.is_dir() else []:
        try:
            rep = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        per_file.append({
            "file_id": p.stem.split("__", 1)[1], "subject_id": rep.get("subject_id"),
            "session_id": rep.get("session_id"), "status": rep.get("status"),
            "epoch_window": rep.get("epoch_window"), "flags": rep.get("flags") or [],
            "strip": "".join("█" if s["p"] < 0.05 else ("▒" if s["p"] < 0.1 else "·")
                             for s in rep.get("segments") or []),
        })
    if not per_file:
        return None
    statuses = {f["status"] for f in per_file}
    overall = next((s for s in ("WARN", "INFO", "PASS") if s in statuses), "SKIPPED")
    out = {"status": overall, "per_file": per_file, "must_present": overall == "WARN"}
    if overall == "WARN":
        out["hint"] = ("Part of the epoch window shows no class difference in some sessions. Tell "
                       "the user which span, and that it should be checked against the dataset's "
                       "trial timeline (documentation / metadata). If the window includes fixation "
                       "or cue phases, offer revise_proposal(epoch_window=...) with the timeline as "
                       "basis. Never choose the window from these p-values.")
    return out
