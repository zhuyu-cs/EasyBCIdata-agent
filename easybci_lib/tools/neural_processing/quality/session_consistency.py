"""Cross-session spectral consistency.

For every subject with >= 2 sessions, compare each session's per-channel
log-PSD of the PREPROCESSED signal against the subject's median spectrum.
Spectral shape is a stable per-subject fingerprint; a channel whose shape
diverges in one session points at an electrode / reference / impedance
change, a broadband shift at a gain or montage change, and an alpha-peak
shift at a state difference. Results are WARN flags — nothing is deleted.

numpy + scipy only. The generated ``qc.py`` inlines the block between the
``# --- session_consistency:`` markers; the agent imports the same code.
"""
# --- session_consistency: begin ---
import numpy as _sc_np

SC_DEFAULTS = {
    "corr_warn": 0.95,
    "session_score_warn": 0.97,
    "band_log2_warn": 1.0,
    "broadband_log2_warn": 1.0,
    "iaf_shift_warn_hz": 1.0,
}
_SC_BANDS = (("delta", 1.0, 4.0), ("theta", 4.0, 8.0), ("alpha", 8.0, 13.0),
             ("beta", 13.0, 30.0), ("gamma", 30.0, 200.0))


def session_psd(data, fs, *, fmin=1.0, fmax=40.0, notch_hz=None, seg_s=2.0,
                block_s=60.0, max_blocks=10, invalid_spans=None):
    """Welch log10-PSD per channel, memory bounded.

    data: (n_ch, n_samples), may be a memmap. Spans listed in
    ``invalid_spans`` [(start_sample, stop_sample), ...] (rejected trials)
    are excluded. Returns (freqs, logpsd[n_ch, n_freq]).
    """
    from scipy.signal import welch

    fs = float(fs)
    n_ch, n = int(data.shape[0]), int(data.shape[1])
    nper = int(max(8, min(round(seg_s * fs), n)))
    blk = int(max(nper, round(block_s * fs)))
    starts = list(range(0, max(n - nper + 1, 1), blk))
    if len(starts) > max_blocks:
        idx = _sc_np.linspace(0, len(starts) - 1, max_blocks).round().astype(int)
        starts = [starts[i] for i in idx]
    bad = sorted((int(a), int(b)) for a, b in (invalid_spans or []))
    acc, wsum, freqs = None, 0.0, None
    for s in starts:
        e = min(s + blk, n)
        seg = _sc_np.asarray(data[:, s:e], dtype=_sc_np.float64)
        mask = _sc_np.isfinite(seg).all(axis=0)
        for a, b in bad:
            if b > s and a < e:
                mask[max(a - s, 0):min(b - s, e - s)] = False
        if mask.sum() < nper:
            continue
        seg = seg[:, mask]
        f, p = welch(seg, fs=fs, nperseg=nper, noverlap=nper // 2, axis=-1)
        w = float(seg.shape[1])
        acc = p * w if acc is None else acc + p * w
        wsum += w
        freqs = f
    if acc is None:
        return None, None
    psd = acc / wsum
    hi = min(float(fmax), fs / 2.0 - 1.0)
    keep = (freqs >= float(fmin)) & (freqs <= hi)
    for nh in (notch_hz or []):
        k = 1
        while nh * k <= hi:
            keep &= _sc_np.abs(freqs - nh * k) > 1.0
            k += 1
    return freqs[keep], _sc_np.log10(_sc_np.maximum(psd[:, keep], 1e-30))


def _sc_iaf(freqs, logpsd, channels):
    post = [i for i, c in enumerate(channels)
            if any(t in str(c).upper() for t in ("O", "P"))]
    rows = logpsd[post] if post else logpsd
    m = (freqs >= 7.0) & (freqs <= 14.0)
    if m.sum() < 3:
        return None
    spec = rows[:, m].mean(axis=0)
    # detrend (1/f) with a line fit so the peak is a true bump
    x = freqs[m]
    spec = spec - _sc_np.polyval(_sc_np.polyfit(x, spec, 1), x)
    k = int(_sc_np.argmax(spec))
    if k in (0, len(spec) - 1):
        return None
    return float(x[k])


def assess_consistency(sessions, *, corr_warn=0.95, session_score_warn=0.97,
                       band_log2_warn=1.0, broadband_log2_warn=1.0,
                       iaf_shift_warn_hz=1.0):
    """sessions: {session_id: (channels, freqs, logpsd)} for ONE subject.

    Returns a JSON-serialisable report with status PASS / WARN / SKIPPED.
    """
    params = {"corr_warn": corr_warn, "session_score_warn": session_score_warn,
              "band_log2_warn": band_log2_warn,
              "broadband_log2_warn": broadband_log2_warn,
              "iaf_shift_warn_hz": iaf_shift_warn_hz}
    valid = {s: v for s, v in sessions.items() if v[1] is not None}
    if len(valid) < 2:
        return {"status": "SKIPPED", "reason": "single_session" if len(sessions) < 2
                else "insufficient_valid_sessions",
                "n_sessions": len(sessions), "params": params, "flags": []}
    sess = sorted(valid)
    common = [c for c in valid[sess[0]][0] if all(c in valid[s][0] for s in sess[1:])]
    not_common = sorted({c for s in sess for c in valid[s][0]} - set(common))
    # common frequency grid (sessions may differ in fs after resample)
    f_ref = valid[sess[0]][1]
    for s in sess[1:]:
        f_ref = f_ref[(f_ref >= valid[s][1].min()) & (f_ref <= valid[s][1].max())]
    if len(common) == 0 or f_ref.size < 5:
        return {"status": "SKIPPED", "reason": "no_common_channels_or_band",
                "n_sessions": len(sess), "params": params, "flags": [],
                "channels_not_common": not_common}
    cube = _sc_np.stack([
        _sc_np.stack([_sc_np.interp(f_ref, valid[s][1], valid[s][2][valid[s][0].index(c)])
                      for c in common])
        for s in sess
    ])  # (n_sess, n_ch, n_f)
    ref = _sc_np.median(cube, axis=0)
    flags = []
    corr = _sc_np.zeros((len(sess), len(common)))
    for i in range(len(sess)):
        for j in range(len(common)):
            a, b = cube[i, j] - cube[i, j].mean(), ref[j] - ref[j].mean()
            den = float(_sc_np.sqrt((a * a).sum() * (b * b).sum()))
            corr[i, j] = float((a * b).sum() / den) if den > 0 else 1.0
    lin = 10.0 ** cube
    total = lin.sum(axis=2)                               # (sess, ch)
    log2_bb = _sc_np.log2(total / _sc_np.median(total, axis=0))
    band_shift = {}
    for name, lo, hi in _SC_BANDS:
        m = (f_ref >= lo) & (f_ref < hi)
        if m.sum() < 2:
            continue
        rel = lin[:, :, m].sum(axis=2) / total
        band_shift[name] = _sc_np.log2(rel / _sc_np.median(rel, axis=0))
    iaf = {s: _sc_iaf(f_ref, cube[i], common) for i, s in enumerate(sess)}
    iaf_vals = [v for v in iaf.values() if v is not None]
    iaf_med = float(_sc_np.median(iaf_vals)) if iaf_vals else None
    per_session = {}
    for i, s in enumerate(sess):
        score = float(_sc_np.median(corr[i]))
        bb = float(_sc_np.median(log2_bb[i]))
        info = {"score": round(score, 4), "broadband_log2_ratio": round(bb, 3),
                "iaf_hz": iaf[s], "flagged_channels": []}
        for j, c in enumerate(common):
            if corr[i, j] < corr_warn:
                flags.append({"session": s, "channel": c, "metric": "spectral_shape_corr",
                              "value": round(float(corr[i, j]), 4), "threshold": corr_warn,
                              "kind": "spectral_shape_drift"})
                info["flagged_channels"].append(c)
        for name, arr in band_shift.items():
            med = float(_sc_np.median(arr[i]))
            if abs(med) > band_log2_warn:
                flags.append({"session": s, "channel": "*", "metric": name + "_rel_power_log2",
                              "value": round(med, 3), "threshold": band_log2_warn,
                              "kind": "band_power_shift"})
        if abs(bb) > broadband_log2_warn:
            flags.append({"session": s, "channel": "*", "metric": "broadband_power_log2",
                          "value": round(bb, 3), "threshold": broadband_log2_warn,
                          "kind": "broadband_power_shift"})
        if iaf_med is not None and iaf[s] is not None and abs(iaf[s] - iaf_med) > iaf_shift_warn_hz:
            flags.append({"session": s, "channel": "*", "metric": "iaf_hz",
                          "value": iaf[s], "threshold": iaf_shift_warn_hz,
                          "kind": "iaf_shift"})
        if score < session_score_warn:
            flags.append({"session": s, "channel": "*", "metric": "session_score",
                          "value": round(score, 4), "threshold": session_score_warn,
                          "kind": "session_outlier"})
        per_session[s] = info
    return {
        "status": "WARN" if flags else "PASS",
        "n_sessions": len(sess),
        "sessions": sess,
        "channels": common,
        "channels_not_common": not_common,
        "band_hz": [float(f_ref.min()), float(f_ref.max())],
        "per_session": per_session,
        "corr_matrix": [[round(float(v), 4) for v in row] for row in corr],
        "band_log2_shift": {k: [[round(float(x), 3) for x in r] for r in v]
                            for k, v in band_shift.items()},
        "flags": flags,
        "params": params,
        "freqs": [round(float(f), 3) for f in f_ref],
        "mean_logpsd": {s: [round(float(x), 4) for x in cube[i].mean(axis=0)]
                        for i, s in enumerate(sess)},
    }


def consistency_markdown(subject_id, rep):
    lines = ["# Cross-session spectral consistency — sub-{}".format(subject_id), ""]
    lines.append("- Status: **{}**".format(rep.get("status")))
    if rep.get("status") == "SKIPPED":
        lines.append("- Reason: {}".format(rep.get("reason")))
        return "\n".join(lines) + "\n"
    lines.append("- Sessions: {}".format(", ".join(rep["sessions"])))
    lines.append("- Band: {:.1f}-{:.1f} Hz, common channels: {}".format(
        rep["band_hz"][0], rep["band_hz"][1], len(rep["channels"])))
    if rep.get("channels_not_common"):
        lines.append("- Not compared (missing in some session): {}".format(
            ", ".join(rep["channels_not_common"])))
    lines += ["", "| session | score (median shape corr) | broadband log2 | IAF Hz | flagged channels |",
              "|---|---|---|---|---|"]
    for s in rep["sessions"]:
        info = rep["per_session"][s]
        lines.append("| {} | {:.4f} | {:+.3f} | {} | {} |".format(
            s, info["score"], info["broadband_log2_ratio"],
            "-" if info["iaf_hz"] is None else "{:.1f}".format(info["iaf_hz"]),
            ", ".join(info["flagged_channels"]) or "-"))
    if rep.get("flags"):
        lines += ["", "## Flags", ""]
        for f in rep["flags"]:
            lines.append("- {kind}: session {session}, channel {channel}, {metric}={value} "
                         "(threshold {threshold})".format(**f))
    return "\n".join(lines) + "\n"
# --- session_consistency: end ---


def inline_source() -> str:
    """Module source between the inline markers (for codegen embedding)."""
    import inspect
    import sys

    src = inspect.getsource(sys.modules[__name__])
    begin = src.index("# --- session_consistency: begin ---")
    end = src.index("# --- session_consistency: end ---") + len("# --- session_consistency: end ---")
    return src[begin:end] + "\n"


def summarize_work_dir(work_dir) -> dict | None:
    """Tool-return block from ``QC_out/sub-*/session_consistency.json``."""
    import json
    from pathlib import Path

    qc_root = Path(work_dir) / "preprocessed_output" / "QC_out"
    if not qc_root.is_dir():
        return None
    per_subject = {}
    any_warn = False
    for p in sorted(qc_root.glob("sub-*/session_consistency.json")):
        try:
            rep = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        status = rep.get("status")
        any_warn = any_warn or status == "WARN"
        flags = rep.get("flags") or []
        per_subject[p.parent.name] = {
            "status": status,
            "reason": rep.get("reason"),
            "n_sessions": rep.get("n_sessions"),
            "session_scores": {s: i.get("score") for s, i in (rep.get("per_session") or {}).items()},
            "flagged": [
                "{}:{}:{}={}".format(f["session"], f["channel"], f["kind"], f["value"])
                for f in flags[:12]
            ],
            "n_flags": len(flags),
            "report_path": str(p),
            "figure_path": str(Path(work_dir) / "preprocessed_output" / "figures"
                               / p.parent.name / "session_consistency_psd.png"),
        }
    if not per_subject:
        return None
    statuses = {v["status"] for v in per_subject.values()}
    overall = "WARN" if "WARN" in statuses else ("PASS" if "PASS" in statuses else "SKIPPED")
    out = {"status": overall, "per_subject": per_subject, "must_present": any_warn}
    if any_warn:
        out["hint"] = (
            "Some sessions are spectrally inconsistent with the same subject's other "
            "sessions. Tell the user which session/channel, the likely causes "
            "(electrode impedance or contact, reference/montage change, amplifier "
            "gain, subject state) and let them decide whether to drop the channel, "
            "exclude the session, or keep it. Do not delete sessions automatically."
        )
    return out
