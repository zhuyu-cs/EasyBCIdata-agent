"""Effective data rank at the precision the data is stored in.

Pure numpy + stdlib. The generated ``pipeline.py`` and ``build_ai_ready.py``
inline this module's source (between the ``# --- data_rank:`` markers) so the
mini-repo stays self-contained; the agent imports the same functions.

An average reference (car) or ICA component removal makes the channels
linearly dependent. Stored as float32, that exact dependency survives only to
about 1e-7 relative precision, so estimators that assume float64 precision
(e.g. MNE ``rank=None``) report full rank and covariance-based methods (CSP,
xDAWN, Riemannian, ICA, source imaging) invert a near-singular matrix. The
rank recorded here is the one downstream code must pass explicitly.
"""
# --- data_rank: begin ---
import numpy as _dr_np

DATA_RANK_PRECISION = "float32"
_DR_MAX_SAMPLES = 200_000
_DR_CHUNK_ELEMS = 20_000_000


def _dr_blocks(arr):
    """Yield float64 ``(n_ch, k)`` blocks of a 2-D or 3-D array, time-subsampled
    so at most ~``_DR_MAX_SAMPLES`` samples are used (cost stays bounded)."""
    if arr.ndim == 3:
        n_tr, _n_ch, n_t = arr.shape
        step = max(1, int(_dr_np.ceil(n_tr * n_t / float(_DR_MAX_SAMPLES))))
        for k in range(n_tr):
            yield _dr_np.asarray(arr[k, :, ::step], dtype=_dr_np.float64)
        return
    n_ch, n_s = arr.shape
    step = max(1, int(_dr_np.ceil(n_s / float(_DR_MAX_SAMPLES))))
    width = max(1000, int(_DR_CHUNK_ELEMS // max(n_ch, 1))) * step
    for s in range(0, n_s, width):
        yield _dr_np.asarray(arr[:, s:s + width:step], dtype=_dr_np.float64)


def _dr_cov(arr, n_ch):
    """Two-pass covariance over finite columns (mean first, so a large DC
    offset cannot swamp the smallest eigenvalue)."""
    total = _dr_np.zeros(n_ch)
    cnt = 0
    for b in _dr_blocks(arr):
        ok = _dr_np.isfinite(b).all(axis=0)
        if ok.any():
            total += b[:, ok].sum(axis=1)
            cnt += int(ok.sum())
    if cnt <= n_ch:
        return None, cnt
    mu = total / cnt
    cov = _dr_np.zeros((n_ch, n_ch))
    for b in _dr_blocks(arr):
        ok = _dr_np.isfinite(b).all(axis=0)
        if ok.any():
            z = b[:, ok] - mu[:, None]
            cov += z @ z.T
    return cov / cnt, cnt


# Reducers that project onto a fixed subspace (car removes the all-ones
# direction): they cap the rank instead of subtracting from it.
_DR_PROJECTORS = ("car",)


def note_rank_reducer(d, op, drop, channels=None):
    """Record a rank-reducing step (``meta["rank_reducers"]``) on a data dict.

    ``channels`` are the rows the step acted on (default: all current
    channels). The reducer only counts while all of them are still present:
    dropping one of them (e.g. a channel after car) restores the rank;
    dropping channels it never touched does not."""
    meta = d.get("meta")
    if not isinstance(meta, dict):
        meta = {}
        d["meta"] = meta
    meta["rank_reducers"] = list(meta.get("rank_reducers") or []) + [{
        "op": str(op), "drop": int(drop),
        "channels": [str(c) for c in (d.get("channels") or [] if channels is None else channels)],
    }]
    return d


def estimate_rank(data, *, channels=None, reducers=None, ch_types=None,
                  default_ch_type="eeg", precision=DATA_RANK_PRECISION):
    """Effective rank of ``(n_ch, n_samp)`` or ``(n_trials, n_ch, n_times)`` data.

    numerical_rank: singular values of the channel-standardised covariance
    above ``n_ch * eps(precision) * s_max`` (the float32 criterion MNE uses
    with ``tol="float32"``). structural_rank: the recorded reducers that still
    apply to this channel set, in order — a fixed projector (car) caps the rank
    at ``n_ch - drop`` (applied after ICA it maps the reduced subspace
    one-to-one, so it does not subtract again), component removal (ica)
    subtracts its drop. ``rank`` is the
    smaller of the two. ``ch_types`` (per-channel MNE types; else every
    channel counts as ``default_ch_type``) is recorded as counts — the MNE
    rank dict is keyed by type. Never raises; failures give ``status: unknown``.
    """
    try:
        arr = _dr_np.asarray(data)
        if arr.ndim not in (2, 3) or 0 in arr.shape:
            return {"status": "unknown", "reason": "bad_shape"}
        n_ch = int(arr.shape[1] if arr.ndim == 3 else arr.shape[0])
        if isinstance(ch_types, (list, tuple)) and len(ch_types) == n_ch:
            type_counts = {}
            for t in ch_types:
                type_counts[str(t)] = type_counts.get(str(t), 0) + 1
        else:
            type_counts = {str(default_ch_type): n_ch}
        cov, n_used = _dr_cov(arr, n_ch)
        if cov is None:
            return {"status": "unknown", "reason": "too_few_finite_samples", "n_channels": n_ch}
        sd = _dr_np.sqrt(_dr_np.clip(_dr_np.diag(cov), 0.0, None))
        scale = _dr_np.where(sd > 0, sd, 1.0)
        lam = _dr_np.linalg.eigvalsh(cov / _dr_np.outer(scale, scale))[::-1]
        sv = _dr_np.sqrt(_dr_np.clip(lam, 0.0, None))
        eps_t = _dr_np.float64 if str(precision) == "float64" else _dr_np.float32
        s_max = float(sv[0]) if sv.size else 0.0
        tol = n_ch * float(_dr_np.finfo(eps_t).eps) * s_max
        numerical = int((sv > tol).sum()) if s_max > 0 else 0

        names = [str(c) for c in channels] if channels is not None else None
        valid, invalid = [], []
        for rec in reducers or []:
            if not isinstance(rec, dict):
                continue
            rch = rec.get("channels")
            if rch is None:
                same = True
            elif names is not None:
                same = set(str(c) for c in rch) <= set(names)
            else:
                same = len(rch) <= n_ch
            (valid if same else invalid).append(rec)
        structural = n_ch
        for r in valid:
            drop = int(r.get("drop") or 0)
            if str(r.get("op")) in _DR_PROJECTORS:
                structural = min(structural, n_ch - drop)
            else:
                structural -= drop
        structural = max(0, structural)
        rank = min(numerical, structural)
        warnings = []
        if numerical > structural:
            warnings.append(
                "rank_mismatch: {} should leave rank {} but the data is numerically rank {} "
                "— a later non-linear step (e.g. clip) broke the exact dependency, the "
                "covariance is near-singular; use rank={}".format(
                    [str(r.get("op")) for r in valid], structural, numerical, structural))
        elif numerical < structural:
            warnings.append(
                "extra_dependency: data is numerically rank {} below the expected {} "
                "(average reference / duplicated, flat or linearly dependent channels)".format(
                    numerical, structural))
        for r in invalid:
            warnings.append("reducer_invalidated: {} was applied to a different channel set".format(
                r.get("op")))
        return {
            "status": "ok", "n_channels": n_ch, "rank": rank,
            "numerical_rank": numerical, "structural_rank": structural,
            "reducers": [str(r.get("op")) for r in valid],
            "reducer_drops": [int(r.get("drop") or 0) for r in valid], "precision": str(precision),
            "ch_types": type_counts,
            "min_sv_ratio": float(sv[-1] / s_max) if s_max > 0 else 0.0,
            "n_samples_used": int(n_used), "warnings": warnings,
        }
    except Exception as exc:  # noqa: BLE001
        return {"status": "unknown", "reason": "{}: {}".format(type(exc).__name__, exc)}
# --- data_rank: end ---


def inline_source() -> str:
    """Module source between the inline markers (for codegen embedding)."""
    import inspect
    import sys

    src = inspect.getsource(sys.modules[__name__])
    begin = src.index("# --- data_rank: begin ---")
    end = src.index("# --- data_rank: end ---") + len("# --- data_rank: end ---")
    return src[begin:end] + "\n"


def default_ch_type(modality) -> str:
    """MNE channel type keying the rank dict when the data carry no channel types."""
    m = str(modality or "").strip().lower()
    return m if m in ("eeg", "meg", "seeg", "ecog", "dbs") else "eeg"


def rank_hint(rank, n_channels, reducers=None, ch_types=None) -> str:
    """Copy-ready instruction for covariance-based methods ("" at full rank).

    The MNE rank dict is keyed by channel type. With one type the hint gives
    the exact dict; with mixed types only the whole-matrix rank is known, so
    the hint says so instead of guessing a key.
    """
    if not isinstance(rank, int) or not isinstance(n_channels, int) or rank >= n_channels:
        return ""
    why = ", ".join(reducers or []) or "linearly dependent channels"
    types = dict(ch_types or {"eeg": n_channels})
    head = (
        "Data rank is {r} of {n} channels ({why}). Covariance-based methods must get it "
        "explicitly — MNE rank=None judges float32 data at float64 precision, reports full "
        "rank and collapses CSP (every trial predicted as one class). "
    ).format(r=rank, n=n_channels, why=why)
    if len(types) == 1:
        k = next(iter(types))
        return head + (
            "Use mne.decoding.CSP(..., rank={{\"{k}\": {r}}}), mne.compute_covariance(..., "
            "rank={{\"{k}\": {r}}}), ICA n_components <= {r}; for a hand-written covariance use "
            "shrinkage or drop one channel. AI_ready epochs.pkl carries 'rank'."
        ).format(k=k, r=rank)
    return head + (
        "The channel types are mixed ({t}): {r} is the rank of the whole matrix — select the "
        "channels used downstream and pass the rank for that type; ICA n_components <= {r}."
    ).format(t=", ".join("{}: {}".format(a, b) for a, b in types.items()), r=rank)


def summarize_work_dir(work_dir) -> dict | None:
    """Tool-return block from ``middle_process/pipeline_status__*.json:data_rank``.

    ``None`` when no file recorded a rank. ``must_present`` when a file is
    rank-deficient, its rank is unknown, or it carries rank warnings.
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
        rec = st.get("data_rank") if isinstance(st, dict) else None
        if not isinstance(rec, dict):
            continue
        per_file.append({
            "file_id": st.get("file_id"), "subject_id": st.get("subject_id"),
            "session_id": st.get("session_id"),
            **{k: rec.get(k) for k in ("status", "n_channels", "rank", "reducers", "ch_types",
                                       "warnings", "reason")},
        })
    if not per_file:
        return None
    ranks = [f["rank"] for f in per_file if isinstance(f.get("rank"), int)]
    deficient = [f for f in per_file
                 if isinstance(f.get("rank"), int) and isinstance(f.get("n_channels"), int)
                 and f["rank"] < f["n_channels"]]
    unknown = [f.get("file_id") for f in per_file if f.get("status") != "ok"]
    block = {"n_files": len(per_file), "rank_min": min(ranks) if ranks else None,
             "rank_deficient": bool(deficient), "per_file": per_file}
    hint = ""
    if deficient:
        worst = min(deficient, key=lambda f: f["rank"])
        hint = rank_hint(worst["rank"], worst["n_channels"], worst.get("reducers"),
                         worst.get("ch_types"))
        if len({(f.get("rank"), f.get("n_channels")) for f in per_file}) > 1:
            hint += " Files differ ({}); when pooling sessions use the minimum, rank={}.".format(
                ", ".join("{}: {}/{}".format(f.get("session_id") or f.get("file_id"),
                                             f.get("rank"), f.get("n_channels"))
                          for f in per_file), block["rank_min"])
    if unknown:
        hint = (hint + " Rank could not be estimated for {}: estimate it before any "
                "covariance-based method — never assume full rank.".format(unknown)).strip()
    if hint:
        block["hint"] = hint
    block["must_present"] = bool(deficient or unknown or any(f.get("warnings") for f in per_file))
    return block
