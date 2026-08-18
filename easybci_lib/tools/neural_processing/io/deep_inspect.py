"""Full-data scan producing inspection_report.json — Phase 1 of the
two-phase pipeline flow.

Trades wall-clock + RAM at Phase 1 for accurate operator/parameter choices
at Phase 2.
"""

from __future__ import annotations

import hashlib
import logging
import signal
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np

from easybci_lib.tools.neural_processing.io.loader import load_neural
from easybci_lib.tools.neural_processing.io.inspection_report import (
    ArtifactSummary,
    ChannelStat,
    ChannelSummary,
    EventsSummary,
    Fingerprint,
    InspectionReport,
    MemoryEstimate,
    PsdSummary,
    compute_file_id,
    load_inspection_report,
    save_inspection_report,
)
from easybci_lib.tools.neural_processing.io.routing_table import (
    RoutingEntry,
    RoutingConflictError,
    stem_safe,
    upsert_routing_entry,
)
from easybci_lib.tools.neural_processing.profile.identity_resolver import (
    resolve_identity,
)

logger = logging.getLogger(__name__)

_REPORT_FILENAME = "inspection_report.json"
_DEFAULT_SAMPLE_PCT = 0.10
_DEFAULT_PSD_RES_HZ = 1.0
_DEFAULT_MAX_PRELOAD_MB = 4096
_DEFAULT_TIMEOUT_S = 300


class _TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):  # noqa: ARG001
    raise _TimeoutError("deep_inspect timeout")


def deep_inspect(
    data_path: str,
    work_dir: str,
    *,
    sample_pct: float = _DEFAULT_SAMPLE_PCT,
    psd_resolution_hz: float = _DEFAULT_PSD_RES_HZ,
    max_preload_mb: int = _DEFAULT_MAX_PRELOAD_MB,
    timeout_s: int = _DEFAULT_TIMEOUT_S,
    cli_subject_id: str | None = None,
    cli_session_id: str | None = None,
) -> dict[str, Any]:
    """Full-data scan; writes work_dir/middle_process/inspection_report.json.

    Returns {success, report_path, report, degraded, elapsed_s}. Never raises
    into the agent loop — degraded paths still return success=True.
    """
    out_path = Path(work_dir) / "middle_process" / _REPORT_FILENAME

    # SIGALRM is POSIX-only and signal.signal() may only be installed from
    # the main thread of the main interpreter. Under the gateway, tool
    # dispatch runs in a worker thread, so skip the alarm there and rely on
    # the caller's wall-clock budget — same fallback path Windows already uses.
    has_alarm = (
        hasattr(signal, "SIGALRM")
        and threading.current_thread() is threading.main_thread()
    )
    if has_alarm:
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(int(timeout_s))

    started = time.monotonic()
    try:
        report = _scan(
            data_path=data_path,
            sample_pct=sample_pct,
            psd_resolution_hz=psd_resolution_hz,
            max_preload_mb=max_preload_mb,
        )
    except _TimeoutError:
        report = _degraded_lightweight(
            data_path, reason="timeout",
        )
    except MemoryError:
        report = _degraded_lightweight(data_path, reason="memory_cap")
    except Exception as exc:
        logger.exception("deep_inspect scan failed; degrading")
        report = _degraded_lightweight(
            data_path, reason=f"loader_error: {type(exc).__name__}: {exc}"
        )
    finally:
        if has_alarm:
            signal.alarm(0)

    # Resolve subject + session identity — single source of truth that all
    # downstream tools (codegen, repo_builder, figure writers) must read
    # instead of re-inferring from data_path.stem.
    try:
        identity = resolve_identity(
            Path(data_path),
            cli_subject_id=cli_subject_id,
            cli_session_id=cli_session_id,
        )
        report.identity = identity
        if identity.fallback_used:
            report.warnings.append(
                f"subject_id defaulted to '{identity.subject_id}' "
                f"({identity.notes})"
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("identity resolution failed: %s", exc)
        identity = None

    # Compute a stable file_id (sha256 of first 1 MiB → 8 hex). Falls back to
    # an md5 of the path string when the file is unreadable so multi-input
    # routing still gets a unique key.
    file_id = compute_file_id(data_path) or hashlib.md5(
        str(data_path).encode("utf-8")
    ).hexdigest()[:8]
    report.file_id = file_id

    # Summarize sidecar events CSV/TSV so the LLM never needs to read raw files.
    events_csv_path = _discover_events_csv(data_path)
    if events_csv_path:
        try:
            report.events_summary = _summarize_events_csv(events_csv_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("events CSV summary failed: %s", exc)

    # Persist BOTH: per-file report under middle_process/inspect/<file_id>/
    # (multi-input single source of truth) AND the root-level path (consumed
    # by plan/codegen as the "representative" report for parameter decisions).
    # Multi-input: the root is written ONLY on the first call (first-writer-wins)
    # so that plan_pipeline/generate_code get a stable, predictable report rather
    # than whichever file happened to be inspected last. Exception: a degraded
    # root is replaced by a non-degraded report (prefer quality data for hints).
    work_dir_path = Path(work_dir)
    per_file_path = (
        work_dir_path / "middle_process" / "inspect" / file_id / _REPORT_FILENAME
    )
    save_inspection_report(report, per_file_path)
    _should_write_root = not out_path.is_file()
    if not _should_write_root and not report.degraded:
        try:
            _existing = load_inspection_report(out_path)
            if _existing.degraded:
                _should_write_root = True
        except Exception:
            pass
    if _should_write_root:
        save_inspection_report(report, out_path)

    # Upsert into the routing table. Conflicts (same (sub, ses, stem) under
    # different file_id) are logged + the per-file report still lands; the
    # routing table is left in its prior state for the caller to reconcile.
    if identity is not None:
        try:
            entry = RoutingEntry(
                data_path=str(data_path),
                stem_safe=stem_safe(data_path),
                sha256_1mb=file_id,
                file_id=file_id,
                subject_id=identity.subject_id,
                session_id=identity.session_id,
                identity_source=identity.source,
                identity_confidence=identity.confidence,
                inspection_report_path=str(
                    per_file_path.relative_to(work_dir_path)
                ),
                events_path=events_csv_path,
                override_script=None,
            )
            upsert_routing_entry(work_dir_path, entry)
        except RoutingConflictError as exc:
            logger.warning("routing table upsert refused: %s", exc)
        except Exception as exc:  # noqa: BLE001
            logger.warning("routing table upsert failed: %s", exc)

    envelope: dict[str, Any] = {
        "success": True,
        "report_path": str(out_path),
        "report": report.to_dict(),
        "degraded": report.degraded,
        "elapsed_s": round(time.monotonic() - started, 2),
    }
    # Touchpoint ①: if this degraded because no loader could read the format
    # (as opposed to a recognised loader crashing), surface an actionable
    # next_action pointing the agent at register_io_loader.
    if report.degraded:
        next_action = _unsupported_format_next_action(data_path)
        if next_action is not None:
            envelope["next_action"] = next_action
            envelope["report"]["degraded_reason"] = "unsupported_format"
    return envelope


def _discover_events_csv(data_path: Path | str) -> Optional[str]:
    """Best-effort lookup of a sidecar event file.

    Covers two naming conventions:

    * **legacy** — ``events_{stem_safe}.csv`` / ``events_{raw_stem}.csv``
      (the original EasyBCI convention; raw stem may contain spaces).
    * **BIDS** — ``{base}_events.tsv`` where ``base`` is the data stem with
      its modality suffix (``_eeg`` / ``_meg`` / ``_ieeg`` / ``_ecog`` /
      ``_nirs`` / ``_beh``) dropped, per the BIDS sidecar rule. Falls back to
      ``{stem}_events.tsv`` when there is no recognised modality suffix.

    For BIDS, also tries **entity-reduced** names: an events sidecar is often
    shared one entity level above the data file (e.g. data
    ``sub-01_ses-1_task-x_run-1_eeg.edf`` -> events
    ``sub-01_ses-1_task-x_events.tsv``, no run). We strip trailing entities
    (run-/acq-/rec-/split-/…) one at a time, but keep the ``sub-``(+``ses-``)
    prefix so a flat multi-subject directory never resolves another subject's
    events file.

    Returns an absolute path string or ``None``. Used by the routing table so
    build_ai_ready never has to re-discover events at run time.
    """
    p = Path(data_path)
    data_dir = p.parent
    if not data_dir.is_dir():
        return None

    # BIDS base = stem minus a trailing modality entity (e.g.
    # "sub-01_task-motor_eeg" -> "sub-01_task-motor").
    _BIDS_MODALITY_SUFFIXES = ("_eeg", "_meg", "_ieeg", "_ecog", "_nirs", "_beh")
    stem = p.stem
    bids_base = stem
    for suffix in _BIDS_MODALITY_SUFFIXES:
        if stem.endswith(suffix):
            bids_base = stem[: -len(suffix)]
            break

    candidates = [
        data_dir / f"events_{stem_safe(p)}.csv",
        data_dir / f"events_{stem}.csv",
        data_dir / f"{bids_base}_events.tsv",
        data_dir / f"{stem}_events.tsv",
    ]
    for c in candidates:
        if c.is_file():
            return str(c)

    # Entity-reduced BIDS fallback: strip trailing "_<key>-<val>" segments from
    # bids_base and retry, but never drop the leading sub-/ses- entities (that
    # would let us match a different subject in a flat directory).
    tokens = bids_base.split("_")
    # Determine how many leading tokens are the protected sub-/ses- prefix.
    protected = 0
    for tok in tokens:
        if tok.startswith("sub-") or tok.startswith("ses-"):
            protected += 1
        else:
            break
    # Peel one trailing entity token at a time down to (but not into) the
    # protected prefix.
    for cut in range(len(tokens) - 1, protected, -1):
        reduced = "_".join(tokens[:cut])
        if not reduced:
            continue
        cand = data_dir / f"{reduced}_events.tsv"
        if cand.is_file():
            return str(cand)

    return None


def _summarize_events_csv(events_path: str | Path) -> Optional[EventsSummary]:
    """Read a sidecar events CSV/TSV and produce a structured summary.

    Returns None on parse failure or if the file is empty.
    Caps at 10000 rows to avoid memory issues on very large files.
    """
    import csv
    import hashlib as _hl

    p = Path(events_path)
    if not p.is_file():
        return None

    try:
        raw = p.read_bytes()
    except OSError:
        return None

    file_hash = _hl.sha256(raw[:1 << 20]).hexdigest()[:8]

    text = raw.decode("utf-8-sig", errors="replace")
    delimiter = "\t" if p.suffix.lower() in (".tsv",) else ","

    try:
        reader = csv.DictReader(text.splitlines(), delimiter=delimiter)
        columns = reader.fieldnames or []
        if not columns:
            return None

        rows: list[dict] = []
        for i, row in enumerate(reader):
            if i >= 10000:
                break
            rows.append(row)
    except Exception:
        return None

    if not rows:
        return None

    n_rows = len(rows)
    sample_rows = [{k: v for k, v in r.items()} for r in rows[:3]]

    time_col = None
    for candidate in ("time", "onset", "time_s", "latency", "sample_index"):
        if candidate in columns:
            time_col = candidate
            break

    time_range: list[float] | None = None
    if time_col:
        try:
            times = [float(r[time_col]) for r in rows if r.get(time_col)]
            if times:
                time_range = [round(min(times), 3), round(max(times), 3)]
        except (ValueError, TypeError):
            pass

    code_col = None
    for candidate in ("event_code", "trigger_code", "type", "value",
                      "description", "trial_type", "stim_type"):
        if candidate in columns:
            code_col = candidate
            break

    event_code_distribution: dict[str, int] = {}
    if code_col:
        for r in rows:
            v = str(r.get(code_col, "") or "").strip()
            if v:
                event_code_distribution[v] = event_code_distribution.get(v, 0) + 1

    trial_col = None
    for candidate in ("trial_id", "trial", "trial_number", "epoch"):
        if candidate in columns:
            trial_col = candidate
            break

    unique_trials: int | None = None
    if trial_col:
        trial_vals = {r.get(trial_col) for r in rows if r.get(trial_col)}
        unique_trials = len(trial_vals)

    return EventsSummary(
        source_file=p.name,
        n_rows=n_rows,
        columns=list(columns),
        time_range_s=time_range,
        event_code_distribution=event_code_distribution,
        unique_trials=unique_trials,
        sample_rows=sample_rows,
        file_hash_prefix=file_hash,
    )


def _scan(
    *,
    data_path: str,
    sample_pct: float,
    psd_resolution_hz: float,
    max_preload_mb: int,
) -> InspectionReport:
    light = load_neural(data_path, modality="auto", inspect_only=True)
    meta = light.get("meta", {}) or {}
    if isinstance(meta, dict) and meta.get("load_error"):
        raise RuntimeError(meta["load_error"])

    n_channels = int(meta.get("n_channels") or len(light.get("channels") or []))
    n_samples_total = int(meta.get("n_samples_total") or 0)
    fs = float(light.get("frequency") or 0.0)
    duration_s = float(light.get("duration") or (n_samples_total / fs if fs else 0.0))
    channels = list(light.get("channels") or [])
    # Backends report modality inside meta (top-level "modality" never exists);
    # fall back to top-level then "auto" for forward-compat.
    modality = meta.get("modality") or light.get("modality") or "auto"
    fmt = (meta.get("format") or Path(data_path).suffix.lstrip(".")).lower()

    preload_mb = (n_channels * n_samples_total * 4) / (1024 * 1024)
    if preload_mb > max_preload_mb:
        raise MemoryError(
            f"preload_mb={preload_mb:.0f} > cap={max_preload_mb}"
        )

    full = load_neural(data_path, modality=modality, inspect_only=False)
    data = full["data"]

    if isinstance(data, list):
        # Spike modality: list-of-arrays — PSD / artifact stats not meaningful.
        return _spike_fingerprint(
            data_path=data_path, channels=channels, modality=modality,
            fs=fs, duration_s=duration_s, fmt=fmt, data=data,
        )

    channel_stats = _per_channel_stats(data, channels)
    psd_summary = _psd_summary(data, fs=fs, resolution_hz=psd_resolution_hz)
    artifact_summary = _artifact_summary(data, fs=fs, sample_pct=sample_pct)
    channel_summary = _classify_channels(
        channel_stats=channel_stats, channels=channels, meta=meta,
        modality=modality,
    )
    warnings = _build_warnings(channel_stats, channel_summary, psd_summary)

    events_block = meta.get("annotations") or {}
    n_events = (
        len(events_block.get("onset", []))
        if isinstance(events_block, dict) else 0
    )
    event_types = (
        sorted({str(d) for d in events_block.get("description", [])})
        if n_events else []
    )

    return InspectionReport(
        generated_at=datetime.utcnow().isoformat(timespec="seconds"),
        data_path=str(data_path),
        degraded=False,
        degraded_reason=None,
        fingerprint=Fingerprint(
            format=fmt, modality=modality,
            n_channels=n_channels, sampling_freq_hz=fs,
            duration_s=round(duration_s, 2),
            n_events=int(n_events), event_types=event_types,
        ),
        channel_stats=channel_stats,
        channel_summary=channel_summary,
        psd_summary=psd_summary,
        artifact_summary=artifact_summary,
        memory_estimate=MemoryEstimate(
            preload_full_mb=round(preload_mb, 1),
            peak_processing_mb_estimate=round(preload_mb * 4, 1),
        ),
        warnings=warnings,
    )


def _per_channel_stats(data: np.ndarray, channels: list[str]) -> list[ChannelStat]:
    if data.ndim != 2:
        return []
    out: list[ChannelStat] = []
    for i in range(data.shape[0]):
        ch = data[i]
        nan_pct = float(np.isnan(ch).mean() * 100.0)
        inf_pct = float(np.isinf(ch).mean() * 100.0)
        flat_pct = float((np.diff(ch) == 0).mean() * 100.0)
        std = float(np.nanstd(ch))
        spike_count = (
            int(np.sum(np.abs(ch - np.nanmean(ch)) > 8 * std)) if std > 0 else 0
        )
        out.append(ChannelStat(
            name=channels[i] if i < len(channels) else f"ch{i}",
            category="data",
            variance=float(np.nanvar(ch)),
            mean=float(np.nanmean(ch)),
            std=std,
            nan_pct=nan_pct, inf_pct=inf_pct, flat_pct=flat_pct,
            spike_count=spike_count,
        ))
    return out


def _psd_summary(data: np.ndarray, *, fs: float, resolution_hz: float) -> PsdSummary:
    from scipy.signal import welch

    empty = PsdSummary(
        power_line_peak_hz=None,
        power_line_peak_db_above_floor=None,
        harmonics_detected_hz=[],
        low_freq_drift_below_1hz_present=False,
        high_freq_noise_above_40hz_present=False,
    )
    if data.ndim != 2 or fs <= 0 or data.shape[1] < int(fs * 4):
        return empty

    nperseg = max(64, int(fs / resolution_hz))
    avg = data.mean(axis=0)
    freqs, pxx = welch(avg, fs=fs, nperseg=min(nperseg, len(avg)))
    pxx_db = 10 * np.log10(pxx + 1e-20)
    median_db = float(np.median(pxx_db))

    band = (freqs >= 48) & (freqs <= 62)
    line_peak = None
    line_db = None
    if band.any():
        idx = int(np.argmax(pxx_db[band]))
        peak_hz = float(freqs[band][idx])
        peak_db = float(pxx_db[band][idx] - median_db)
        if peak_db > 6.0:
            line_peak = peak_hz
            line_db = peak_db

    harmonics: list[float] = []
    if line_peak:
        for k in (2, 3, 4):
            h = line_peak * k
            if h >= freqs[-1]:
                break
            i = int(np.argmin(np.abs(freqs - h)))
            if pxx_db[i] - median_db > 6.0:
                harmonics.append(round(float(freqs[i]), 1))

    low_drift = bool(np.any((freqs < 1.0) & (pxx_db > median_db + 10)))
    high_noise = bool(
        np.any((freqs > 40) & (freqs < min(fs / 2, 100)) & (pxx_db > median_db + 6))
    )

    return PsdSummary(
        power_line_peak_hz=line_peak,
        power_line_peak_db_above_floor=(
            round(line_db, 1) if line_db is not None else None
        ),
        harmonics_detected_hz=harmonics,
        low_freq_drift_below_1hz_present=low_drift,
        high_freq_noise_above_40hz_present=high_noise,
    )


def _artifact_summary(
    data: np.ndarray, *, fs: float, sample_pct: float,
) -> ArtifactSummary:
    empty = ArtifactSummary(
        sample_pct=sample_pct, blink_rate_per_min=0.0,
        muscle_artifact_pct=0.0, saturation_pct=0.0,
    )
    if data.ndim != 2 or fs <= 0:
        return empty
    n_total = data.shape[1]
    n_sample = max(int(fs * 4), int(n_total * sample_pct))
    n_sample = min(n_sample, n_total)
    if n_sample <= 1:
        return empty
    start = int((n_total - n_sample) / 2)
    seg = data[:, start:start + n_sample]

    mx = float(np.nanmax(np.abs(seg)))
    saturation_pct = (
        float((np.abs(seg) > 0.999 * mx).mean() * 100.0) if mx > 0 else 0.0
    )

    z = (seg[0] - np.nanmean(seg[0])) / (np.nanstd(seg[0]) + 1e-12)
    blink_events = int(np.sum(np.abs(z) > 4))
    blink_rate_per_min = (
        float(blink_events / (n_sample / fs / 60.0)) if n_sample else 0.0
    )

    diff_var = np.var(np.diff(seg, axis=1), axis=1)
    if diff_var.size:
        muscle_pct = float(
            (diff_var > np.percentile(diff_var, 90)).mean() * 100.0
        )
    else:
        muscle_pct = 0.0

    return ArtifactSummary(
        sample_pct=sample_pct,
        blink_rate_per_min=round(blink_rate_per_min, 2),
        muscle_artifact_pct=round(muscle_pct, 2),
        saturation_pct=round(saturation_pct, 2),
    )


def _classify_channels(
    *,
    channel_stats: list[ChannelStat],
    channels: list[str],
    meta: dict,
    modality: str,
) -> ChannelSummary:
    from easybci_lib.tools.neural_tools import _build_channel_summary
    try:
        legacy = _build_channel_summary(channels, meta, modality) or {}
    except Exception:
        legacy = {"must_drop": [], "suggest_drop": []}

    must = list(legacy.get("must_drop") or [])
    suggest = list(legacy.get("suggest_drop") or [])

    variances = [c.variance for c in channel_stats if c.variance > 0]
    median_var = float(np.median(variances)) if variances else 0.0
    high_var = [
        c.name for c in channel_stats
        if median_var and c.variance > 5 * median_var
    ]
    flat = [c.name for c in channel_stats if c.flat_pct > 50.0]
    spiky = [c.name for c in channel_stats if c.spike_count > 100]

    return ChannelSummary(
        must_drop=must, suggest_drop=suggest,
        bad_candidates_high_variance=high_var,
        bad_candidates_flat=flat,
        bad_candidates_spike=spiky,
    )


def _build_warnings(
    channel_stats: list[ChannelStat],
    channel_summary: ChannelSummary,
    psd_summary: PsdSummary,
) -> list[str]:
    warns: list[str] = []
    for ch in channel_summary.bad_candidates_high_variance:
        warns.append(
            f"Channel {ch} variance much higher than median — likely bad channel"
        )
    for ch in channel_summary.bad_candidates_flat:
        warns.append(
            f"Channel {ch} is flat (>50% identical samples) — disconnected electrode"
        )
    if psd_summary.low_freq_drift_below_1hz_present:
        warns.append("Low-frequency drift (<1 Hz) detected — recommend high-pass filter")
    if psd_summary.high_freq_noise_above_40hz_present:
        warns.append(
            "High-frequency noise above 40 Hz — recommend low-pass filter or muscle ICA"
        )
    if psd_summary.power_line_peak_hz:
        warns.append(
            f"Power line peak at {psd_summary.power_line_peak_hz} Hz "
            f"({psd_summary.power_line_peak_db_above_floor} dB above floor) — recommend notch"
        )
    return warns


def _unsupported_format_next_action(data_path: str) -> Optional[dict[str, Any]]:
    """Distinguish 'no loader for this format' from 'loader crashed'.

    Touchpoint ① of the extensible-io design: when a light load returns the
    unknown-format sentinel (meta.format=='unknown', set only by
    _load_unknown_format after built-ins AND registered plugins both declined),
    return a next_action pointing the agent at register_io_loader. A genuine
    loader crash (a recognised format that failed to parse) returns None so the
    caller keeps the plain degraded path — the two failures have different
    remedies and must not be conflated.
    """
    try:
        light = load_neural(data_path, modality="auto", inspect_only=True)
    except Exception:  # noqa: BLE001 — any load error here is not 'unsupported'
        return None
    meta = light.get("meta", {}) or {}
    if not isinstance(meta, dict) or meta.get("format") != "unknown":
        return None
    return {
        "next_tool": "register_io_loader",
        "must_present": True,
        "reason": "unsupported_format",
        "hint": (
            f"No built-in or registered loader can read '{Path(data_path).name}'. "
            "Read the file's structure, write a loader exposing matches(path) and "
            "load(path, inspect_only=False) that returns the standard dict, register "
            "it with register_io_loader (it auto-probes + validates), then re-run "
            "deep_inspect."
        ),
        "supported_formats": meta.get("supported_formats") or [],
    }


def _degraded_lightweight(data_path: str, *, reason: str) -> InspectionReport:
    """Fallback: header-only fingerprint, neutral zero stats."""
    try:
        light = load_neural(data_path, modality="auto", inspect_only=True)
    except Exception as exc:
        light = {
            "meta": {"load_error": str(exc)},
            "data": np.zeros((0, 0)),
            "channels": [], "frequency": 0.0, "duration": 0.0,
            "modality": "auto",
        }

    meta = light.get("meta", {}) or {}
    n_channels = int(meta.get("n_channels") or 0)
    n_samples = int(meta.get("n_samples_total") or 0)
    fs = float(light.get("frequency") or 0.0)
    duration_s = float(light.get("duration") or 0.0)
    # Backends report modality inside meta (top-level "modality" never exists);
    # fall back to top-level then "auto" for forward-compat.
    modality = meta.get("modality") or light.get("modality") or "auto"
    fmt = (meta.get("format") or Path(data_path).suffix.lstrip(".")).lower()

    return InspectionReport(
        generated_at=datetime.utcnow().isoformat(timespec="seconds"),
        data_path=str(data_path),
        degraded=True,
        degraded_reason=reason,
        fingerprint=Fingerprint(
            format=fmt, modality=modality,
            n_channels=n_channels, sampling_freq_hz=fs,
            duration_s=round(duration_s, 2),
            n_events=0, event_types=[],
        ),
        channel_stats=[],
        channel_summary=ChannelSummary(
            must_drop=[], suggest_drop=[],
            bad_candidates_high_variance=[],
            bad_candidates_flat=[], bad_candidates_spike=[],
        ),
        psd_summary=PsdSummary(
            power_line_peak_hz=None,
            power_line_peak_db_above_floor=None,
            harmonics_detected_hz=[],
            low_freq_drift_below_1hz_present=False,
            high_freq_noise_above_40hz_present=False,
        ),
        artifact_summary=ArtifactSummary(
            sample_pct=0.0, blink_rate_per_min=0.0,
            muscle_artifact_pct=0.0, saturation_pct=0.0,
        ),
        memory_estimate=MemoryEstimate(
            preload_full_mb=round((n_channels * n_samples * 4) / (1024 * 1024), 1),
            peak_processing_mb_estimate=0.0,
        ),
        warnings=[
            f"deep_inspect ran in degraded mode (reason={reason!r}); "
            "channel stats / PSD / artifact rate are unavailable. "
            "Treat operator/parameter choices as best-effort.",
        ],
    )


def _spike_fingerprint(
    *, data_path, channels, modality, fs, duration_s, fmt, data,
) -> InspectionReport:
    n_channels = len(data)
    return InspectionReport(
        generated_at=datetime.utcnow().isoformat(timespec="seconds"),
        data_path=str(data_path),
        degraded=True,
        degraded_reason="spike_modality_no_psd",
        fingerprint=Fingerprint(
            format=fmt, modality=modality,
            n_channels=n_channels, sampling_freq_hz=fs,
            duration_s=round(duration_s, 2),
            n_events=0, event_types=[],
        ),
        channel_stats=[],
        channel_summary=ChannelSummary(
            must_drop=[], suggest_drop=[],
            bad_candidates_high_variance=[],
            bad_candidates_flat=[], bad_candidates_spike=[],
        ),
        psd_summary=PsdSummary(
            power_line_peak_hz=None,
            power_line_peak_db_above_floor=None,
            harmonics_detected_hz=[],
            low_freq_drift_below_1hz_present=False,
            high_freq_noise_above_40hz_present=False,
        ),
        artifact_summary=ArtifactSummary(
            sample_pct=0.0, blink_rate_per_min=0.0,
            muscle_artifact_pct=0.0, saturation_pct=0.0,
        ),
        memory_estimate=MemoryEstimate(
            preload_full_mb=0.0, peak_processing_mb_estimate=0.0,
        ),
        warnings=["spike modality: deep_inspect returns fingerprint only"],
    )
