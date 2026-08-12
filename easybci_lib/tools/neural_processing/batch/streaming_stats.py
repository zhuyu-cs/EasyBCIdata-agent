"""Streaming (chunked) QC-metric computation for the batch summary step.

Rationale
---------
The batch summary used to compute three scalar QC metrics per subject by
loading the ENTIRE preprocessed NWB into RAM (``data[:]`` → one file ≈ 7 GB
float32 for 259ch × 7.18M samples), then allocating several more full-size
copies (``.T``, ``np.diff``, ``np.abs(data - median)``). Peak RSS reached
~59 GB and the kernel OOM-killed the process mid-batch, taking the tmux
session with it.

This module computes the SAME three metrics — per-channel variance (mean/std
across channels), global SNR in dB, and artifact ratio — by reading the source
in time windows and reducing per-channel in float64. Peak memory is bounded by
a budget (default a small fraction of available RAM), independent of recording
length.

Numerical equivalence (see ``tests/batch/test_streaming_stats.py``)
-------------------------------------------------------------------
Source layout is ``(n_samples, n_channels)`` (NWB on-disk order); time is axis
0 of the source, and the reference implementations treated time as axis -1 of
``(n_channels, n_samples)`` — the two are transposes of each other.

* channel variance: ``var_c = E[x_c^2] - E[x_c]^2`` from per-channel
  ``sum`` / ``sumsq`` / ``N`` (float64). mean/std across the length-n_ch vector.
* SNR dB: the reference's ``np.var(data)`` and ``np.var(np.diff(data))`` are
  over the WHOLE flattened set (scalars), so we pool per-channel raw moments:
  ``total_var = ΣsumSq/M - (Σsum/M)^2`` (M = total elements) and, since
  ``np.diff`` never crosses a channel boundary, the flattened diff set is the
  disjoint union of per-channel diffs → ``diff_var`` from pooled per-channel
  ``dsum`` / ``dsumsq`` over ``D = Σ(N_c - 1)``. ``noise_var = diff_var/2``.
  A per-channel ``prev_last`` carry folds the cross-window seam difference so
  windowing does not drop any diff term.
* artifact ratio: median and 5*MAD threshold are strictly per-channel, and the
  reference's ``np.mean(bool)`` over all elements equals ``Σcount_c / ΣN_c``.
  Exact medians need each channel's full series, so channels are processed in
  groups sized to the budget (~28 MB per channel-second-hour scale).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

__all__ = ["StreamingMetrics", "compute_streaming_metrics"]


@dataclass
class StreamingMetrics:
    channel_variance_mean: float
    channel_variance_std: float
    snr_db: float
    artifact_ratio: float
    n_channels: int
    n_samples: int


def _resolve_budget_mb(memory_budget_mb: Optional[float]) -> float:
    """Budget (MB) for per-channel accumulators. Small fraction of available RAM."""
    if memory_budget_mb is not None and memory_budget_mb > 0:
        return float(memory_budget_mb)
    try:
        from easybci_lib.tools.neural_processing.preprocess.memory_strategy import (
            _get_available_memory_mb,
        )
        available = _get_available_memory_mb()
    except Exception:
        available = 8000.0
    # Budget sizes the per-channel-group series buffers; measured peak RSS is
    # ~2-3x this (buffer + median/dev temporaries + the row read block). 512 MB
    # → ~1.3 GB peak on a 259ch/7.18M sEEG file, and it's actually faster than
    # a larger budget (cache locality). Cap low and independent of host size so
    # a big box doesn't balloon to 8+ GB for no speed gain; shrink only on
    # genuinely tiny hosts.
    return float(min(512.0, max(128.0, available * 0.02)))


def _as_2d_samp_ch(dataset: Any) -> Any:
    """Return an object with ``.shape == (n_samples, n_channels)`` and
    ``__getitem__`` slicing. Accepts an h5py Dataset, a lazy shim, or an
    in-memory ndarray. A 1-D array is treated as a single channel; a
    ``(n_ch, n_samp)`` array whose caller already transposed is NOT assumed —
    callers pass the on-disk ``(n_samp, n_ch)`` orientation.
    """
    shape = getattr(dataset, "shape", None)
    if shape is None:
        dataset = np.asarray(dataset)
        shape = dataset.shape
    if len(shape) == 1:
        # single channel, samples along axis 0
        return dataset
    if len(shape) != 2:
        raise ValueError(f"streaming metrics need a 2-D source, got shape {shape}")
    return dataset


def compute_streaming_metrics(
    dataset: Any,
    *,
    memory_budget_mb: Optional[float] = None,
    window_samples: Optional[int] = None,
) -> StreamingMetrics:
    """Compute channel-variance / SNR / artifact-ratio without full load.

    Parameters
    ----------
    dataset:
        Sliceable source with layout ``(n_samples, n_channels)`` — e.g. the
        h5py ``Dataset`` behind an NWB ``ElectricalSeries.data``. Time windows
        are read as ``dataset[t0:t1, :]`` (contiguous sequential I/O for
        C-order storage). An in-memory ndarray is also accepted.
    memory_budget_mb:
        Budget for per-channel accumulator buffers. ``None`` → auto from host.
    window_samples:
        Rows read per I/O block. ``None`` → derived from the budget.
    """
    ds = _as_2d_samp_ch(dataset)
    shape = ds.shape
    if len(shape) == 1:
        n_samp, n_ch = int(shape[0]), 1
        single_channel = True
    else:
        n_samp, n_ch = int(shape[0]), int(shape[1])
        single_channel = False

    if n_samp == 0 or n_ch == 0:
        return StreamingMetrics(0.0, 0.0, 0.0, 0.0, n_ch, n_samp)

    budget_mb = _resolve_budget_mb(memory_budget_mb)
    budget_bytes = budget_mb * 1024 * 1024

    # Channel group size G: each held channel keeps a full float32 series buffer
    # (n_samp * 4 bytes) for exact median/MAD.
    per_channel_bytes = max(1, n_samp * 4)
    group_size = max(1, min(n_ch, int(budget_bytes // per_channel_bytes)))

    # I/O window (rows). Keep the read block within ~1/4 of budget too.
    if window_samples is None:
        row_bytes = max(1, n_ch * 4)
        window_samples = max(1, int((budget_bytes / 4) // row_bytes))
    window_samples = max(1, min(int(window_samples), n_samp))

    # Global (all-channel) additive accumulators for variance & SNR pooling.
    g_sum = np.zeros(n_ch, dtype=np.float64)
    g_sumsq = np.zeros(n_ch, dtype=np.float64)
    g_n = np.zeros(n_ch, dtype=np.float64)
    d_sum = np.zeros(n_ch, dtype=np.float64)
    d_sumsq = np.zeros(n_ch, dtype=np.float64)
    d_n = np.zeros(n_ch, dtype=np.float64)

    artifact_count = 0.0
    artifact_total = 0.0

    def _read_window(t0: int, t1: int, ch0: int, ch1: int) -> np.ndarray:
        if single_channel:
            block = np.asarray(ds[t0:t1])
            return block.reshape(-1, 1)
        block = np.asarray(ds[t0:t1, ch0:ch1])
        if block.ndim == 1:
            block = block.reshape(-1, 1)
        return block

    # Process channels in groups; fuse the additive pass into the first group's
    # sweep is not possible cheaply (groups partition channels), so we run the
    # additive accumulation per group over the same windowed reads.
    for ch0 in range(0, n_ch, group_size):
        ch1 = min(n_ch, ch0 + group_size)
        gwidth = ch1 - ch0

        # Per-channel full-series buffers (float32) for exact median/MAD.
        buffers = np.empty((gwidth, n_samp), dtype=np.float32)
        prev_last = np.full(gwidth, np.nan, dtype=np.float64)

        for t0 in range(0, n_samp, window_samples):
            t1 = min(n_samp, t0 + window_samples)
            block = _read_window(t0, t1, ch0, ch1)  # (w, gwidth)
            w = block.shape[0]
            blk64 = block.astype(np.float64, copy=False)

            # store for median/MAD
            buffers[:, t0:t1] = block.T if block.shape[1] == gwidth else block

            # additive moments (per channel over this window)
            g_sum[ch0:ch1] += blk64.sum(axis=0)
            g_sumsq[ch0:ch1] += np.square(blk64).sum(axis=0)
            g_n[ch0:ch1] += w

            # within-window diffs along time (axis 0)
            if w >= 2:
                dwin = np.diff(blk64, axis=0)  # (w-1, gwidth)
                d_sum[ch0:ch1] += dwin.sum(axis=0)
                d_sumsq[ch0:ch1] += np.square(dwin).sum(axis=0)
                d_n[ch0:ch1] += (w - 1)
            # seam diff: first sample of this window minus last of previous
            first = blk64[0]  # (gwidth,)
            seam_valid = ~np.isnan(prev_last)
            if seam_valid.any():
                seam = first - prev_last
                idx = np.where(seam_valid)[0]
                d_sum[ch0 + idx] += seam[idx]
                d_sumsq[ch0 + idx] += np.square(seam[idx])
                d_n[ch0 + idx] += 1
            prev_last = blk64[-1].copy()

        # Exact per-channel median / MAD / artifact count. Loop per channel so
        # the abs-deviation temporary is one channel (~n_samp*4 bytes), not the
        # whole group — otherwise np.abs(group - med) would triple the group
        # buffer's footprint and dominate peak RSS.
        for gi in range(gwidth):
            ch = buffers[gi]
            med = np.median(ch)
            dev = np.abs(ch - med)
            mad = max(float(np.median(dev)), 1e-12)
            artifact_count += float(np.count_nonzero(dev > 5.0 * mad))
        artifact_total += float(gwidth * n_samp)
        del buffers

    # ---- channel variance (mean/std across channels) ----
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_c = g_sum / g_n
        var_c = g_sumsq / g_n - np.square(mean_c)
    var_c = np.maximum(var_c, 0.0)  # guard tiny negative from cancellation
    channel_variance_mean = float(np.mean(var_c))
    channel_variance_std = float(np.std(var_c))

    # ---- SNR dB (global pooled) ----
    M = float(g_n.sum())
    total_sum = float(g_sum.sum())
    total_sumsq = float(g_sumsq.sum())
    total_var = total_sumsq / M - (total_sum / M) ** 2
    total_var = max(total_var, 0.0)

    D = float(d_n.sum())
    if D > 0:
        dd_sum = float(d_sum.sum())
        dd_sumsq = float(d_sumsq.sum())
        diff_var = dd_sumsq / D - (dd_sum / D) ** 2
        diff_var = max(diff_var, 0.0)
        noise_var = diff_var / 2.0
    else:
        noise_var = 0.0

    signal_var = max(total_var - noise_var, 1e-12)
    if noise_var < 1e-12:
        snr_db = 30.0
    else:
        snr = signal_var / noise_var
        snr_db = float(10 * math.log10(max(snr, 1e-6)))

    # ---- artifact ratio ----
    artifact_ratio = float(artifact_count / artifact_total) if artifact_total else 0.0

    return StreamingMetrics(
        channel_variance_mean=channel_variance_mean,
        channel_variance_std=channel_variance_std,
        snr_db=snr_db,
        artifact_ratio=artifact_ratio,
        n_channels=n_ch,
        n_samples=n_samp,
    )
