---
name: waveform_snippet
description: "Extract per-spike waveform windows (-0.5 to +1.5 ms) around threshold-detected events for QC / offline-sorting bootstrap"
layer: L3
group: spike
metadata:
  tags: [operator, spike, waveform, snippet, qc, sorting_bootstrap]
  modalities: [spike]
  step_string: "waveform_snippet"
  analysis_goal_allowed: [feature_extraction, exploratory]
  analysis_goal_forbidden: [source_localization, online_inference]
---
# Spike Waveform Snippet

## Function

Cuts fixed-width waveform windows around each spike time produced by
upstream `threshold_spike`, yielding a 3D ``(n_channels, n_spikes_max,
n_window_samples)`` array (ragged-padded with NaN) for downstream
visualization, QC, or offline-sorting bootstrap.

The default window is **-0.5 ms to +1.5 ms** relative to the spike peak
(60 samples at 30 kHz), matching the SpikeInterface / Kilosort default
``ms_before=0.5, ms_after=1.5`` waveform extractor.

**Input / Output**:

```
in  : data_dict["data"] (n_ch, n_t)  + meta["spike_times"]: list[ndarray]
out : meta["waveforms"]: ndarray (n_ch, n_spikes_padded, n_window_samples)
      meta["waveform_window_s"]: (t_before_s, t_after_s)
      meta["waveform_spike_counts"]: ndarray (n_ch,)
```

Continuous ``data_dict["data"]`` is unchanged (Rule 5).

## Algorithm & Math

### Window cut

For each channel `c` with spike times `t_c = [t_0, t_1, ...]` (sample
indices), window half-widths in samples:

```
n_before = round(ms_before · sfreq / 1000)
n_after  = round(ms_after  · sfreq / 1000)
window_len = n_before + n_after
```

Then per spike `t_i`:

```
W[c, i, :] = data[c, t_i - n_before : t_i + n_after]
```

Spikes whose window would extend past the recording boundary are
**dropped silently** (waveform clipped at boundary is meaningless for
QC); the original count is preserved in `meta["mua_counts"]` from
`threshold_spike` for reference.

### Ragged-padded output

Because per-channel spike counts vary, the 3D output is padded with
`NaN` along the spike axis to `max(spike_counts)`. Downstream code reads
the valid count from `meta["waveform_spike_counts"][c]`.

For memory-bounded sessions, set `max_spikes_per_channel=N` to cap the
output at the first N detections per channel (random selection optional via `EASYBCI_SEED`).

### Spike-triggered average (STA)

The mean waveform per channel is a useful QC artefact:

```
sta[c, :] = mean(W[c, :n_spikes_c, :], axis=0)
```

Stored in `meta["spike_triggered_average"]`.

## Parameter Format & Defaults

`waveform_snippet:{ms_before},{ms_after},{max_per_channel}` — comma-separated:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ms_before` | float | 0.5 | Pre-peak window (ms). |
| `ms_after` | float | 1.5 | Post-peak window (ms). |
| `max_per_channel` | int or None | None | Cap waveform count per channel for memory. `None` = no cap. |
| `align_peak` | bool (kw) | True | Re-center on the negative peak within ±0.2 ms (corrects threshold-crossing vs peak offset). |

Examples:

- `waveform_snippet` — Kilosort/SpikeInterface default 2 ms total window.
- `waveform_snippet:1.0,2.0,500` — wider 3 ms window, cap 500 per channel.
- `waveform_snippet:0.5,1.5,None` with `align_peak=True` — full extraction with peak alignment.

## Modality-Specific Considerations

| Modality | Window | Notes |
|----------|--------|-------|
| Neuropixels (cortex) | 0.5 ms before / 1.5 ms after | Default. Captures full AP including after-hyperpolarization. |
| Neuropixels (hippocampus) | 0.5 ms / 2.0 ms | Slightly longer post for pyramidal cell ADP. |
| Utah array (motor) | 0.3 ms / 1.2 ms | Tighter; spikes are sharper in motor M1. |
| Tetrode | 0.5 ms / 1.5 ms | Same as Neuropixels — tetrode density is lower but waveforms similar. |

### Hard exclusions

EEG / MEG / sEEG-macro / ECoG / fNIRS / LFP-only band → no spike content;
`threshold_spike` upstream would have rejected the modality already. If
this op runs without `meta["spike_times"]`, raises
`EasyBCIOperatorError(recoverable=True, fallback_step="threshold_spike → waveform_snippet")`.

## When to Use / NOT to Use

**Use** when:

- Building **QC plots** of mean waveform per channel (spike-triggered average).
- Bootstrapping **offline sorting** training set from threshold detections.
- Debugging upstream `threshold_spike` (waveform shape sanity check).
- Cell-type-by-waveform classification on top of threshold MUA.

**Don't use** when:

- Target is **online BCI** — waveform extraction adds latency and memory
  pressure with little online value.
- You have **`spike_sorting`** outputs already — Kilosort's `waveforms.npy`
  is the right artefact; do not double-extract.
- Recording is very long (> 30 min, 384 ch, 30 kHz) and `max_per_channel`
  is unset — output can balloon past memory.

## Constraints & Ordering

- **Required upstream**: `threshold_spike` (writes `meta["spike_times"]`).
- **`sfreq >= 20 kHz`** for usable waveform timing — same constraint as `threshold_spike`.
- **Memory floor**: at 384 ch × 100k spikes × 60 samples × 4 bytes ≈ 9.2 GB. Cap with `max_per_channel=2000`.
- **Mutually exclusive with `spike_sorting`** on the same input: sorting produces its own waveforms; don't run both.

Canonical ordering:

```
bandpass:300,6000 → car → threshold_spike → waveform_snippet → qc_plot
                                          → mua_binning      → decoder
```

`waveform_snippet` and `mua_binning` can run **in parallel** — they consume the same `spike_times` and don't interact.

## Failure Modes & Detection

| Failure | Symptom | Detection |
|---------|---------|-----------|
| **Missing spike_times** | KeyError on `meta["spike_times"]`. | Pre-check; raise recoverable suggesting upstream `threshold_spike`. |
| **Boundary clip drop** | `len(W) < len(spike_times)` for some channels. | Reported in `meta["waveform_spike_counts"]`; logged at WARNING when drop > 1%. |
| **Out-of-memory on huge sessions** | MemoryError mid-extraction. | Detect via pre-allocation size check; if `n_ch · max_spikes · n_samples · 4 > 4 GB`, raise recoverable suggesting `max_per_channel=2000`. |
| **Misaligned peak** | STA peak does not sit at sample `n_before`; downstream sorting fails. | Compute `argmax(|sta[c]|)` deviation from `n_before` across channels; if median > 3 samples → `align_peak=False` likely set; warn. |
| **Waveform is just noise** | STA looks like white noise. | `mean(|sta|) / mean(|noise|) < 1.5` where `mean(|noise|)` is the MAD σ̂ → upstream threshold was too liberal; suggest k↑1. |

Auto-detection helper:

```python
import numpy as np


def diagnose_waveforms(waveforms, spike_times, n_before):
    """Return (status, details). `waveforms` shape (n_ch, n_spk, n_win)."""
    n_ch, n_spk, n_win = waveforms.shape
    if n_spk == 0:
        return "empty", {}
    valid = ~np.isnan(waveforms[..., 0])  # mask of present spikes
    counts = valid.sum(axis=1)
    if (counts == 0).all():
        return "all_clipped", {"hint": "ms_before/ms_after too large vs duration"}
    sta = np.nanmean(waveforms, axis=1)  # (n_ch, n_win)
    peak_offset = np.abs(np.nanargmax(np.abs(sta), axis=1) - n_before)
    median_offset = float(np.median(peak_offset))
    if median_offset > 3:
        return "misaligned", {"median_offset_samples": median_offset}
    return "ok", {"mean_count": float(counts.mean()), "sta_peak_offset": median_offset}
```

## Common Issues

- **"My waveforms look like white noise."** Upstream threshold was too
  liberal — many false positives. Raise `k` (e.g., 4 → 5) and re-run.
- **"Different channels have very different mean waveforms."** Expected —
  Neuropixels covers multiple brain regions with different cell types;
  Utah arrays have less variability but still meaningful.
- **"Memory blew up."** Set `max_per_channel=2000` for long sessions; that
  caps output at ~75 MB per channel.
- **"Mean waveform peak is not centered."** Set `align_peak=True`
  (default) — threshold crossings are slightly before the actual peak;
  the operator re-centers per spike within a ±0.2 ms window.

## Reference Implementation

### Standalone (no EasyBCI dependencies)

```python
"""Standalone waveform snippet extraction — drop into any environment."""
from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np


def waveform_snippet(
    data: np.ndarray,
    spike_times: List[np.ndarray],
    sfreq: float,
    *,
    ms_before: float = 0.5,
    ms_after: float = 1.5,
    max_per_channel: int | None = None,
    align_peak: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Cut waveform windows around each spike.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
        Band-pass filtered AP-band signal.
    spike_times : list[ndarray]
        Per-channel spike sample indices (output of threshold_spike).
    sfreq : float
        Sampling rate (Hz).
    ms_before, ms_after : float
        Window half-widths in milliseconds (default 0.5 / 1.5).
    max_per_channel : int or None
        Cap waveform count per channel for memory; None = no cap.
    align_peak : bool
        Re-center on negative-peak within ±0.2 ms (default True).

    Returns
    -------
    waveforms : ndarray, shape (n_channels, n_spikes_max, n_window_samples)
        Padded with NaN beyond per-channel valid count.
    counts : ndarray, shape (n_channels,)
        Valid (non-NaN) spike count per channel.
    """
    n_ch, n_t = data.shape
    n_before = int(round(ms_before * sfreq / 1000))
    n_after = int(round(ms_after * sfreq / 1000))
    n_win = n_before + n_after
    align_window = int(round(0.2 * sfreq / 1000))  # ±0.2 ms peak search

    rng = np.random.default_rng(int(os.environ.get("EASYBCI_SEED", "0")))

    valid_indices: List[np.ndarray] = []
    for c, idx in enumerate(spike_times):
        good = idx[(idx >= n_before) & (idx + n_after < n_t)]
        if max_per_channel is not None and len(good) > max_per_channel:
            sel = rng.choice(len(good), size=max_per_channel, replace=False)
            good = np.sort(good[sel])
        valid_indices.append(good)

    n_spk_max = max((len(v) for v in valid_indices), default=0)
    waveforms = np.full((n_ch, n_spk_max, n_win), np.nan, dtype=np.float32)
    counts = np.zeros(n_ch, dtype=np.int64)

    for c, idx in enumerate(valid_indices):
        for i, t in enumerate(idx):
            if align_peak:
                lo = max(0, t - align_window)
                hi = min(n_t, t + align_window)
                t = lo + int(np.argmax(np.abs(data[c, lo:hi])))
                if t < n_before or t + n_after >= n_t:
                    continue
            waveforms[c, i, :] = data[c, t - n_before : t + n_after]
            counts[c] += 1

    return waveforms, counts
```

### EasyBCI-Adapted (in-framework)

```python
from typing import Any, Dict, List
import os
import time

import numpy as np

from easybci_lib.tools.neural_processing.operator_errors import EasyBCIOperatorError
from easybci_lib.tools.neural_processing.preprocess.step_cache import record_step_elapsed


def operator_waveform_snippet(
    data_dict: Dict[str, Any],
    *,
    ms_before: float = 0.5,
    ms_after: float = 1.5,
    max_per_channel: int | None = None,
    align_peak: bool = True,
) -> Dict[str, Any]:
    """EasyBCI-adapted waveform-snippet extraction.

    Parameters
    ----------
    data_dict : dict
        OperatorIO; requires ``meta["spike_times"]`` from `threshold_spike`.
    ms_before, ms_after : float
        Window half-widths in milliseconds (default 0.5 / 1.5).
    max_per_channel : int or None
        Cap per-channel waveform count for memory (default None).
    align_peak : bool
        Re-center on negative peak within ±0.2 ms (default True).

    Returns
    -------
    dict
        OperatorIO with continuous ``data`` unchanged plus:
        - ``meta["waveforms"]``: ndarray (n_ch, n_spk_max, n_win) float32
        - ``meta["waveform_spike_counts"]``: ndarray (n_ch,) int64
        - ``meta["waveform_window_s"]``: (t_before_s, t_after_s) tuple
        - ``meta["spike_triggered_average"]``: ndarray (n_ch, n_win) float32

    Raises
    ------
    EasyBCIOperatorError
        ``recoverable=True`` if ``meta["spike_times"]`` missing.
        ``recoverable=True`` (fallback to capped max_per_channel) if
        estimated memory exceeds 4 GB.

    Modality coverage
    -----------------
    spike: yes (Neuropixels / Utah / tetrode / sEEG micro-wire).
    EEG / MEG / sEEG-macro / ECoG / fNIRS: forbidden (no spike_times upstream).

    References
    ----------
    Pachitariu 2024; SpikeInterface waveform extractor docs.
    """
    spike_times: List[np.ndarray] = (data_dict.get("meta") or {}).get("spike_times")
    if spike_times is None:
        raise EasyBCIOperatorError(
            operator="waveform_snippet",
            reason="meta['spike_times'] missing — run threshold_spike upstream",
            recoverable=True,
            fallback_step="threshold_spike → waveform_snippet",
        )

    data = data_dict["data"]
    sfreq = float(data_dict["frequency"])
    n_ch, n_t = data.shape
    n_before = int(round(ms_before * sfreq / 1000))
    n_after = int(round(ms_after * sfreq / 1000))
    n_win = n_before + n_after
    align_window = int(round(0.2 * sfreq / 1000))

    total_spikes = sum(len(s) for s in spike_times)
    est_bytes = n_ch * max(1, total_spikes // max(1, n_ch)) * n_win * 4
    if max_per_channel is None and est_bytes > 4 * (1 << 30):
        suggested = max(500, 4 * (1 << 30) // (n_ch * n_win * 4))
        raise EasyBCIOperatorError(
            operator="waveform_snippet",
            reason=f"estimated output ~{est_bytes / (1 << 30):.1f} GB > 4 GB",
            recoverable=True,
            fallback_step=f"waveform_snippet:{ms_before},{ms_after},{suggested}",
        )

    t0 = time.monotonic()
    rng = np.random.default_rng(int(os.environ.get("EASYBCI_SEED", "0")))

    valid_indices = []
    for idx in spike_times:
        good = idx[(idx >= n_before) & (idx + n_after < n_t)]
        if max_per_channel is not None and len(good) > max_per_channel:
            sel = rng.choice(len(good), size=max_per_channel, replace=False)
            good = np.sort(good[sel])
        valid_indices.append(good)

    n_spk_max = max((len(v) for v in valid_indices), default=0)
    waveforms = np.full((n_ch, n_spk_max, n_win), np.nan, dtype=np.float32)
    counts = np.zeros(n_ch, dtype=np.int64)

    for c, idx in enumerate(valid_indices):
        for i, t in enumerate(idx):
            t_int = int(t)
            if align_peak:
                lo = max(0, t_int - align_window)
                hi = min(n_t, t_int + align_window)
                t_int = lo + int(np.argmax(np.abs(data[c, lo:hi])))
                if t_int < n_before or t_int + n_after >= n_t:
                    continue
            waveforms[c, i, :] = data[c, t_int - n_before : t_int + n_after]
            counts[c] += 1

    sta = np.nanmean(waveforms, axis=1).astype(np.float32) if n_spk_max > 0 else np.zeros((n_ch, n_win), dtype=np.float32)
    elapsed = time.monotonic() - t0

    out = dict(data_dict)
    out["data"] = data  # unchanged
    out["elapsed_s"] = elapsed
    new_meta = dict(out.get("meta") or {})
    new_meta["waveforms"] = waveforms
    new_meta["waveform_spike_counts"] = counts
    new_meta["waveform_window_s"] = (ms_before / 1000.0, ms_after / 1000.0)
    new_meta["spike_triggered_average"] = sta
    new_meta["waveform_snippet"] = {
        "ms_before": ms_before, "ms_after": ms_after,
        "max_per_channel": max_per_channel, "align_peak": align_peak,
    }
    out["meta"] = new_meta
    record_step_elapsed(
        "waveform_snippet", elapsed,
        (data_dict.get("meta") or {}).get("step_cache_key"),
    )
    return out
```

## References

1. Pachitariu, M., Sridhar, S., & Stringer, C. (2024). *Spike sorting with
   Kilosort4*. Nature Methods 21(5): 914–921. doi:10.1038/s41592-024-02232-7
   — waveform extraction defaults that match this op's window.
2. Buccino, A. P. et al. (2020). *SpikeInterface, a unified framework for
   spike sorting*. eLife 9: e61834. doi:10.7554/eLife.61834 — `extract_waveforms`
   parameter conventions; aligns with `ms_before / ms_after` of this op.
3. Lee, J. et al. (2017). *YASS: Yet Another Spike Sorter*. NeurIPS 2017 —
   peak-alignment heuristic; we use a ±0.2 ms window for cheap re-centering.
4. Rey, H. G., Pedreira, C., & Quiroga, R. Q. (2015). *Past, present and
   future of spike sorting techniques*. Brain Research Bulletin 119:
   106–117. doi:10.1016/j.brainresbull.2015.04.007 — context on waveform
   features in spike-by-waveform classification.
