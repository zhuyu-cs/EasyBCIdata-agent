---
name: mua_binning
description: "Bin MUA spike times into firing-rate arrays (1/10/50/100 ms windows) for decoding / population analysis"
layer: L3
group: spike
metadata:
  tags: [operator, spike, mua, binning, firing_rate, decode, neuropixels, utah]
  modalities: [spike]
  step_string: "mua_binning"
  analysis_goal_allowed: [classification, feature_extraction, exploratory, generic, online_inference]
  analysis_goal_forbidden: [source_localization]
---
# MUA Binning

## Function

Converts a per-channel spike-time list (from `threshold_spike`) into a
2D firing-rate matrix sampled on a regular time grid. Operates on the
``meta["spike_times"]`` produced by `threshold_spike` and writes
``meta["mua_train"]`` (shape `(n_channels, n_bins)`) plus
``meta["bin_centers"]`` for downstream alignment.

This is the canonical "spike times → continuous-valued feature" step for
population decoding, online BCI, and any analysis that needs spike
activity at a fixed temporal resolution (e.g., LSTMs over 25 ms bins).

**Input / Output**:

```
in  : data_dict with meta["spike_times"]: list[ndarray]   # output of threshold_spike
out : meta["mua_train"]: ndarray (n_channels, n_bins)     # spike count or firing rate
      meta["bin_centers"]: ndarray (n_bins,)              # seconds, midpoints
```

The continuous ``data_dict["data"]`` is left unchanged (Rule 5); binning
augments meta only.

## Algorithm & Math

### Histogram-based binning

For channel `c` with spike times `t_c = [t_0, t_1, ..., t_N]` (sample
indices), and bin width `Δ` in seconds:

```
count_c[i] = |{ t ∈ t_c : i·Δ ≤ t/sfreq < (i+1)·Δ }|
n_bins = floor(duration / Δ)
```

The output unit defaults to **firing rate** (`spikes / second`):

```
rate_c[i] = count_c[i] / Δ
```

When `output="counts"` the raw count is kept (lower precision; smaller
file).

### Gaussian smoothing kernel

Bin sizes < 50 ms produce sparse / spiky rate estimates. The optional
Gaussian kernel σ = `Δ / 2` smooths bins without introducing significant
latency:

```
rate_c[i] ← (rate_c * g_σ)[i]            where g_σ(t) = exp(-t²/(2σ²)) / (σ √2π)
```

This matches Cunningham & Yu (2014)'s recommendation for MUA decoding:
σ slightly less than the bin width prevents over-smoothing while
suppressing histogram noise. Disable for online streaming (kernel
introduces ½-window lookahead unless causal-truncated).

### Bin size → downstream task

| Bin Δ | Use case | Notes |
|-------|----------|-------|
| 1 ms | High-precision timing analysis; rate-PSTH | Very sparse; requires N>10 trials averaging. |
| 10 ms | Reach-decoding / fast motor BCI | BrainGate cursor decoder default. |
| **25 ms** | **Default** for motor / cognitive decoders | Trautmann 2019; LSTM input cadence. |
| 50 ms | Slow MI / cognitive load | Smooths to behavioral timescale. |
| 100 ms | Drug-effect / state estimation | Hides fast dynamics; use only when needed. |

### Edge handling

- Spikes with `t > duration` are dropped (defensive — should not happen if `threshold_spike` is upstream on the same data).
- Spikes within the **last `Δ/2`** of the recording are kept; the last bin is potentially under-sampled but reported with the full count.

## Parameter Format & Defaults

`mua_binning:{bin_ms},{smooth},{output}` — comma-separated:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `bin_ms` | float | 25.0 | Bin width in milliseconds. |
| `smooth` | bool | True | Apply Gaussian kernel (σ = bin/2). Disable for online causal. |
| `output` | "rate" \| "counts" | "rate" | Firing rate (spikes/s) vs raw count. |
| `causal` | bool (kw) | False | If True, the kernel is half-Gaussian (one-sided, no lookahead) — for online use. |

Examples:

- `mua_binning` — 25 ms bins, smoothed, firing rate (default for offline decoding).
- `mua_binning:10,True,rate` — BrainGate-style 10 ms bins.
- `mua_binning:25,False,counts` — raw counts at 25 ms; downstream model handles smoothing.
- `mua_binning:25,True,rate` with `causal=True` — online BCI variant.

See `parameter_uncertainty/mua_binning.yaml` for paradigm-specific defaults.

## Modality-Specific Considerations

This operator is **spike-only**; it operates on `meta["spike_times"]`
(produced by `threshold_spike`) and ignores the continuous `data`. Modality
table below indicates which upstream pipelines feed sensible spike trains.

| Upstream modality | Bin size | Smooth | Notes |
|---|---|---|---|
| Neuropixels (motor / hippocampus / visual) | 25 ms | True | Default; matches IBL decoders. |
| Utah array (motor cortex) | 10 ms | True | BrainGate / Hochberg cursor control default. |
| Tetrode (hippocampus place cells) | 50 ms | True | Slower task timescale. |
| sEEG micro-wire (MTL single units) | 100 ms | True | Sparse firing; aggregate to behavioral timescale. |

### Hard exclusions

`mua_binning` requires `meta["spike_times"]` from upstream `threshold_spike`
or equivalent. If absent the operator raises
`EasyBCIOperatorError(recoverable=True, fallback_step="threshold_spike → mua_binning")`.

## When to Use / NOT to Use

**Use** when:

- Downstream is a **temporal model** (LSTM / GRU / Transformer / Kalman) needing fixed-rate inputs.
- Decoder is **linear / Riemannian** over firing-rate windows.
- Target is **online BCI** with latency budget — small bins (10–25 ms) + causal kernel.
- Population-level analysis (mean ± SEM PSTH, dimensionality reduction).

**Don't use** when:

- Target is **single-spike timing** (cross-correlogram, spike-triggered LFP — use `cross_correlogram` / `autocorrelogram`).
- Target is **PAC / phase-amplitude coupling** — needs phase, not rate.
- Upstream sorting produced **unit-labelled** trains and you want per-unit features — use `bin_spikes` (unit-level) instead; `mua_binning` aggregates across channel.
- `analysis_goal == source_localization` — source models do not consume rate matrices.

## Constraints & Ordering

- **Required upstream**: `threshold_spike` (or equivalent that writes `meta["spike_times"]`).
- **Strongly recommended downstream**: a decoder operator (`riemannian_covariance`, linear) or feature aggregator (`firing_rate`).
- **Bin size lower bound**: `bin_ms >= 1000 / sfreq` (cannot be smaller than the sample period — would alias).
- **Conflict with `bin_spikes`**: both produce `mua_train`; pick one. `bin_spikes` is for **sorted** spike data (per-unit), `mua_binning` is for **threshold MUA** (per-channel).

Canonical ordering:

```
bandpass:300,6000 → car → threshold_spike → mua_binning → riemannian_covariance → classifier
```

## Failure Modes & Detection

| Failure | Symptom | Detection |
|---------|---------|-----------|
| **Empty spike_times** | `mua_train` is all zeros. | `meta["mua_train"].sum() == 0` → log warning + raise recoverable error suggesting upstream threshold-k decrease. |
| **Bin too small (aliasing)** | `bin_ms · sfreq / 1000 < 1` → bin period < sample period. | Pre-check at top of op; raise `EasyBCIOperatorError(recoverable=True, fallback_step="mua_binning:10")`. |
| **Bin too large (collapse)** | All variance collapsed; downstream decoder accuracy drops. | If `bin_ms > 500` log warning (likely user error). |
| **Smoothing introduces latency** | Online-pipeline accuracy degrades because of look-ahead. | If `causal=False` and `analysis_goal == "online_inference"` log warning suggesting `causal=True`. |
| **Mismatched n_bins downstream** | Decoder expects 4096 bins, got 4095. | Last-bin under-sampling — document boundary; consider trimming last bin in downstream skill. |

Auto-detection helper:

```python
import numpy as np


def diagnose_binning(mua_train, bin_ms, sfreq):
    """Return (status, details)."""
    if mua_train.sum() == 0:
        return "empty", {"hint": "upstream threshold_spike produced no spikes"}
    if bin_ms * sfreq / 1000.0 < 1.0:
        return "aliased", {"bin_ms": bin_ms, "sfreq": sfreq}
    nonzero_bins = float((mua_train > 0).mean())
    if nonzero_bins < 0.05:
        return "sparse", {"nonzero_bin_fraction": nonzero_bins, "hint": "consider larger bin_ms"}
    return "ok", {"nonzero_bin_fraction": nonzero_bins, "mean_rate": float(mua_train.mean())}
```

## Common Issues

- **"My MUA train is all zeros."** Upstream `threshold_spike` produced no
  events. Check `meta["mean_firing_rate_hz"]` — if ≈ 0, decrease `k`
  (e.g., 4 → 3) or check `direction` for inverted polarity.
- **"Decoder accuracy is poor at small bins."** 1–10 ms bins are very
  sparse; the decoder needs averaging. Either (a) enable Gaussian smoothing,
  (b) go to 25 ms bins, or (c) feed a temporal model that accumulates
  variance internally.
- **"Online BCI is jittery."** You likely have `smooth=True` with the
  symmetric (default) kernel — that introduces a ½-bin lookahead.
  Set `causal=True` (one-sided kernel).
- **"My `mua_binning` is slower than `threshold_spike`."** Smoothing on
  Neuropixels (384 ch × 240k bins) via `scipy.ndimage.gaussian_filter1d`
  is heavier than the threshold itself. Disable smoothing for
  perf-critical online use.

## Reference Implementation

### Standalone (no EasyBCI dependencies)

```python
"""Standalone MUA spike-time binning — drop into any environment."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d


def mua_binning(
    spike_times: List[np.ndarray],
    sfreq: float,
    duration: float,
    *,
    bin_ms: float = 25.0,
    smooth: bool = True,
    output: str = "rate",
    causal: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Bin per-channel spike times into a (n_channels, n_bins) firing-rate array.

    Parameters
    ----------
    spike_times : list[ndarray]
        len == n_channels; each entry is integer sample indices.
    sfreq : float
        Sampling rate (Hz).
    duration : float
        Recording duration (s). Used to compute n_bins.
    bin_ms : float
        Bin width in milliseconds (default 25).
    smooth : bool
        Apply Gaussian kernel σ = bin_ms / 2 (default True).
    output : {"rate", "counts"}
        Spikes per second vs raw count.
    causal : bool
        If True, kernel is one-sided (online use; default False).

    Returns
    -------
    mua_train : ndarray, shape (n_channels, n_bins), float32
    bin_centers : ndarray, shape (n_bins,), float64 — bin midpoints in seconds.
    """
    if bin_ms * sfreq / 1000.0 < 1.0:
        raise ValueError(
            f"mua_binning: bin_ms={bin_ms} ms below sample period "
            f"(sfreq={sfreq} Hz → {1000.0 / sfreq:.3f} ms/sample)"
        )
    bin_s = bin_ms / 1000.0
    n_bins = max(1, int(np.floor(duration / bin_s)))
    n_ch = len(spike_times)
    edges = np.arange(n_bins + 1) * bin_s
    centers = (edges[:-1] + edges[1:]) / 2.0

    mua = np.zeros((n_ch, n_bins), dtype=np.float32)
    for c, idx in enumerate(spike_times):
        if len(idx) == 0:
            continue
        seconds = idx.astype(np.float64) / sfreq
        counts, _ = np.histogram(seconds, bins=edges)
        mua[c] = counts.astype(np.float32)

    if output == "rate":
        mua = mua / bin_s

    if smooth:
        sigma_bins = max(0.5, bin_ms / 2.0 / bin_ms)  # σ = 0.5 bins by construction
        if causal:
            # Half-Gaussian: pad-and-clip the lookahead half.
            kernel_width = max(3, int(round(3 * sigma_bins)))
            half = np.exp(-0.5 * (np.arange(kernel_width) / sigma_bins) ** 2)
            half = half / half.sum()
            for c in range(n_ch):
                mua[c] = np.convolve(mua[c], half[::-1], mode="same")
        else:
            mua = gaussian_filter1d(mua, sigma=sigma_bins, axis=1, mode="reflect")

    return mua, centers
```

### EasyBCI-Adapted (in-framework)

```python
from typing import Any, Dict, List
import time

import numpy as np

from easybci_lib.tools.neural_processing.operator_errors import EasyBCIOperatorError
from easybci_lib.tools.neural_processing.preprocess.step_cache import record_step_elapsed


def operator_mua_binning(
    data_dict: Dict[str, Any],
    *,
    bin_ms: float = 25.0,
    smooth: bool = True,
    output: str = "rate",
    causal: bool = False,
) -> Dict[str, Any]:
    """EasyBCI-adapted MUA binning.

    Parameters
    ----------
    data_dict : dict
        OperatorIO; requires ``meta["spike_times"]`` (output of `threshold_spike`).
    bin_ms : float
        Bin width in milliseconds (default 25.0).
    smooth : bool
        Apply Gaussian smoothing kernel (default True).
    output : {"rate", "counts"}
        Firing rate (spikes/s) vs raw count (default "rate").
    causal : bool
        Half-Gaussian (one-sided) kernel for online use (default False).

    Returns
    -------
    dict
        OperatorIO with continuous ``data`` unchanged plus:
        - ``meta["mua_train"]``: ndarray (n_channels, n_bins) float32
        - ``meta["bin_centers"]``: ndarray (n_bins,) float64 — seconds
        - ``meta["mua_binning"]``: parameter record

    Raises
    ------
    EasyBCIOperatorError
        ``recoverable=True`` (fallback to upstream threshold_spike) if
        spike_times missing; bin-aliasing parameter errors.

    Modality coverage
    -----------------
    spike: yes. EEG / MEG / sEEG / ECoG / fNIRS: forbidden (no spike_times).

    References
    ----------
    Cunningham & Yu 2014 (MUA decoding); Trautmann 2019.
    """
    spike_times: List[np.ndarray] = (data_dict.get("meta") or {}).get("spike_times")
    if spike_times is None:
        raise EasyBCIOperatorError(
            operator="mua_binning",
            reason="meta['spike_times'] missing — run threshold_spike upstream",
            recoverable=True,
            fallback_step="threshold_spike → mua_binning",
        )

    sfreq = float(data_dict["frequency"])
    duration = float(data_dict.get("duration") or data_dict["data"].shape[-1] / sfreq)
    if bin_ms * sfreq / 1000.0 < 1.0:
        raise EasyBCIOperatorError(
            operator="mua_binning",
            reason=f"bin_ms={bin_ms} below sample period",
            recoverable=True,
            fallback_step=f"mua_binning:{max(10, int(2000 / sfreq))}",
        )

    t0 = time.monotonic()
    bin_s = bin_ms / 1000.0
    n_bins = max(1, int(np.floor(duration / bin_s)))
    n_ch = len(spike_times)
    edges = np.arange(n_bins + 1) * bin_s
    centers = (edges[:-1] + edges[1:]) / 2.0

    mua = np.zeros((n_ch, n_bins), dtype=np.float32)
    for c, idx in enumerate(spike_times):
        if len(idx) == 0:
            continue
        seconds = idx.astype(np.float64) / sfreq
        counts, _ = np.histogram(seconds, bins=edges)
        mua[c] = counts.astype(np.float32)

    if output == "rate":
        mua = mua / bin_s

    if smooth:
        from scipy.ndimage import gaussian_filter1d  # local: scipy heavy
        sigma_bins = 0.5  # σ = bin/2 expressed in bin units
        if causal:
            kernel_width = max(3, int(round(3 * sigma_bins)))
            half = np.exp(-0.5 * (np.arange(kernel_width) / sigma_bins) ** 2)
            half = half / half.sum()
            for c in range(n_ch):
                mua[c] = np.convolve(mua[c], half[::-1], mode="same")
        else:
            mua = gaussian_filter1d(mua, sigma=sigma_bins, axis=1, mode="reflect")

    elapsed = time.monotonic() - t0
    out = dict(data_dict)
    out["data"] = data_dict["data"]  # unchanged
    out["elapsed_s"] = elapsed
    new_meta = dict(out.get("meta") or {})
    new_meta["mua_train"] = mua
    new_meta["bin_centers"] = centers
    new_meta["mua_binning"] = {
        "bin_ms": bin_ms, "smooth": smooth,
        "output": output, "causal": causal,
    }
    out["meta"] = new_meta
    record_step_elapsed(
        "mua_binning", elapsed,
        (data_dict.get("meta") or {}).get("step_cache_key"),
    )
    return out
```

## References

1. Cunningham, J. P., & Yu, B. M. (2014). *Dimensionality reduction for
   large-scale neural recordings*. Nature Neuroscience 17(11): 1500–1509.
   doi:10.1038/nn.3776 — population analysis pipelines and the role of
   bin size in MUA decoding.
2. Trautmann, E. M. et al. (2019). *Accurate Estimation of Neural Population
   Dynamics without Spike Sorting*. Neuron 103(2): 292–308.e4.
   doi:10.1016/j.neuron.2019.05.003 — 25 ms-bin MUA decoder reaches sorted
   parity in motor cortex.
3. Hochberg, L. R. et al. (2012). *Reach and grasp by people with
   tetraplegia using a neurally controlled robotic arm*. Nature 485: 372–375.
   doi:10.1038/nature11076 — BrainGate Utah-array decoder; 10 ms bins.
4. Pandarinath, C. et al. (2018). *Inferring single-trial neural population
   dynamics using sequential auto-encoders*. Nature Methods 15: 805–815.
   doi:10.1038/s41592-018-0109-9 — LFADS over binned spike counts; bin-size
   sensitivity analysis.
