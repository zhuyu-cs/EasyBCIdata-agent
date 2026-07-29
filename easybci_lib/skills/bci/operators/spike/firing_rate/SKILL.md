---
name: firing_rate
description: "Instantaneous firing rate from sorted spike times — Gaussian-kernel smoothed"
layer: L3
group: spike
metadata:
  tags: [operator, spike, firing_rate, sorted_unit, kde, gaussian]
  modalities: [spike]
  step_string: "firing_rate"
  analysis_goal_allowed: [classification, feature_extraction, exploratory, generic, online_inference]
  analysis_goal_forbidden: [source_localization]
---
# Firing Rate

## Function

Estimates instantaneous firing rate (Hz) from **sorted** spike-unit
times via Gaussian-kernel density estimation. Equivalent of `mua_binning`
but operates on unit-level (post-spike-sort) data, producing
`(n_units, n_bins)` rate matrix.

Input / Output: requires `meta["units"]: dict[unit_id, spike_times_s]`
→ `meta["firing_rate"]: (n_units, n_bins)`.

## Algorithm & Math

For unit `u` with spike times `t_u = [t_0, ..., t_N]`:
```
rate_u(t) = Σ_i (1 / (σ √(2π))) · exp(- (t − t_i)² / (2σ²))
```

Equivalent to histogram-binning + Gaussian smoothing.

## Parameter Format & Defaults

`firing_rate:{sigma_ms},{bin_ms}`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `sigma_ms` | float | 50.0 | Gaussian σ in milliseconds. |
| `bin_ms` | float | 10.0 | Output bin resolution. |
| `causal` (kw) | bool | False | One-sided kernel for online use. |

## Modality-Specific Considerations

Sorted spike data only. Use `mua_binning` for threshold MUA (unit-less).

## When to Use / NOT to Use

**Use** when: sorted-unit per-neuron rate features; PSTH; trial-averaging
per neuron.

**Don't use** when: MUA threshold (use `mua_binning`); broadband LFP
analysis (use `multitaper_psd`); online inference (smoothing latency).

## Constraints & Ordering

- Apply after `spike_sort` (or load NWB with `/units` already present).
- `bin_ms <= sigma_ms / 2` for adequate sampling.

## Failure Modes & Detection

| Failure | Symptom | Detection |
|---|---|---|
| Empty units | `meta["units"]` is empty. | Raise recoverable. |
| Bin > sigma | Aliasing; output flat. | Pre-check; warn. |

## Common Issues

- **"My rate has gaps."** Sparse-firing unit; widen σ.

## Reference Implementation

### Standalone

```python
from __future__ import annotations
import numpy as np


def firing_rate(
    units: dict, duration: float, sigma_ms: float = 50.0, bin_ms: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-unit Gaussian-smoothed firing rate. Returns (rate, bin_centers)."""
    bin_s = bin_ms / 1000.0
    sigma_s = sigma_ms / 1000.0
    n_bins = int(np.ceil(duration / bin_s))
    bin_centers = (np.arange(n_bins) + 0.5) * bin_s
    rate = np.zeros((len(units), n_bins), dtype=np.float32)
    for ui, (uid, times) in enumerate(units.items()):
        times = np.asarray(times, dtype=np.float64)
        for t in times:
            kernel = np.exp(-((bin_centers - t) ** 2) / (2 * sigma_s ** 2))
            kernel /= sigma_s * np.sqrt(2 * np.pi)
            rate[ui] += kernel
    return rate, bin_centers
```

### EasyBCI-Adapted

```python
from typing import Any, Dict
import time
import numpy as np
from easybci_lib.tools.neural_processing.operator_errors import EasyBCIOperatorError
from easybci_lib.tools.neural_processing.preprocess.step_cache import record_step_elapsed


def operator_firing_rate(
    data_dict: Dict[str, Any], *,
    sigma_ms: float = 50.0, bin_ms: float = 10.0, causal: bool = False,
) -> Dict[str, Any]:
    """Gaussian-smoothed firing rate per sorted unit.

    Parameters
    ----------
    data_dict : dict
        OperatorIO; `meta["units"]: dict[int, ndarray]` required.
    sigma_ms : float
    bin_ms : float
    causal : bool

    Returns
    -------
    dict — `meta["firing_rate"]: (n_units, n_bins)`.

    Raises
    ------
    EasyBCIOperatorError
        recoverable=True if units missing.

    Modality coverage
    -----------------
    Spike: yes. Others: forbidden.

    References
    ----------
    Cunningham & Yu 2014.
    """
    units = (data_dict.get("meta") or {}).get("units")
    if not units:
        raise EasyBCIOperatorError(
            operator="firing_rate", reason="meta['units'] empty / missing",
            recoverable=True, fallback_step="spike_sort then firing_rate",
        )

    t0 = time.monotonic()
    duration = float(data_dict.get("duration") or 0.0)
    bin_s = bin_ms / 1000.0
    sigma_s = sigma_ms / 1000.0
    n_bins = max(1, int(np.ceil(duration / bin_s)))
    bin_centers = (np.arange(n_bins) + 0.5) * bin_s
    rate = np.zeros((len(units), n_bins), dtype=np.float32)
    for ui, (uid, times) in enumerate(units.items()):
        times = np.asarray(times, dtype=np.float64)
        for t in times:
            kernel = np.exp(-((bin_centers - t) ** 2) / (2 * sigma_s ** 2))
            if causal:
                kernel[bin_centers > t] = 0
            kernel /= sigma_s * np.sqrt(2 * np.pi)
            rate[ui] += kernel
    elapsed = time.monotonic() - t0

    out = dict(data_dict)
    out["elapsed_s"] = elapsed
    out["meta"] = {
        **out.get("meta", {}),
        "firing_rate": rate,
        "firing_rate_bin_centers": bin_centers,
        "firing_rate_meta": {"sigma_ms": sigma_ms, "bin_ms": bin_ms, "causal": causal},
    }
    record_step_elapsed("firing_rate", elapsed, (data_dict.get("meta") or {}).get("step_cache_key"))
    return out
```

## References

1. Cunningham, J. P., & Yu, B. M. (2014). *Dimensionality reduction for
   large-scale neural recordings*. Nature Neuroscience 17(11): 1500–1509.
   doi:10.1038/nn.3776.
2. Shimazaki, H., & Shinomoto, S. (2010). *Kernel bandwidth optimization
   in spike rate estimation*. J. Comput. Neurosci. 29: 171–182.
   doi:10.1007/s10827-009-0180-4.
