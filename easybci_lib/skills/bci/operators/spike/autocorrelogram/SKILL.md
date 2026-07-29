---
name: autocorrelogram
description: "Spike autocorrelogram — per-unit ISI / refractory / bursting structure"
layer: L3
group: spike
metadata:
  tags: [operator, spike, autocorrelogram, isi, refractory, burst, qc]
  modalities: [spike]
  step_string: "autocorr"
  analysis_goal_allowed: [feature_extraction, exploratory, generic]
  analysis_goal_forbidden: [source_localization, online_inference]
---
# Spike Autocorrelogram

## Function

Computes the per-unit auto-correlogram (ACG) — histogram of inter-spike
time differences. Reveals refractory period, bursting, and cell-type
signatures (narrow vs broad).

Input / Output: `meta["units"]` → `meta["autocorr"]: (n_units, n_lag_bins)`.

## Algorithm & Math

For unit `u` with spike times `t_u`:
```
ACG_u(τ) = | { (i, j) : t_u[j] − t_u[i] = τ, i ≠ j } |
```

Symmetric around 0; typically display ±50 ms.

## Parameter Format & Defaults

`autocorr:{max_lag_ms},{bin_ms}`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `max_lag_ms` | float | 50.0 | Window half-width (ms). |
| `bin_ms` | float | 0.5 | Histogram bin width. |

## Modality-Specific Considerations

Sorted spike data only.

## When to Use / NOT to Use

**Use** when: QC of single units (refractory < 1 ms dip indicates good
isolation); cell-type classification; burst detection.

**Don't use** when: threshold MUA (no unit concept); online.

## Constraints & Ordering

After `spike_sort` or NWB load with `/units`. `max_lag_ms / bin_ms >= 20`.

## Failure Modes & Detection

| Failure | Symptom | Detection |
|---|---|---|
| Sparse unit | All-zero ACG. | If unit has < 10 spikes warn. |

## Common Issues

- **"No refractory dip at 1 ms."** Unit may be multi-unit contamination
  or noise; check via `spike_sort` quality metrics.

## Reference Implementation

### Standalone

```python
from __future__ import annotations
import numpy as np


def autocorrelogram(
    units: dict, max_lag_ms: float = 50.0, bin_ms: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """Per-unit autocorrelogram. Returns (acg, lag_centers)."""
    max_lag_s = max_lag_ms / 1000.0
    bin_s = bin_ms / 1000.0
    edges = np.arange(-max_lag_s, max_lag_s + bin_s, bin_s)
    centers = (edges[:-1] + edges[1:]) / 2
    acg = np.zeros((len(units), len(centers)), dtype=np.float32)
    for ui, (uid, times) in enumerate(units.items()):
        times = np.asarray(times, dtype=np.float64)
        for i, t in enumerate(times):
            in_window = times - t
            in_window = in_window[(in_window >= -max_lag_s) & (in_window <= max_lag_s) & (in_window != 0)]
            counts, _ = np.histogram(in_window, bins=edges)
            acg[ui] += counts
    return acg, centers
```

### EasyBCI-Adapted

```python
from typing import Any, Dict
import time
import numpy as np
from easybci_lib.tools.neural_processing.operator_errors import EasyBCIOperatorError
from easybci_lib.tools.neural_processing.preprocess.step_cache import record_step_elapsed


def operator_autocorr(
    data_dict: Dict[str, Any], *, max_lag_ms: float = 50.0, bin_ms: float = 0.5,
) -> Dict[str, Any]:
    """Per-unit autocorrelogram.

    Parameters
    ----------
    data_dict : dict
    max_lag_ms, bin_ms : float

    Returns
    -------
    dict — `meta["autocorr"]`, `meta["autocorr_centers"]`.

    Raises
    ------
    EasyBCIOperatorError
        recoverable=True if units missing.

    Modality coverage
    -----------------
    Spike (sorted): yes.

    References
    ----------
    Bartho et al. 2004; Buzsaki 2002.
    """
    units = (data_dict.get("meta") or {}).get("units")
    if not units:
        raise EasyBCIOperatorError(
            operator="autocorr", reason="meta['units'] missing",
            recoverable=True, fallback_step="spike_sort first",
        )

    t0 = time.monotonic()
    max_lag_s = max_lag_ms / 1000.0
    bin_s = bin_ms / 1000.0
    edges = np.arange(-max_lag_s, max_lag_s + bin_s, bin_s)
    centers = (edges[:-1] + edges[1:]) / 2
    acg = np.zeros((len(units), len(centers)), dtype=np.float32)
    for ui, (uid, times) in enumerate(units.items()):
        times = np.asarray(times, dtype=np.float64)
        for t in times:
            diff = times - t
            mask = (diff >= -max_lag_s) & (diff <= max_lag_s) & (diff != 0)
            counts, _ = np.histogram(diff[mask], bins=edges)
            acg[ui] += counts
    elapsed = time.monotonic() - t0

    out = dict(data_dict)
    out["elapsed_s"] = elapsed
    out["meta"] = {
        **out.get("meta", {}),
        "autocorr": acg, "autocorr_centers": centers,
        "autocorr_meta": {"max_lag_ms": max_lag_ms, "bin_ms": bin_ms},
    }
    record_step_elapsed("autocorr", elapsed, (data_dict.get("meta") or {}).get("step_cache_key"))
    return out
```

## References

1. Bartho, P. et al. (2004). *Characterization of neocortical principal
   cells and interneurons by network interactions and extracellular
   features*. J. Neurophysiol. 92(1): 600–608. doi:10.1152/jn.01170.2003.
2. Buzsaki, G. (2002). *Theta oscillations in the hippocampus*. Neuron
   33(3): 325–340. doi:10.1016/S0896-6273(02)00586-X.
