---
name: cross_correlogram
description: "Cross-correlogram between unit pairs — synaptic coupling / co-firing structure"
layer: L3
group: spike
metadata:
  tags: [operator, spike, cross_correlogram, ccg, synaptic, coupling]
  modalities: [spike]
  step_string: "crosscorr"
  analysis_goal_allowed: [feature_extraction, exploratory, connectivity, generic]
  analysis_goal_forbidden: [source_localization, online_inference]
---
# Spike Cross-Correlogram

## Function

Pairwise cross-correlogram (CCG) between sorted units — reveals
monosynaptic peaks (0.5–3 ms lag), common input (0-lag peak), and
oscillatory co-firing.

Input / Output: `meta["units"]` → `meta["crosscorr"]: (n_units, n_units, n_lag_bins)`.

## Algorithm & Math

For unit-pair `(i, j)`:
```
CCG_{i,j}(τ) = | { (a, b) : t_j[b] − t_i[a] = τ } |
```

Negative τ: j fires before i. Positive: j fires after i.

## Parameter Format & Defaults

`crosscorr:{max_lag_ms},{bin_ms}`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `max_lag_ms` | float | 50.0 | Window half-width (ms). |
| `bin_ms` | float | 0.5 | Bin width. |

## Modality-Specific Considerations

Sorted spike data only.

## When to Use / NOT to Use

**Use** when: synaptic detection; co-firing networks; oscillatory coupling.

**Don't use** when: threshold MUA; online inference.

## Constraints & Ordering

After `spike_sort`. `n_units^2` cost — use only when n_units is reasonable.

## Failure Modes & Detection

| Failure | Symptom | Detection |
|---|---|---|
| n_units > 256 | Memory explosion. | Pre-check; raise recoverable. |
| Sparse units | All-zero CCG. | Warn for units with < 10 spikes. |

## Common Issues

- **"My CCG runtime exploded."** O(n_units² · n_spikes²); subset units
  (top firing) first.

## Reference Implementation

### Standalone

```python
from __future__ import annotations
import numpy as np


def cross_correlogram(
    units: dict, max_lag_ms: float = 50.0, bin_ms: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """Pairwise CCG. Returns (ccg, lag_centers)."""
    max_lag_s = max_lag_ms / 1000.0
    bin_s = bin_ms / 1000.0
    edges = np.arange(-max_lag_s, max_lag_s + bin_s, bin_s)
    centers = (edges[:-1] + edges[1:]) / 2
    n_u = len(units)
    ccg = np.zeros((n_u, n_u, len(centers)), dtype=np.float32)
    arr = {uid: np.asarray(t, dtype=np.float64) for uid, t in units.items()}
    keys = list(arr.keys())
    for i in range(n_u):
        for j in range(n_u):
            if i == j: continue
            ti = arr[keys[i]]; tj = arr[keys[j]]
            for a in ti:
                diff = tj - a
                mask = (diff >= -max_lag_s) & (diff <= max_lag_s)
                counts, _ = np.histogram(diff[mask], bins=edges)
                ccg[i, j] += counts
    return ccg, centers
```

### EasyBCI-Adapted

```python
from typing import Any, Dict
import time
import numpy as np
from easybci_lib.tools.neural_processing.operator_errors import EasyBCIOperatorError
from easybci_lib.tools.neural_processing.preprocess.step_cache import record_step_elapsed


def operator_crosscorr(
    data_dict: Dict[str, Any], *, max_lag_ms: float = 50.0, bin_ms: float = 0.5,
) -> Dict[str, Any]:
    """Pairwise cross-correlogram.

    Parameters
    ----------
    data_dict : dict
    max_lag_ms, bin_ms : float

    Returns
    -------
    dict — `meta["crosscorr"]: (n_u, n_u, n_lag)`.

    Raises
    ------
    EasyBCIOperatorError
        recoverable=True if units missing or > 256.

    Modality coverage
    -----------------
    Spike (sorted): yes.

    References
    ----------
    Perkel et al. 1967.
    """
    units = (data_dict.get("meta") or {}).get("units")
    if not units:
        raise EasyBCIOperatorError(
            operator="crosscorr", reason="meta['units'] missing",
            recoverable=True, fallback_step="spike_sort first",
        )
    if len(units) > 256:
        raise EasyBCIOperatorError(
            operator="crosscorr", reason=f"{len(units)} units > 256 — too expensive",
            recoverable=True, fallback_step="subset to top-N firing units",
        )

    t0 = time.monotonic()
    max_lag_s = max_lag_ms / 1000.0
    bin_s = bin_ms / 1000.0
    edges = np.arange(-max_lag_s, max_lag_s + bin_s, bin_s)
    centers = (edges[:-1] + edges[1:]) / 2
    n_u = len(units)
    ccg = np.zeros((n_u, n_u, len(centers)), dtype=np.float32)
    arr = {uid: np.asarray(t, dtype=np.float64) for uid, t in units.items()}
    keys = list(arr.keys())
    for i in range(n_u):
        for j in range(n_u):
            if i == j: continue
            ti = arr[keys[i]]; tj = arr[keys[j]]
            for a in ti:
                diff = tj - a
                mask = (diff >= -max_lag_s) & (diff <= max_lag_s)
                counts, _ = np.histogram(diff[mask], bins=edges)
                ccg[i, j] += counts
    elapsed = time.monotonic() - t0

    out = dict(data_dict)
    out["elapsed_s"] = elapsed
    out["meta"] = {
        **out.get("meta", {}),
        "crosscorr": ccg, "crosscorr_centers": centers,
        "crosscorr_meta": {"max_lag_ms": max_lag_ms, "bin_ms": bin_ms},
    }
    record_step_elapsed("crosscorr", elapsed, (data_dict.get("meta") or {}).get("step_cache_key"))
    return out
```

## References

1. Perkel, D. H. et al. (1967). *Neuronal spike trains and stochastic
   point processes: I. The single spike train*. Biophysical Journal
   7(4): 391–418. doi:10.1016/S0006-3495(67)86596-2.
2. English, D. F. et al. (2017). *Pyramidal cell-interneuron circuit
   architecture and dynamics in hippocampal networks*. Neuron 96(2):
   505–520.e7. doi:10.1016/j.neuron.2017.09.033.
