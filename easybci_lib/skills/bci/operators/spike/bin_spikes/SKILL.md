---
name: bin_spikes
description: "Bin sorted-unit spike times into discrete count windows (per-unit, fixed grid)"
layer: L3
group: spike
metadata:
  tags: [operator, spike, binning, sorted_unit, decoder_input]
  modalities: [spike]
  step_string: "bin_spikes"
  analysis_goal_allowed: [classification, feature_extraction, exploratory, generic, online_inference]
  analysis_goal_forbidden: [source_localization]
---
# Bin Spikes (Unit-Level)

## Function

Per-unit histogram binning of sorted spike times. Sister to
`mua_binning` (which is per-channel MUA from threshold output).

Input / Output: `meta["units"]: dict[id, times_s]` → `meta["binned_units"]: (n_units, n_bins)`.

## Algorithm & Math

For unit `u`, bin size `Δ`:
```
count_u[i] = |{ t ∈ t_u : i·Δ ≤ t < (i+1)·Δ }|
```

Output unit selectable: "counts" (raw integer) or "rate" (counts/Δ).

## Parameter Format & Defaults

`bin_spikes:{bin_ms},{output}`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `bin_ms` | float | 25.0 | Bin width (ms). |
| `output` | str | "rate" | "rate" / "counts". |
| `smooth` (kw) | bool | False | Optional Gaussian smoothing post-bin. |

## Modality-Specific Considerations

Sorted spike data only.

## When to Use / NOT to Use

**Use** when: decoder expects per-unit binned input; PSTH; LFADS.

**Don't use** when: threshold MUA (use `mua_binning`); rate KDE (use `firing_rate`).

## Constraints & Ordering

- Apply after `spike_sort` or NWB-load with `/units`.
- `bin_ms · sfreq / 1000 >= 1` to avoid aliasing.

## Failure Modes & Detection

| Failure | Symptom | Detection |
|---|---|---|
| Empty units | Output all zero. | Raise recoverable. |
| Bin < sample period | Aliased. | Pre-check; raise recoverable. |

## Common Issues

- **"Difference vs `firing_rate`?"** `bin_spikes` is hard binning (sharp
  edges); `firing_rate` is KDE (smooth). Use bin_spikes for decoders, KDE
  for visualization.

## Reference Implementation

### Standalone

```python
from __future__ import annotations
import numpy as np


def bin_spikes(
    units: dict, duration: float, bin_ms: float = 25.0, output: str = "rate",
) -> tuple[np.ndarray, np.ndarray]:
    """Per-unit binning. Returns (binned, bin_centers)."""
    bin_s = bin_ms / 1000.0
    n_bins = max(1, int(np.floor(duration / bin_s)))
    edges = np.arange(n_bins + 1) * bin_s
    centers = (edges[:-1] + edges[1:]) / 2
    binned = np.zeros((len(units), n_bins), dtype=np.float32)
    for ui, (uid, times) in enumerate(units.items()):
        times = np.asarray(times, dtype=np.float64)
        counts, _ = np.histogram(times, bins=edges)
        binned[ui] = counts
    if output == "rate":
        binned = binned / bin_s
    return binned, centers
```

### EasyBCI-Adapted

```python
from typing import Any, Dict
import time
import numpy as np
from easybci_lib.tools.neural_processing.operator_errors import EasyBCIOperatorError
from easybci_lib.tools.neural_processing.preprocess.step_cache import record_step_elapsed


def operator_bin_spikes(
    data_dict: Dict[str, Any], *, bin_ms: float = 25.0, output: str = "rate",
) -> Dict[str, Any]:
    """Per-unit spike binning.

    Parameters
    ----------
    data_dict : dict
        OperatorIO; `meta["units"]` required.
    bin_ms : float
    output : str

    Returns
    -------
    dict — `meta["binned_units"]: (n_units, n_bins)`.

    Raises
    ------
    EasyBCIOperatorError
        recoverable=True if units missing.

    Modality coverage
    -----------------
    Spike (sorted): yes. Others: forbidden.

    References
    ----------
    Cunningham & Yu 2014; Pandarinath 2018 (LFADS).
    """
    units = (data_dict.get("meta") or {}).get("units")
    if not units:
        raise EasyBCIOperatorError(
            operator="bin_spikes", reason="meta['units'] missing",
            recoverable=True, fallback_step="spike_sort first",
        )

    t0 = time.monotonic()
    duration = float(data_dict.get("duration") or 0.0)
    bin_s = bin_ms / 1000.0
    sfreq = float(data_dict["frequency"])
    if bin_s * sfreq < 1.0:
        raise EasyBCIOperatorError(
            operator="bin_spikes", reason=f"bin_ms={bin_ms} below sample period",
            recoverable=True, fallback_step=f"bin_spikes:{1000/sfreq*2:.1f}",
        )
    n_bins = max(1, int(np.floor(duration / bin_s)))
    edges = np.arange(n_bins + 1) * bin_s
    centers = (edges[:-1] + edges[1:]) / 2
    binned = np.zeros((len(units), n_bins), dtype=np.float32)
    for ui, (uid, times) in enumerate(units.items()):
        times = np.asarray(times, dtype=np.float64)
        counts, _ = np.histogram(times, bins=edges)
        binned[ui] = counts
    if output == "rate":
        binned = binned / bin_s
    elapsed = time.monotonic() - t0

    out = dict(data_dict)
    out["elapsed_s"] = elapsed
    out["meta"] = {
        **out.get("meta", {}),
        "binned_units": binned,
        "binned_units_centers": centers,
        "bin_spikes": {"bin_ms": bin_ms, "output": output},
    }
    record_step_elapsed("bin_spikes", elapsed, (data_dict.get("meta") or {}).get("step_cache_key"))
    return out
```

## References

1. Cunningham, J. P., & Yu, B. M. (2014). *Dimensionality reduction for
   large-scale neural recordings*. Nature Neuroscience 17(11): 1500–1509.
   doi:10.1038/nn.3776.
2. Pandarinath, C. et al. (2018). *Inferring single-trial neural population
   dynamics using sequential auto-encoders*. Nature Methods 15: 805–815.
   doi:10.1038/s41592-018-0109-9.
