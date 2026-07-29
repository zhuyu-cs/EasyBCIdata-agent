---
name: short_separation_regression
description: "fNIRS short-separation channel regression — remove systemic (extracerebral) artifact"
layer: L3
group: fnirs
metadata:
  tags: [operator, fnirs, short_separation, ssr, systemic, gagnon]
  modalities: [fnirs]
  step_string: "short_sep_regression"
  analysis_goal_allowed: [classification, source_localization, feature_extraction, clinical_screening, exploratory, generic, connectivity, online_inference]
  analysis_goal_forbidden: []
---
# Short-Separation Regression

## Function

Regresses out **systemic (extracerebral) artifact** from fNIRS long-separation
channels using paired short-separation (< 1.5 cm) channels. The
short-separation signal is dominated by scalp + skull blood flow; the
long-separation signal contains cortical + scalp. Subtracting (regressed)
short from long isolates the cortical response.

Input / Output: `(n_channels_HbO + n_channels_HbR, n_times)` post-mBLL →
same shape with systemic component removed.

## Algorithm & Math

For long-channel `c_long` with paired short-channel `c_short`:
```
β_c = argmin || c_long - β · c_short ||²
out_c = c_long - β_c · c_short
```

Per HbO and HbR separately.

## Parameter Format & Defaults

`short_sep_regression:{short_indices}`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `short_indices` | list[int] | required | Indices of short-separation channels. |

## Modality-Specific Considerations

fNIRS HbO/HbR only.

## When to Use / NOT to Use

**Use** when: short-separation channels present (recommended in any
fNIRS rig); reducing systemic noise.

**Don't use** when: no short-separation channels; long channels alone
(no regressor).

## Constraints & Ordering

- Apply **after** `modified_beer_lambert`.
- Apply **before** bandpass to HRF band.

## Failure Modes & Detection

| Failure | Symptom | Detection |
|---|---|---|
| No short channels | KeyError. | Raise recoverable. |
| Saturated short | Inflated β; over-correction. | Pre-check σ of short channels. |

## Common Issues

- **"My HbO response is now negative."** Over-correction; short was
  dominated by something other than systemic (movement). Check QC.

## Reference Implementation

### Standalone

```python
from __future__ import annotations
import numpy as np


def short_separation_regression(
    data: np.ndarray, short_indices: list[int],
) -> np.ndarray:
    """Regress short-separation channels out of long channels."""
    n_ch = data.shape[0]
    short_signals = data[short_indices]                       # (n_short, n_t)
    out = data.copy()
    for c in range(n_ch):
        if c in short_indices: continue
        long_sig = data[c]
        # Solve β = (X^T X)^{-1} X^T y for each short channel
        X = short_signals.T                                    # (n_t, n_short)
        beta, *_ = np.linalg.lstsq(X, long_sig, rcond=None)
        out[c] = long_sig - X @ beta
    return out
```

### EasyBCI-Adapted

```python
from typing import Any, Dict, List
import time
import numpy as np
from easybci_lib.tools.neural_processing.operator_errors import EasyBCIOperatorError
from easybci_lib.tools.neural_processing.preprocess.step_cache import record_step_elapsed


def operator_short_sep_regression(
    data_dict: Dict[str, Any], *, short_indices: List[int],
) -> Dict[str, Any]:
    """Short-separation regression.

    Parameters
    ----------
    data_dict : dict
    short_indices : list[int]
        Channel indices of short-separation channels (already mBLL-processed).

    Returns
    -------
    dict — long-channel systemic component removed.

    Raises
    ------
    EasyBCIOperatorError
        recoverable=True if short_indices empty.

    Modality coverage
    -----------------
    fNIRS HbO/HbR: yes. Others: forbidden.

    References
    ----------
    Gagnon et al. 2012; Scholkmann et al. 2014.
    """
    if not short_indices:
        raise EasyBCIOperatorError(
            operator="short_sep_regression", reason="short_indices required",
            recoverable=True, fallback_step="skip if no short channels",
        )

    t0 = time.monotonic()
    data = data_dict["data"]
    short_signals = data[short_indices]
    new_data = data.copy()
    n_ch = data.shape[0]
    X = short_signals.T
    for c in range(n_ch):
        if c in short_indices: continue
        long_sig = data[c]
        beta, *_ = np.linalg.lstsq(X, long_sig, rcond=None)
        new_data[c] = long_sig - X @ beta
    new_data = new_data.astype(data.dtype, copy=False)
    elapsed = time.monotonic() - t0

    out = dict(data_dict)
    out["data"] = new_data
    out["elapsed_s"] = elapsed
    out["meta"] = {**out.get("meta", {}),
                   "short_sep_regression": {"short_indices": list(short_indices)}}
    record_step_elapsed("short_sep_regression", elapsed, (data_dict.get("meta") or {}).get("step_cache_key"))
    return out
```

## References

1. Gagnon, L. et al. (2012). *Short separation channel location impacts
   the performance of short channel regression in NIRS*. NeuroImage
   59(3): 2518–2528. doi:10.1016/j.neuroimage.2011.08.095.
2. Scholkmann, F. et al. (2014). *A review on continuous wave functional
   near-infrared spectroscopy*. NeuroImage 85: 6–27.
   doi:10.1016/j.neuroimage.2013.05.004.
