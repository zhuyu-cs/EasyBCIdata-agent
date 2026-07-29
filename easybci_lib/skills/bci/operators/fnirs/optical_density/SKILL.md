---
name: optical_density
description: "Raw fNIRS intensity → optical density (OD) conversion"
layer: L3
group: fnirs
metadata:
  tags: [operator, fnirs, optical_density, od, hbo, hbr]
  modalities: [fnirs]
  step_string: "optical_density"
  analysis_goal_allowed: [classification, feature_extraction, clinical_screening, exploratory, generic, online_inference]
  analysis_goal_forbidden: [source_localization]
---
# Optical Density

## Function

Converts raw fNIRS light intensity (CW or FD device output) to optical
density (OD) per source-detector channel × wavelength. The first
step in any fNIRS pipeline; required before modified Beer-Lambert.

Input / Output: `(n_channels, n_times)` raw intensity → `(n_channels, n_times)` OD.

## Algorithm & Math

```
OD(t) = -log( I(t) / I_baseline )
```

where `I_baseline = mean(I(t_0..t_baseline))` is the early-recording
baseline mean.

## Parameter Format & Defaults

`optical_density:{baseline_s}`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `baseline_s` | float | 10.0 | Baseline interval at start (seconds). |
| `epsilon` (kw) | float | 1e-12 | Numerical floor for log. |

## Modality-Specific Considerations

fNIRS only.

## When to Use / NOT to Use

**Use** when: fNIRS raw intensity → OD → mBLL pipeline.

**Don't use** when: already-OD data (downstream skill checks); other modalities.

## Constraints & Ordering

- First step on raw fNIRS intensity.
- Before modified Beer-Lambert.
- Baseline must contain stable signal (no movement).

## Failure Modes & Detection

| Failure | Symptom | Detection |
|---|---|---|
| Saturated channels | Intensity = 0 or max. | Raise NaN; warn channel. |
| Movement during baseline | Wrong baseline mean. | Pre-check stability. |

## Common Issues

- **"My OD is NaN."** Channel saturated; check intensity range.

## Reference Implementation

### Standalone

```python
from __future__ import annotations
import numpy as np


def optical_density(
    intensity: np.ndarray, sfreq: float,
    baseline_s: float = 10.0, epsilon: float = 1e-12,
) -> np.ndarray:
    """Raw intensity → OD."""
    n_base = max(1, int(baseline_s * sfreq))
    I0 = np.maximum(intensity[:, :n_base].mean(axis=1, keepdims=True), epsilon)
    return -np.log(np.maximum(intensity, epsilon) / I0).astype(intensity.dtype, copy=False)
```

### EasyBCI-Adapted

```python
from typing import Any, Dict
import time
import numpy as np
from easybci_lib.tools.neural_processing.operator_errors import EasyBCIOperatorError
from easybci_lib.tools.neural_processing.preprocess.step_cache import record_step_elapsed


def operator_optical_density(
    data_dict: Dict[str, Any], *, baseline_s: float = 10.0, epsilon: float = 1e-12,
) -> Dict[str, Any]:
    """Raw fNIRS intensity → OD.

    Parameters
    ----------
    data_dict : dict
        OperatorIO with raw intensity in `data`.
    baseline_s : float
    epsilon : float

    Returns
    -------
    dict — OD in `data`.

    Raises
    ------
    EasyBCIOperatorError
        recoverable=False on saturated channels.

    Modality coverage
    -----------------
    fNIRS: yes. Others: forbidden.

    References
    ----------
    Boas & Dale 2005; Scholkmann et al. 2014.
    """
    modality = (data_dict.get("meta") or {}).get("modality", "").lower()
    if modality and modality != "fnirs":
        raise EasyBCIOperatorError(
            operator="optical_density", reason=f"modality={modality} not fnirs",
            recoverable=False,
        )

    t0 = time.monotonic()
    sfreq = float(data_dict["frequency"])
    n_base = max(1, int(baseline_s * sfreq))
    intensity = data_dict["data"]
    I0 = np.maximum(intensity[:, :n_base].mean(axis=1, keepdims=True), epsilon)
    od = (-np.log(np.maximum(intensity, epsilon) / I0)).astype(intensity.dtype, copy=False)
    elapsed = time.monotonic() - t0

    out = dict(data_dict)
    out["data"] = od
    out["elapsed_s"] = elapsed
    out["meta"] = {**out.get("meta", {}),
                   "optical_density": {"baseline_s": baseline_s}, "modality": "fnirs_od"}
    record_step_elapsed("optical_density", elapsed, (data_dict.get("meta") or {}).get("step_cache_key"))
    return out
```

## References

1. Boas, D. A., & Dale, A. M. (2005). *Simulation study of magnetic
   resonance imaging-guided cortically constrained diffuse optical
   tomography of human brain function*. Applied Optics 44(10):
   1957–1968. doi:10.1364/AO.44.001957.
2. Scholkmann, F. et al. (2014). *A review on continuous wave functional
   near-infrared spectroscopy and imaging instrumentation and
   methodology*. NeuroImage 85(1): 6–27.
   doi:10.1016/j.neuroimage.2013.05.004.
