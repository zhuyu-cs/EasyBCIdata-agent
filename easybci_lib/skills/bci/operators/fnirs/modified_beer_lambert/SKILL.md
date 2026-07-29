---
name: modified_beer_lambert
description: "Modified Beer–Lambert law — OD → HbO / HbR concentration"
layer: L3
group: fnirs
metadata:
  tags: [operator, fnirs, mbll, beer_lambert, hbo, hbr, hemoglobin]
  modalities: [fnirs]
  step_string: "mBLL"
  analysis_goal_allowed: [classification, source_localization, feature_extraction, clinical_screening, exploratory, generic, connectivity, online_inference]
  analysis_goal_forbidden: []
---
# Modified Beer-Lambert Law (mBLL)

## Function

Converts dual-wavelength optical density to **oxy-hemoglobin (HbO)** and
**deoxy-hemoglobin (HbR)** concentration change per channel. The
canonical second step in fNIRS preprocessing.

Input / Output: `(2 · n_channels, n_times)` OD (one row per λ per
channel-pair) → `(2 · n_channels, n_times)` (HbO + HbR per channel).

## Algorithm & Math

For wavelengths `λ1, λ2`:
```
[Δ[HbO]]    1                 [ε_HbR(λ2)  −ε_HbR(λ1)]   [OD(λ1) / (L · DPF(λ1))]
[Δ[HbR]] = ─────────────────── [−ε_HbO(λ2)  ε_HbO(λ1)] · [OD(λ2) / (L · DPF(λ2))]
            det(ε)
```

Where `L = source-detector distance`, `DPF = differential pathlength factor`.

## Parameter Format & Defaults

`mBLL:{distance_cm},{dpf_730},{dpf_850}`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `distance_cm` | float | 3.0 | Source-detector distance (cm). |
| `dpf_lo` (kw) | float | 6.0 | DPF for shorter wavelength. |
| `dpf_hi` (kw) | float | 5.0 | DPF for longer wavelength. |

## Modality-Specific Considerations

fNIRS only (CW dual-wavelength setups).

## When to Use / NOT to Use

**Use** when: standard fNIRS pipeline; downstream task expects HbO / HbR.

**Don't use** when: triple-wavelength (use multi-wavelength mBLL); FD-fNIRS
(different physics).

## Constraints & Ordering

- Apply after `optical_density`.
- Before `bandpass:0.01,0.5` for HRF-band isolation.
- DPF must match probe geometry / age / wavelength.

## Failure Modes & Detection

| Failure | Symptom | Detection |
|---|---|---|
| Wrong wavelength pairing | HbO / HbR swapped. | Sanity: HbO peak > HbR peak during task. |
| Wrong DPF | Concentration scale off. | Compare against literature. |

## Common Issues

- **"HbO and HbR look the same sign."** Wrong DPF or channel order; check
  `data` row layout (must be λ1 followed by λ2 per channel pair).

## Reference Implementation

### Standalone

```python
from __future__ import annotations
import numpy as np


# Extinction coefficients at 730 nm and 850 nm (Cope 1991)
EPSILON = {
    730: {"hbo": 0.428, "hbr": 1.102},
    850: {"hbo": 1.058, "hbr": 0.691},
}


def mbll(
    od_pairs: np.ndarray, distance_cm: float = 3.0,
    wavelengths: tuple[int, int] = (730, 850),
    dpf: tuple[float, float] = (6.0, 5.0),
) -> np.ndarray:
    """OD pairs → HbO/HbR.

    Parameters
    ----------
    od_pairs : (2 * n_channels, n_times) — interleaved λ1, λ2 per channel.

    Returns
    -------
    (2 * n_channels, n_times) — interleaved HbO, HbR per channel.
    """
    n_pairs = od_pairs.shape[0] // 2
    e1_hbo, e1_hbr = EPSILON[wavelengths[0]]["hbo"], EPSILON[wavelengths[0]]["hbr"]
    e2_hbo, e2_hbr = EPSILON[wavelengths[1]]["hbo"], EPSILON[wavelengths[1]]["hbr"]
    det_e = e1_hbo * e2_hbr - e1_hbr * e2_hbo
    inv = np.array([[e2_hbr, -e1_hbr], [-e2_hbo, e1_hbo]]) / det_e
    out = np.zeros_like(od_pairs)
    for i in range(n_pairs):
        od_lambda1 = od_pairs[2*i] / (distance_cm * dpf[0])
        od_lambda2 = od_pairs[2*i + 1] / (distance_cm * dpf[1])
        out[2*i] = inv[0, 0] * od_lambda1 + inv[0, 1] * od_lambda2       # HbO
        out[2*i + 1] = inv[1, 0] * od_lambda1 + inv[1, 1] * od_lambda2   # HbR
    return out
```

### EasyBCI-Adapted

```python
from typing import Any, Dict
import time
import numpy as np
from easybci_lib.tools.neural_processing.operator_errors import EasyBCIOperatorError
from easybci_lib.tools.neural_processing.preprocess.step_cache import record_step_elapsed


_EPSILON = {
    730: {"hbo": 0.428, "hbr": 1.102},
    850: {"hbo": 1.058, "hbr": 0.691},
}


def operator_modified_beer_lambert(
    data_dict: Dict[str, Any], *,
    distance_cm: float = 3.0,
    wavelengths: tuple = (730, 850), dpf: tuple = (6.0, 5.0),
) -> Dict[str, Any]:
    """OD → HbO / HbR via mBLL.

    Parameters
    ----------
    data_dict : dict
    distance_cm : float
    wavelengths, dpf : tuple

    Returns
    -------
    dict — `data` is HbO/HbR interleaved.

    Raises
    ------
    EasyBCIOperatorError
        recoverable=False if wavelength unknown.

    Modality coverage
    -----------------
    fNIRS: yes. Others: forbidden.

    References
    ----------
    Cope 1991; Delpy 1988.
    """
    modality = (data_dict.get("meta") or {}).get("modality", "").lower()
    if modality and not modality.startswith("fnirs"):
        raise EasyBCIOperatorError(
            operator="modified_beer_lambert", reason=f"modality={modality} not fnirs",
            recoverable=False,
        )
    for w in wavelengths:
        if w not in _EPSILON:
            raise EasyBCIOperatorError(
                operator="modified_beer_lambert",
                reason=f"wavelength {w} nm not in EPSILON {list(_EPSILON.keys())}",
                recoverable=False,
            )

    t0 = time.monotonic()
    od_pairs = data_dict["data"]
    n_pairs = od_pairs.shape[0] // 2
    e1_hbo, e1_hbr = _EPSILON[wavelengths[0]]["hbo"], _EPSILON[wavelengths[0]]["hbr"]
    e2_hbo, e2_hbr = _EPSILON[wavelengths[1]]["hbo"], _EPSILON[wavelengths[1]]["hbr"]
    det_e = e1_hbo * e2_hbr - e1_hbr * e2_hbo
    inv = np.array([[e2_hbr, -e1_hbr], [-e2_hbo, e1_hbo]]) / det_e
    new_data = np.zeros_like(od_pairs)
    for i in range(n_pairs):
        od_l1 = od_pairs[2*i] / (distance_cm * dpf[0])
        od_l2 = od_pairs[2*i + 1] / (distance_cm * dpf[1])
        new_data[2*i] = inv[0, 0] * od_l1 + inv[0, 1] * od_l2
        new_data[2*i + 1] = inv[1, 0] * od_l1 + inv[1, 1] * od_l2
    elapsed = time.monotonic() - t0

    out = dict(data_dict)
    out["data"] = new_data
    out["elapsed_s"] = elapsed
    out["meta"] = {**out.get("meta", {}),
                   "modified_beer_lambert": {
                       "distance_cm": distance_cm, "wavelengths": list(wavelengths), "dpf": list(dpf),
                   },
                   "modality": "fnirs_hbo_hbr"}
    record_step_elapsed("modified_beer_lambert", elapsed, (data_dict.get("meta") or {}).get("step_cache_key"))
    return out
```

## References

1. Cope, M., & Delpy, D. T. (1988). *System for long-term measurement of
   cerebral blood and tissue oxygenation on newborn infants by near
   infra-red transillumination*. Medical & Biological Engineering &
   Computing 26(3): 289–294. doi:10.1007/BF02447083.
2. Scholkmann, F. et al. (2014). *A review on continuous wave functional
   near-infrared spectroscopy*. NeuroImage 85: 6–27.
   doi:10.1016/j.neuroimage.2013.05.004.
