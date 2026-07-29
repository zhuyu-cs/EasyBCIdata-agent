---
name: laplacian_ref
description: "Surface Laplacian (CSD) — focal source enhancement, complementary to CAR"
layer: L3
group: reference
metadata:
  tags: [operator, reference, laplacian, csd, surface, focal]
  modalities: [eeg, ecog]
  step_string: "laplacian_ref"
  analysis_goal_allowed: [classification, feature_extraction, clinical_screening, exploratory, generic, connectivity, phase_amplitude_coupling, online_inference]
  analysis_goal_forbidden: []
---
# Surface Laplacian (CSD)

## Function

The surface Laplacian (a.k.a. Current Source Density / CSD) is a
reference-free spatial filter that enhances focal cortical sources by
computing the spatial second derivative of the voltage map across the
scalp. Each channel becomes the difference between itself and a
weighted average of its nearest neighbours.

Input / Output: `(n_channels, n_times)` → `(n_channels, n_times)`.

## Algorithm & Math

For each electrode `c`:
```
out[c] = data[c] − mean(data[neighbours_of(c)])
```

Small Laplacian: 4 nearest electrodes (~5 cm). Large Laplacian: 8
neighbours (~10 cm). Alternative: spline-Laplacian (Perrin 1989) uses
spherical-spline interpolation of the full montage.

## Parameter Format & Defaults

`laplacian_ref:{knn}`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `knn` | int | 4 | Number of nearest neighbours. |
| `method` (kw) | str | "knn" | "knn" or "spline". |

## Modality-Specific Considerations

| Modality | knn | Notes |
|---|---|---|
| EEG (10-20, 32 ch) | 4 | Small Laplacian. |
| EEG (10-10, 64 ch) | 4–6 | More montage density. |
| EEG (high-density, 128+) | 8 | Larger neighbourhood viable. |
| ECoG | 4 | Small grid spacing. |
| sEEG (depth) | n/a | Use `bipolar_ref` instead. |

Hard exclusion: MEG / fNIRS (no electrode-position-based Laplacian);
spike (per-channel spike train; no spatial smoothing semantics).

## When to Use / NOT to Use

**Use** when: focal source enhancement (motor mu suppression, P300 over
Pz, gamma over visual cortex); complement to / replacement of CAR;
connectivity / PAC where you want a reference-free representation.

**Don't use** when: scalp position metadata absent; sparse montage (< 16
ch); depth electrodes (use bipolar).

## Constraints & Ordering

- Apply **after** bandpass / notch.
- Apply **before** decoder feature extraction.
- Requires electrode positions in `data_dict["meta"]["electrode_positions"]`
  or fall back to label-based 10-10 lookup.

## Failure Modes & Detection

| Failure | Symptom | Detection |
|---|---|---|
| Missing positions | KeyError. | Raise recoverable; suggest CAR. |
| Sparse montage | Laplacian dominated by 1–2 neighbours. | If `min_neighbours < knn` warn. |
| Edge channels | Boundary electrodes get one-sided average. | Note in meta. |

## Common Issues

- **"My alpha decreased after Laplacian."** Expected — Laplacian
  removes spatially diffuse signal (which alpha tends to be) and
  enhances focal sources.

## Reference Implementation

### Standalone

```python
from __future__ import annotations
import numpy as np


def laplacian_ref(
    data: np.ndarray, positions: np.ndarray, knn: int = 4,
) -> np.ndarray:
    """Spatial Laplacian via k-nearest neighbours.

    Parameters
    ----------
    data : ndarray (n_channels, n_times)
    positions : ndarray (n_channels, 3) — 3D electrode coords.
    """
    n_ch = data.shape[0]
    # Compute pairwise distances
    diff = positions[:, None, :] - positions[None, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dists, np.inf)
    knn_idx = np.argpartition(dists, knn, axis=-1)[:, :knn]
    out = np.empty_like(data)
    for c in range(n_ch):
        neighbour_mean = data[knn_idx[c]].mean(axis=0)
        out[c] = data[c] - neighbour_mean
    return out
```

### EasyBCI-Adapted

```python
from typing import Any, Dict
import time
import numpy as np
from easybci_lib.tools.neural_processing.operator_errors import EasyBCIOperatorError
from easybci_lib.tools.neural_processing.preprocess.step_cache import record_step_elapsed


def operator_laplacian_ref(
    data_dict: Dict[str, Any], *, knn: int = 4, method: str = "knn",
) -> Dict[str, Any]:
    """Surface Laplacian re-reference.

    Parameters
    ----------
    data_dict : dict
        OperatorIO; `meta["electrode_positions"]` (n_ch, 3) required.
    knn : int
        Nearest-neighbour count (default 4).
    method : str
        "knn" (default) or "spline".

    Returns
    -------
    dict
        OperatorIO with `data` Laplacian-filtered.

    Raises
    ------
    EasyBCIOperatorError
        recoverable=True if electrode positions missing.

    Modality coverage
    -----------------
    EEG / ECoG: yes. MEG / fNIRS / sEEG / spike: forbidden.

    References
    ----------
    Perrin et al. 1989; Nunez et al. 1994; Kayser & Tenke 2006.
    """
    positions = (data_dict.get("meta") or {}).get("electrode_positions")
    if positions is None:
        raise EasyBCIOperatorError(
            operator="laplacian_ref", reason="meta['electrode_positions'] required",
            recoverable=True, fallback_step="car:median",
        )
    positions = np.asarray(positions)
    if positions.shape[0] != data_dict["data"].shape[0]:
        raise EasyBCIOperatorError(
            operator="laplacian_ref",
            reason=f"positions n_ch={positions.shape[0]} != data n_ch={data_dict['data'].shape[0]}",
            recoverable=False,
        )

    t0 = time.monotonic()
    n_ch = data_dict["data"].shape[0]
    diff = positions[:, None, :] - positions[None, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dists, np.inf)
    knn_idx = np.argpartition(dists, knn, axis=-1)[:, :knn]
    new_data = np.empty_like(data_dict["data"])
    for c in range(n_ch):
        neighbour_mean = data_dict["data"][knn_idx[c]].mean(axis=0)
        new_data[c] = data_dict["data"][c] - neighbour_mean
    elapsed = time.monotonic() - t0

    out = dict(data_dict)
    out["data"] = new_data
    out["elapsed_s"] = elapsed
    out["meta"] = {**out.get("meta", {}), "laplacian_ref": {"knn": knn, "method": method}}
    record_step_elapsed("laplacian_ref", elapsed, (data_dict.get("meta") or {}).get("step_cache_key"))
    return out
```

## References

1. Perrin, F. et al. (1989). *Spherical splines for scalp potential and
   current density mapping*. EEG Clin. Neurophysiol. 72(2): 184–187.
   doi:10.1016/0013-4694(89)90180-6 — spline Laplacian.
2. Nunez, P. L. et al. (1994). *A theoretical and experimental study of
   high resolution EEG based on surface Laplacians and cortical imaging*.
   EEG Clin. Neurophysiol. 90(1): 40–57.
   doi:10.1016/0013-4694(94)90112-0.
3. Kayser, J., & Tenke, C. E. (2006). *Principal components analysis of
   Laplacian waveforms as a generic method for identifying ERP
   generator patterns*. Clinical Neurophysiology 117(2): 348–368.
   doi:10.1016/j.clinph.2005.08.034.
