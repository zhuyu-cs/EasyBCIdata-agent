---
name: rest_reference
description: "REST (Reference Electrode Standardization Technique, Yao 2001) — infinity-reference EEG"
layer: L3
group: reference
metadata:
  tags: [operator, reference, rest, yao, infinity_reference, source_model]
  modalities: [eeg]
  step_string: "rest_reference"
  analysis_goal_allowed: [classification, source_localization, feature_extraction, exploratory, generic, connectivity, phase_amplitude_coupling]
  analysis_goal_forbidden: []
---
# REST — Reference Electrode Standardization Technique

## Function

REST (Yao 2001) re-references EEG to a virtual point at infinity by
inverting a forward head model. This is the only mathematically
reference-free EEG montage and is the recommended choice for
**connectivity / PAC / source-localization** analyses where common-mode
references (CAR, mastoid) bias the result.

Input / Output: `(n_channels, n_times)` → `(n_channels, n_times)`.

## Algorithm & Math

Given the forward leadfield `G` (from a sphere or BEM head model) of
shape `(n_channels, n_sources)`:

```
G_avg = G − mean(G, axis=0)            # average-referenced leadfield
out = data + (G · G_avg^+) · data       # add back the spatial average bias
```

Where `^+` denotes pseudo-inverse. Intuitively: subtract the projection
of `data` onto the average-reference subspace of the leadfield.

## Parameter Format & Defaults

`rest_reference:{model}`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | str | "sphere" | "sphere" (3-shell sphere) / "bem" (boundary-element). |
| `lambda_reg` (kw) | float | 0.05 | Tikhonov regularization. |

## Modality-Specific Considerations

EEG only. MEG has no reference problem; sEEG / ECoG use bipolar.

## When to Use / NOT to Use

**Use** when: connectivity / PAC / source-localization analyses; you
want a reference-free montage with theoretical justification.

**Don't use** when: dense electrode geometry unknown (sphere fallback OK
but loses fidelity); MEG / sEEG / ECoG.

## Constraints & Ordering

- Apply **after** drop_bads (REST is rank-sensitive).
- Apply **before** epoching / decoder.
- Electrode positions in `meta["electrode_positions"]` required.

## Failure Modes & Detection

| Failure | Symptom | Detection |
|---|---|---|
| Missing positions | KeyError. | Pre-check; suggest CAR fallback. |
| Singular leadfield | LinAlgError. | Increase `lambda_reg`. |

## Common Issues

- **"My PSD shape changed substantially."** Expected — REST preserves the
  absolute scale of the source field, unlike CAR.

## Reference Implementation

### Standalone

```python
from __future__ import annotations
import numpy as np


def _sphere_leadfield(positions: np.ndarray, n_sources: int = 200) -> np.ndarray:
    """3-shell sphere forward model leadfield (simplified)."""
    # Assume positions on unit sphere; n_sources at random interior points.
    rng = np.random.default_rng(0)
    sources = rng.uniform(-0.7, 0.7, size=(n_sources, 3))
    sources /= np.linalg.norm(sources, axis=1, keepdims=True) + 1e-30
    sources *= 0.8  # interior radius
    # Leadfield ~ 1 / |r - r_source|² (very simplified)
    n_ch = positions.shape[0]
    G = np.zeros((n_ch, n_sources))
    for c in range(n_ch):
        diffs = positions[c, None, :] - sources
        d = np.linalg.norm(diffs, axis=-1) + 1e-6
        G[c] = 1.0 / d ** 2
    return G


def rest_reference(
    data: np.ndarray, positions: np.ndarray, lambda_reg: float = 0.05
) -> np.ndarray:
    """REST infinity reference."""
    G = _sphere_leadfield(positions)
    G_avg = G - G.mean(axis=0, keepdims=True)
    # Pseudo-inverse with Tikhonov
    GtG = G_avg.T @ G_avg + lambda_reg * np.eye(G_avg.shape[1])
    G_pinv = np.linalg.solve(GtG, G_avg.T)
    correction = G @ G_pinv
    return (data + correction @ data).astype(data.dtype, copy=False)
```

### EasyBCI-Adapted

```python
from typing import Any, Dict
import time
import numpy as np
from easybci_lib.tools.neural_processing.operator_errors import EasyBCIOperatorError
from easybci_lib.tools.neural_processing.preprocess.step_cache import record_step_elapsed


def operator_rest_reference(
    data_dict: Dict[str, Any], *,
    model: str = "sphere", lambda_reg: float = 0.05,
) -> Dict[str, Any]:
    """REST infinity reference.

    Parameters
    ----------
    data_dict : dict
        OperatorIO; `meta["electrode_positions"]` required.
    model : str
        "sphere" or "bem" (default "sphere").
    lambda_reg : float
        Tikhonov regularization (default 0.05).

    Returns
    -------
    dict
        OperatorIO with REST-referenced data.

    Raises
    ------
    EasyBCIOperatorError
        recoverable=True if positions missing.

    Modality coverage
    -----------------
    EEG: yes. Others: forbidden.

    References
    ----------
    Yao 2001; Liu et al. 2017.
    """
    positions = (data_dict.get("meta") or {}).get("electrode_positions")
    if positions is None:
        raise EasyBCIOperatorError(
            operator="rest_reference", reason="meta['electrode_positions'] required",
            recoverable=True, fallback_step="car:median",
        )
    positions = np.asarray(positions)

    t0 = time.monotonic()
    rng = np.random.default_rng(0)
    n_sources = 200
    sources = rng.uniform(-0.7, 0.7, size=(n_sources, 3))
    sources /= np.linalg.norm(sources, axis=1, keepdims=True) + 1e-30
    sources *= 0.8
    n_ch = positions.shape[0]
    G = np.zeros((n_ch, n_sources))
    for c in range(n_ch):
        diffs = positions[c, None, :] - sources
        d = np.linalg.norm(diffs, axis=-1) + 1e-6
        G[c] = 1.0 / d ** 2
    G_avg = G - G.mean(axis=0, keepdims=True)
    GtG = G_avg.T @ G_avg + lambda_reg * np.eye(G_avg.shape[1])
    G_pinv = np.linalg.solve(GtG, G_avg.T)
    correction = G @ G_pinv
    new_data = (data_dict["data"] + correction @ data_dict["data"]).astype(
        data_dict["data"].dtype, copy=False
    )
    elapsed = time.monotonic() - t0

    out = dict(data_dict)
    out["data"] = new_data
    out["elapsed_s"] = elapsed
    out["meta"] = {**out.get("meta", {}), "rest_reference": {"model": model, "lambda": lambda_reg}}
    record_step_elapsed("rest_reference", elapsed, (data_dict.get("meta") or {}).get("step_cache_key"))
    return out
```

## References

1. Yao, D. (2001). *A method to standardize a reference of scalp EEG
   recordings to a point at infinity*. Physiological Measurement 22(4):
   693–711. doi:10.1088/0967-3334/22/4/305 — the original REST paper.
2. Liu, Q. et al. (2017). *A novel reference-free method for analysis of
   EEG: REST*. NeuroImage 161: 219–230. doi:10.1016/j.neuroimage.2017.08.034.
3. Yao, D. et al. (2019). *Which reference should we use for EEG and ERP
   practice?* Brain Topography 32(4): 530–549.
   doi:10.1007/s10548-019-00707-x — recommendations.
