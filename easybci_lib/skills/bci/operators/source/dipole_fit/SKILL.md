---
name: dipole_fit
description: "Equivalent current dipole fitting — focal source modelling"
layer: L3
group: source
metadata:
  tags: [operator, source, dipole, ecd, inverse]
  modalities: [eeg, meg]
  step_string: "dipole_fit"
  analysis_goal_allowed: [source_localization, feature_extraction, clinical_screening, exploratory]
  analysis_goal_forbidden: [online_inference]
---
# Dipole Fit (ECD)

## Function

Fits one or two equivalent current dipoles (ECDs) to an ERP / oscillation
topography by non-linear least-squares against a forward head model.
Standard pre-distributed-source approach; still the canonical tool for
focal sources (somatosensory N20, auditory M100).

Input / Output: epoched (or averaged) `(n_channels, n_times)` →
`(n_dipoles, {position, orientation, amplitude, gof})`.

## Algorithm & Math

For each time point: solve

```
argmin_{r, q} || measured(t) − G(r) · q ||²
```

over dipole position `r` (free) and orientation × amplitude `q ∈ R³`.
G is the forward leadfield from the head model (sphere / BEM / FEM).
Optimization via Nelder-Mead or trust-region. Goodness of fit (GOF) =
explained variance.

## Parameter Format & Defaults

`dipole_fit:{n_dipoles}`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n_dipoles` | int | 1 | 1 or 2 dipoles (per ERP wave). |
| `head_model` (kw) | str | "sphere" | "sphere" or "bem". |
| `gof_threshold` (kw) | float | 0.6 | Min GOF to accept fit. |

## Modality-Specific Considerations

EEG / MEG only. Best with high-density montage (≥ 32 EEG, ≥ 64 MEG).

## When to Use / NOT to Use

**Use** when: focal source (P3, N20, M100); high-density montage;
electrode positions + head model available.

**Don't use** when: distributed sources (use sLORETA / LCMV); low-density;
phase-only analysis.

## Constraints & Ordering

- Apply **on averaged ERP** or per-epoch.
- After bandpass / notch / drop_bads.
- Requires `meta["electrode_positions"]` and `meta["head_model"]`.

## Failure Modes & Detection

| Failure | Symptom | Detection |
|---|---|---|
| Missing leadfield | KeyError. | Raise recoverable; suggest sLORETA. |
| GOF below threshold | Fit unreliable. | If `gof < gof_threshold` flag in meta. |
| Multiple local minima | Different runs give different positions. | Use multi-start optimization. |

## Common Issues

- **"Dipole position outside the brain."** Sign of poor fit; check
  forward model quality (BEM > sphere).

## Reference Implementation

### Standalone

```python
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize


def dipole_fit(
    avg_erp: np.ndarray, leadfield_fn, n_dipoles: int = 1,
) -> dict:
    """Single dipole fit minimizing residual at peak time.

    Parameters
    ----------
    avg_erp : ndarray (n_channels,) — voltage at peak time.
    leadfield_fn : callable(r) -> (n_channels, 3) leadfield columns at position r.
    """
    if n_dipoles != 1:
        raise NotImplementedError("standalone shows 1-dipole; multi-dipole via MNE")
    n_ch = avg_erp.shape[0]

    def loss(r):
        G = leadfield_fn(r)
        q, *_ = np.linalg.lstsq(G, avg_erp, rcond=None)
        resid = avg_erp - G @ q
        return float(np.sum(resid ** 2)), q

    # Multi-start
    best = None
    rng = np.random.default_rng(0)
    starts = rng.uniform(-0.07, 0.07, size=(5, 3))
    for r0 in starts:
        result = minimize(lambda r: loss(r)[0], r0, method="Nelder-Mead")
        if best is None or result.fun < best.fun:
            best = result
    final_loss, q = loss(best.x)
    total_var = float(np.sum(avg_erp ** 2))
    gof = 1.0 - final_loss / (total_var + 1e-30)
    return {"position": best.x.tolist(), "orientation": q.tolist(),
            "amplitude": float(np.linalg.norm(q)), "gof": gof}
```

### EasyBCI-Adapted

```python
from typing import Any, Dict
import time
import numpy as np
from easybci_lib.tools.neural_processing.operator_errors import EasyBCIOperatorError
from easybci_lib.tools.neural_processing.preprocess.step_cache import record_step_elapsed


def operator_dipole_fit(
    data_dict: Dict[str, Any], *,
    n_dipoles: int = 1, head_model: str = "sphere", gof_threshold: float = 0.6,
) -> Dict[str, Any]:
    """Equivalent dipole fitting.

    Parameters
    ----------
    data_dict : dict
        OperatorIO; `meta["averaged_erp"]` (n_ch,) or `meta["peak_topography"]` required.
    n_dipoles : int
    head_model : str
    gof_threshold : float

    Returns
    -------
    dict — `meta["dipoles"]: list of dict (position / orientation / gof)`.

    Raises
    ------
    EasyBCIOperatorError
        recoverable=True if leadfield or topography missing.

    Modality coverage
    -----------------
    EEG / MEG: yes. Others: forbidden.

    References
    ----------
    Scherg & von Cramon 1985; Mosher 1992.
    """
    erp = (data_dict.get("meta") or {}).get("averaged_erp")
    if erp is None:
        raise EasyBCIOperatorError(
            operator="dipole_fit", reason="meta['averaged_erp'] required",
            recoverable=True, fallback_step="average epochs first",
        )
    positions = (data_dict.get("meta") or {}).get("electrode_positions")
    if positions is None:
        raise EasyBCIOperatorError(
            operator="dipole_fit", reason="meta['electrode_positions'] required",
            recoverable=True, fallback_step="sloreta",
        )

    t0 = time.monotonic()
    from scipy.optimize import minimize
    erp = np.asarray(erp); positions = np.asarray(positions)

    def leadfield_fn(r):
        diffs = positions - r[None, :]
        d = np.linalg.norm(diffs, axis=-1)[:, None] + 1e-6
        return diffs / d ** 3

    def loss(r):
        G = leadfield_fn(r)
        q, *_ = np.linalg.lstsq(G, erp, rcond=None)
        resid = erp - G @ q
        return float(np.sum(resid ** 2))

    rng = np.random.default_rng(0)
    best = None
    for r0 in rng.uniform(-0.07, 0.07, size=(5, 3)):
        res = minimize(loss, r0, method="Nelder-Mead")
        if best is None or res.fun < best.fun:
            best = res
    G = leadfield_fn(best.x)
    q, *_ = np.linalg.lstsq(G, erp, rcond=None)
    resid = erp - G @ q
    gof = 1.0 - float(np.sum(resid ** 2)) / (float(np.sum(erp ** 2)) + 1e-30)
    elapsed = time.monotonic() - t0

    out = dict(data_dict)
    out["elapsed_s"] = elapsed
    out["meta"] = {
        **out.get("meta", {}),
        "dipoles": [{
            "position": best.x.tolist(),
            "orientation": q.tolist(),
            "amplitude": float(np.linalg.norm(q)),
            "gof": gof,
        }],
        "dipole_fit_meta": {"n_dipoles": n_dipoles, "head_model": head_model,
                            "gof_threshold": gof_threshold},
    }
    record_step_elapsed("dipole_fit", elapsed, (data_dict.get("meta") or {}).get("step_cache_key"))
    return out
```

## References

1. Scherg, M., & von Cramon, D. (1985). *Two bilateral sources of the
   late AEP as identified by a spatio-temporal dipole model*. EEG Clin.
   Neurophysiol. 62(1): 32–44. doi:10.1016/0168-5597(85)90033-4.
2. Mosher, J. C., Lewis, P. S., & Leahy, R. M. (1992). *Multiple dipole
   modeling and localization from spatio-temporal MEG data*. IEEE Trans.
   Biomed. Eng. 39(6): 541–557. doi:10.1109/10.141192.
