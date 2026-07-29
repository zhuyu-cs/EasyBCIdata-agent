---
name: granger_causality
description: "Granger causality via multivariate autoregressive (MVAR) model — directed connectivity"
layer: L3
group: connectivity
metadata:
  tags: [operator, connectivity, granger, mvar, directed, causality]
  modalities: [eeg, meg, seeg, ecog, lfp]
  step_string: "granger"
  analysis_goal_allowed: [feature_extraction, exploratory, connectivity]
  analysis_goal_forbidden: [source_localization, online_inference]
---
# Granger Causality

## Function

Granger causality (GC) measures **directed** connectivity: signal X
"Granger-causes" Y if past X improves prediction of present Y beyond
past Y alone. Implemented here via MVAR (multivariate autoregressive)
modelling.

Input / Output: `(n_channels, n_times)` → `(n_channels, n_channels)`
(asymmetric — direction matters).

## Algorithm & Math

Fit a vector autoregressive model of order p:
```
x(t) = Σ_{k=1..p} A_k · x(t−k) + ε(t)
```

GC from j → i is the log-ratio of restricted (excluding j) to unrestricted
residual variances of channel i.

## Parameter Format & Defaults

`granger:{order}`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `order` | int | 10 | MVAR model order. |
| `regularize` (kw) | float | 0.0 | Ridge on regression. |

## Modality-Specific Considerations

| Modality | Order | Notes |
|---|---|---|
| EEG | 8–15 | Standard. |
| MEG | 10 | Same. |
| sEEG / ECoG | 8–10 | Localized signals; order matters less. |
| LFP | 5–10 | Slow oscillations. |

## When to Use / NOT to Use

**Use** when: directionality matters; few channels (< 64); reasonably
stationary signal segments.

**Don't use** when: > 64 channels (parameter count explodes); non-stationary;
online inference.

## Constraints & Ordering

- Apply on **stationary** segments (epoched).
- After bandpass / drop_bads.
- Channel count limited; consider PCA / region averaging first.

## Failure Modes & Detection

| Failure | Symptom | Detection |
|---|---|---|
| Order > samples / n_ch | Singular regression. | Pre-check; raise recoverable. |
| Non-stationary segment | Order-selection BIC unstable. | Check residual stationarity. |

## Common Issues

- **"GC values are tiny."** Common — GC's absolute scale is small; use
  values relative within session for interpretation.

## Reference Implementation

### Standalone

```python
from __future__ import annotations
import numpy as np


def granger_causality(
    data: np.ndarray, order: int = 10, regularize: float = 0.0,
) -> np.ndarray:
    """MVAR-based pairwise Granger causality."""
    n_ch, n_t = data.shape
    if n_ch > n_t / (order + 1):
        raise ValueError("granger: too few samples for n_ch and order")
    gc = np.zeros((n_ch, n_ch), dtype=np.float32)

    def _fit_ar(X, p):
        # X: (n_signals, n_t)
        n_s, n_t = X.shape
        # Build design matrix: rows = t > p, cols = stacked lagged X.
        rows = []
        targets = []
        for t in range(p, n_t):
            row = []
            for k in range(1, p + 1):
                row.append(X[:, t - k])
            rows.append(np.concatenate(row))
            targets.append(X[:, t])
        D = np.stack(rows)
        T = np.stack(targets)
        # OLS with optional ridge
        DtD = D.T @ D + regularize * np.eye(D.shape[1])
        beta = np.linalg.solve(DtD, D.T @ T)
        resid = T - D @ beta
        return resid

    # Unrestricted: full model
    resid_full = _fit_ar(data, order)
    var_full = np.var(resid_full, axis=0)
    # Restricted: drop channel j from regressors, refit per target i
    for j in range(n_ch):
        idx_others = [k for k in range(n_ch) if k != j]
        X_rest = data[idx_others]
        # Need to predict channel i using X_rest only
        for i in range(n_ch):
            if i == j: continue
            resid_rest = _fit_ar(np.vstack([data[i:i+1], X_rest]), order)
            var_rest = float(np.var(resid_rest[:, 0]))
            ratio = np.log((var_rest + 1e-30) / (var_full[i] + 1e-30))
            gc[j, i] = max(0.0, float(ratio))
    return gc
```

### EasyBCI-Adapted

```python
from typing import Any, Dict
import time
import numpy as np
from easybci_lib.tools.neural_processing.operator_errors import EasyBCIOperatorError
from easybci_lib.tools.neural_processing.preprocess.step_cache import record_step_elapsed


def operator_granger(
    data_dict: Dict[str, Any], *, order: int = 10, regularize: float = 0.0,
) -> Dict[str, Any]:
    """MVAR-based Granger causality matrix.

    Parameters
    ----------
    data_dict : dict
    order : int
    regularize : float

    Returns
    -------
    dict — `meta["granger"]: (n_ch, n_ch)` (j→i direction).

    Raises
    ------
    EasyBCIOperatorError
        recoverable=True for too-many channels relative to samples.

    Modality coverage
    -----------------
    EEG / MEG / sEEG / ECoG / LFP: yes.

    References
    ----------
    Granger 1969; Geweke 1982; Bressler & Seth 2011.
    """
    data = data_dict["data"]
    n_ch, n_t = data.shape
    if n_ch * (order + 1) > n_t:
        raise EasyBCIOperatorError(
            operator="granger",
            reason=f"n_ch * order > n_t; rank-deficient",
            recoverable=True, fallback_step=f"granger:{order // 2}",
        )

    t0 = time.monotonic()
    def _resid_var(X, p):
        n_s, n_t_ = X.shape
        rows, targets = [], []
        for t in range(p, n_t_):
            row = np.concatenate([X[:, t - k] for k in range(1, p + 1)])
            rows.append(row); targets.append(X[:, t])
        D = np.stack(rows); T = np.stack(targets)
        DtD = D.T @ D + regularize * np.eye(D.shape[1])
        beta = np.linalg.solve(DtD, D.T @ T)
        resid = T - D @ beta
        return np.var(resid, axis=0), resid

    var_full, _ = _resid_var(data, order)
    gc = np.zeros((n_ch, n_ch), dtype=np.float32)
    for j in range(n_ch):
        idx_others = [k for k in range(n_ch) if k != j]
        X_rest = data[idx_others]
        for i in range(n_ch):
            if i == j: continue
            resid_var_i, _ = _resid_var(np.vstack([data[i:i+1], X_rest]), order)
            ratio = np.log((float(resid_var_i[0]) + 1e-30) / (float(var_full[i]) + 1e-30))
            gc[j, i] = max(0.0, float(ratio))
    elapsed = time.monotonic() - t0

    out = dict(data_dict)
    out["elapsed_s"] = elapsed
    out["meta"] = {**out.get("meta", {}), "granger": gc, "granger_order": order}
    record_step_elapsed("granger", elapsed, (data_dict.get("meta") or {}).get("step_cache_key"))
    return out
```

## Related Skills

- See `bci/paradigms/analysis/connectivity/SKILL.md`.

## References

1. Granger, C. W. J. (1969). *Investigating causal relations by
   econometric models and cross-spectral methods*. Econometrica 37(3):
   424–438. doi:10.2307/1912791 — original Granger paper.
2. Geweke, J. (1982). *Measurement of linear dependence and feedback
   between multiple time series*. JASA 77(378): 304–313.
   doi:10.1080/01621459.1982.10477803 — MVAR-Granger formulation.
3. Bressler, S. L., & Seth, A. K. (2011). *Wiener-Granger causality: A
   well established methodology*. NeuroImage 58(2): 323–329.
   doi:10.1016/j.neuroimage.2010.02.059 — neuroscience usage guide.
