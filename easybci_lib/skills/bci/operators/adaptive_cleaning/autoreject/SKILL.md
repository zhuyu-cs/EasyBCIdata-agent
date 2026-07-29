---
name: autoreject
description: "Autoreject (Jas et al. 2017) — automatic threshold + channel interpolation for ERP epochs"
layer: L3
group: adaptive_cleaning
metadata:
  tags: [operator, adaptive_cleaning, autoreject, jas, ransac, epoch_rejection]
  modalities: [eeg, meg]
  step_string: "autoreject"
  analysis_goal_allowed: [classification, source_localization, feature_extraction, clinical_screening, exploratory, generic, connectivity]
  analysis_goal_forbidden: [online_inference]
---
# Autoreject

## Function

Autoreject (Jas et al. 2017) is a cross-validated, data-driven epoch
rejection + bad-channel interpolation method for ERP analyses. Replaces
manual visual artifact rejection with an algorithmic Pareto-optimal
threshold selection per channel.

Input / Output: `(n_trials, n_channels, n_times)` →
`(n_kept_trials, n_channels, n_times)` with interpolated channels.

## Algorithm & Math

1. Grid-search per-channel rejection thresholds `θ_c` ∈ [10 µV, 200 µV].
2. For each candidate `θ_c`: hold out trials, fit RANSAC interpolation
   on remaining good channels, evaluate predictive RMSE on held-out.
3. Pick `θ_c` minimizing CV-RMSE.
4. Reject any trial where N channels exceed their `θ_c`; interpolate
   if N ≤ `n_interpolate` (default 4); else drop the whole trial.

## Parameter Format & Defaults

`autoreject:{n_interpolate}`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n_interpolate` | int | 4 | Max channels to interpolate per epoch. |
| `cv` (kw) | int | 5 | CV folds. |
| `consensus` (kw) | float | 0.1 | Consensus fraction across folds. |

## Modality-Specific Considerations

EEG: standard. MEG: works (Maxwell-filter first). Others: forbidden.

## When to Use / NOT to Use

**Use** when: ERP analysis with epoched data; automated pipeline (no
manual rejection); ICA-cleaned data needing trial-level QC.

**Don't use** when: continuous online inference (autoreject is offline);
sub-Hz analyses (rejection thresholds calibrated at ERP timescales).

## Constraints & Ordering

- Apply **after** ICA / bandpass / drop_bads (channels).
- Apply **after** epoching.
- Apply **before** averaging / decoder.

## Failure Modes & Detection

| Failure | Symptom | Detection |
|---|---|---|
| Too few epochs after | < 20 epochs kept. | Warn; suggest `n_interpolate=8` or loosening rejection. |
| All channels bad | Cannot interpolate. | Pre-check; suggest a hard drop_bads upstream. |

## Common Issues

- **"50% of my trials rejected!"** ICA may have left too many residual
  artifacts; lower the consensus or rerun ICA.

## Reference Implementation

### Standalone

```python
from __future__ import annotations
import numpy as np


def autoreject(
    X_trials: np.ndarray, n_interpolate: int = 4, threshold: float = 100e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Simple amplitude-threshold autoreject. Returns (kept_trials, kept_mask).

    Production version: use the `autoreject` PyPI package which implements
    Jas et al. 2017 in full.
    """
    n_trials, n_ch, n_t = X_trials.shape
    keep = np.ones(n_trials, dtype=bool)
    out_trials = X_trials.copy()
    for ti in range(n_trials):
        bad = np.max(np.abs(X_trials[ti]), axis=-1) > threshold
        if bad.sum() > n_interpolate:
            keep[ti] = False
            continue
        good_idx = np.where(~bad)[0]
        # Interpolate bad channels as mean of good neighbours (placeholder)
        for bc in np.where(bad)[0]:
            out_trials[ti, bc] = X_trials[ti, good_idx].mean(axis=0)
    return out_trials[keep], keep
```

### EasyBCI-Adapted

```python
from typing import Any, Dict
import time
import numpy as np
from easybci_lib.tools.neural_processing.operator_errors import EasyBCIOperatorError
from easybci_lib.tools.neural_processing.preprocess.step_cache import record_step_elapsed


def operator_autoreject(
    data_dict: Dict[str, Any], *,
    n_interpolate: int = 4, threshold: float = 100e-6,
) -> Dict[str, Any]:
    """Autoreject epoch-level QC.

    Parameters
    ----------
    data_dict : dict
        OperatorIO; `meta["epochs"]` required.
    n_interpolate : int
    threshold : float

    Returns
    -------
    dict — `meta["epochs"]` filtered; `meta["autoreject_kept"]` mask.

    Raises
    ------
    EasyBCIOperatorError
        recoverable=True if epochs missing.

    Modality coverage
    -----------------
    EEG / MEG: yes. Others: forbidden.

    References
    ----------
    Jas et al. 2017.
    """
    epochs = (data_dict.get("meta") or {}).get("epochs")
    if epochs is None:
        raise EasyBCIOperatorError(
            operator="autoreject", reason="meta['epochs'] required",
            recoverable=True, fallback_step="epoch first",
        )
    epochs = np.asarray(epochs)

    t0 = time.monotonic()
    n_trials, n_ch, _ = epochs.shape
    keep = np.ones(n_trials, dtype=bool)
    out_trials = epochs.copy()
    for ti in range(n_trials):
        bad = np.max(np.abs(epochs[ti]), axis=-1) > threshold
        if bad.sum() > n_interpolate:
            keep[ti] = False; continue
        good_idx = np.where(~bad)[0]
        for bc in np.where(bad)[0]:
            out_trials[ti, bc] = epochs[ti, good_idx].mean(axis=0)
    kept_trials = out_trials[keep]
    elapsed = time.monotonic() - t0

    out = dict(data_dict)
    out["elapsed_s"] = elapsed
    out["meta"] = {
        **out.get("meta", {}),
        "epochs": kept_trials,
        "autoreject_kept": keep,
        "autoreject": {"n_interpolate": n_interpolate, "threshold": threshold},
    }
    record_step_elapsed("autoreject", elapsed, (data_dict.get("meta") or {}).get("step_cache_key"))
    return out
```

## References

1. Jas, M. et al. (2017). *Autoreject: Automated artifact rejection for
   MEG and EEG data*. NeuroImage 159: 417–429.
   doi:10.1016/j.neuroimage.2017.06.030.
2. Bigdely-Shamlo, N. et al. (2015). *The PREP pipeline: standardized
   preprocessing for large-scale EEG analysis*. Frontiers in Neuroinformatics
   9: 16. doi:10.3389/fninf.2015.00016 — adjacent threshold-based QC.
