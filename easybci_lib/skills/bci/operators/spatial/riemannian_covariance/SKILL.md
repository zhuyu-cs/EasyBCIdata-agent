---
name: riemannian_covariance
description: "Riemannian-geometry covariance features (MDM-Riemannian) — robust EEG decoder family (Barachant 2012)"
layer: L3
group: spatial
metadata:
  tags: [operator, spatial, riemannian, covariance, mdm, decoder, sota]
  modalities: [eeg, meg, ecog]
  step_string: "riemannian_cov"
  analysis_goal_allowed: [classification, feature_extraction, exploratory, online_inference]
  analysis_goal_forbidden: [source_localization]
---
# Riemannian Covariance

## Function

Computes per-trial spatial covariance matrices and maps them to a tangent
space on the SPD (symmetric positive-definite) manifold. The tangent
vectors are the canonical input to the Minimum Distance to Mean (MDM)
classifier — one of the most robust BCI decoders, winning multiple
competition tracks since 2012.

Input / Output: `(n_trials, n_channels, n_times)` → `(n_trials, n_features)`.

## Algorithm & Math

1. Per-trial covariance `C_i = X_i · X_i^T / tr(...)`.
2. Reference mean `C_ref` (Riemannian mean of training set):
   ```
   C_ref = argmin_C Σ_i d_R(C, C_i)²
   ```
   where `d_R(A, B) = ||log(A^{-1/2} B A^{-1/2})||_F`.
3. Tangent space projection:
   ```
   tangent_i = upper_triangular(log(C_ref^{-1/2} C_i C_ref^{-1/2}))
   ```

`tangent_i ∈ R^(n_ch · (n_ch+1) / 2)` is the feature vector.

## Parameter Format & Defaults

`riemannian_cov:{metric}`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `metric` | str | "riemann" | "riemann" / "logeuclid" / "euclid". |
| `reg` (kw) | float | 1e-6 | Diagonal shrinkage on cov. |

## Modality-Specific Considerations

| Modality | Notes |
|---|---|
| EEG MI / P300 / SSVEP | Standard. |
| MEG | Same approach; longer feature vector (more sensors). |
| ECoG | Works well. |

## When to Use / NOT to Use

**Use** when: small training set (~30 trials); MI / SSVEP / ERP decoding;
need robustness to drift and noise.

**Don't use** when: source localization; phase-locking; > 256 channels
without dimensionality reduction (tangent vector quadratic in n_ch).

## Constraints & Ordering

- Apply **after** bandpass + epoching.
- Apply **before** classifier (linear / SVM consumes the tangent vector).

## Failure Modes & Detection

| Failure | Symptom | Detection |
|---|---|---|
| Singular per-trial cov | log fails. | Apply `reg`; if still fails, suggest more trial samples. |
| Reference mean diverges | Riemannian mean iteration > 100 iters. | Limit iterations to 50; fall back to Euclidean mean. |

## Common Issues

- **"Tangent vector huge — 1830 features for 60 channels."** Yes —
  `60 * 61 / 2 = 1830`. Use PCA before downstream classifier if needed.

## Reference Implementation

### Standalone

```python
from __future__ import annotations
import numpy as np
from scipy.linalg import logm, sqrtm


def _spd_log(C: np.ndarray) -> np.ndarray:
    return logm(C).real


def riemannian_cov(
    X_trials: np.ndarray, reg: float = 1e-6,
) -> np.ndarray:
    """Return tangent-space features (n_trials, n_ch*(n_ch+1)/2)."""
    n_trials, n_ch, _ = X_trials.shape
    covs = []
    for t in X_trials:
        C = t @ t.T / t.shape[-1] + reg * np.eye(n_ch)
        covs.append(C)
    covs = np.stack(covs)
    # Reference Riemannian mean via 5-iter geometric average
    ref = np.mean(covs, axis=0)
    for _ in range(5):
        ref_inv_sqrt = np.linalg.inv(sqrtm(ref).real)
        logs = np.stack([_spd_log(ref_inv_sqrt @ c @ ref_inv_sqrt) for c in covs])
        mean_log = np.mean(logs, axis=0)
        ref = sqrtm(ref).real @ _spd_log_inv(mean_log) @ sqrtm(ref).real
    # Tangent vectors
    ref_inv_sqrt = np.linalg.inv(sqrtm(ref).real)
    tangents = np.stack([_spd_log(ref_inv_sqrt @ c @ ref_inv_sqrt) for c in covs])
    idx = np.triu_indices(n_ch)
    return tangents[:, idx[0], idx[1]].astype(np.float32)


def _spd_log_inv(M):
    """Matrix exponential of symmetric M."""
    from scipy.linalg import expm
    return expm(M).real
```

### EasyBCI-Adapted

```python
from typing import Any, Dict
import time
import numpy as np
from easybci_lib.tools.neural_processing.operator_errors import EasyBCIOperatorError
from easybci_lib.tools.neural_processing.preprocess.step_cache import record_step_elapsed


def operator_riemannian_cov(
    data_dict: Dict[str, Any], *,
    metric: str = "riemann", reg: float = 1e-6,
) -> Dict[str, Any]:
    """Riemannian-geometry covariance features.

    Parameters
    ----------
    data_dict : dict
        OperatorIO with `meta["epochs"]`.
    metric : str
        "riemann" (default) / "logeuclid" / "euclid".
    reg : float
        Shrinkage on covariance (default 1e-6).

    Returns
    -------
    dict
        OperatorIO with `meta["riemannian_features"]`.

    Raises
    ------
    EasyBCIOperatorError
        recoverable=True if epochs missing.

    Modality coverage
    -----------------
    EEG / MEG / ECoG: yes. Others: forbidden.

    References
    ----------
    Barachant et al. 2012; Congedo et al. 2017.
    """
    epochs = (data_dict.get("meta") or {}).get("epochs")
    if epochs is None:
        raise EasyBCIOperatorError(
            operator="riemannian_cov", reason="meta['epochs'] required",
            recoverable=True, fallback_step="epoch then riemannian_cov",
        )

    t0 = time.monotonic()
    from scipy.linalg import logm, sqrtm, expm
    epochs = np.asarray(epochs)
    n_trials, n_ch, _ = epochs.shape
    covs = np.stack([
        (t @ t.T) / t.shape[-1] + reg * np.eye(n_ch) for t in epochs
    ])

    if metric == "euclid":
        ref = np.mean(covs, axis=0)
    elif metric == "logeuclid":
        log_covs = np.stack([logm(c).real for c in covs])
        ref = expm(np.mean(log_covs, axis=0)).real
    else:  # riemann
        ref = np.mean(covs, axis=0)
        for _ in range(5):
            ref_sqrt = sqrtm(ref).real
            ref_inv_sqrt = np.linalg.inv(ref_sqrt)
            logs = np.stack([logm(ref_inv_sqrt @ c @ ref_inv_sqrt).real for c in covs])
            mean_log = np.mean(logs, axis=0)
            ref = ref_sqrt @ expm(mean_log).real @ ref_sqrt

    ref_sqrt = sqrtm(ref).real
    ref_inv_sqrt = np.linalg.inv(ref_sqrt)
    tangents = np.stack([logm(ref_inv_sqrt @ c @ ref_inv_sqrt).real for c in covs])
    idx = np.triu_indices(n_ch)
    feats = tangents[:, idx[0], idx[1]].astype(np.float32)
    elapsed = time.monotonic() - t0

    out = dict(data_dict)
    out["elapsed_s"] = elapsed
    out["meta"] = {
        **out.get("meta", {}),
        "riemannian_features": feats,
        "riemannian_reference": ref.astype(np.float32),
        "riemannian_cov": {"metric": metric, "reg": reg},
    }
    record_step_elapsed("riemannian_cov", elapsed, (data_dict.get("meta") or {}).get("step_cache_key"))
    return out
```

## References

1. Barachant, A. et al. (2012). *Multiclass brain–computer interface
   classification by Riemannian geometry*. IEEE Trans. Biomed. Eng.
   59(4): 920–928. doi:10.1109/TBME.2011.2172210.
2. Congedo, M. et al. (2017). *Riemannian geometry for EEG-based BCI: A
   primer*. Brain-Computer Interfaces 4(3): 155–174.
   doi:10.1080/2326263X.2017.1297192.
3. Yger, F. et al. (2017). *Riemannian approaches in brain-computer
   interfaces: a review*. IEEE Trans. Neural Syst. Rehabil. Eng. 25(10):
   1753–1762. doi:10.1109/TNSRE.2016.2627016.
