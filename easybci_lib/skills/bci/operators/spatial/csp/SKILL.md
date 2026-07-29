---
name: csp
description: "Common Spatial Patterns — supervised spatial filter for motor imagery / two-class EEG decoding"
layer: L3
group: spatial
metadata:
  tags: [operator, spatial, csp, motor_imagery, supervised, decoder]
  modalities: [eeg, meg, ecog]
  step_string: "csp"
  analysis_goal_allowed: [classification, feature_extraction, exploratory, online_inference]
  analysis_goal_forbidden: [source_localization, connectivity, phase_amplitude_coupling]
---
# Common Spatial Patterns (CSP)

## Function

A **supervised** spatial filter that learns linear combinations of channels
maximizing the variance ratio between two classes — the de-facto baseline
for two-class motor-imagery EEG decoding (Ramoser 2000; Blankertz 2008).

Input / Output: `(n_trials, n_channels, n_times)` + labels → `(n_trials, n_components)` log-variance features.

## Algorithm & Math

Given covariance matrices `Σ_1` and `Σ_2` (per class average of trial
covariances), solve the generalized eigenproblem:

```
Σ_1 · w = λ · (Σ_1 + Σ_2) · w
```

Eigenvectors `w_1 ... w_n_ch` with eigenvalues `λ_1 > ... > λ_n_ch`.
Keep top `n` + bottom `n` (the extreme eigenvectors maximize class
separability). Features per trial:

```
y_k(trial) = log( var( w_k^T · X(trial) ) )
```

## Parameter Format & Defaults

`csp:{n_components}`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n_components` | int | 4 | Top + bottom pairs to keep (4 → 8 features total). |
| `reg` (kw) | str/float | None | Regularization (e.g. "oas" for Oracle Approximating Shrinkage). |
| `log` (kw) | bool | True | Apply log to variance features. |
| `random_state` (kw) | int | from `EASYBCI_SEED` | RNG for cov-shuffling helpers. |

## Modality-Specific Considerations

| Modality | n_components | Notes |
|---|---|---|
| EEG (motor imagery, 64+ ch) | 4–6 | Standard 8–12 features. |
| EEG (low-density, ≤ 32 ch) | 2–3 | Higher n overfits. |
| MEG | 4 | Same as high-density EEG. |
| ECoG | 4–8 | More channels available. |

Hard exclusion: requires **labelled trials** (supervised). Cannot be
applied to unlabelled continuous data without a paradigm-defined event
table.

## When to Use / NOT to Use

**Use** when: 2-class classification (left vs right MI most common); ≥ 30
trials per class; ≥ 16 channels.

**Don't use** when: > 2 classes (use multi-class CSP variants);
unsupervised analysis; phase-locking required (variance loses phase).

## Constraints & Ordering

- Apply **after** bandpass to the discriminative band (8–30 Hz for MI).
- Apply **after** ICA / EOG-removal.
- Apply **after** epoching.
- Train on training trials; **freeze** spatial filters; project test trials.

## Failure Modes & Detection

| Failure | Symptom | Detection |
|---|---|---|
| Singular cov (n_ch > trials) | LinAlgError on eig. | Pre-check `n_trials >= 2 * n_ch`; raise recoverable; suggest `reg="oas"`. |
| Class imbalance (>10:1) | Worse-class eigenvectors poorly estimated. | Warn if `min(class_counts) / max < 0.3`. |
| One trivial class | All eigenvalues ≈ 0.5; no separability. | Warn if `λ_top - λ_bottom < 0.1`. |

## Common Issues

- **"CSP works on training but fails on test."** Spatial overfit; use
  `reg="oas"` or shrinkage cov estimator.
- **"My 4-class MI task — CSP gives weird results."** Standard CSP is
  binary; use one-vs-rest or pair-wise multi-class CSP.

## Reference Implementation

### Standalone

```python
from __future__ import annotations
import os
import numpy as np
from scipy.linalg import eigh


def csp(
    X_trials: np.ndarray, y: np.ndarray, n_components: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Common Spatial Patterns.

    Parameters
    ----------
    X_trials : ndarray (n_trials, n_channels, n_times)
    y : ndarray (n_trials,)  with two unique class labels.

    Returns
    -------
    W : ndarray (2 * n_components, n_channels) — spatial filters.
    eigenvalues : ndarray (2 * n_components,)
    """
    classes = np.unique(y)
    if classes.size != 2:
        raise ValueError(f"csp: expected 2 classes, got {classes}")

    rng = np.random.default_rng(int(os.environ.get("EASYBCI_SEED", "0")))  # noqa: F841 — for downstream

    def class_cov(X_class):
        # Average over trials of (X · X^T / trace).
        covs = []
        for trial in X_class:
            cov = trial @ trial.T
            cov /= np.trace(cov) + 1e-30
            covs.append(cov)
        return np.mean(covs, axis=0)

    Sigma1 = class_cov(X_trials[y == classes[0]])
    Sigma2 = class_cov(X_trials[y == classes[1]])
    eigvals, eigvecs = eigh(Sigma1, Sigma1 + Sigma2)
    # Pick top + bottom n_components
    order = np.concatenate([np.arange(n_components), np.arange(-n_components, 0)])
    W = eigvecs[:, order].T
    return W.astype(np.float32), eigvals[order]
```

### EasyBCI-Adapted

```python
from typing import Any, Dict
import os
import time
import numpy as np
from easybci_lib.tools.neural_processing.operator_errors import EasyBCIOperatorError
from easybci_lib.tools.neural_processing.preprocess.step_cache import record_step_elapsed


def operator_csp(
    data_dict: Dict[str, Any], *,
    n_components: int = 4, log: bool = True,
) -> Dict[str, Any]:
    """Common Spatial Patterns.

    Parameters
    ----------
    data_dict : dict
        OperatorIO with `meta["epochs"]: (n_trials, n_ch, n_t)` and
        `meta["labels"]: (n_trials,)`.
    n_components : int
        Top+bottom eigenvectors to keep (default 4 → 8 features).
    log : bool
        Apply log to variance features (default True).

    Returns
    -------
    dict
        OperatorIO with:
        - `meta["csp_filters"]`: (2*n_components, n_ch)
        - `meta["csp_features"]`: (n_trials, 2*n_components)

    Raises
    ------
    EasyBCIOperatorError
        recoverable=True if trial / class count insufficient.

    Modality coverage
    -----------------
    EEG / MEG / ECoG (multi-trial labelled): yes.
    sEEG / fNIRS / spike: forbidden.

    References
    ----------
    Ramoser 2000; Blankertz et al. 2008.
    """
    epochs = (data_dict.get("meta") or {}).get("epochs")
    labels = (data_dict.get("meta") or {}).get("labels")
    if epochs is None or labels is None:
        raise EasyBCIOperatorError(
            operator="csp", reason="meta['epochs'] / meta['labels'] required",
            recoverable=True, fallback_step="epoch first then csp",
        )
    epochs = np.asarray(epochs)
    labels = np.asarray(labels)
    classes = np.unique(labels)
    if classes.size != 2:
        raise EasyBCIOperatorError(
            operator="csp", reason=f"expected 2 classes, got {classes.tolist()}",
            recoverable=False,
        )
    n_ch = epochs.shape[1]
    if epochs.shape[0] < 2 * n_ch:
        raise EasyBCIOperatorError(
            operator="csp",
            reason=f"{epochs.shape[0]} trials < {2*n_ch}; cov rank-deficient",
            recoverable=True, fallback_step=f"csp with reg='oas'",
        )

    t0 = time.monotonic()
    rng = np.random.default_rng(int(os.environ.get("EASYBCI_SEED", "0")))  # noqa: F841
    from scipy.linalg import eigh
    def class_cov(Xc):
        covs = [(t @ t.T) / (np.trace(t @ t.T) + 1e-30) for t in Xc]
        return np.mean(covs, axis=0)
    S1 = class_cov(epochs[labels == classes[0]])
    S2 = class_cov(epochs[labels == classes[1]])
    eigvals, eigvecs = eigh(S1, S1 + S2)
    order = np.concatenate([np.arange(n_components), np.arange(-n_components, 0)])
    W = eigvecs[:, order].T.astype(np.float32)
    feats = np.zeros((epochs.shape[0], 2 * n_components), dtype=np.float32)
    for i, trial in enumerate(epochs):
        proj = W @ trial
        var = np.var(proj, axis=-1)
        feats[i] = np.log(var + 1e-30) if log else var
    elapsed = time.monotonic() - t0

    out = dict(data_dict)
    out["elapsed_s"] = elapsed
    out["meta"] = {
        **out.get("meta", {}),
        "csp_filters": W,
        "csp_features": feats,
        "csp_eigenvalues": eigvals[order],
        "csp": {"n_components": n_components, "log": log},
    }
    record_step_elapsed("csp", elapsed, (data_dict.get("meta") or {}).get("step_cache_key"))
    return out
```

## References

1. Ramoser, H., Müller-Gerking, J., & Pfurtscheller, G. (2000). *Optimal
   spatial filtering of single trial EEG during imagined hand movement*.
   IEEE Trans. Rehabil. Eng. 8(4): 441–446. doi:10.1109/86.895946 —
   the canonical CSP paper.
2. Blankertz, B. et al. (2008). *Optimizing spatial filters for robust
   EEG single-trial analysis*. IEEE Signal Processing Magazine 25(1):
   41–56. doi:10.1109/MSP.2008.4408441 — CSP regularization variants.
3. Lotte, F., & Guan, C. (2011). *Regularizing common spatial patterns
   to improve BCI designs*. IEEE Trans. Biomed. Eng. 58(2): 355–362.
   doi:10.1109/TBME.2010.2082539 — small-sample-friendly variants.
