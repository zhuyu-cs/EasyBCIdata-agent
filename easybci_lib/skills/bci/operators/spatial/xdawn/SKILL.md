---
name: xdawn
description: "xDAWN supervised spatial filter — P300 / ERP decoding (Rivet 2009)"
layer: L3
group: spatial
metadata:
  tags: [operator, spatial, xdawn, p300, erp, supervised]
  modalities: [eeg, meg]
  step_string: "xdawn"
  analysis_goal_allowed: [classification, feature_extraction, exploratory, online_inference]
  analysis_goal_forbidden: [source_localization, connectivity, phase_amplitude_coupling]
---
# xDAWN Spatial Filter

## Function

Supervised spatial filter optimized for **ERP** detection (P300 oddball,
target-vs-nontarget). Maximizes the signal-to-signal-plus-noise ratio
where signal = trial-average ERP. The de-facto baseline for P300 BCI
speller decoders since Rivet 2009.

Input / Output: `(n_trials, n_channels, n_times)` + labels → `(n_trials, n_components, n_times)` filtered trials.

## Algorithm & Math

Solve the generalized eigenproblem `Σ_signal · w = λ · Σ_total · w`,
where `Σ_signal` is the average-ERP covariance and `Σ_total` the
single-trial covariance. Top `n_components` eigenvectors carry the
target ERP; project trials through them.

## Parameter Format & Defaults

`xdawn:{n_components}`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n_components` | int | 4 | Filters per class. |
| `reg` (kw) | str | None | "oas" / "ledoit_wolf" for small samples. |

## Modality-Specific Considerations

| Modality | n_components | Notes |
|---|---|---|
| EEG (P300, 32+ ch) | 3–5 | Standard P300 BCI speller. |
| EEG (low-density) | 2 | Overfit risk. |
| MEG | 5 | More channels available. |

Hard exclusion: requires labelled trials (supervised); no continuous data.

## When to Use / NOT to Use

**Use** when: P300 / target-detection paradigm; trial-averaged ERP is the
signal of interest.

**Don't use** when: oscillation-based decoding (use CSP); unlabelled data;
source localization (eigenvectors are not anatomically interpretable).

## Constraints & Ordering

- Apply **after** bandpass (0.1–30 Hz for P300).
- Apply **after** epoching with the ERP window aligned to stimulus.
- Train on training trials; freeze filters; apply to test.

## Failure Modes & Detection

| Failure | Symptom | Detection |
|---|---|---|
| Singular cov | LinAlgError. | Pre-check `n_trials >= 2·n_ch`; suggest reg. |
| Weak ERP (no target SNR) | Top eigenvalue ≈ 0. | If `λ_top < 0.1` warn — ERP may not exist or trials misaligned. |

## Common Issues

- **"Filters look like noise."** Average ERP too weak or trial count too
  low; need ≥ 50 target trials.

## Reference Implementation

### Standalone

```python
from __future__ import annotations
import os
import numpy as np
from scipy.linalg import eigh


def xdawn(
    X_trials: np.ndarray, y: np.ndarray, n_components: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """xDAWN spatial filter for ERP. Returns (filters, eigenvalues)."""
    target_class = np.unique(y)[-1]
    rng = np.random.default_rng(int(os.environ.get("EASYBCI_SEED", "0")))  # noqa: F841
    erp = np.mean(X_trials[y == target_class], axis=0)
    Sigma_signal = erp @ erp.T / (erp.shape[1] + 1e-30)
    Sigma_total = sum(t @ t.T for t in X_trials) / (X_trials.shape[0] * X_trials.shape[2] + 1e-30)
    eigvals, eigvecs = eigh(Sigma_signal, Sigma_total)
    order = np.argsort(eigvals)[::-1][:n_components]
    return eigvecs[:, order].T.astype(np.float32), eigvals[order]
```

### EasyBCI-Adapted

```python
from typing import Any, Dict
import os
import time
import numpy as np
from easybci_lib.tools.neural_processing.operator_errors import EasyBCIOperatorError
from easybci_lib.tools.neural_processing.preprocess.step_cache import record_step_elapsed


def operator_xdawn(
    data_dict: Dict[str, Any], *, n_components: int = 4,
) -> Dict[str, Any]:
    """xDAWN spatial filter for ERP decoding.

    Parameters
    ----------
    data_dict : dict
        OperatorIO with `meta["epochs"]` and `meta["labels"]`.
    n_components : int
        Filters to keep (default 4).

    Returns
    -------
    dict
        OperatorIO with `meta["xdawn_filters"]` and `meta["xdawn_filtered"]`.

    Raises
    ------
    EasyBCIOperatorError
        recoverable=True if missing labels / too few trials.

    Modality coverage
    -----------------
    EEG / MEG: yes (labelled-trials). Others: forbidden.

    References
    ----------
    Rivet et al. 2009; Congedo et al. 2017.
    """
    epochs = (data_dict.get("meta") or {}).get("epochs")
    labels = (data_dict.get("meta") or {}).get("labels")
    if epochs is None or labels is None:
        raise EasyBCIOperatorError(
            operator="xdawn", reason="epochs + labels required",
            recoverable=True, fallback_step="epoch then xdawn",
        )
    epochs = np.asarray(epochs)
    labels = np.asarray(labels)
    n_ch = epochs.shape[1]
    if epochs.shape[0] < 2 * n_ch:
        raise EasyBCIOperatorError(
            operator="xdawn", reason="insufficient trials",
            recoverable=True, fallback_step="xdawn with reg='oas'",
        )

    t0 = time.monotonic()
    rng = np.random.default_rng(int(os.environ.get("EASYBCI_SEED", "0")))  # noqa: F841
    from scipy.linalg import eigh
    target_class = np.unique(labels)[-1]
    erp = np.mean(epochs[labels == target_class], axis=0)
    Sigma_signal = erp @ erp.T / (erp.shape[1] + 1e-30)
    Sigma_total = sum(t @ t.T for t in epochs) / (
        epochs.shape[0] * epochs.shape[2] + 1e-30
    )
    eigvals, eigvecs = eigh(Sigma_signal, Sigma_total)
    order = np.argsort(eigvals)[::-1][:n_components]
    W = eigvecs[:, order].T.astype(np.float32)
    filtered = np.einsum("ck,tnk...->tc...", W, epochs.transpose(0, 2, 1))
    # Above transpose-and-einsum is a per-trial W @ trial; clearer form:
    filtered = np.stack([W @ trial for trial in epochs])
    elapsed = time.monotonic() - t0

    out = dict(data_dict)
    out["elapsed_s"] = elapsed
    out["meta"] = {
        **out.get("meta", {}),
        "xdawn_filters": W,
        "xdawn_filtered": filtered.astype(np.float32),
        "xdawn_eigenvalues": eigvals[order],
        "xdawn": {"n_components": n_components},
    }
    record_step_elapsed("xdawn", elapsed, (data_dict.get("meta") or {}).get("step_cache_key"))
    return out
```

## References

1. Rivet, B. et al. (2009). *xDAWN algorithm to enhance evoked potentials*.
   IEEE Trans. Biomed. Eng. 56(8): 2035–2043. doi:10.1109/TBME.2009.2012869.
2. Congedo, M. et al. (2017). *Riemannian geometry for EEG-based BCI: A
   primer*. Brain-Computer Interfaces 4(3): 155–174.
   doi:10.1080/2326263X.2017.1297192.
