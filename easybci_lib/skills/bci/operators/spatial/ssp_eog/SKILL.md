---
name: ssp_eog
description: "Signal Space Projection for EOG (Uusitalo & Ilmoniemi 1997) — MEG gold standard for ocular artifact removal"
layer: L3
group: spatial
metadata:
  tags: [operator, spatial, ssp, eog, meg, mne, ocular]
  modalities: [eeg, meg]
  step_string: "ssp_eog"
  analysis_goal_allowed: [classification, source_localization, feature_extraction, clinical_screening, exploratory, generic, connectivity, online_inference]
  analysis_goal_forbidden: []
---
# SSP — EOG Removal

## Function

Signal Space Projection (SSP) computes a projection matrix from EOG
events that removes the ocular artifact subspace from the data. Faster
and simpler than ICA, and the MEG gold standard for routine ocular
artifact removal.

Input / Output: `(n_channels, n_times)` → same shape, ocular subspace projected out.

## Algorithm & Math

1. Find blink / saccade onsets from an EOG channel (peak detection).
2. Average epochs around onsets → topography of the ocular artifact.
3. PCA / SVD on the average → top `n_proj` eigenvectors form the
   "ocular subspace".
4. Apply orthogonal projector `P = I − UU^T` to all data (where `U` is
   the top eigenvectors).

## Parameter Format & Defaults

`ssp_eog:{n_proj},{eog_ch}`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n_proj` | int | 2 | Projection components (1–2 typically). |
| `eog_ch` | str | "EOG" | EOG channel name (substring match). |
| `tmin`, `tmax` (kw) | float, float | -0.2, 0.4 | Epoch around blink. |

## Modality-Specific Considerations

| Modality | n_proj | Notes |
|---|---|---|
| MEG | 2 | Gold-standard ocular removal; ICA optional. |
| EEG | 1–2 | ICA usually preferred; SSP for online use. |

## When to Use / NOT to Use

**Use** when: dedicated EOG channels present; online BCI (cheap); MEG
preprocessing.

**Don't use** when: no EOG channel; PAC analysis (SSP can strip oscillation
content if too aggressive); blinks rare (fewer than 20 events).

## Constraints & Ordering

- Apply **after** notch / bandpass.
- Apply **before** ICA / epoching.
- EOG channel must be present in `data_dict["channels"]`.

## Failure Modes & Detection

| Failure | Symptom | Detection |
|---|---|---|
| EOG channel missing | KeyError. | Pre-check; raise recoverable, suggest `regression_eog`. |
| Too few blinks | Topography noisy; projector unreliable. | If detected events < 20 warn. |
| n_proj too high | Strips brain signal. | Compare pre/post PSD; if > 30% drop in alpha/beta warn. |

## Common Issues

- **"My alpha disappeared."** n_proj too high; reduce to 1.

## Reference Implementation

### Standalone

```python
from __future__ import annotations
import numpy as np
from scipy.signal import find_peaks


def ssp_eog(
    data: np.ndarray, sfreq: float, eog_idx: int, n_proj: int = 2,
    tmin: float = -0.2, tmax: float = 0.4,
) -> np.ndarray:
    """SSP projection from EOG events."""
    eog = data[eog_idx]
    threshold = 4 * np.median(np.abs(eog)) / 0.6745
    peaks, _ = find_peaks(np.abs(eog), height=threshold, distance=int(0.5 * sfreq))
    if len(peaks) < 20:
        raise ValueError(f"ssp_eog: only {len(peaks)} blinks; need >= 20")
    n_pre = int(-tmin * sfreq)
    n_post = int(tmax * sfreq)
    n_t = n_pre + n_post
    avg = np.zeros((data.shape[0], n_t))
    cnt = 0
    for p in peaks:
        if p - n_pre < 0 or p + n_post >= data.shape[1]:
            continue
        avg += data[:, p - n_pre : p + n_post]
        cnt += 1
    avg /= max(cnt, 1)
    U, _, _ = np.linalg.svd(avg, full_matrices=False)
    U = U[:, :n_proj]
    projector = np.eye(data.shape[0]) - U @ U.T
    return (projector @ data).astype(data.dtype, copy=False)
```

### EasyBCI-Adapted

```python
from typing import Any, Dict
import time
import numpy as np
from easybci_lib.tools.neural_processing.operator_errors import EasyBCIOperatorError
from easybci_lib.tools.neural_processing.preprocess.step_cache import record_step_elapsed


def operator_ssp_eog(
    data_dict: Dict[str, Any], *,
    n_proj: int = 2, eog_ch: str = "EOG",
    tmin: float = -0.2, tmax: float = 0.4,
) -> Dict[str, Any]:
    """SSP-EOG ocular artifact removal.

    Parameters
    ----------
    data_dict : dict
        OperatorIO; `channels` must contain an EOG-named channel.
    n_proj : int
        Projection components (default 2).
    eog_ch : str
        EOG channel name substring (default "EOG").
    tmin, tmax : float
        Blink epoch in seconds (default -0.2, 0.4).

    Returns
    -------
    dict
        OperatorIO with ocular subspace projected out.

    Raises
    ------
    EasyBCIOperatorError
        recoverable=True if no EOG channel or too few blinks.

    Modality coverage
    -----------------
    EEG / MEG: yes. sEEG / ECoG / fNIRS / spike: forbidden (no EOG channel).

    References
    ----------
    Uusitalo & Ilmoniemi 1997.
    """
    channels = data_dict.get("channels", [])
    eog_idx = next((i for i, c in enumerate(channels) if eog_ch.lower() in c.lower()), None)
    if eog_idx is None:
        raise EasyBCIOperatorError(
            operator="ssp_eog", reason=f"no channel matching {eog_ch!r}",
            recoverable=True, fallback_step="regression_eog",
        )

    t0 = time.monotonic()
    sfreq = float(data_dict["frequency"])
    from scipy.signal import find_peaks
    data = data_dict["data"]
    eog = data[eog_idx]
    threshold = 4 * np.median(np.abs(eog)) / 0.6745
    peaks, _ = find_peaks(np.abs(eog), height=threshold, distance=int(0.5 * sfreq))
    if len(peaks) < 20:
        raise EasyBCIOperatorError(
            operator="ssp_eog", reason=f"only {len(peaks)} blinks (need >= 20)",
            recoverable=True, fallback_step="regression_eog",
        )
    n_pre = int(-tmin * sfreq)
    n_post = int(tmax * sfreq)
    avg = np.zeros((data.shape[0], n_pre + n_post))
    cnt = 0
    for p in peaks:
        if p - n_pre < 0 or p + n_post >= data.shape[1]:
            continue
        avg += data[:, p - n_pre : p + n_post]
        cnt += 1
    avg /= max(cnt, 1)
    U, _, _ = np.linalg.svd(avg, full_matrices=False)
    U = U[:, :n_proj]
    projector = np.eye(data.shape[0]) - U @ U.T
    new_data = (projector @ data).astype(data.dtype, copy=False)
    elapsed = time.monotonic() - t0

    out = dict(data_dict)
    out["data"] = new_data
    out["elapsed_s"] = elapsed
    out["meta"] = {
        **out.get("meta", {}),
        "ssp_eog": {"n_proj": n_proj, "n_blinks": int(cnt), "eog_ch": eog_ch},
    }
    record_step_elapsed("ssp_eog", elapsed, (data_dict.get("meta") or {}).get("step_cache_key"))
    return out
```

## References

1. Uusitalo, M. A., & Ilmoniemi, R. J. (1997). *Signal-space projection
   method for separating MEG or EEG into components*. Medical & Biological
   Engineering & Computing 35(2): 135–140. doi:10.1007/BF02534144.
2. Gramfort, A. et al. (2013). *MEG and EEG data analysis with MNE-Python*.
   Frontiers in Neuroscience 7: 267. doi:10.3389/fnins.2013.00267.
