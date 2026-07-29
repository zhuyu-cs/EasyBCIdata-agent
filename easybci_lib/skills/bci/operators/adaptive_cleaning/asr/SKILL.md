---
name: asr
description: "Artifact Subspace Reconstruction (Kothe-Makeig 2013) — online BCI gold standard, ICA alternative"
layer: L3
group: adaptive_cleaning
metadata:
  tags: [operator, adaptive_cleaning, asr, kothe, online_bci, real_time]
  modalities: [eeg, meg]
  step_string: "asr"
  analysis_goal_allowed: [classification, feature_extraction, exploratory, generic, online_inference]
  analysis_goal_forbidden: [source_localization, phase_amplitude_coupling]
---
# ASR — Artifact Subspace Reconstruction

## Function

Artifact Subspace Reconstruction (Kothe & Makeig 2013, EEGLAB plug-in) —
online-capable adaptive cleaning that identifies "calibration" epochs of
clean data, learns their PCA subspace, and rejects any subspace
component exceeding `k · σ` of the calibration norm.

Input / Output: `(n_channels, n_times)` → `(n_channels, n_times)`.

## Algorithm & Math

1. Sliding-window calibration: pick clean 1-min segments from baseline.
2. Compute calibration covariance `C_cal` and eigendecomposition.
3. For each sliding 0.5–1 s test window:
   - Project onto calibration eigenvectors.
   - Per-component variance > k · σ_cal? → mark as artifact subspace.
   - Reconstruct data from non-artifact eigenvectors only.

## Parameter Format & Defaults

`asr:{k}`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `k` | float | 20.0 | Std-multiplier (Mullen 2015 recommends 20; lower = more aggressive). |
| `window_s` (kw) | float | 0.5 | Sliding window seconds. |
| `calib_s` (kw) | float | 60.0 | Calibration window seconds. |

## Modality-Specific Considerations

EEG: standard. MEG: viable for online closed-loop. sEEG / ECoG / fNIRS:
not standard.

## When to Use / NOT to Use

**Use** when: online BCI; closed-loop experiment; ICA too slow; episodic
artifacts (eye movements, jaw clench).

**Don't use** when: PAC (subspace removal may strip oscillations);
source localization (alters spatial pattern non-trivially); chronic
clean recording with no artifacts.

## Constraints & Ordering

- Apply **after** notch / bandpass.
- Apply **before** epoching / decoder.
- Calibration window must contain clean data (manual or auto-detect).

## Failure Modes & Detection

| Failure | Symptom | Detection |
|---|---|---|
| Calibration contaminated | Threshold k inflated; later artifacts not removed. | Pre-validate calibration with auto-clean detector. |
| k too low (over-aggressive) | Brain signal removed; alpha disappears. | If post-PSD alpha drop > 50% → raise k. |

## Common Issues

- **"My alpha dropped after ASR."** k too low (e.g. 5–10). Raise to 20+.

## Reference Implementation

### Standalone

```python
from __future__ import annotations
import numpy as np


def asr(
    data: np.ndarray, sfreq: float, k: float = 20.0,
    window_s: float = 0.5, calib_s: float = 60.0,
) -> np.ndarray:
    """ASR — calibration-based subspace cleaning."""
    n_ch, n_t = data.shape
    n_calib = min(int(calib_s * sfreq), n_t)
    calib = data[:, :n_calib]
    C_cal = calib @ calib.T / n_calib
    eigvals, eigvecs = np.linalg.eigh(C_cal)
    eigvals = np.maximum(eigvals, 1e-30)
    sigmas = np.sqrt(eigvals)
    n_window = int(window_s * sfreq)
    out = data.copy()
    for start in range(0, n_t, n_window):
        end = min(start + n_window, n_t)
        seg = data[:, start:end]
        if seg.shape[1] < 2: continue
        proj = eigvecs.T @ seg
        component_vars = np.var(proj, axis=1)
        keep_mask = component_vars < (k * sigmas) ** 2
        proj_filtered = proj * keep_mask[:, None]
        out[:, start:end] = eigvecs @ proj_filtered
    return out.astype(data.dtype, copy=False)
```

### EasyBCI-Adapted

```python
from typing import Any, Dict
import time
import numpy as np
from easybci_lib.tools.neural_processing.operator_errors import EasyBCIOperatorError
from easybci_lib.tools.neural_processing.preprocess.step_cache import record_step_elapsed


def operator_asr(
    data_dict: Dict[str, Any], *, k: float = 20.0,
    window_s: float = 0.5, calib_s: float = 60.0,
) -> Dict[str, Any]:
    """ASR adaptive cleaning.

    Parameters
    ----------
    data_dict : dict
    k : float
    window_s, calib_s : float

    Returns
    -------
    dict — cleaned data.

    Raises
    ------
    EasyBCIOperatorError
        recoverable=True if recording too short for calibration window.

    Modality coverage
    -----------------
    EEG / MEG: yes. Others: forbidden.

    References
    ----------
    Kothe & Makeig 2013; Mullen et al. 2015.
    """
    sfreq = float(data_dict["frequency"])
    n_ch, n_t = data_dict["data"].shape
    if n_t / sfreq < calib_s:
        calib_s = max(10.0, (n_t / sfreq) / 2)

    t0 = time.monotonic()
    n_calib = int(calib_s * sfreq)
    calib = data_dict["data"][:, :n_calib]
    C_cal = calib @ calib.T / n_calib
    eigvals, eigvecs = np.linalg.eigh(C_cal)
    eigvals = np.maximum(eigvals, 1e-30)
    sigmas = np.sqrt(eigvals)
    n_window = int(window_s * sfreq)
    new_data = data_dict["data"].copy()
    for start in range(0, n_t, n_window):
        end = min(start + n_window, n_t)
        seg = data_dict["data"][:, start:end]
        if seg.shape[1] < 2: continue
        proj = eigvecs.T @ seg
        component_vars = np.var(proj, axis=1)
        keep_mask = component_vars < (k * sigmas) ** 2
        proj_filtered = proj * keep_mask[:, None]
        new_data[:, start:end] = eigvecs @ proj_filtered
    new_data = new_data.astype(data_dict["data"].dtype, copy=False)
    elapsed = time.monotonic() - t0

    out = dict(data_dict)
    out["data"] = new_data
    out["elapsed_s"] = elapsed
    out["meta"] = {**out.get("meta", {}), "asr": {"k": k, "window_s": window_s, "calib_s": calib_s}}
    record_step_elapsed("asr", elapsed, (data_dict.get("meta") or {}).get("step_cache_key"))
    return out
```

## References

1. Kothe, C. A. E., & Makeig, S. (2013). *BCILAB: A platform for
   brain-computer interface development*. J. Neural Eng. 10(5): 056014.
   doi:10.1088/1741-2560/10/5/056014.
2. Mullen, T. R. et al. (2015). *Real-time neuroimaging and cognitive
   monitoring using wearable dry EEG*. IEEE Trans. Biomed. Eng. 62(11):
   2553–2567. doi:10.1109/TBME.2015.2481482 — k parameter guidance.
3. Chang, C.-Y. et al. (2020). *Evaluation of Artifact Subspace
   Reconstruction for Automatic Artifact Components Removal in
   Multi-Channel EEG Recordings*. IEEE Trans. Biomed. Eng. 67(4):
   1114–1121. doi:10.1109/TBME.2019.2930186.
