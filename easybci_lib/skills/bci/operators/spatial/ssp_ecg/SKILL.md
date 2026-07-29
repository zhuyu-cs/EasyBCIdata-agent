---
name: ssp_ecg
description: "Signal Space Projection for ECG — cardiac artifact removal (MEG / EEG)"
layer: L3
group: spatial
metadata:
  tags: [operator, spatial, ssp, ecg, mcg, cardiac, meg]
  modalities: [eeg, meg]
  step_string: "ssp_ecg"
  analysis_goal_allowed: [classification, source_localization, feature_extraction, clinical_screening, exploratory, generic, connectivity, online_inference]
  analysis_goal_forbidden: []
---
# SSP — ECG Removal

## Function

Signal Space Projection for cardiac artifact: detects QRS complexes
from an ECG channel and removes the cardiac magnetic / electric
artifact subspace from MEG / EEG via orthogonal projection. The MEG
gold standard for cardiac artifact (the cardiac artifact is strong in
MEG due to the magnetic field of blood circulation).

Input / Output: `(n_channels, n_times)` → same shape, cardiac subspace projected out.

## Algorithm & Math

QRS detection (Pan–Tompkins-style: `bandpass:5,15 → derivative → square → moving_avg → peak_find`).
Average epoch around QRS → top `n_proj` SVD components form the cardiac
subspace; orthogonal projector `P = I − UU^T`.

## Parameter Format & Defaults

`ssp_ecg:{n_proj},{ecg_ch}`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n_proj` | int | 2 | Projection components. |
| `ecg_ch` | str | "ECG" | ECG channel substring. |
| `tmin`, `tmax` (kw) | float, float | -0.1, 0.4 | QRS epoch in seconds. |

## Modality-Specific Considerations

MEG: critical. EEG: minor (cardiac usually small); ICA often suffices.

## When to Use / NOT to Use

**Use** when: MEG with ECG channel; cardiac artifact visible in PSD.

**Don't use** when: no ECG channel; cardiac artifact not present.

## Constraints & Ordering

Apply after notch/bandpass, before ICA/epoching.

## Failure Modes & Detection

| Failure | Symptom | Detection |
|---|---|---|
| ECG channel missing | KeyError. | Pre-check; raise recoverable. |
| Wrong QRS detection | Topography is noise. | If detected QRS rate < 30 BPM or > 200 BPM warn. |

## Common Issues

- **"Heart-rate detection failed."** ECG-channel polarity or saturation;
  check raw ECG trace.

## Reference Implementation

### Standalone

```python
from __future__ import annotations
import numpy as np
from scipy.signal import find_peaks, filtfilt, butter


def ssp_ecg(
    data: np.ndarray, sfreq: float, ecg_idx: int, n_proj: int = 2,
    tmin: float = -0.1, tmax: float = 0.4,
) -> np.ndarray:
    """SSP from QRS events."""
    ecg = data[ecg_idx]
    b, a = butter(2, [5 / (sfreq / 2), 15 / (sfreq / 2)], btype="bandpass")
    filtered = filtfilt(b, a, ecg)
    diff = np.diff(filtered, prepend=filtered[0])
    energy = diff ** 2
    win = max(1, int(0.15 * sfreq))
    energy = np.convolve(energy, np.ones(win) / win, mode="same")
    threshold = 4 * np.median(np.abs(energy)) / 0.6745
    peaks, _ = find_peaks(energy, height=threshold, distance=int(0.3 * sfreq))
    if len(peaks) < 20:
        raise ValueError(f"ssp_ecg: only {len(peaks)} QRS detections")
    n_pre = int(-tmin * sfreq); n_post = int(tmax * sfreq)
    avg = np.zeros((data.shape[0], n_pre + n_post)); cnt = 0
    for p in peaks:
        if p - n_pre < 0 or p + n_post >= data.shape[1]: continue
        avg += data[:, p - n_pre : p + n_post]; cnt += 1
    avg /= max(cnt, 1)
    U, _, _ = np.linalg.svd(avg, full_matrices=False)
    return ((np.eye(data.shape[0]) - U[:, :n_proj] @ U[:, :n_proj].T) @ data).astype(data.dtype, copy=False)
```

### EasyBCI-Adapted

```python
from typing import Any, Dict
import time
import numpy as np
from easybci_lib.tools.neural_processing.operator_errors import EasyBCIOperatorError
from easybci_lib.tools.neural_processing.preprocess.step_cache import record_step_elapsed


def operator_ssp_ecg(
    data_dict: Dict[str, Any], *,
    n_proj: int = 2, ecg_ch: str = "ECG",
    tmin: float = -0.1, tmax: float = 0.4,
) -> Dict[str, Any]:
    """SSP-ECG cardiac artifact removal.

    Parameters
    ----------
    data_dict : dict
    n_proj : int
    ecg_ch : str
    tmin, tmax : float

    Returns
    -------
    dict — projected continuous data.

    Raises
    ------
    EasyBCIOperatorError
        recoverable=True if ECG channel absent or QRS detection fails.

    Modality coverage
    -----------------
    EEG / MEG: yes. Others: forbidden.

    References
    ----------
    Uusitalo & Ilmoniemi 1997; Pan & Tompkins 1985.
    """
    channels = data_dict.get("channels", [])
    ecg_idx = next((i for i, c in enumerate(channels) if ecg_ch.lower() in c.lower()), None)
    if ecg_idx is None:
        raise EasyBCIOperatorError(
            operator="ssp_ecg", reason=f"no channel matching {ecg_ch!r}",
            recoverable=True, fallback_step="ica",
        )

    t0 = time.monotonic()
    sfreq = float(data_dict["frequency"])
    from scipy.signal import find_peaks, filtfilt, butter
    data = data_dict["data"]
    ecg = data[ecg_idx]
    b, a = butter(2, [5 / (sfreq / 2), 15 / (sfreq / 2)], btype="bandpass")
    filtered = filtfilt(b, a, ecg)
    diff = np.diff(filtered, prepend=filtered[0])
    energy = diff ** 2
    win = max(1, int(0.15 * sfreq))
    energy = np.convolve(energy, np.ones(win) / win, mode="same")
    threshold = 4 * np.median(np.abs(energy)) / 0.6745
    peaks, _ = find_peaks(energy, height=threshold, distance=int(0.3 * sfreq))
    if len(peaks) < 20:
        raise EasyBCIOperatorError(
            operator="ssp_ecg", reason=f"only {len(peaks)} QRS detections",
            recoverable=True, fallback_step="ica",
        )
    n_pre = int(-tmin * sfreq); n_post = int(tmax * sfreq)
    avg = np.zeros((data.shape[0], n_pre + n_post)); cnt = 0
    for p in peaks:
        if p - n_pre < 0 or p + n_post >= data.shape[1]: continue
        avg += data[:, p - n_pre : p + n_post]; cnt += 1
    avg /= max(cnt, 1)
    U, _, _ = np.linalg.svd(avg, full_matrices=False)
    new_data = ((np.eye(data.shape[0]) - U[:, :n_proj] @ U[:, :n_proj].T) @ data).astype(data.dtype, copy=False)
    elapsed = time.monotonic() - t0

    out = dict(data_dict)
    out["data"] = new_data
    out["elapsed_s"] = elapsed
    out["meta"] = {**out.get("meta", {}), "ssp_ecg": {"n_proj": n_proj, "n_qrs": int(cnt)}}
    record_step_elapsed("ssp_ecg", elapsed, (data_dict.get("meta") or {}).get("step_cache_key"))
    return out
```

## References

1. Uusitalo, M. A., & Ilmoniemi, R. J. (1997). *Signal-space projection method*.
   Medical & Biological Engineering & Computing 35(2): 135–140. doi:10.1007/BF02534144.
2. Pan, J., & Tompkins, W. J. (1985). *A real-time QRS detection algorithm*.
   IEEE Trans. Biomed. Eng. 32(3): 230–236. doi:10.1109/TBME.1985.325532.
