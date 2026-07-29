---
name: peakdetect_qrs
description: "Pan-Tompkins QRS detector — extract heart-beat events from EOG/ECG channels"
layer: L3
group: adaptive_cleaning
metadata:
  tags: [operator, adaptive_cleaning, qrs, pan_tompkins, ecg, heart_rate]
  modalities: [eeg, meg]
  step_string: "peakdetect_qrs"
  analysis_goal_allowed: [feature_extraction, clinical_screening, exploratory, generic, online_inference]
  analysis_goal_forbidden: [source_localization, phase_amplitude_coupling]
---
# Pan-Tompkins QRS Detector

## Function

Detects QRS complexes (R-wave peaks) in an ECG channel using the
Pan-Tompkins (1985) algorithm. Produces event timestamps suitable for
downstream `ssp_ecg`, heart-rate variability (HRV), or cardiac-event
exclusion in epoched analyses.

Input / Output: data with ECG-named channel → `meta["qrs_times"]` (event
times in seconds) + `meta["heart_rate_bpm"]`.

## Algorithm & Math

1. Bandpass 5–15 Hz (QRS band).
2. Differentiate.
3. Square.
4. Moving-window integration (150 ms).
5. Adaptive threshold + 200 ms refractory peak detection.

## Parameter Format & Defaults

`peakdetect_qrs:{ecg_ch}`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `ecg_ch` | str | "ECG" | ECG channel substring. |
| `refractory_ms` (kw) | float | 200.0 | Min interval between QRS peaks. |

## Modality-Specific Considerations

EEG / MEG (with ECG channel): yes. Direct ECG recording: yes.
Other modalities without ECG: forbidden.

## When to Use / NOT to Use

**Use** when: extracting QRS events for SSP-ECG; computing HRV features;
masking cardiac epochs in PAC.

**Don't use** when: no ECG channel; cardiac artifact subspace removal is
handled by SSP-ECG directly.

## Constraints & Ordering

- Apply on raw or band-passed data with ECG channel preserved.
- Output drives downstream `ssp_ecg` or `epoch_event_masking`.

## Failure Modes & Detection

| Failure | Symptom | Detection |
|---|---|---|
| ECG saturated | Bandpass output flat. | If `np.ptp(filtered) < 0.001` raise recoverable. |
| HR < 30 or > 200 BPM | Likely false detection. | Warn. |

## Common Issues

- **"Heart rate doubled."** T-wave being detected as QRS; tighten band
  to 8–20 Hz.

## Reference Implementation

### Standalone

```python
from __future__ import annotations
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks


def peakdetect_qrs(
    ecg: np.ndarray, sfreq: float, refractory_ms: float = 200.0,
) -> np.ndarray:
    """Pan-Tompkins QRS detection. Returns sample indices."""
    b, a = butter(2, [5 / (sfreq/2), 15 / (sfreq/2)], btype="bandpass")
    filtered = filtfilt(b, a, ecg)
    diff = np.diff(filtered, prepend=filtered[0])
    energy = diff ** 2
    win = max(1, int(0.15 * sfreq))
    energy = np.convolve(energy, np.ones(win) / win, mode="same")
    threshold = 4 * np.median(np.abs(energy)) / 0.6745
    refractory_samples = int(refractory_ms * sfreq / 1000)
    peaks, _ = find_peaks(energy, height=threshold, distance=refractory_samples)
    return peaks
```

### EasyBCI-Adapted

```python
from typing import Any, Dict
import time
import numpy as np
from easybci_lib.tools.neural_processing.operator_errors import EasyBCIOperatorError
from easybci_lib.tools.neural_processing.preprocess.step_cache import record_step_elapsed


def operator_peakdetect_qrs(
    data_dict: Dict[str, Any], *, ecg_ch: str = "ECG", refractory_ms: float = 200.0,
) -> Dict[str, Any]:
    """Pan-Tompkins QRS peak detection.

    Parameters
    ----------
    data_dict : dict
    ecg_ch : str
    refractory_ms : float

    Returns
    -------
    dict — `meta["qrs_times"]`, `meta["heart_rate_bpm"]`.

    Raises
    ------
    EasyBCIOperatorError
        recoverable=True if ECG channel absent or signal flat.

    Modality coverage
    -----------------
    EEG / MEG / ECG-as-data: yes.

    References
    ----------
    Pan & Tompkins 1985.
    """
    channels = data_dict.get("channels", [])
    ecg_idx = next((i for i, c in enumerate(channels) if ecg_ch.lower() in c.lower()), None)
    if ecg_idx is None:
        raise EasyBCIOperatorError(
            operator="peakdetect_qrs", reason=f"no channel matching {ecg_ch!r}",
            recoverable=True, fallback_step=f"acquire ECG or skip",
        )

    t0 = time.monotonic()
    sfreq = float(data_dict["frequency"])
    from scipy.signal import butter, filtfilt, find_peaks
    ecg = data_dict["data"][ecg_idx]
    b, a = butter(2, [5 / (sfreq/2), 15 / (sfreq/2)], btype="bandpass")
    filtered = filtfilt(b, a, ecg)
    if float(np.ptp(filtered)) < 0.001:
        raise EasyBCIOperatorError(
            operator="peakdetect_qrs", reason="ECG signal flat after bandpass",
            recoverable=True, fallback_step="verify ECG channel",
        )
    diff = np.diff(filtered, prepend=filtered[0])
    energy = diff ** 2
    win = max(1, int(0.15 * sfreq))
    energy = np.convolve(energy, np.ones(win) / win, mode="same")
    threshold = 4 * np.median(np.abs(energy)) / 0.6745
    refractory_samples = int(refractory_ms * sfreq / 1000)
    peaks, _ = find_peaks(energy, height=threshold, distance=refractory_samples)
    qrs_times = peaks.astype(np.float64) / sfreq
    duration = data_dict["data"].shape[-1] / sfreq
    hr_bpm = float(len(peaks) / duration * 60.0) if duration > 0 else 0.0
    elapsed = time.monotonic() - t0

    out = dict(data_dict)
    out["elapsed_s"] = elapsed
    out["meta"] = {
        **out.get("meta", {}),
        "qrs_times": qrs_times,
        "heart_rate_bpm": hr_bpm,
        "peakdetect_qrs": {"ecg_ch": ecg_ch, "refractory_ms": refractory_ms},
    }
    record_step_elapsed("peakdetect_qrs", elapsed, (data_dict.get("meta") or {}).get("step_cache_key"))
    return out
```

## References

1. Pan, J., & Tompkins, W. J. (1985). *A real-time QRS detection
   algorithm*. IEEE Trans. Biomed. Eng. 32(3): 230–236.
   doi:10.1109/TBME.1985.325532.
2. Hamilton, P. (2002). *Open source ECG analysis*. Computers in
   Cardiology: 101–104. doi:10.1109/CIC.2002.1166717.
