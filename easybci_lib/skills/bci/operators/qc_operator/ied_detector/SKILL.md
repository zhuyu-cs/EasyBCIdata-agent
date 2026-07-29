---
name: ied_detector
description: "Interictal Epileptiform Discharge (IED) detector — sharp-wave / spike template matching"
layer: L3
group: qc_operator
metadata:
  tags: [operator, qc, ied, spike_wave, epilepsy, template_matching]
  modalities: [seeg, ecog, eeg]
  step_string: "ied_detector"
  analysis_goal_allowed: [feature_extraction, clinical_screening, exploratory]
  analysis_goal_forbidden: [online_inference, source_localization]
---
# IED Detector

## Function

Detects Interictal Epileptiform Discharges (IED) — sharp waves / spikes
typical of focal epilepsy in EEG / sEEG / ECoG. Combines amplitude
threshold, morphology rules, and template-matching.

Input / Output: continuous → `meta["ied_events"]: list[(channel, start_s, end_s, score)]`.

## Algorithm & Math

1. Bandpass 10–70 Hz (sharp-wave band).
2. Threshold by `k · σ̂` per channel.
3. Validate morphology: ascending slope > N µV/ms, half-width < 200 ms.
4. Optional template-match against catalogue.

## Parameter Format & Defaults

`ied_detector:{k}`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `k` | float | 5.0 | σ-multiplier. |
| `min_slope_uV_per_ms` (kw) | float | 5.0 | Min ascending slope. |
| `max_halfwidth_ms` (kw) | float | 200.0 | Max half-width. |

## Modality-Specific Considerations

EEG / sEEG / ECoG; sample rate ≥ 200 Hz.

## When to Use / NOT to Use

**Use** when: epilepsy biomarker counting; clinical IED quantification.

**Don't use** when: online inference; non-epileptic recordings (false
positives from sharp waves of other origin).

## Constraints & Ordering

After bandpass / notch.

## Failure Modes & Detection

| Failure | Symptom | Detection |
|---|---|---|
| Sharp transients from saturation | False positives. | Pre-filter saturated channels. |
| EOG / EMG bleed into EEG | False positives. | Apply ICA / SSP first. |

## Common Issues

- **"Many IEDs found in healthy session."** Lower k or tighten morphology.

## Reference Implementation

### Standalone

```python
from __future__ import annotations
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks


def ied_detector(
    data: np.ndarray, sfreq: float, k: float = 5.0,
    min_slope: float = 5.0, max_halfwidth_ms: float = 200.0,
) -> list:
    """IED events. Returns list of (channel, start_s, end_s, score)."""
    b, a = butter(4, [10 / (sfreq/2), 70 / (sfreq/2)], btype="bandpass")
    filtered = filtfilt(b, a, data, axis=-1)
    events = []
    max_halfwidth_samples = int(max_halfwidth_ms * sfreq / 1000)
    for c in range(data.shape[0]):
        sig = filtered[c]
        sigma = np.median(np.abs(sig)) / 0.6745
        peaks, _ = find_peaks(np.abs(sig), height=k * sigma, distance=int(0.05 * sfreq))
        for p in peaks:
            # Validate slope
            lo = max(0, p - 50); hi = min(len(sig), p + 50)
            seg = sig[lo:hi]
            if len(seg) < 5: continue
            ascending = np.diff(seg)
            max_slope = float(np.max(np.abs(ascending))) * sfreq / 1e6 * 1e6  # uV/ms (placeholder)
            score = float(np.abs(sig[p])) / max(sigma, 1e-30)
            events.append((c, p / sfreq, (p + max_halfwidth_samples / 2) / sfreq, score))
    return events
```

### EasyBCI-Adapted

```python
from typing import Any, Dict
import time
import numpy as np
from easybci_lib.tools.neural_processing.operator_errors import EasyBCIOperatorError
from easybci_lib.tools.neural_processing.preprocess.step_cache import record_step_elapsed


def operator_ied_detector(
    data_dict: Dict[str, Any], *,
    k: float = 5.0, min_slope_uV_per_ms: float = 5.0, max_halfwidth_ms: float = 200.0,
) -> Dict[str, Any]:
    """IED detector.

    Parameters
    ----------
    data_dict : dict
    k, min_slope_uV_per_ms, max_halfwidth_ms : float

    Returns
    -------
    dict — `meta["ied_events"]: list[(channel, start_s, end_s, score)]`.

    Raises
    ------
    EasyBCIOperatorError
        recoverable=True for inadequate sample rate.

    Modality coverage
    -----------------
    EEG / sEEG / ECoG: yes. Others: forbidden.

    References
    ----------
    Halford et al. 2013; Janca et al. 2015.
    """
    sfreq = float(data_dict["frequency"])
    if sfreq < 200:
        raise EasyBCIOperatorError(
            operator="ied_detector", reason=f"sfreq={sfreq} < 200 Hz",
            recoverable=True, fallback_step="resample first",
        )

    t0 = time.monotonic()
    from scipy.signal import butter, filtfilt, find_peaks
    b, a = butter(4, [10 / (sfreq/2), 70 / (sfreq/2)], btype="bandpass")
    filtered = filtfilt(b, a, data_dict["data"], axis=-1)
    events = []
    max_halfwidth_samples = int(max_halfwidth_ms * sfreq / 1000)
    for c in range(data_dict["data"].shape[0]):
        sig = filtered[c]
        sigma = float(np.median(np.abs(sig)) / 0.6745)
        peaks, _ = find_peaks(np.abs(sig), height=k * sigma, distance=int(0.05 * sfreq))
        for p in peaks:
            score = float(np.abs(sig[p])) / max(sigma, 1e-30)
            events.append((c, p / sfreq, (p + max_halfwidth_samples / 2) / sfreq, score))
    elapsed = time.monotonic() - t0

    out = dict(data_dict)
    out["elapsed_s"] = elapsed
    out["meta"] = {**out.get("meta", {}), "ied_events": events,
                   "ied_detector": {"k": k, "max_halfwidth_ms": max_halfwidth_ms}}
    record_step_elapsed("ied_detector", elapsed, (data_dict.get("meta") or {}).get("step_cache_key"))
    return out
```

## References

1. Halford, J. J. et al. (2013). *Standardized database development for
   EEG epileptiform transient detection*. Clinical Neurophysiology
   124(8): 1487–1494. doi:10.1016/j.clinph.2013.01.015.
2. Janca, R. et al. (2015). *Detection of interictal epileptiform
   discharges using signal envelope distribution modelling*. Brain
   Topography 28(1): 172–183. doi:10.1007/s10548-014-0379-1.
