---
name: plv
description: "Phase-Locking Value — pairwise phase synchrony (Lachaux 1999)"
layer: L3
group: connectivity
metadata:
  tags: [operator, connectivity, plv, phase, lachaux]
  modalities: [eeg, meg, seeg, ecog, lfp]
  step_string: "plv"
  analysis_goal_allowed: [feature_extraction, exploratory, connectivity]
  analysis_goal_forbidden: [source_localization, online_inference]
---
# Phase-Locking Value (PLV)

## Function

Computes pairwise PLV across channels — a measure of how consistently
the instantaneous phase of two signals locks in time. Range [0, 1]:
0 = independent, 1 = perfectly phase-locked.

Input / Output: `(n_channels, n_times)` → `(n_channels, n_channels)`.

## Algorithm & Math

For two channels x, y:
```
φ_x(t), φ_y(t) = instantaneous phase via Hilbert (or bandpassed Hilbert).
PLV(x, y) = | (1/T) · Σ_t exp(j · (φ_x(t) − φ_y(t))) |
```

Bandpass before PLV — the analysis is meaningful only for narrow-band
phase.

## Parameter Format & Defaults

`plv:{fmin},{fmax}`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `fmin`, `fmax` | float, float | required | Narrow band edges. |
| `n_segments` (kw) | int | 1 | Segment count for time-averaged PLV. |

## Modality-Specific Considerations

| Modality | Bands | Notes |
|---|---|---|
| EEG / MEG | alpha (8–13), beta (13–30), gamma (30–80) | Standard cognition. |
| sEEG / ECoG | gamma / HFO | Functional mapping. |
| LFP | theta / gamma | Hippocampal coupling. |

## When to Use / NOT to Use

**Use** when: functional connectivity / phase coupling; PAC carrier
analysis (per-band step).

**Don't use** when: amplitude-only analyses; online inference (cost);
wide-band (need narrow band for meaningful phase).

## Constraints & Ordering

- Apply **after** bandpass to target band.
- Apply on continuous data; segment downstream.

## Failure Modes & Detection

| Failure | Symptom | Detection |
|---|---|---|
| Wide-band input | PLV is mean ~ 1/sqrt(T) (random phase). | If `fmax - fmin > 5 Hz` warn. |
| Volume conduction (EEG) | All-pair PLV ≈ 1. | Use REST or Laplacian first. |

## Common Issues

- **"My EEG PLV is uniformly high."** Volume conduction. Re-reference to
  REST or apply Laplacian first.

## Reference Implementation

### Standalone

```python
from __future__ import annotations
import numpy as np
from scipy.signal import hilbert, butter, filtfilt


def plv(
    data: np.ndarray, sfreq: float, fmin: float, fmax: float
) -> np.ndarray:
    """Pairwise PLV across channels."""
    b, a = butter(4, [fmin / (sfreq/2), fmax / (sfreq/2)], btype="bandpass")
    filtered = filtfilt(b, a, data, axis=-1)
    phases = np.angle(hilbert(filtered, axis=-1))
    n_ch = data.shape[0]
    plv_mat = np.zeros((n_ch, n_ch), dtype=np.float32)
    for i in range(n_ch):
        for j in range(i, n_ch):
            diff = phases[i] - phases[j]
            plv_mat[i, j] = plv_mat[j, i] = float(np.abs(np.mean(np.exp(1j * diff))))
    return plv_mat
```

### EasyBCI-Adapted

```python
from typing import Any, Dict
import time
import numpy as np
from easybci_lib.tools.neural_processing.operator_errors import EasyBCIOperatorError
from easybci_lib.tools.neural_processing.preprocess.step_cache import record_step_elapsed


def operator_plv(
    data_dict: Dict[str, Any], *, fmin: float, fmax: float,
) -> Dict[str, Any]:
    """Phase-locking value pairwise matrix.

    Parameters
    ----------
    data_dict : dict
    fmin, fmax : float
        Narrow band.

    Returns
    -------
    dict — `meta["plv"]: (n_ch, n_ch)`.

    Raises
    ------
    EasyBCIOperatorError
        recoverable=True for wide-band input or fmax >= Nyquist.

    Modality coverage
    -----------------
    EEG / MEG / sEEG / ECoG / LFP: yes. fNIRS / spike: forbidden.

    References
    ----------
    Lachaux et al. 1999.
    """
    sfreq = float(data_dict["frequency"])
    if fmax >= sfreq / 2 or fmin >= fmax:
        raise EasyBCIOperatorError(
            operator="plv", reason=f"invalid band fmin={fmin}, fmax={fmax}, sfreq={sfreq}",
            recoverable=False,
        )

    t0 = time.monotonic()
    from scipy.signal import hilbert, butter, filtfilt
    b, a = butter(4, [fmin / (sfreq/2), fmax / (sfreq/2)], btype="bandpass")
    filtered = filtfilt(b, a, data_dict["data"], axis=-1)
    phases = np.angle(hilbert(filtered, axis=-1))
    n_ch = data_dict["data"].shape[0]
    plv_mat = np.zeros((n_ch, n_ch), dtype=np.float32)
    for i in range(n_ch):
        for j in range(i, n_ch):
            diff = phases[i] - phases[j]
            plv_mat[i, j] = plv_mat[j, i] = float(np.abs(np.mean(np.exp(1j * diff))))
    elapsed = time.monotonic() - t0

    out = dict(data_dict)
    out["elapsed_s"] = elapsed
    out["meta"] = {**out.get("meta", {}), "plv": plv_mat, "plv_band": [fmin, fmax]}
    record_step_elapsed("plv", elapsed, (data_dict.get("meta") or {}).get("step_cache_key"))
    return out
```

## Related Skills

- See `bci/paradigms/analysis/connectivity/SKILL.md` for
  paradigm-level guidance.

## References

1. Lachaux, J.-P. et al. (1999). *Measuring phase synchrony in brain
   signals*. Human Brain Mapping 8(4): 194–208.
   doi:10.1002/(SICI)1097-0193(1999)8:4<194::AID-HBM4>3.0.CO;2-C.
2. Cohen, M. X. (2014). *Analyzing Neural Time Series Data*. MIT Press.
