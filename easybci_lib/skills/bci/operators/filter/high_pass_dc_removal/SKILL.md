---
name: high_pass_dc_removal
description: "FIR-only DC-removal high-pass (< 0.1 Hz) — preserves ERP slow components, no IIR ringing"
layer: L3
group: filter
metadata:
  tags: [operator, filter, highpass, dc, drift]
  modalities: [eeg, meg, seeg, ecog, fnirs]
  step_string: "hp_dc"
  analysis_goal_allowed: [classification, source_localization, feature_extraction, clinical_screening, exploratory, generic, connectivity, phase_amplitude_coupling, online_inference]
  analysis_goal_forbidden: []
---
# High-Pass DC Removal

## Function

Very-slow-drift high-pass (cut-off typically 0.01–0.1 Hz) implemented as
a **linear-phase FIR** rather than the more common IIR. Removes DC and
electrode drift without distorting the slow ERP components (CNV, contingent
negative variation, sleep slow waves) that an IIR high-pass typically warps.

Input / Output: `(n_channels, n_times)` → `(n_channels, n_times)`. Same
shape and dtype; only spectral content shaped below the cut-off.

## Algorithm & Math

Linear-phase FIR via `firwin` with Hamming window. Filter length is set
by the sample rate and cut-off — at 1 kHz with `low=0.1` the filter is
~30k samples per channel — long but zero-phase via `filtfilt` (forward +
reverse), so no group delay introduced.

When `low >= 0.5 Hz`, the operator delegates to the regular `bandpass:low,`
form. The unique value-add here is the sub-0.5 Hz regime where IIR
high-pass distorts ERP latencies (Tanner 2015).

## Parameter Format & Defaults

`hp_dc:{low}`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `low` | float | 0.05 | High-pass cut-off (Hz). Range 0.01–0.5. |
| `numtaps` (kw) | int | auto | FIR length; auto = `next_odd(8 * sfreq / low)`. |

## Modality-Specific Considerations

| Modality | Cut-off (Hz) | Notes |
|---|---|---|
| EEG (ERP / CNV / P3 / N400) | 0.05–0.1 | Tanner 2015: ≥ 0.5 Hz distorts ERP latency by tens of ms. |
| EEG (sleep) | 0.05 | Preserve slow waves (0.5–4 Hz) intact. |
| MEG | 0.1 | Tighter than EEG; less drift but baseline noise still removed. |
| sEEG | 0.1 | DC drift from contact polarization. |
| fNIRS | 0.01 | Hemodynamic response is itself slow. |

Hard exclusion: spike-AP (handled by `bandpass:300,6000`).

## When to Use / NOT to Use

**Use** when: target task is ERP / sleep / CNV; you need slow drifts
removed without distorting < 5 Hz components.

**Don't use** when: `low >= 0.5 Hz` → just use `bandpass:low,`; spike
data (no DC drift content in the spike band).

## Constraints & Ordering

- Apply **before** ICA — ICA is more robust on DC-removed data.
- Apply **before** epoching — drift segmentation depends on cut-off.
- Long FIR makes this expensive on multi-hour recordings; cache the
  output via step_cache.

## Failure Modes & Detection

| Failure | Symptom | Detection |
|---|---|---|
| FIR too short relative to `low` | Cut-off not actually realized; PSD shows energy below `low`. | Compute PSD; if `mean(PSD[f < low/2]) > 0.3 * mean(PSD[f > low])` raise log warning. |
| Recording too short for FIR length | filtfilt pads + reflects; edge artefacts > 30 s. | If `n_t < 3 · numtaps` raise recoverable, suggest `bandpass:{low},` form. |

## Common Issues

- **"My P300 looks shifted in time."** You used a generic `bandpass` with
  `low >= 0.5` instead of this op with `low=0.1`. Switch.
- **"Filter takes forever."** Long FIR. Trade off via `low=0.1` (faster)
  vs `low=0.05` (slower); offline batch is the expected mode.

## Reference Implementation

### Standalone

```python
from __future__ import annotations
import numpy as np
from scipy.signal import firwin, filtfilt


def high_pass_dc_removal(
    data: np.ndarray, sfreq: float, low: float = 0.05, *, numtaps: int | None = None
) -> np.ndarray:
    """Linear-phase FIR high-pass at very low cut-off (default 0.05 Hz)."""
    if low <= 0 or low > 0.5:
        raise ValueError(f"hp_dc: low={low} outside (0, 0.5] Hz")
    if numtaps is None:
        numtaps = int(8 * sfreq / low)
        numtaps |= 1
    h = firwin(numtaps, low, fs=sfreq, pass_zero=False, window="hamming")
    return filtfilt(h, [1.0], data, axis=-1, padtype="even").astype(data.dtype, copy=False)
```

### EasyBCI-Adapted

```python
from typing import Any, Dict
import time
import numpy as np
from easybci_lib.tools.neural_processing.operator_errors import EasyBCIOperatorError
from easybci_lib.tools.neural_processing.preprocess.step_cache import record_step_elapsed


def operator_hp_dc(
    data_dict: Dict[str, Any], *, low: float = 0.05, numtaps: int | None = None,
) -> Dict[str, Any]:
    """FIR-only DC-removal high-pass.

    Parameters
    ----------
    data_dict : dict
        OperatorIO.
    low : float
        High-pass cut-off (default 0.05 Hz). Must be in (0, 0.5].
    numtaps : int or None
        FIR length (default auto).

    Returns
    -------
    dict
        OperatorIO with filtered `data`.

    Raises
    ------
    EasyBCIOperatorError
        recoverable=True if recording too short for required FIR.

    Modality coverage
    -----------------
    EEG / MEG / sEEG / ECoG / fNIRS: yes.
    Spike: forbidden (spike band is > 300 Hz; this op makes no sense there).

    References
    ----------
    Tanner et al. 2015 — high-pass cut-off and ERP latency distortion.
    """
    if low <= 0 or low > 0.5:
        raise EasyBCIOperatorError(
            operator="hp_dc",
            reason=f"low={low} outside (0, 0.5]; use bandpass:{low}, for >=0.5",
            recoverable=True,
            fallback_step=f"bandpass:{low},",
        )
    sfreq = float(data_dict["frequency"])
    if numtaps is None:
        numtaps = int(8 * sfreq / low) | 1
    if data_dict["data"].shape[-1] < 3 * numtaps:
        raise EasyBCIOperatorError(
            operator="hp_dc",
            reason=f"recording too short ({data_dict['data'].shape[-1]} samples) for FIR length {numtaps}",
            recoverable=True,
            fallback_step=f"bandpass:0.5,",
        )

    t0 = time.monotonic()
    from scipy.signal import firwin, filtfilt
    h = firwin(numtaps, low, fs=sfreq, pass_zero=False, window="hamming")
    filtered = filtfilt(h, [1.0], data_dict["data"], axis=-1, padtype="even")
    new_data = filtered.astype(data_dict["data"].dtype, copy=False)
    elapsed = time.monotonic() - t0

    out = dict(data_dict)
    out["data"] = new_data
    out["elapsed_s"] = elapsed
    out["meta"] = {**out.get("meta", {}), "hp_dc": {"low": low, "numtaps": numtaps}}
    record_step_elapsed("hp_dc", elapsed, (data_dict.get("meta") or {}).get("step_cache_key"))
    return out
```

## References

1. Tanner, D. et al. (2015). *How inappropriate high-pass filters can
   produce artifactual effects and incorrect conclusions in ERP studies
   of language and cognition*. Psychophysiology 52(8): 997–1009.
   doi:10.1111/psyp.12437 — proves the < 0.5 Hz regime matters for ERP.
2. Widmann, A. et al. (2015). *Digital filter design for electrophysiological
   data*. Journal of Neuroscience Methods 250: 34–46.
   doi:10.1016/j.jneumeth.2014.08.002 — FIR design reference.
3. de Cheveigné, A. & Nelken, I. (2019). *Filters: When, Why, and How
   (Not) to Use Them*. Neuron 102(2): 280–293.
   doi:10.1016/j.neuron.2019.02.039 — high-pass and ERP distortion.
