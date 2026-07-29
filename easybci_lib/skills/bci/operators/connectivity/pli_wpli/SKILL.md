---
name: pli_wpli
description: "Phase Lag Index / weighted PLI — volume-conduction-robust phase connectivity (Stam 2007 / Vinck 2011)"
layer: L3
group: connectivity
metadata:
  tags: [operator, connectivity, pli, wpli, phase, volume_conduction]
  modalities: [eeg, meg, seeg, ecog, lfp]
  step_string: "pli_wpli"
  analysis_goal_allowed: [feature_extraction, clinical_screening, exploratory, connectivity]
  analysis_goal_forbidden: [source_localization, online_inference]
---
# PLI / wPLI — Phase Lag Index

## Function

PLI (Stam 2007) and weighted PLI (wPLI, Vinck 2011) are phase-connectivity
metrics specifically designed to be **robust to volume conduction**.
Where PLV bleeds zero-lag (instantaneous) coupling — which is dominated
by volume conduction in EEG — PLI / wPLI weight that out.

Input / Output: `(n_channels, n_times)` → `(n_channels, n_channels)`.

## Algorithm & Math

PLI:
```
PLI(x, y) = | mean_t sign( imag( H(x) · conj(H(y)) ) ) |
```

wPLI weights by imaginary magnitude (more sensitive at low lag):
```
wPLI(x, y) = | mean_t imag(H(x)·conj(H(y))) | / mean_t |imag(H(x)·conj(H(y)))|
```

Where `H(·)` is the analytic signal (Hilbert).

## Parameter Format & Defaults

`pli_wpli:{fmin},{fmax},{kind}`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `fmin`, `fmax` | float, float | required | Band. |
| `kind` | str | "wpli" | "pli" or "wpli". |

## Modality-Specific Considerations

EEG: strongly recommended over PLV for non-source-localized data.
MEG / sEEG / ECoG / LFP: PLV is fine; PLI / wPLI is also valid.

## When to Use / NOT to Use

**Use** when: EEG connectivity at sensor level; epilepsy connectivity
networks; resting-state EEG graphs.

**Don't use** when: source-localized data (PLI suppresses zero-lag which
may be meaningful at source level); broadband.

## Constraints & Ordering

Same as PLV — bandpass first, continuous data.

## Failure Modes & Detection

| Failure | Symptom | Detection |
|---|---|---|
| Zero variance | NaN. | If `imag(H(x)·conj(H(y)))` is all zero → log warning. |
| Wide-band | Noisy estimate. | Same as PLV. |

## Common Issues

- **"My PLI is much lower than PLV."** Expected — PLI suppresses
  instantaneous coupling (volume conduction).

## Reference Implementation

### Standalone

```python
from __future__ import annotations
import numpy as np
from scipy.signal import hilbert, butter, filtfilt


def pli_wpli(
    data: np.ndarray, sfreq: float, fmin: float, fmax: float, kind: str = "wpli",
) -> np.ndarray:
    """PLI or wPLI."""
    b, a = butter(4, [fmin / (sfreq/2), fmax / (sfreq/2)], btype="bandpass")
    filtered = filtfilt(b, a, data, axis=-1)
    H = hilbert(filtered, axis=-1)
    n_ch = data.shape[0]
    out = np.zeros((n_ch, n_ch), dtype=np.float32)
    for i in range(n_ch):
        for j in range(i, n_ch):
            cross = H[i] * np.conj(H[j])
            im = np.imag(cross)
            if kind == "wpli":
                num = np.abs(np.mean(im))
                denom = np.mean(np.abs(im)) + 1e-30
                v = float(num / denom)
            else:  # pli
                v = float(np.abs(np.mean(np.sign(im))))
            out[i, j] = out[j, i] = v
    return out
```

### EasyBCI-Adapted

```python
from typing import Any, Dict
import time
import numpy as np
from easybci_lib.tools.neural_processing.operator_errors import EasyBCIOperatorError
from easybci_lib.tools.neural_processing.preprocess.step_cache import record_step_elapsed


def operator_pli_wpli(
    data_dict: Dict[str, Any], *, fmin: float, fmax: float, kind: str = "wpli",
) -> Dict[str, Any]:
    """PLI / wPLI pairwise connectivity.

    Parameters
    ----------
    data_dict : dict
    fmin, fmax : float
    kind : str

    Returns
    -------
    dict — `meta["pli_wpli"]: (n_ch, n_ch)`.

    Raises
    ------
    EasyBCIOperatorError
        recoverable=False on invalid band or kind.

    Modality coverage
    -----------------
    EEG / MEG / sEEG / ECoG / LFP: yes.

    References
    ----------
    Stam et al. 2007; Vinck et al. 2011.
    """
    sfreq = float(data_dict["frequency"])
    if fmax >= sfreq / 2 or fmin >= fmax or kind not in ("pli", "wpli"):
        raise EasyBCIOperatorError(
            operator="pli_wpli",
            reason=f"invalid params fmin={fmin}, fmax={fmax}, kind={kind}",
            recoverable=False,
        )

    t0 = time.monotonic()
    from scipy.signal import hilbert, butter, filtfilt
    b, a = butter(4, [fmin / (sfreq/2), fmax / (sfreq/2)], btype="bandpass")
    filtered = filtfilt(b, a, data_dict["data"], axis=-1)
    H = hilbert(filtered, axis=-1)
    n_ch = data_dict["data"].shape[0]
    pli_mat = np.zeros((n_ch, n_ch), dtype=np.float32)
    for i in range(n_ch):
        for j in range(i, n_ch):
            cross = H[i] * np.conj(H[j])
            im = np.imag(cross)
            if kind == "wpli":
                v = float(np.abs(np.mean(im)) / (np.mean(np.abs(im)) + 1e-30))
            else:
                v = float(np.abs(np.mean(np.sign(im))))
            pli_mat[i, j] = pli_mat[j, i] = v
    elapsed = time.monotonic() - t0

    out = dict(data_dict)
    out["elapsed_s"] = elapsed
    out["meta"] = {**out.get("meta", {}), "pli_wpli": pli_mat,
                   "pli_wpli_kind": kind, "pli_wpli_band": [fmin, fmax]}
    record_step_elapsed("pli_wpli", elapsed, (data_dict.get("meta") or {}).get("step_cache_key"))
    return out
```

## Related Skills

- See `bci/paradigms/analysis/connectivity/SKILL.md`.

## References

1. Stam, C. J. et al. (2007). *Phase lag index: assessment of functional
   connectivity from multi channel EEG and MEG with diminished bias from
   common sources*. Human Brain Mapping 28(11): 1178–1193.
   doi:10.1002/hbm.20346.
2. Vinck, M. et al. (2011). *An improved index of phase-synchronization
   for electrophysiological data in the presence of volume-conduction,
   noise and sample-size bias*. NeuroImage 55(4): 1548–1565.
   doi:10.1016/j.neuroimage.2011.01.055.
