---
name: coherence
description: "Magnitude-squared / imaginary coherence — spectral-domain connectivity"
layer: L3
group: connectivity
metadata:
  tags: [operator, connectivity, coherence, imaginary_coherence, csd]
  modalities: [eeg, meg, seeg, ecog, lfp]
  step_string: "coherence"
  analysis_goal_allowed: [feature_extraction, clinical_screening, exploratory, connectivity, online_inference]
  analysis_goal_forbidden: [source_localization]
---
# Coherence

## Function

Computes magnitude-squared coherence and (optionally) imaginary coherence
across channel pairs at a given frequency band. Imaginary coherence
(Nolte 2004) is volume-conduction robust, like wPLI but in the spectral
domain.

Input / Output: `(n_channels, n_times)` → `(n_channels, n_channels)`.

## Algorithm & Math

```
C_xy(f) = |S_xy(f)|² / (S_xx(f) · S_yy(f))                # magnitude-squared
C_imag_xy(f) = imag(S_xy(f)) / sqrt(S_xx(f) · S_yy(f))   # imaginary
```

Where `S_xy` is the cross-spectral density via Welch.

## Parameter Format & Defaults

`coherence:{fmin},{fmax},{kind}`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `fmin`, `fmax` | float, float | required | Band. |
| `kind` | str | "mag2" | "mag2" or "imaginary". |
| `nperseg` (kw) | int | sfreq | Welch segment length. |

## Modality-Specific Considerations

EEG: prefer "imaginary" to suppress volume conduction.
MEG / sEEG / ECoG / LFP: "mag2" or "imaginary" both valid.

## When to Use / NOT to Use

**Use** when: spectral-domain connectivity; band-power coupling.

**Don't use** when: phase-only analysis (use PLV/PLI); source-level data
where volume conduction is gone.

## Constraints & Ordering

After bandpass / notch / drop_bads; on continuous data.

## Failure Modes & Detection

| Failure | Symptom | Detection |
|---|---|---|
| Recording too short | Welch fails. | If `n_t < 4 · nperseg` raise recoverable. |
| Empty band | All zero. | Pre-check `fmax < sfreq/2`. |

## Common Issues

- **"My imaginary coherence is much smaller than magnitude-squared."**
  Expected — imag suppresses zero-lag (volume conduction).

## Reference Implementation

### Standalone

```python
from __future__ import annotations
import numpy as np
from scipy.signal import csd


def coherence(
    data: np.ndarray, sfreq: float, fmin: float, fmax: float,
    kind: str = "mag2", nperseg: int | None = None,
) -> np.ndarray:
    """Pairwise coherence at band-mean."""
    if nperseg is None:
        nperseg = int(sfreq)
    n_ch = data.shape[0]
    out = np.zeros((n_ch, n_ch), dtype=np.float32)
    for i in range(n_ch):
        for j in range(i, n_ch):
            f, Sxy = csd(data[i], data[j], fs=sfreq, nperseg=nperseg)
            _, Sxx = csd(data[i], data[i], fs=sfreq, nperseg=nperseg)
            _, Syy = csd(data[j], data[j], fs=sfreq, nperseg=nperseg)
            mask = (f >= fmin) & (f <= fmax)
            if kind == "imaginary":
                num = np.imag(Sxy[mask])
                denom = np.sqrt(np.abs(Sxx[mask] * Syy[mask])) + 1e-30
                v = float(np.mean(np.abs(num / denom)))
            else:
                num = np.abs(Sxy[mask]) ** 2
                denom = np.abs(Sxx[mask] * Syy[mask]) + 1e-30
                v = float(np.mean(num / denom))
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


def operator_coherence(
    data_dict: Dict[str, Any], *,
    fmin: float, fmax: float, kind: str = "mag2",
    nperseg: int | None = None,
) -> Dict[str, Any]:
    """Pairwise coherence.

    Parameters
    ----------
    data_dict : dict
    fmin, fmax : float
    kind : str
        "mag2" / "imaginary".
    nperseg : int or None

    Returns
    -------
    dict — `meta["coherence"]: (n_ch, n_ch)`.

    Raises
    ------
    EasyBCIOperatorError
        recoverable=False for invalid band; recoverable=True for short recordings.

    Modality coverage
    -----------------
    EEG / MEG / sEEG / ECoG / LFP: yes.

    References
    ----------
    Nolte et al. 2004; Welch 1967.
    """
    sfreq = float(data_dict["frequency"])
    if fmax >= sfreq / 2 or fmin >= fmax or kind not in ("mag2", "imaginary"):
        raise EasyBCIOperatorError(
            operator="coherence",
            reason=f"invalid fmin={fmin}, fmax={fmax}, kind={kind}",
            recoverable=False,
        )
    if nperseg is None:
        nperseg = int(sfreq)
    if data_dict["data"].shape[-1] < 4 * nperseg:
        raise EasyBCIOperatorError(
            operator="coherence",
            reason="recording too short for Welch",
            recoverable=True, fallback_step="coherence with smaller nperseg",
        )

    t0 = time.monotonic()
    from scipy.signal import csd
    data = data_dict["data"]
    n_ch = data.shape[0]
    coh = np.zeros((n_ch, n_ch), dtype=np.float32)
    for i in range(n_ch):
        for j in range(i, n_ch):
            f, Sxy = csd(data[i], data[j], fs=sfreq, nperseg=nperseg)
            _, Sxx = csd(data[i], data[i], fs=sfreq, nperseg=nperseg)
            _, Syy = csd(data[j], data[j], fs=sfreq, nperseg=nperseg)
            mask = (f >= fmin) & (f <= fmax)
            if kind == "imaginary":
                num = np.imag(Sxy[mask])
                denom = np.sqrt(np.abs(Sxx[mask] * Syy[mask])) + 1e-30
                v = float(np.mean(np.abs(num / denom)))
            else:
                num = np.abs(Sxy[mask]) ** 2
                denom = np.abs(Sxx[mask] * Syy[mask]) + 1e-30
                v = float(np.mean(num / denom))
            coh[i, j] = coh[j, i] = v
    elapsed = time.monotonic() - t0

    out = dict(data_dict)
    out["elapsed_s"] = elapsed
    out["meta"] = {**out.get("meta", {}), "coherence": coh,
                   "coherence_band": [fmin, fmax], "coherence_kind": kind}
    record_step_elapsed("coherence", elapsed, (data_dict.get("meta") or {}).get("step_cache_key"))
    return out
```

## Related Skills

- `bci/paradigms/analysis/connectivity/SKILL.md`.

## References

1. Nolte, G. et al. (2004). *Identifying true brain interaction from EEG
   data using the imaginary part of coherency*. Clinical Neurophysiology
   115(10): 2292–2307. doi:10.1016/j.clinph.2004.04.029.
2. Welch, P. D. (1967). *The use of fast Fourier transform for the
   estimation of power spectra*. IEEE Trans. Audio Electroacoustics
   15(2): 70–73. doi:10.1109/TAU.1967.1161901.
