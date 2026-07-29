---
name: log_band_power
description: "Log-transformed band-power features (multi-band) for classification feature vectors"
layer: L3
group: spectral
metadata:
  tags: [operator, spectral, band_power, feature, log, decoder_input]
  modalities: [eeg, meg, seeg, ecog, lfp]
  step_string: "log_band_power"
  analysis_goal_allowed: [classification, feature_extraction, clinical_screening, exploratory, generic, online_inference]
  analysis_goal_forbidden: []
---
# Log Band Power

## Function

Computes log-power in canonical EEG bands (or user-specified bands)
per channel, producing a feature vector or feature matrix suitable for
downstream classifiers. Log transform stabilizes the variance of
right-skewed power distributions and makes Gaussian classifiers more
applicable.

Input / Output: `(n_channels, n_times)` → `(n_channels, n_bands)` (or
`(n_channels, n_bands, n_segments)` when segmented).

## Algorithm & Math

For each band `[f_lo, f_hi]`:
```
power_band[c] = mean over f ∈ [f_lo, f_hi] of PSD_welch(data[c], f)
log_band_power[c] = log(power_band[c] + ε)        # ε = 1e-20
```

PSD uses Welch's method (segmented periodogram + Hann window + median
averaging) by default; switch to multitaper via `method="multitaper"`.

## Parameter Format & Defaults

`log_band_power:{band_set}`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `bands` | list of (lo, hi) tuples | EEG canonical | Default `[(1,4),(4,8),(8,13),(13,30),(30,45)]` (δ θ α β γ). |
| `method` (kw) | str | "welch" | "welch" or "multitaper". |
| `n_per_seg` (kw) | int | sfreq | Welch segment length. |
| `epsilon` (kw) | float | 1e-20 | Log-stability term. |

## Modality-Specific Considerations

| Modality | Default bands |
|---|---|
| EEG | δ(1-4) θ(4-8) α(8-13) β(13-30) γ(30-45) |
| MEG | Same as EEG, optionally extend γ to 80 Hz. |
| sEEG / ECoG | Add HFO(80-200) + ripple(80-150) bands. |
| LFP | δ(0.5-4) θ(4-8) β(13-30) γ(30-100). |
| fNIRS | n/a — use HbO / HbR concentration directly. |

## When to Use / NOT to Use

**Use** when: downstream classifier expects feature vector; standard
band-power EEG baseline; online BCI cheap feature extraction.

**Don't use** when: phase information matters (PAC, connectivity);
single-frequency analysis (use bandpass + hilbert envelope).

## Constraints & Ordering

- Apply **after** all temporal preprocessing (bandpass, notch, ICA).
- Apply **after** segmentation if per-epoch features needed.

## Failure Modes & Detection

| Failure | Symptom | Detection |
|---|---|---|
| Empty band (lo > Nyquist) | log(0) → -inf or NaN. | Pre-check; raise recoverable. |
| Numerical underflow | Some channels return -inf. | epsilon ≥ 1e-30; warn if any output is -inf. |

## Common Issues

- **"My classifier expects raw power, not log."** Drop the log — but
  classifier accuracy is usually 5–10% better with log on EEG band-power.

## Reference Implementation

### Standalone

```python
from __future__ import annotations
import numpy as np
from scipy.signal import welch


DEFAULT_BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]


def log_band_power(
    data: np.ndarray, sfreq: float, bands=None, epsilon: float = 1e-20
) -> np.ndarray:
    """Log band-power features. Returns (n_ch, n_bands)."""
    bands = bands or DEFAULT_BANDS
    f, p = welch(data, fs=sfreq, nperseg=min(int(sfreq), data.shape[-1]), axis=-1)
    out = np.zeros((data.shape[0], len(bands)), dtype=np.float32)
    for bi, (lo, hi) in enumerate(bands):
        mask = (f >= lo) & (f <= hi)
        if not mask.any():
            raise ValueError(f"band {lo}-{hi} Hz contains no samples (Nyquist={sfreq/2})")
        out[:, bi] = np.log(np.mean(p[..., mask], axis=-1) + epsilon)
    return out
```

### EasyBCI-Adapted

```python
from typing import Any, Dict, List, Tuple
import time
import numpy as np
from easybci_lib.tools.neural_processing.operator_errors import EasyBCIOperatorError
from easybci_lib.tools.neural_processing.preprocess.step_cache import record_step_elapsed


_DEFAULT_BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]


def operator_log_band_power(
    data_dict: Dict[str, Any], *,
    bands: List[Tuple[float, float]] | None = None,
    method: str = "welch",
    epsilon: float = 1e-20,
) -> Dict[str, Any]:
    """Log band-power features.

    Parameters
    ----------
    data_dict : dict
        OperatorIO.
    bands : list of (lo, hi) tuples
        Default `[(1,4),(4,8),(8,13),(13,30),(30,45)]`.
    method : {"welch", "multitaper"}
        PSD method (default "welch").
    epsilon : float
        Log-stability term (default 1e-20).

    Returns
    -------
    dict
        OperatorIO; `data` unchanged; meta populated:
        - `meta["log_band_power"]`: ndarray (n_ch, n_bands)
        - `meta["log_band_power_bands"]`: list of (lo, hi)

    Raises
    ------
    EasyBCIOperatorError
        recoverable=True for any band beyond Nyquist.

    Modality coverage
    -----------------
    EEG / MEG / sEEG / ECoG / LFP: yes.
    fNIRS: forbidden (no oscillatory power).
    Spike: forbidden.

    References
    ----------
    Pfurtscheller & Aranibar 1977; Blankertz et al. 2008.
    """
    bands = bands or _DEFAULT_BANDS
    sfreq = float(data_dict["frequency"])
    nyq = sfreq / 2
    for lo, hi in bands:
        if hi > nyq:
            raise EasyBCIOperatorError(
                operator="log_band_power",
                reason=f"band {lo}-{hi} exceeds Nyquist {nyq}",
                recoverable=True,
                fallback_step=f"log_band_power (dropping {lo}-{hi})",
            )

    t0 = time.monotonic()
    from scipy.signal import welch
    f, p = welch(
        data_dict["data"], fs=sfreq,
        nperseg=min(int(sfreq), data_dict["data"].shape[-1]), axis=-1,
    )
    out_arr = np.zeros((data_dict["data"].shape[0], len(bands)), dtype=np.float32)
    for bi, (lo, hi) in enumerate(bands):
        mask = (f >= lo) & (f <= hi)
        out_arr[:, bi] = np.log(np.mean(p[..., mask], axis=-1) + epsilon)
    elapsed = time.monotonic() - t0

    out = dict(data_dict)
    out["elapsed_s"] = elapsed
    out["meta"] = {
        **out.get("meta", {}),
        "log_band_power": out_arr,
        "log_band_power_bands": list(bands),
        "log_band_power_meta": {"method": method, "epsilon": epsilon},
    }
    record_step_elapsed("log_band_power", elapsed, (data_dict.get("meta") or {}).get("step_cache_key"))
    return out
```

## References

1. Pfurtscheller, G., & Aranibar, A. (1977). *Event-related cortical
   desynchronization detected by power measurements of scalp EEG*.
   Electroencephalography and Clinical Neurophysiology 42(6): 817–826.
   doi:10.1016/0013-4694(77)90235-8 — band-power feature origin.
2. Blankertz, B. et al. (2008). *Optimizing spatial filters for robust
   EEG single-trial analysis*. IEEE Signal Processing Magazine 25(1):
   41–56. doi:10.1109/MSP.2008.4408441 — log-band-power + CSP standard
   pipeline.
