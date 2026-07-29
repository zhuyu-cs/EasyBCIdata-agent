---
name: cwt_morlet
description: "Continuous Wavelet Transform (Morlet wavelet) — time-frequency with logarithmic frequency spacing"
layer: L3
group: spectral
metadata:
  tags: [operator, spectral, wavelet, morlet, time_frequency]
  modalities: [eeg, meg, seeg, ecog, lfp]
  step_string: "cwt_morlet"
  analysis_goal_allowed: [classification, feature_extraction, clinical_screening, exploratory, connectivity, phase_amplitude_coupling]
  analysis_goal_forbidden: [online_inference, source_localization]
---
# CWT — Morlet Wavelet

## Function

Continuous wavelet transform with complex Morlet wavelets — produces a
time-frequency representation with **logarithmic** frequency spacing
(default) and adaptive time window per frequency (longer window at low
frequencies, shorter at high). Strong alternative to STFT for ERP /
oscillation analysis.

Input / Output: `(n_channels, n_times)` → `(n_channels, n_freqs, n_times)` complex64.

## Algorithm & Math

Complex Morlet wavelet:
```
ψ(t, f, n) = (1/(σ_t √π)) · exp(-t² / (2σ_t²)) · exp(j 2π f t)
σ_t = n / (2π f)        # n = "number of cycles"
```

Convolve each channel with `ψ(t, f, n)` at each `f` — the result is the
complex amplitude × phase at frequency `f`, time `t`.

Trade-offs:
- Low `n` (3–5 cycles): wider band, sharper time resolution.
- High `n` (7–10 cycles): narrower band, coarser time resolution.

## Parameter Format & Defaults

`cwt_morlet:{fmin},{fmax},{n_freqs},{n_cycles}`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `fmin`, `fmax` | float, float | 2.0, 80.0 | Frequency range (Hz). |
| `n_freqs` | int | 30 | Log-spaced frequencies. |
| `n_cycles` | float \| array | 7.0 | Cycle count (scalar = constant; vary as `n_cycles · f/fmin` for log-scaling). |

## Modality-Specific Considerations

| Modality | fmin | fmax | n_cycles | Notes |
|---|---|---|---|---|
| EEG (alpha/beta) | 4 | 40 | 7 | Standard ERS/ERD analysis. |
| EEG (gamma) | 30 | 80 | 10 | Long n_cycles to resolve gamma. |
| MEG | 4 | 80 | 7 | Same as EEG. |
| sEEG (HFO) | 80 | 200 | 12 | Long n_cycles for HFO. |
| LFP | 1 | 100 | 5 | Wider band; slower oscillations. |

## When to Use / NOT to Use

**Use** when: log-spaced time-frequency required; ERS/ERD per band;
PAC carrier extraction at multiple bands.

**Don't use** when: online inference (computational cost); single band
sufficient (use bandpass + hilbert); linear-frequency analysis (use STFT).

## Constraints & Ordering

- Apply **after** bandpass / notch / drop_bads.
- Recording must be ≥ `5 · n_cycles / fmin` seconds; otherwise the lowest
  frequencies suffer edge effects.

## Failure Modes & Detection

| Failure | Symptom | Detection |
|---|---|---|
| Recording too short for fmin | Edge effects swamp low-freq output. | If `n_t / sfreq < 5 · n_cycles / fmin` raise recoverable, suggest higher fmin. |
| n_cycles too low | Spectral leakage; bands smear together. | If wavelet's bandwidth at fmax > (fmax − fmin) / 2 warn. |

## Common Issues

- **"My gamma response is smeared in time."** n_cycles too high; lower to
  5 for finer time resolution at the cost of frequency.
- **"Edges of the spectrogram show ringing."** Recording boundary;
  exclude the first / last `n_cycles / fmin` seconds from analysis.

## Reference Implementation

### Standalone

```python
from __future__ import annotations
import numpy as np


def _morlet_wavelet(f: float, sfreq: float, n_cycles: float) -> np.ndarray:
    """Complex Morlet wavelet at frequency f."""
    sigma_t = n_cycles / (2 * np.pi * f)
    t = np.arange(-3.5 * sigma_t, 3.5 * sigma_t, 1.0 / sfreq)
    return (1.0 / (sigma_t * np.sqrt(np.pi))) * np.exp(-(t ** 2) / (2 * sigma_t ** 2)) * np.exp(
        1j * 2 * np.pi * f * t
    )


def cwt_morlet(
    data: np.ndarray, sfreq: float, fmin: float = 2.0, fmax: float = 80.0,
    n_freqs: int = 30, n_cycles: float = 7.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (freqs, tfr) — tfr shape (n_ch, n_freqs, n_t)."""
    freqs = np.logspace(np.log10(fmin), np.log10(fmax), n_freqs)
    n_ch, n_t = data.shape
    tfr = np.zeros((n_ch, n_freqs, n_t), dtype=np.complex64)
    for fi, f in enumerate(freqs):
        w = _morlet_wavelet(f, sfreq, n_cycles)
        for ci in range(n_ch):
            tfr[ci, fi, :] = np.convolve(data[ci], w, mode="same")
    return freqs, tfr
```

### EasyBCI-Adapted

```python
from typing import Any, Dict
import time
import numpy as np
from easybci_lib.tools.neural_processing.operator_errors import EasyBCIOperatorError
from easybci_lib.tools.neural_processing.preprocess.step_cache import record_step_elapsed


def operator_cwt_morlet(
    data_dict: Dict[str, Any], *,
    fmin: float = 2.0, fmax: float = 80.0, n_freqs: int = 30, n_cycles: float = 7.0,
) -> Dict[str, Any]:
    """Complex Morlet CWT.

    Parameters
    ----------
    data_dict : dict
        OperatorIO.
    fmin, fmax : float
        Frequency range (default 2 / 80 Hz).
    n_freqs : int
        Number of log-spaced frequencies (default 30).
    n_cycles : float
        Wavelet cycle count (default 7.0).

    Returns
    -------
    dict
        OperatorIO; continuous `data` unchanged; meta populated:
        - `meta["tfr"]`: complex64 (n_ch, n_freqs, n_t)
        - `meta["tfr_freqs"]`: ndarray (n_freqs,)

    Raises
    ------
    EasyBCIOperatorError
        recoverable=True for too-short recordings.

    Modality coverage
    -----------------
    EEG / MEG / sEEG / ECoG / LFP: yes.
    Spike: forbidden.

    References
    ----------
    Tallon-Baudry & Bertrand 1999; Cohen 2014.
    """
    sfreq = float(data_dict["frequency"])
    n_t = data_dict["data"].shape[-1]
    if n_t / sfreq < 5 * n_cycles / fmin:
        raise EasyBCIOperatorError(
            operator="cwt_morlet",
            reason=f"recording {n_t/sfreq:.2f}s < {5*n_cycles/fmin:.1f}s required for fmin={fmin}",
            recoverable=True, fallback_step=f"cwt_morlet:{fmin*2},{fmax},{n_freqs},{n_cycles}",
        )

    t0 = time.monotonic()
    freqs = np.logspace(np.log10(fmin), np.log10(fmax), n_freqs)
    data = data_dict["data"]
    n_ch = data.shape[0]
    tfr = np.zeros((n_ch, n_freqs, n_t), dtype=np.complex64)
    for fi, f in enumerate(freqs):
        sigma_t = n_cycles / (2 * np.pi * f)
        t_arr = np.arange(-3.5 * sigma_t, 3.5 * sigma_t, 1.0 / sfreq)
        w = (1.0 / (sigma_t * np.sqrt(np.pi))) * np.exp(-(t_arr ** 2) / (2 * sigma_t ** 2)) * np.exp(
            1j * 2 * np.pi * f * t_arr
        )
        for ci in range(n_ch):
            tfr[ci, fi, :] = np.convolve(data[ci], w, mode="same")
    elapsed = time.monotonic() - t0

    out = dict(data_dict)
    out["elapsed_s"] = elapsed
    out["meta"] = {
        **out.get("meta", {}),
        "tfr": tfr,
        "tfr_freqs": freqs,
        "cwt_morlet": {"fmin": fmin, "fmax": fmax, "n_freqs": n_freqs, "n_cycles": n_cycles},
    }
    record_step_elapsed("cwt_morlet", elapsed, (data_dict.get("meta") or {}).get("step_cache_key"))
    return out
```

## References

1. Tallon-Baudry, C., & Bertrand, O. (1999). *Oscillatory gamma activity
   in humans and its role in object representation*. Trends in Cognitive
   Sciences 3(4): 151–162. doi:10.1016/S1364-6613(99)01299-1 — Morlet CWT
   in neuroscience.
2. Cohen, M. X. (2014). *Analyzing Neural Time Series Data*. MIT Press —
   wavelet practical guide.
3. Torrence, C., & Compo, G. P. (1998). *A practical guide to wavelet
   analysis*. Bulletin of the American Meteorological Society 79(1):
   61–78. doi:10.1175/1520-0477(1998)079<0061:APGTWA>2.0.CO;2.
