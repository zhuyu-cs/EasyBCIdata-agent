---
name: stft
description: "Short-Time Fourier Transform — complex spectrogram for time-frequency analysis"
layer: L3
group: spectral
metadata:
  tags: [operator, spectral, stft, time_frequency, fourier]
  modalities: [eeg, meg, seeg, ecog, lfp]
  step_string: "stft"
  analysis_goal_allowed: [classification, feature_extraction, exploratory, generic, connectivity, phase_amplitude_coupling]
  analysis_goal_forbidden: []
---
# Short-Time Fourier Transform (STFT)

## Function

Computes the complex-valued STFT spectrogram (magnitude + phase) of
continuous neural data. Distinguished from `spectrogram` (magnitude
squared) by returning the complex array — required for phase-based
analyses (PLV, PAC carrier extraction, complex demodulation).

Input / Output: `(n_channels, n_times)` → `(n_channels, n_freqs, n_segments)` complex64.

## Algorithm & Math

`scipy.signal.stft(data, fs=sfreq, nperseg=window, noverlap=overlap)`,
returning complex `(channel, freq, time)`.

- Window: Hann by default (good main-lobe / side-lobe trade-off).
- nperseg: 1 s @ source sfreq → frequency resolution `Δf = sfreq / nperseg`.
- Hop: `nperseg − noverlap`; default 50% overlap.

## Parameter Format & Defaults

`stft:{window_s},{overlap_frac}`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `window_s` | float | 1.0 | Window length in seconds. |
| `overlap_frac` | float | 0.5 | Overlap fraction. |
| `window_type` (kw) | str | "hann" | scipy window name. |

## Modality-Specific Considerations

| Modality | window_s | Notes |
|---|---|---|
| EEG (mu/beta) | 0.5–1.0 | Δf = 1–2 Hz; matches band granularity. |
| EEG (gamma) | 0.25–0.5 | Δf = 2–4 Hz; trade off for time resolution. |
| sEEG / ECoG | 0.25 | High-gamma analysis. |
| MEG | 0.5–1.0 | Same as EEG. |
| LFP | 1.0–2.0 | Slow oscillations. |
| Spike AP | n/a | Use waveform-domain ops, not STFT. |

## When to Use / NOT to Use

**Use** when: time-frequency representation needed; PAC carrier
extraction; complex demodulation; computing instantaneous phase across
multiple frequencies.

**Don't use** when: a single band suffices (use `bandpass + hilbert`);
PSD averaged across time only (use `multitaper_psd`); spike waveform
domain.

## Constraints & Ordering

- Apply **after** bandpass / notch.
- Window length sets the time-frequency trade-off: long window =
  fine `Δf` / coarse `Δt`.

## Failure Modes & Detection

| Failure | Symptom | Detection |
|---|---|---|
| Window > recording | STFT returns 1 segment, no time info. | Check `n_t > 2 * nperseg`; raise recoverable. |
| Δf coarser than needed | Bin width too large to resolve target. | If `target_resolution < sfreq / nperseg` warn and suggest longer window. |

## Common Issues

- **"My phase looks random."** Edges of segments cause phase wraps; trim
  the first/last 10% of segments.

## Reference Implementation

### Standalone

```python
from __future__ import annotations
import numpy as np
from scipy.signal import stft as _stft


def stft(
    data: np.ndarray, sfreq: float, window_s: float = 1.0, overlap_frac: float = 0.5
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Complex STFT spectrogram. Returns (freqs, times, Z)."""
    nperseg = max(8, int(window_s * sfreq))
    noverlap = int(overlap_frac * nperseg)
    f, t, Z = _stft(data, fs=sfreq, nperseg=nperseg, noverlap=noverlap, axis=-1)
    return f, t, Z.astype(np.complex64)
```

### EasyBCI-Adapted

```python
from typing import Any, Dict
import time
import numpy as np
from easybci_lib.tools.neural_processing.operator_errors import EasyBCIOperatorError
from easybci_lib.tools.neural_processing.preprocess.step_cache import record_step_elapsed


def operator_stft(
    data_dict: Dict[str, Any], *, window_s: float = 1.0, overlap_frac: float = 0.5,
    window_type: str = "hann",
) -> Dict[str, Any]:
    """Short-Time Fourier Transform.

    Parameters
    ----------
    data_dict : dict
        OperatorIO.
    window_s : float
        Window length in seconds (default 1.0).
    overlap_frac : float
        Overlap fraction (default 0.5).
    window_type : str
        Window name (default "hann").

    Returns
    -------
    dict
        OperatorIO with continuous `data` unchanged; meta populated:
        - `meta["stft_Z"]`: complex64 (n_ch, n_freqs, n_segments)
        - `meta["stft_freqs"]`: ndarray (n_freqs,)
        - `meta["stft_times"]`: ndarray (n_segments,)

    Raises
    ------
    EasyBCIOperatorError
        recoverable=True if window > recording.

    Modality coverage
    -----------------
    EEG / MEG / sEEG / ECoG / LFP: yes.
    Spike: forbidden (use waveform-domain ops).

    References
    ----------
    Cohen 2014 — Analyzing Neural Time Series Data; scipy.signal.stft docs.
    """
    sfreq = float(data_dict["frequency"])
    nperseg = max(8, int(window_s * sfreq))
    n_t = data_dict["data"].shape[-1]
    if n_t < 2 * nperseg:
        raise EasyBCIOperatorError(
            operator="stft",
            reason=f"recording {n_t/sfreq:.2f}s < 2 windows of {window_s}s",
            recoverable=True,
            fallback_step=f"stft:{max(0.1, n_t / sfreq / 4):.2f},{overlap_frac}",
        )

    t0 = time.monotonic()
    from scipy.signal import stft as _stft
    noverlap = int(overlap_frac * nperseg)
    f, t_arr, Z = _stft(
        data_dict["data"], fs=sfreq, window=window_type,
        nperseg=nperseg, noverlap=noverlap, axis=-1,
    )
    elapsed = time.monotonic() - t0

    out = dict(data_dict)
    out["elapsed_s"] = elapsed
    out["meta"] = {
        **out.get("meta", {}),
        "stft_Z": Z.astype(np.complex64),
        "stft_freqs": f,
        "stft_times": t_arr,
        "stft": {"window_s": window_s, "overlap_frac": overlap_frac, "window_type": window_type},
    }
    record_step_elapsed("stft", elapsed, (data_dict.get("meta") or {}).get("step_cache_key"))
    return out
```

## References

1. Cohen, M. X. (2014). *Analyzing Neural Time Series Data: Theory and
   Practice*. MIT Press — comprehensive STFT / time-frequency reference.
2. Bruns, A. (2004). *Fourier-, Hilbert- and wavelet-based signal
   analysis: are they really different approaches?* Journal of
   Neuroscience Methods 137(2): 321–332.
   doi:10.1016/j.jneumeth.2004.03.002 — STFT vs alternatives.
