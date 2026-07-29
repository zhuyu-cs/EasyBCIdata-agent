---
name: anti_alias_resample
description: "Anti-alias FIR filter followed by integer-factor decimation — explicit pre-roll for cross-band pipelines"
layer: L3
group: filter
metadata:
  tags: [operator, filter, resample, anti_alias, decimate, downsample]
  modalities: [eeg, meg, seeg, ecog, fnirs, lfp, spike]
  step_string: "anti_alias_resample"
  analysis_goal_allowed: [classification, source_localization, feature_extraction, clinical_screening, exploratory, generic, connectivity, phase_amplitude_coupling, online_inference]
  analysis_goal_forbidden: []
---
# Anti-Alias Resample

## Function

Two-step downsampling: apply a low-pass FIR anti-alias filter at the
target Nyquist, then decimate by an integer factor. Unlike the generic
`resample` operator (which fuses both steps in a single library call),
this operator exposes the anti-alias cut-off explicitly — important when
the downstream analysis crosses bands (e.g., AP / LF dual stream on
Neuropixels) and you want to control the transition band manually.

Input / Output: `(n_channels, n_times)` → `(n_channels, n_times // factor)`,
`frequency` updated to `frequency / factor`.

## Algorithm & Math

- Low-pass FIR with cut-off `f_c = target_rate · 0.45` (with default 10%
  guard below Nyquist) using `scipy.signal.firwin` + Hamming window.
- Stop-band attenuation ≥ 60 dB by construction (Hamming = 53 dB; the
  longer filter length used here gains the extra margin).
- After the FIR, `data[:, ::factor]` selects every Nth sample. The phase
  is preserved (FIR is linear-phase).

Choice vs. `scipy.signal.resample_poly`:

| Path | When |
|---|---|
| `firwin + decimate` (this op) | Integer factor; cross-band pipelines where you need the explicit anti-alias breakpoint. |
| `resample_poly` (generic resample) | Rational factor (e.g., 2048 → 1000); single library call; cleaner code. |

## Parameter Format & Defaults

`anti_alias_resample:{target_rate}` — comma-separated:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_rate` | float | required | Output sampling rate (Hz). |
| `numtaps` (kw) | int | auto | FIR length; auto = `max(101, round(50 * source / target))`. |
| `guard_fraction` (kw) | float | 0.10 | Cut-off as `(target/2) · (1 − guard_fraction)`. |

Examples:

- `anti_alias_resample:1000` — Neuropixels AP 30 kHz → 1 kHz LFP-band-grade.
- `anti_alias_resample:256` — EEG 1024 Hz → 256 Hz (ML model input rate).

## Modality-Specific Considerations

| Modality | Source rate | Target rate | Notes |
|---|---|---|---|
| EEG | 1024 / 2048 | 256 | Standard ML target. |
| MEG | 2 kHz | 1 kHz | Halve for offline analysis. |
| sEEG | 2 kHz | 500 | When only LFP-band needed. |
| Spike AP | 30 kHz | 30 kHz (no-op) | Don't downsample spike-band; threshold needs full rate. |
| LFP | 2.5 kHz | 1 kHz | Common downsample. |

### Hard exclusions

- Spike AP-band data should **not** be downsampled — waveform timing
  precision lost; raise `EasyBCIOperatorError(recoverable=False)` when
  `target_rate < 20 kHz` and the input is flagged spike-AP modality.

## When to Use / NOT to Use

**Use** when: target rate is an integer divisor; you need an explicit
anti-alias cut-off (cross-band).

**Don't use** when: factor is non-integer (use `resample`); target is
spike AP band; you've already low-passed at the target Nyquist (no-op).

## Constraints & Ordering

- **Integer factor only.** Non-integer → falls back to `resample`.
- Apply **before** `bandpass` whose `high` ≤ target Nyquist.
- **After** `notch` — line-noise harmonics must be removed before
  the anti-alias cuts above target Nyquist (otherwise harmonics fold).

## Failure Modes & Detection

| Failure | Symptom | Detection |
|---------|---------|-----------|
| Non-integer factor | factor = source / target is non-integer | Pre-check; raise recoverable with `fallback_step="resample:{target}"`. |
| Spike-band downsample | sfreq dropping below 20 kHz on spike modality | Hard error; recoverable=False. |
| Aliased output | PSD shows energy folding above target Nyquist | Compare PSD pre/post; flag if folded energy > 10% of in-band. |

## Common Issues

- **"My PSD shows weird spectral wrap-around."** Bandpass after the
  resample (instead of before) lets near-Nyquist content alias. Move
  anti-alias before any bandpass that touches the target Nyquist.

## Reference Implementation

### Standalone

```python
from __future__ import annotations
import numpy as np
from scipy.signal import firwin, filtfilt


def anti_alias_resample(
    data: np.ndarray,
    sfreq: float,
    target_rate: float,
    *,
    numtaps: int | None = None,
    guard_fraction: float = 0.10,
) -> tuple[np.ndarray, float]:
    """Anti-alias then decimate.

    Returns
    -------
    out : ndarray (n_channels, n_times // factor), float32
    new_sfreq : float
    """
    factor = sfreq / target_rate
    if abs(factor - round(factor)) > 1e-6:
        raise ValueError(
            f"anti_alias_resample: factor {factor:.3f} not integer; "
            f"use polyphase resample"
        )
    factor = int(round(factor))
    if numtaps is None:
        numtaps = max(101, int(round(50 * factor)))
        numtaps |= 1  # odd
    cutoff = (target_rate / 2.0) * (1.0 - guard_fraction)
    h = firwin(numtaps, cutoff, fs=sfreq, window="hamming")
    filtered = filtfilt(h, [1.0], data, axis=-1, padtype="even")
    out = filtered[..., ::factor].astype(data.dtype, copy=False)
    return out, sfreq / factor
```

### EasyBCI-Adapted

```python
from typing import Any, Dict
import time
import numpy as np
from easybci_lib.tools.neural_processing.operator_errors import EasyBCIOperatorError
from easybci_lib.tools.neural_processing.preprocess.step_cache import record_step_elapsed


def operator_anti_alias_resample(
    data_dict: Dict[str, Any],
    *,
    target_rate: float,
    numtaps: int | None = None,
    guard_fraction: float = 0.10,
) -> Dict[str, Any]:
    """Anti-alias FIR + integer decimate.

    Parameters
    ----------
    data_dict : dict
        OperatorIO.
    target_rate : float
        Output sampling rate (Hz).
    numtaps : int or None
        FIR length (default auto).
    guard_fraction : float
        Cut-off guard below Nyquist (default 0.10).

    Returns
    -------
    dict
        OperatorIO with `data` downsampled, `frequency` updated.

    Raises
    ------
    EasyBCIOperatorError
        recoverable=True for non-integer factor; recoverable=False if
        downsampling spike-AP below 20 kHz.

    Modality coverage
    -----------------
    EEG / MEG / sEEG / ECoG / fNIRS / LFP: yes.
    Spike AP-band (sfreq >= 20 kHz to < 20 kHz): forbidden.

    References
    ----------
    Oppenheim & Schafer (1989) DSP textbook; scipy.signal.firwin docs.
    """
    src = float(data_dict["frequency"])
    modality = (data_dict.get("meta") or {}).get("modality", "").lower()
    if modality == "spike" and target_rate < 20_000:
        raise EasyBCIOperatorError(
            operator="anti_alias_resample",
            reason=f"refusing to downsample spike-AP to {target_rate} Hz (< 20 kHz)",
            recoverable=False,
        )
    factor = src / target_rate
    if abs(factor - round(factor)) > 1e-6:
        raise EasyBCIOperatorError(
            operator="anti_alias_resample",
            reason=f"non-integer factor {factor:.3f}; use generic resample",
            recoverable=True,
            fallback_step=f"resample:{int(target_rate)}",
        )

    t0 = time.monotonic()
    from scipy.signal import firwin, filtfilt
    factor_int = int(round(factor))
    if numtaps is None:
        numtaps = max(101, int(round(50 * factor_int))) | 1
    cutoff = (target_rate / 2.0) * (1.0 - guard_fraction)
    h = firwin(numtaps, cutoff, fs=src, window="hamming")
    filtered = filtfilt(h, [1.0], data_dict["data"], axis=-1, padtype="even")
    new_data = filtered[..., ::factor_int].astype(data_dict["data"].dtype, copy=False)
    elapsed = time.monotonic() - t0

    out = dict(data_dict)
    out["data"] = new_data
    out["frequency"] = src / factor_int
    out["duration"] = new_data.shape[-1] / out["frequency"]
    out["elapsed_s"] = elapsed
    out["meta"] = {**out.get("meta", {}), "anti_alias_resample": {
        "target_rate": target_rate, "numtaps": numtaps, "factor": factor_int,
    }}
    record_step_elapsed(
        "anti_alias_resample", elapsed,
        (data_dict.get("meta") or {}).get("step_cache_key"),
    )
    return out
```

## References

1. Oppenheim, A. V., & Schafer, R. W. (1989). *Discrete-Time Signal
   Processing*. Prentice Hall — anti-alias decimation reference.
2. Vaidyanathan, P. P. (1990). *Multirate Systems and Filter Banks*.
   Prentice Hall — polyphase / multirate theory.
3. de Cheveigné, A. & Nelken, I. (2019). *Filters: When, Why, and How
   (Not) to Use Them*. Neuron 102: 280–293.
   doi:10.1016/j.neuron.2019.02.039 — anti-alias breakpoint and
   distortion guidance.
