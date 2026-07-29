---
name: bandpass_filter
description: "Band-pass filter to isolate frequency range of interest with Nyquist guard"
layer: L3
group: filter
metadata:
  tags: [operator, filter, bandpass, highpass, lowpass]
  modalities: [eeg, seeg, ecog, meg, fnirs]
  step_string: "bandpass"
  analysis_goal_allowed: [classification, source_localization, feature_extraction, clinical_screening, exploratory, generic, connectivity, phase_amplitude_coupling, online_inference]
  analysis_goal_forbidden: []
---
# Band-pass Filter

> **Gold-standard reference template (T2 Sub-phase A.1).** All other operator
> SKILL.md files should mirror this section structure: Function → Algorithm
> & Math → Parameter Format & Defaults → Modality-Specific Considerations →
> When to Use / NOT to Use → Constraints & Ordering → Failure Modes &
> Detection → Common Issues → Reference Implementation → References.

## Function

FIR (or optionally IIR) band-pass filter that isolates frequencies within a
specified `[low, high]` range, removing slow drifts below ``low`` and
high-frequency noise above ``high``.  Includes automatic Nyquist guard:
when ``high >= sfreq/2`` the low-pass arm is disabled with a logged warning
rather than silently returning NaN.

**Input / Output tensor shape.** Operates per-channel on continuous
recordings: ``(n_channels, n_times)`` → ``(n_channels, n_times)``.  The
sampling rate is unchanged; only spectral content is reshaped.  Edge
samples are zero-phase compensated by MNE's default ``filter_length='auto'``
+ ``phase='zero-double'`` so the filter does not introduce a group delay.

## Algorithm & Math

### Transfer function

A linear-phase FIR designed by MNE's default
``firwin`` window method has impulse response

```
h[n] = w[n] · sinc(2 f_c (n - N/2)) − sinc(2 f_l (n - N/2))
```

with Hamming window ``w[n]``, low cutoff ``f_l``, high cutoff ``f_c``, and
length ``N`` chosen by ``mne.filter.next_fast_len`` to give a transition
band of ``min(max(low/4, 2), 50)`` Hz on the high-pass side and
``min(max(high/4, 2), 50)`` Hz on the low-pass side.  Stop-band attenuation
is ≥ 53 dB by construction.

When ``method="iir"`` is used, the filter is a 4th-order Butterworth
band-pass with bilinear-transform coefficients

```
H(z) = b(z) / a(z),     order 4 per arm,      transition slope −24 dB/oct
```

applied with ``scipy.signal.sosfiltfilt`` for zero phase.

### FIR vs IIR trade-off

| Property | FIR (default) | IIR (Butter, order 4) |
|---|---|---|
| Phase | Linear (no group delay) | Non-linear; warps phase |
| Computational cost | O(N · L) per channel | O(L) per channel (very cheap) |
| Edge artefacts | Mild ringing at filter length boundary | Settling on first ~2/f_l samples |
| Transition steepness | Configurable; long FIR → sharp | 4th-order ≈ −24 dB/oct |
| Phase-sensitive analyses (PAC, CFC, connectivity) | **Required** | **Forbidden** |
| Online inference | Pre-roll required (causal FIR alt.) | **Required** (causal, low latency) |

The default in EasyBCI is FIR for offline; the only legitimate use of IIR
is the ``online_inference`` analysis_goal where causal filtering avoids
look-ahead.

## Parameter Format & Defaults

`bandpass:{low},{high}` — comma-separated, both optional but not both
empty:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| ``low`` | float or empty | None | High-pass cutoff (Hz). Empty disables high-pass arm. |
| ``high`` | float or empty | None | Low-pass cutoff (Hz). Empty disables low-pass arm. |
| ``method`` (kw) | "fir" \| "iir" | "fir" | FIR for analysis, IIR only for online_inference. |
| ``order`` (kw, IIR only) | int | 4 | Butterworth order per arm. |

Examples:
- `bandpass:1,40` — standard EEG analysis band.
- `bandpass:0.5,100` — wider band for sEEG/ECoG.
- `bandpass:,40` — low-pass only (drop high-frequency EMG without
  touching DC/drift).
- `bandpass:1,` — high-pass only (remove drift; keep all high freq).

See ``parameter_uncertainty/bandpass_filter.yaml`` for per-paradigm
empirical defaults with citations.

## Modality-Specific Considerations

Bandpass is one of the most modality-agnostic operators in the registry,
but cutoff selection is **paradigm- and modality-dependent**.  Sensible
defaults:

| Modality | Paradigm | Band (Hz) | Rationale |
|----------|----------|-----------|-----------|
| EEG | Motor Imagery | 0.5–40 | Preserves mu (8–12), beta (13–30); drops slow drifts. |
| EEG | P300 / ERP | 0.1–30 | Low high-pass preserves slow ERP components (e.g., CNV). Aggressive high-pass distorts P3 amplitude — see Tanner et al. 2015. |
| EEG | SSVEP | 5–45 | Stimulation frequencies are typically 5–40 Hz; harmonics up to 80 Hz. |
| EEG | Sleep | 0.3–35 | Preserves slow oscillations (0.5–4) + spindles (12–14) + theta. |
| EEG | Emotion | 1–50 | All bands except gamma; aggressive high-pass distorts theta-asymmetry. |
| sEEG | Epilepsy / general | 1–200 (or 1–500) | High-gamma (70–150) is the canonical functional marker; HFO (80–500) is the IED biomarker. |
| ECoG | General | 1–200 | High-gamma carries spatially-precise functional information. |
| MEG | General | 1–100 | Wider than EEG; less muscle artifact above 60 Hz. |
| fNIRS | General | 0.01–0.5 | Hemodynamic response is very slow; passband on the order of the HRF time constant. |
| Spike (LFP only) | General | 0.5–300 | Local field potential band; spike waveforms (300–6000) handled by `spike_sorting`. |
| Spike (waveform) | — | **Forbidden** | Spike waveforms are sub-millisecond — bandpass would mangle the action-potential shape. Use ``spike_sorting`` directly. |

**Hard exclusions.** Bandpass cannot be applied to:
- Spike waveforms (use spike_sorting instead).
- fNIRS data **before** the modified Beer–Lambert transform (filter HbO/HbR,
  not raw optical density).
- Signals where ``low`` or ``high`` is below the recording's intrinsic
  high-pass (e.g. AC-coupled amplifiers with 0.5 Hz hardware cutoff and
  ``low=0.1`` requested).

## When to Use / NOT to Use

**Use** when:
- Removing slow drifts (< 0.5 Hz) from electrode movement, sweat, or thermal
  drift in the amplifier.
- Removing high-frequency EMG / muscle artefact above paradigm-relevant
  bands (typically > 40–50 Hz for scalp EEG).
- Isolating a frequency band for paradigm-specific feature extraction
  (e.g., 8–12 Hz band-power for mu suppression in motor imagery).
- Anti-aliasing before a downstream `resample` step (the resample
  operator runs its own anti-alias internally, but applying bandpass
  first sharpens the cutoff).

**Don't use** when:
- The downstream task is phase-amplitude coupling, phase-locking, or any
  cross-frequency analysis with **wide-band** input — bandpass narrows
  the band before Hilbert, which is *required* for PAC but the cutoffs
  must match the PAC frequency pair, not the EEG analysis defaults.
  See `phase_amplitude_coupling.md`.
- The data is already pre-filtered at acquisition and the requested band
  is wider than the hardware band — no-op at best, ringing at worst.
- The downstream task is online inference with strict latency budget —
  use the IIR variant or a causal FIR with explicit pre-roll.

## Constraints & Ordering

- **Nyquist guard.** If ``high >= sfreq/2``, the low-pass arm is silently
  dropped (logged at WARNING).  ``low`` is also checked against
  ``sfreq/2`` for symmetry but a high ``low`` is operationally fine.
- **``low < high``.** Reversed cutoffs raise ``ValueError`` from MNE.
- **At least one cutoff must be specified.** ``bandpass:,`` is rejected.
- **Narrow band < 2 Hz width** may cause ringing.  Use a longer
  ``filter_length`` or a wavelet-based approach (`cwt_morlet`) when
  narrow-band extraction is needed.

**Ordering relative to other operators.**

```
notch_filter → bandpass_filter → drop_bads → ICA → resample → ...
```

- Apply **after** ``notch_filter`` so power-line peaks are gone before the
  bandpass shapes the spectrum.  Otherwise the notch's residual
  side-lobes survive into your analysis band.
- Apply **before** ``ICA`` — ICA convergence is meaningfully better on
  band-limited data (FastICA likelihood is shaped by the spectral
  envelope).
- Apply **before** ``resample`` — bandpass is a stronger anti-alias than
  resample's internal filter when your target rate is close to your
  upper cutoff.
- Apply **after** ``bipolar_ref`` for sEEG / ECoG — bipolar reference
  flattens common-mode noise; bandpass cleans what remains.

## Failure Modes & Detection

| Failure | Symptom | Detection |
|---------|---------|-----------|
| Cutoffs above Nyquist | Output looks unfiltered above ``sfreq/2`` | Compute PSD; expect ≥ 53 dB attenuation in stop-band — if absent, the lowpass was silently disabled. |
| Reversed cutoffs (``low > high``) | ValueError raised by MNE | Caught at parse; surfaces ``EasyBCIOperatorError(recoverable=True)`` with ``fallback_step="bandpass:1,40"``. |
| Narrow band on short data | Ringing at signal start/end | First/last ``filter_length / sfreq`` seconds show oscillation. Reject when ``duration < 4·filter_length / sfreq``. |
| Bandpass on AC-coupled data with low < hardware cutoff | High-pass becomes a no-op | Compare PSD slope below ``low`` before/after — a non-flat slope means the hardware filter dominates. |
| Bandpass before notch | Power-line peak survives | Compute PSD; peak at 50 / 60 Hz with > 6 dB prominence after the bandpass. |
| Bandpass on spike waveforms | Action potentials become ringing oscillations | Mean spike-triggered amplitude drops > 50%; spike-detection thresholds break. Reject upstream by routing waveforms to `spike_sorting`. |

Auto-detection helper:

```python
import numpy as np
from scipy.signal import welch

def diagnose_bandpass(data, sfreq, low, high):
    """Return (status, details) — 'ok' / 'nyquist_exceeded' / 'unfiltered'."""
    f, p = welch(data, sfreq, nperseg=min(2048, data.shape[-1]))
    if high is not None and high >= sfreq / 2:
        return "nyquist_exceeded", {"requested_high": high, "nyquist": sfreq / 2}
    in_band = (f >= (low or 0)) & (f <= (high or sfreq / 2))
    if in_band.any():
        in_band_db = 10 * np.log10(np.median(p[..., in_band]) + 1e-20)
        out_band_db = 10 * np.log10(np.median(p[..., ~in_band]) + 1e-20)
        if (in_band_db - out_band_db) < 20:
            return "unfiltered", {"in_db": in_band_db, "out_db": out_band_db}
    return "ok", {}
```

## Common Issues

- **"My P300 looks shifted in time."** A high ``low`` (≥ 1 Hz) high-pass
  introduces phase distortion in the < 5 Hz range that visibly shifts
  ERP latencies. Use ``low=0.1`` or zero-phase IIR.
- **"Filter takes forever on long sEEG."** FIR filter length scales with
  ``sfreq / low``.  At 2048 Hz with ``low=0.5``, the filter is ≥ 8000
  samples per channel — switch to ``method="iir"`` for offline analysis
  if you don't need linear phase, or process in `mne.io.read_raw_*`
  chunks via `preload=False` + `filter()` resume.
- **"My PSD has a 50 Hz spike after bandpass:1,80."** Bandpass alone does
  not remove discrete narrow-band peaks; chain ``notch_filter:50`` first.

## Reference Implementation

### Standalone (no EasyBCI dependencies)

```python
"""Standalone band-pass filter — drop into any environment."""
from __future__ import annotations

import logging
import numpy as np
import mne

logger = logging.getLogger(__name__)


def bandpass_filter(
    data: np.ndarray,
    sfreq: float,
    low: float | None,
    high: float | None,
    *,
    method: str = "fir",
    order: int = 4,
) -> np.ndarray:
    """Band-pass filter ``data``.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
        Continuous neural data, float32 or float64.
    sfreq : float
        Sampling rate (Hz).
    low, high : float or None
        High-pass / low-pass cutoffs.  Both may be None individually but
        not simultaneously.
    method : {"fir", "iir"}
        FIR is the default; IIR is for online_inference only.

    Returns
    -------
    ndarray
        Same shape and dtype as input.
    """
    if low is None and high is None:
        raise ValueError("bandpass_filter: at least one of low/high must be set")
    if low is not None and high is not None and low >= high:
        raise ValueError(f"bandpass_filter: low ({low}) must be < high ({high})")
    if high is not None and high >= sfreq / 2:
        logger.warning(
            "bandpass_filter: high=%.2f >= Nyquist=%.2f; disabling lowpass",
            high, sfreq / 2,
        )
        high = None

    info = mne.create_info(
        [f"ch{i}" for i in range(data.shape[0])], sfreq, ch_types="eeg",
    )
    raw = mne.io.RawArray(data.astype(np.float64, copy=False), info, verbose="ERROR")
    raw.filter(l_freq=low, h_freq=high, method=method, iir_params={"order": order} if method == "iir" else None, verbose="ERROR")
    return raw.get_data().astype(data.dtype, copy=False)
```

### EasyBCI-Adapted (in-framework)

The framework adapter wraps the standalone implementation in the
``data_dict`` schema and emits step elapsed time for the four-stage
progress tracker:

```python
from typing import Any, Dict
import numpy as np
from easybci_lib.tools.neural_processing.preprocess.step_cache import record_step_elapsed
from easybci_lib.tools.neural_processing.operator_errors import EasyBCIOperatorError

def operator_bandpass_filter(
    data_dict: Dict[str, Any],
    *,
    low: float | None = None,
    high: float | None = None,
    method: str = "fir",
    order: int = 4,
) -> Dict[str, Any]:
    """EasyBCI-adapted bandpass — see Standalone above for parameter semantics.

    Modality coverage
    -----------------
    EEG: yes / MEG: yes / sEEG: yes / ECoG: yes / fNIRS: caveat (apply
    AFTER modified_beer_lambert) / spike: NO (waveforms only — use
    spike_sorting; LFP band is fine).
    """
    import time
    if data_dict.get("modality") == "spike_waveform":
        raise EasyBCIOperatorError(
            operator="bandpass_filter",
            reason="bandpass would distort action-potential waveforms; route through spike_sorting instead.",
            recoverable=False,
        )
    t0 = time.monotonic()
    try:
        new_data = bandpass_filter(
            data_dict["data"],
            data_dict["frequency"],
            low, high,
            method=method, order=order,
        )
    except ValueError as exc:
        raise EasyBCIOperatorError(
            operator="bandpass_filter",
            reason=str(exc),
            recoverable=True,
            fallback_step=f"bandpass:1,40",
        ) from exc
    elapsed = time.monotonic() - t0
    out = dict(data_dict)
    out["data"] = new_data
    out["elapsed_s"] = elapsed
    out["meta"] = {**out.get("meta", {}), "bandpass": {"low": low, "high": high, "method": method}}
    record_step_elapsed("bandpass_filter", elapsed, data_dict.get("meta", {}).get("step_cache_key"))
    return out
```

## References

1. Widmann, A., Schröger, E., & Maess, B. (2015). *Digital filter design
   for electrophysiological data — a practical approach*. Journal of
   Neuroscience Methods 250: 34–46. doi:10.1016/j.jneumeth.2014.08.002 —
   the canonical "what cutoffs to use" reference.
2. Tanner, D., Morgan-Short, K., & Luck, S. J. (2015). *How inappropriate
   high-pass filters can produce artifactual effects and incorrect
   conclusions in ERP studies of language and cognition*. Psychophysiology
   52(8): 997–1009. doi:10.1111/psyp.12437 — proves why ``low=0.1`` is
   the safe default for ERP work.
3. Gramfort, A. et al. (2013). *MEG and EEG data analysis with MNE-Python*.
   Frontiers in Neuroscience 7: 267. doi:10.3389/fnins.2013.00267 — MNE
   filter implementation reference.
4. de Cheveigné, A. & Nelken, I. (2019). *Filters: When, Why, and How
   (Not) to Use Them*. Neuron 102(2): 280–293.
   doi:10.1016/j.neuron.2019.02.039 — practical guidance on when filtering
   distorts the very effect you're trying to measure.
5. Cole, S. & Voytek, B. (2019). *Cycle-by-cycle analysis of neural
   oscillations*. Journal of Neurophysiology 122(2): 849–861.
   doi:10.1152/jn.00273.2019 — caution: bandpass + Hilbert shapes
   waveform-asymmetry estimates; consider cycle-based methods for
   non-sinusoidal rhythms.
