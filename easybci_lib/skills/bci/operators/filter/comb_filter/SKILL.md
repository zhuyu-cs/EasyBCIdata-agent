---
name: comb_filter
description: "Multi-harmonic comb filter — remove power-line fundamental + all harmonics in one pass"
layer: L3
group: filter
metadata:
  tags: [operator, filter, comb, powerline, harmonics]
  modalities: [eeg, meg, seeg, ecog, fnirs]
  step_string: "comb"
  analysis_goal_allowed: [classification, feature_extraction, clinical_screening, exploratory, generic, online_inference]
  analysis_goal_forbidden: [phase_amplitude_coupling]
---
# Comb Filter

## Function

A single-pass digital comb filter that creates equispaced notches at a
fundamental frequency (e.g., 50 Hz) and **all** its integer harmonics
up to Nyquist (100, 150, 200, ... 50 · k Hz). Cheaper than chaining 5–10
`notch_filter` calls, and the harmonic alignment guarantees that the
notch widths are uniformly Q-bound.

Input / Output: `(n_channels, n_times)` → same shape.

## Algorithm & Math

A digital comb (notch-mode) with fundamental `f0` and N harmonics is

```
H(z) = (1 − r^N · z^(−N)) / (1 − r · z^(−1))     # band-stop variant
```

implemented via `scipy.signal.iircomb(w0=f0/(sfreq/2), Q=Q, ftype="notch")`.
The `Q` parameter sets the notch width — higher Q is a sharper notch
(less neighbour bleeding) but rings longer.

Zero-phase is enforced via `filtfilt`.

## Parameter Format & Defaults

`comb:{f0},{Q}`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `f0` | float | 50.0 | Fundamental frequency (50 EU / 60 US). |
| `Q` | float | 30.0 | Notch sharpness; 20 = wider, 50 = tighter. |
| `max_harmonics` (kw) | int | None | Stop after N harmonics; default = up to Nyquist. |

## Modality-Specific Considerations

| Modality | Q | Notes |
|---|---|---|
| EEG | 30 | Standard line-noise removal; minimal alpha/beta impact. |
| MEG | 30 | SQUID baseline; line noise dominant. |
| sEEG | 25 | High-gamma analysis: keep Q moderate so 100 / 150 Hz notch doesn't drink into the analysis band. |
| fNIRS | n/a | Hemodynamic band has no line noise. |

Hard exclusion for `phase_amplitude_coupling`: comb's harmonic notches
land at frequencies that may be the PAC carrier; use selective `notch:f0`
on the fundamental only instead.

## When to Use / NOT to Use

**Use** when: line noise + many harmonics dominate; you want one-pass
cleanup; downstream is classification / ERP / band-power.

**Don't use** when: PAC / cross-frequency analysis (use single notch);
the harmonic frequencies coincide with paradigm-relevant content (SSVEP
at 25/50/75 Hz hits 50 Hz fundamental).

## Constraints & Ordering

- Apply **before** `bandpass` to avoid notch side-lobes folding into the
  analysis band.
- Apply **before** ICA — line noise hurts ICA convergence.
- Q < 10 widens notches into neighbouring bands; Q > 50 rings.

## Failure Modes & Detection

| Failure | Symptom | Detection |
|---|---|---|
| Notch too wide (Q low) | Visible dip in PSD around `f0 ± 2 Hz`. | Compute PSD; if `mean(PSD[f0-2:f0+2]) < 0.7 * neighbour` raise warning. |
| Harmonic above signal of interest | SSVEP / response peaks attenuated. | If SSVEP paradigm + `f0` divides response → log warning. |
| Recording too short | Edge transients dominate. | If `n_t · sfreq / 1000 < 50 · Q / f0` raise recoverable. |

## Common Issues

- **"My SSVEP response at 25 Hz disappeared."** 25 Hz × 2 = 50 Hz; comb's
  notch at 50 Hz aliasing into 25 Hz subharmonic energy. Use `notch:60` if
  US power, or `notch_filter:50,Q=50` (high-Q single notch).
- **"PSD shows ripples near 100 Hz."** Q too high; lower to 20–30.

## Reference Implementation

### Standalone

```python
from __future__ import annotations
import numpy as np
from scipy.signal import iircomb, filtfilt


def comb_filter(
    data: np.ndarray, sfreq: float, f0: float = 50.0, Q: float = 30.0
) -> np.ndarray:
    """Multi-harmonic notch via digital comb."""
    w0 = f0 / (sfreq / 2)
    if w0 >= 1.0:
        raise ValueError(f"comb: f0={f0} >= Nyquist {sfreq/2}")
    b, a = iircomb(w0, Q, ftype="notch", fs=sfreq)
    return filtfilt(b, a, data, axis=-1, padtype="even").astype(data.dtype, copy=False)
```

### EasyBCI-Adapted

```python
from typing import Any, Dict
import time
import numpy as np
from easybci_lib.tools.neural_processing.operator_errors import EasyBCIOperatorError
from easybci_lib.tools.neural_processing.preprocess.step_cache import record_step_elapsed


def operator_comb(
    data_dict: Dict[str, Any], *, f0: float = 50.0, Q: float = 30.0,
) -> Dict[str, Any]:
    """Multi-harmonic comb notch.

    Parameters
    ----------
    data_dict : dict
        OperatorIO.
    f0 : float
        Fundamental frequency (default 50.0 Hz).
    Q : float
        Notch sharpness (default 30.0).

    Returns
    -------
    dict
        OperatorIO with line-noise + harmonics removed.

    Raises
    ------
    EasyBCIOperatorError
        recoverable=True for f0 >= Nyquist.

    Modality coverage
    -----------------
    EEG / MEG / sEEG / ECoG: yes.
    fNIRS: no (no line noise in hemodynamic band).
    PAC tasks: forbidden — use selective `notch:f0` instead.

    References
    ----------
    Mitra & Bokil 2007 — comb filter for line noise harmonics.
    """
    sfreq = float(data_dict["frequency"])
    if f0 >= sfreq / 2:
        raise EasyBCIOperatorError(
            operator="comb", reason=f"f0={f0} >= Nyquist {sfreq/2}",
            recoverable=True, fallback_step=f"notch:{f0}",
        )

    t0 = time.monotonic()
    from scipy.signal import iircomb, filtfilt
    w0 = f0 / (sfreq / 2)
    b, a = iircomb(w0, Q, ftype="notch", fs=sfreq)
    new_data = filtfilt(b, a, data_dict["data"], axis=-1, padtype="even").astype(
        data_dict["data"].dtype, copy=False
    )
    elapsed = time.monotonic() - t0

    out = dict(data_dict)
    out["data"] = new_data
    out["elapsed_s"] = elapsed
    out["meta"] = {**out.get("meta", {}), "comb": {"f0": f0, "Q": Q}}
    record_step_elapsed("comb", elapsed, (data_dict.get("meta") or {}).get("step_cache_key"))
    return out
```

## References

1. Mitra, P. P., & Bokil, H. (2007). *Observed Brain Dynamics*. Oxford
   University Press — line-noise removal via comb filtering.
2. Widmann, A. et al. (2015). *Digital filter design for electrophysiological
   data*. Journal of Neuroscience Methods 250: 34–46.
   doi:10.1016/j.jneumeth.2014.08.002 — notch / comb practical guidance.
