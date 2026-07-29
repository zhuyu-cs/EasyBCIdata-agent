---
name: multitaper_psd
description: "Multitaper PSD via Slepian DPSS tapers — gold-standard narrow-band power estimate"
layer: L3
group: spectral
metadata:
  tags: [operator, spectral, psd, multitaper, dpss, slepian]
  modalities: [eeg, meg, seeg, ecog, lfp]
  step_string: "multitaper_psd"
  analysis_goal_allowed: [classification, feature_extraction, clinical_screening, exploratory, generic, connectivity, phase_amplitude_coupling]
  analysis_goal_forbidden: [online_inference]
---
# Multitaper PSD (Slepian DPSS)

## Function

Computes the power spectral density (PSD) via the **multitaper method**
using discrete prolate spheroidal sequence (DPSS / Slepian) tapers.
Variance is reduced by averaging K independent taper estimates without
sacrificing bias control; this is the **gold-standard** narrow-band PSD
estimator for neuroscience.

Input / Output: `(n_channels, n_times)` → `(n_channels, n_freqs)`.

## Algorithm & Math

Pick a half-bandwidth parameter `NW` (time-bandwidth product). DPSS
tapers `v_k(t)` for `k = 0, ..., K−1` with `K = 2·NW − 1` are
orthogonal, maximally concentrated within `±W = NW / T` of the center
frequency.

PSD estimate:

```
P̂(f) = (1/K) · Σ_k |Σ_t v_k(t) · x(t) · e^{-j 2π f t}|²
```

Bias is constrained to `±W`; variance reduced by 1/K vs single-taper
periodogram.

## Parameter Format & Defaults

`multitaper_psd:{NW}`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `NW` | float | 4.0 | Time-bandwidth product. K = 2·NW − 1. |
| `fmin`, `fmax` (kw) | float, float | 0.5, 100 | Output frequency range. |
| `n_per_seg` (kw) | int | None | Segment length; default = full recording. |

## Modality-Specific Considerations

| Modality | NW | Notes |
|---|---|---|
| EEG | 3–4 | K = 5–7 tapers. |
| MEG | 4 | Same as EEG. |
| sEEG / ECoG | 4–5 | Higher NW for narrow-band gamma. |
| LFP | 3 | Long records; lower NW sufficient. |
| Spike | n/a | Use spike-train PSD instead (different op). |

## When to Use / NOT to Use

**Use** when: precise band-power needed (resting-state alpha, theta
spectral peaks); statistical claims on PSD shape; QC for line-noise
residue.

**Don't use** when: online inference (multitaper is too slow);
time-resolved analysis (use `cwt_morlet` or `stft`); spike data.

## Constraints & Ordering

- Apply **after** notch / bandpass / drop_bads.
- Apply on **continuous** data; epoching downstream is fine.
- Recording must be ≥ 10 s for stable estimates at low freq.

## Failure Modes & Detection

| Failure | Symptom | Detection |
|---|---|---|
| NW too high | K too many tapers; recording too short. | If `K > n_t · NW / T_window` warn. |
| Recording too short | Bias from edge effects. | If `n_t / sfreq < 10` raise recoverable. |

## Common Issues

- **"My alpha peak is broad."** Multitaper bandwidth `2W = 2 · NW / T`;
  lower NW gives sharper peaks but higher variance.

## Reference Implementation

### Standalone

```python
from __future__ import annotations
import numpy as np
from scipy.signal.windows import dpss


def multitaper_psd(
    data: np.ndarray, sfreq: float, NW: float = 4.0,
    fmin: float = 0.5, fmax: float = 100.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Multitaper PSD via DPSS tapers. Returns (freqs, psd)."""
    n_t = data.shape[-1]
    K = int(2 * NW - 1)
    tapers, _ = dpss(n_t, NW, Kmax=K, return_ratios=True)
    # tapers: (K, n_t)
    psd_per_taper = []
    for v in tapers:
        x = data * v
        X = np.fft.rfft(x, axis=-1)
        psd_per_taper.append(np.abs(X) ** 2)
    psd = np.mean(psd_per_taper, axis=0) / (sfreq * n_t)
    freqs = np.fft.rfftfreq(n_t, d=1.0 / sfreq)
    mask = (freqs >= fmin) & (freqs <= fmax)
    return freqs[mask], psd[..., mask]
```

### EasyBCI-Adapted

```python
from typing import Any, Dict
import time
import numpy as np
from easybci_lib.tools.neural_processing.operator_errors import EasyBCIOperatorError
from easybci_lib.tools.neural_processing.preprocess.step_cache import record_step_elapsed


def operator_multitaper_psd(
    data_dict: Dict[str, Any], *, NW: float = 4.0, fmin: float = 0.5, fmax: float = 100.0,
) -> Dict[str, Any]:
    """Multitaper PSD.

    Parameters
    ----------
    data_dict : dict
        OperatorIO.
    NW : float
        Time-bandwidth product (default 4.0).
    fmin, fmax : float
        Output frequency range (default 0.5, 100).

    Returns
    -------
    dict
        OperatorIO; continuous `data` unchanged; meta populated:
        - `meta["psd"]`: ndarray (n_ch, n_freqs)
        - `meta["psd_freqs"]`: ndarray (n_freqs,)

    Raises
    ------
    EasyBCIOperatorError
        recoverable=True if recording < 10 s.

    Modality coverage
    -----------------
    EEG / MEG / sEEG / ECoG / LFP: yes.
    Spike: forbidden (use spike-train PSD).

    References
    ----------
    Thomson 1982; Percival & Walden 1993 (textbook).
    """
    sfreq = float(data_dict["frequency"])
    n_t = data_dict["data"].shape[-1]
    if n_t / sfreq < 10:
        raise EasyBCIOperatorError(
            operator="multitaper_psd",
            reason=f"recording {n_t/sfreq:.2f}s < 10 s; PSD unreliable",
            recoverable=True, fallback_step="resample then multitaper_psd",
        )

    t0 = time.monotonic()
    from scipy.signal.windows import dpss
    K = int(2 * NW - 1)
    tapers, _ = dpss(n_t, NW, Kmax=K, return_ratios=True)
    psd_acc = None
    for v in tapers:
        x = data_dict["data"] * v
        X = np.fft.rfft(x, axis=-1)
        contrib = np.abs(X) ** 2
        psd_acc = contrib if psd_acc is None else psd_acc + contrib
    psd = psd_acc / (K * sfreq * n_t)
    freqs = np.fft.rfftfreq(n_t, d=1.0 / sfreq)
    mask = (freqs >= fmin) & (freqs <= fmax)
    elapsed = time.monotonic() - t0

    out = dict(data_dict)
    out["elapsed_s"] = elapsed
    out["meta"] = {
        **out.get("meta", {}),
        "psd": psd[..., mask].astype(np.float32),
        "psd_freqs": freqs[mask],
        "multitaper_psd": {"NW": NW, "K": K, "fmin": fmin, "fmax": fmax},
    }
    record_step_elapsed("multitaper_psd", elapsed, (data_dict.get("meta") or {}).get("step_cache_key"))
    return out
```

## References

1. Thomson, D. J. (1982). *Spectrum estimation and harmonic analysis*.
   Proceedings of the IEEE 70(9): 1055–1096.
   doi:10.1109/PROC.1982.12433 — the original multitaper paper.
2. Percival, D. B., & Walden, A. T. (1993). *Spectral Analysis for
   Physical Applications: Multitaper and Conventional Univariate
   Techniques*. Cambridge University Press.
3. Mitra, P. P., & Bokil, H. (2007). *Observed Brain Dynamics*. Oxford
   University Press — neural-data multitaper application.
