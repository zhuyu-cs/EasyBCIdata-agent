---
name: spectral_edge_freq
description: "Spectral edge frequency (SEF-95 / configurable %) — anaesthesia depth / seizure marker"
layer: L3
group: spectral
metadata:
  tags: [operator, spectral, sef, edge_freq, anaesthesia, seizure, clinical]
  modalities: [eeg, meg, seeg, ecog, lfp]
  step_string: "sef"
  analysis_goal_allowed: [feature_extraction, clinical_screening, exploratory, generic]
  analysis_goal_forbidden: [source_localization, online_inference]
---
# Spectral Edge Frequency (SEF)

## Function

The spectral edge frequency (SEF-p) is the frequency below which `p%` of
the total spectral power lies. SEF-95 is the canonical anaesthesia-depth
metric (lower under deep anaesthesia); SEF-50 (median frequency) is a
robust seizure-detection feature.

Input / Output: `(n_channels, n_times)` → `(n_channels,)`.

## Algorithm & Math

1. Compute PSD via Welch.
2. Cumulative sum across frequency:
   ```
   CDF[c, f] = cumsum(PSD[c, :f]) / sum(PSD[c, :])
   ```
3. SEF-p = first frequency where `CDF >= p/100`.

## Parameter Format & Defaults

`sef:{percentile}`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `percentile` | float | 95.0 | Percent of cumulative power. |
| `fmin`, `fmax` (kw) | float, float | 0.5, 40 | Frequency range. |

## Modality-Specific Considerations

| Modality | percentile | fmax | Notes |
|---|---|---|---|
| EEG (anaesthesia) | 95 | 30 | SEF-95 + amplitude as anaesthesia depth. |
| EEG (seizure) | 50 | 40 | Median freq drops at seizure onset. |
| sEEG (HFO/seizure) | 50 | 200 | Median in higher band. |
| LFP | 95 | 100 | General spectral characterization. |

## When to Use / NOT to Use

**Use** when: anaesthesia depth monitoring; seizure detection feature;
spectral summary (1 number per channel).

**Don't use** when: full PSD shape matters (use `multitaper_psd`);
online inference (Welch overhead).

## Constraints & Ordering

- After bandpass / notch / drop_bads.
- Recording ≥ 5 s for stable Welch.

## Failure Modes & Detection

| Failure | Symptom | Detection |
|---|---|---|
| Empty PSD | All zero output. | If `sum(psd) == 0` raise recoverable. |
| Fmax < expected SEF | Output clipped at fmax. | If output equals fmax warn (clipping). |

## Common Issues

- **"My SEF-95 is at the boundary."** Either fmax too low or the
  recording has broadband content; raise fmax (then SEF-95 reflects
  actual data).

## Reference Implementation

### Standalone

```python
from __future__ import annotations
import numpy as np
from scipy.signal import welch


def spectral_edge_freq(
    data: np.ndarray, sfreq: float, percentile: float = 95.0,
    fmin: float = 0.5, fmax: float = 40.0,
) -> np.ndarray:
    """SEF-p per channel."""
    f, p = welch(data, fs=sfreq, nperseg=min(int(sfreq), data.shape[-1]), axis=-1)
    mask = (f >= fmin) & (f <= fmax)
    p_band = p[..., mask]
    f_band = f[mask]
    cumulative = np.cumsum(p_band, axis=-1)
    totals = cumulative[..., -1:]
    cdf = cumulative / np.maximum(totals, 1e-30)
    idx = np.argmax(cdf >= percentile / 100.0, axis=-1)
    return f_band[idx]
```

### EasyBCI-Adapted

```python
from typing import Any, Dict
import time
import numpy as np
from easybci_lib.tools.neural_processing.operator_errors import EasyBCIOperatorError
from easybci_lib.tools.neural_processing.preprocess.step_cache import record_step_elapsed


def operator_sef(
    data_dict: Dict[str, Any], *,
    percentile: float = 95.0, fmin: float = 0.5, fmax: float = 40.0,
) -> Dict[str, Any]:
    """Spectral edge frequency.

    Parameters
    ----------
    data_dict : dict
        OperatorIO.
    percentile : float
        Cumulative power percent (default 95.0).
    fmin, fmax : float
        Frequency range (default 0.5, 40).

    Returns
    -------
    dict
        OperatorIO; `data` unchanged; meta populated:
        - `meta["sef"]`: ndarray (n_ch,) — SEF-percentile per channel.

    Raises
    ------
    EasyBCIOperatorError
        recoverable=True for zero-energy input.

    Modality coverage
    -----------------
    EEG / MEG / sEEG / ECoG / LFP: yes.
    fNIRS / spike: forbidden (no oscillatory power).

    References
    ----------
    Bickford 1950; Schwender et al. 1996 — SEF and anaesthesia depth.
    """
    sfreq = float(data_dict["frequency"])
    if fmax >= sfreq / 2:
        raise EasyBCIOperatorError(
            operator="sef",
            reason=f"fmax {fmax} >= Nyquist {sfreq/2}",
            recoverable=True,
            fallback_step=f"sef:{percentile}",
        )

    t0 = time.monotonic()
    from scipy.signal import welch
    f, p = welch(
        data_dict["data"], fs=sfreq,
        nperseg=min(int(sfreq), data_dict["data"].shape[-1]), axis=-1,
    )
    mask = (f >= fmin) & (f <= fmax)
    p_band = p[..., mask]
    f_band = f[mask]
    cumulative = np.cumsum(p_band, axis=-1)
    totals = cumulative[..., -1:]
    if np.any(totals == 0):
        raise EasyBCIOperatorError(
            operator="sef", reason="zero-energy channel in band",
            recoverable=True, fallback_step=f"drop_bads",
        )
    cdf = cumulative / totals
    idx = np.argmax(cdf >= percentile / 100.0, axis=-1)
    sef = f_band[idx].astype(np.float32)
    elapsed = time.monotonic() - t0

    out = dict(data_dict)
    out["elapsed_s"] = elapsed
    out["meta"] = {
        **out.get("meta", {}),
        "sef": sef,
        "sef_meta": {"percentile": percentile, "fmin": fmin, "fmax": fmax},
    }
    record_step_elapsed("sef", elapsed, (data_dict.get("meta") or {}).get("step_cache_key"))
    return out
```

## References

1. Bickford, R. G. et al. (1950). *Effect of barbiturates on the EEG of
   humans*. EEG Clin. Neurophysiol. 2: 65 — early SEF observation.
2. Schwender, D. et al. (1996). *Spectral edge frequency of the
   electroencephalogram to monitor depth of anaesthesia with isoflurane
   or propofol*. Br. J. Anaesthesia 77(2): 179–184.
   doi:10.1093/bja/77.2.179 — modern SEF-95 clinical reference.
3. Drongelen, W. van et al. (2003). *Seizure anticipation in pediatric
   epilepsy*. J. Clin. Neurophysiol. 20(2): 137–146.
   doi:10.1097/00004691-200304000-00008 — median freq for seizure.
