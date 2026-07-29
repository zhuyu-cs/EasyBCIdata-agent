---
name: zapline
description: "Zapline (de Cheveigné 2020) — non-stationary line-noise removal via spectral PCA"
layer: L3
group: filter
metadata:
  tags: [operator, filter, zapline, line_noise, spectral_pca, pac_safe, connectivity_safe]
  modalities: [eeg, meg, seeg, ecog]
  step_string: "zapline"
  analysis_goal_allowed: [classification, source_localization, feature_extraction, clinical_screening, exploratory, generic, connectivity, phase_amplitude_coupling]
  analysis_goal_forbidden: []
---
# Zapline

## Function

**Zapline** (de Cheveigné, NeuroImage 2020) removes line noise via a
**spectral PCA** that isolates and subtracts components dominated by
the line frequency — without notching neighbouring bands. Strongly
recommended for **PAC / connectivity** analyses where a hard notch
distorts the very phase content being measured.

Input / Output: `(n_channels, n_times)` → same shape.

## Algorithm & Math

1. Split data into spectral domain via STFT (1 s window default).
2. Compute spatial PCA at the line frequency `f_line`.
3. The top `n_remove` components (default 1–4) carry the line-noise; the
   residual is the line-free signal.
4. Inverse transform → time domain.

Mathematically: at frequency `f_line`, decompose the cross-channel
covariance matrix `Σ(f_line)` via PCA. The largest eigenvalue carries
the rank-1 component shared across electrodes (line noise is by nature
spatially coherent). Subtract that subspace from the full STFT, inverse
transform.

Crucially, this **does not zero out** energy at `f_line ± δ` (a hard
notch's collateral); the spatial filter is selective.

## Parameter Format & Defaults

`zapline:{f_line},{n_remove}`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `f_line` | float | 50.0 | Line frequency (Hz). |
| `n_remove` | int | 1 | Components to subtract; 2–4 for very noisy data. |
| `n_keep` (kw) | int | None | Optional rank cap on PCA. |

## Modality-Specific Considerations

| Modality | n_remove | Notes |
|---|---|---|
| EEG (high-density, > 64 ch) | 1 | Rank-1 typically suffices. |
| EEG (low-density, ≤ 32 ch) | 2 | Spatial filter less selective; need more components. |
| MEG | 2–4 | Multiple gradient channels → line noise has higher spatial rank. |
| sEEG | 1 | Implanted electrodes; line noise common-mode. |

Hard exclusion: `fNIRS` (no electrical line noise in hemodynamic band).

## When to Use / NOT to Use

**Use** when:
- PAC / connectivity analysis (you need wide-band phase preserved).
- Non-stationary line noise (frequency drifting at sub-Hz, e.g., elevator
  motors).
- High-density montage (> 32 ch) where rank-1 spatial filter is clean.

**Don't use** when:
- < 16 channels — spatial rank inadequate.
- Line noise is below the noise floor (no-op; just adds cost).
- Online inference (Zapline needs full segments; not stream-friendly —
  use `notch:f_line` instead).

## Constraints & Ordering

- Apply **before** any spectral analysis.
- Apply **before** ICA.
- Apply **after** drop-bads if any channels are saturated (PCA on
  saturated input is degenerate).

## Failure Modes & Detection

| Failure | Symptom | Detection |
|---|---|---|
| n_remove too high | Genuine neural signal at `f_line` (e.g., 50 Hz gamma) removed. | PSD comparison at `f_line ± 5 Hz`: pre vs post drop > 30% → too aggressive; reduce `n_remove`. |
| n_remove too low | Line peak survives. | PSD at exactly `f_line`: post > 10 dB above neighbour → increase `n_remove`. |
| Rank-deficient input | PCA fails (saturated channels). | Pre-check `np.linalg.matrix_rank(data)`; if < n_remove + 1 raise recoverable. |

## Common Issues

- **"Notch was simpler — why use Zapline?"** Because hard notch at 50 Hz
  destroys the PAC carrier if your slow phase is at ~1–4 Hz and your
  fast amplitude is at 50 Hz (or hits a notch harmonic). Zapline keeps
  the analysis-frequency content intact.
- **"Computational cost is high."** Yes — Zapline's STFT + per-frequency
  PCA is O(n_channels² · n_freq) per segment. Cache via step_cache.

## Reference Implementation

### Standalone

```python
from __future__ import annotations
import numpy as np
from scipy.signal import stft, istft


def zapline(
    data: np.ndarray, sfreq: float, f_line: float = 50.0, n_remove: int = 1
) -> np.ndarray:
    """De Cheveigné 2020 zapline — spectral-PCA line noise removal."""
    nperseg = int(sfreq)
    f, t, Z = stft(data, fs=sfreq, nperseg=nperseg, axis=-1)
    target_idx = int(np.argmin(np.abs(f - f_line)))
    # Per spectral frequency near f_line: PCA across channels, subtract top components
    out = Z.copy()
    for i in range(max(0, target_idx - 1), min(len(f), target_idx + 2)):
        slice_ = Z[:, i, :]                       # (n_ch, n_t_stft)
        cov = (slice_ @ slice_.conj().T).real
        # eigendecomp; remove top n_remove eigenvectors
        vals, vecs = np.linalg.eigh(cov)
        keep = vecs[:, : -n_remove] if n_remove > 0 else vecs
        projector = keep @ keep.conj().T
        out[:, i, :] = projector @ slice_
    _, recon = istft(out, fs=sfreq, nperseg=nperseg)
    n_t = data.shape[-1]
    return recon[..., :n_t].astype(data.dtype, copy=False)
```

### EasyBCI-Adapted

```python
from typing import Any, Dict
import time
import numpy as np
from easybci_lib.tools.neural_processing.operator_errors import EasyBCIOperatorError
from easybci_lib.tools.neural_processing.preprocess.step_cache import record_step_elapsed


def operator_zapline(
    data_dict: Dict[str, Any], *, f_line: float = 50.0, n_remove: int = 1,
) -> Dict[str, Any]:
    """Zapline spectral-PCA line-noise removal.

    Parameters
    ----------
    data_dict : dict
        OperatorIO.
    f_line : float
        Line frequency (Hz, default 50.0).
    n_remove : int
        Components to subtract (default 1).

    Returns
    -------
    dict
        OperatorIO with line-noise components removed.

    Raises
    ------
    EasyBCIOperatorError
        recoverable=True if channel count < 16 (rank inadequate).

    Modality coverage
    -----------------
    EEG / MEG / sEEG / ECoG: yes.
    fNIRS: no (no electrical line noise in hemodynamic band).

    References
    ----------
    de Cheveigné A. (2020). NeuroImage 207: 116356.
    """
    n_ch = data_dict["data"].shape[0]
    if n_ch < 16:
        raise EasyBCIOperatorError(
            operator="zapline",
            reason=f"only {n_ch} channels; need >= 16 for spatial PCA",
            recoverable=True, fallback_step=f"notch:{f_line}",
        )

    sfreq = float(data_dict["frequency"])
    t0 = time.monotonic()
    from scipy.signal import stft, istft

    nperseg = int(sfreq)
    f, _, Z = stft(data_dict["data"], fs=sfreq, nperseg=nperseg, axis=-1)
    target_idx = int(np.argmin(np.abs(f - f_line)))
    out = Z.copy()
    for i in range(max(0, target_idx - 1), min(len(f), target_idx + 2)):
        slc = Z[:, i, :]
        cov = (slc @ slc.conj().T).real
        vals, vecs = np.linalg.eigh(cov)
        keep_vecs = vecs[:, : -n_remove] if n_remove > 0 else vecs
        projector = keep_vecs @ keep_vecs.conj().T
        out[:, i, :] = projector @ slc
    _, recon = istft(out, fs=sfreq, nperseg=nperseg)
    n_t = data_dict["data"].shape[-1]
    new_data = recon[..., :n_t].astype(data_dict["data"].dtype, copy=False)
    elapsed = time.monotonic() - t0

    res = dict(data_dict)
    res["data"] = new_data
    res["elapsed_s"] = elapsed
    res["meta"] = {**res.get("meta", {}), "zapline": {"f_line": f_line, "n_remove": n_remove}}
    record_step_elapsed("zapline", elapsed, (data_dict.get("meta") or {}).get("step_cache_key"))
    return res
```

## References

1. de Cheveigné, A. (2020). *ZapLine: A simple and effective method to
   remove power line artifacts*. NeuroImage 207: 116356.
   doi:10.1016/j.neuroimage.2019.116356 — the original Zapline paper.
2. de Cheveigné, A. & Arzounian, D. (2018). *Robust detrending,
   rereferencing, outlier detection, and inpainting for multichannel
   data*. NeuroImage 172: 903–912. doi:10.1016/j.neuroimage.2018.01.035 —
   adjacent spectral-PCA methodology.
