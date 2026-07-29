---
name: dtf_pdc
description: "Directed Transfer Function / Partial Directed Coherence — frequency-domain directed connectivity"
layer: L3
group: connectivity
metadata:
  tags: [operator, connectivity, dtf, pdc, mvar, frequency_directed]
  modalities: [eeg, meg, seeg, ecog, lfp]
  step_string: "dtf_pdc"
  analysis_goal_allowed: [feature_extraction, exploratory, connectivity]
  analysis_goal_forbidden: [source_localization, online_inference]
---
# DTF / PDC

## Function

DTF (Directed Transfer Function, Kaminski 1991) and PDC (Partial
Directed Coherence, Baccala 2001) are **frequency-resolved** directed
connectivity measures derived from the same MVAR model that underlies
Granger causality. Output: connectivity matrix per frequency bin.

Input / Output: `(n_channels, n_times)` → `(n_channels, n_channels, n_freqs)`.

## Algorithm & Math

Fit MVAR of order p, transform AR coefficients to frequency:
```
A(f) = I - Σ_k A_k · e^{-j 2π f k / sfreq}
H(f) = A(f)^{-1}                                # transfer function
DTF_{ij}(f) = |H_{ij}(f)|² / Σ_k |H_{ik}(f)|²    # row-normalized
PDC_{ij}(f) = |A_{ij}(f)|² / Σ_k |A_{kj}(f)|²    # column-normalized
```

## Parameter Format & Defaults

`dtf_pdc:{order},{kind},{fmin},{fmax}`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `order` | int | 10 | MVAR order. |
| `kind` | str | "dtf" | "dtf" / "pdc". |
| `fmin`, `fmax` | float, float | 1, 50 | Output frequency range. |
| `n_freqs` (kw) | int | 50 | Frequency bin count. |

## Modality-Specific Considerations

Same as Granger: < 64 channels, stationary segments.

## When to Use / NOT to Use

**Use** when: frequency-specific directed connectivity needed; comparing
band-specific information flow.

**Don't use** when: > 64 channels; non-stationary; online inference.

## Constraints & Ordering

Same as Granger — stationary, bandpassed, < 64 channels.

## Failure Modes & Detection

| Failure | Symptom | Detection |
|---|---|---|
| MVAR underdetermined | LinAlgError. | Pre-check; suggest lower order. |
| Numerical instability at high freq | Inf in H. | Add small regularization. |

## Common Issues

- **"PDC and DTF disagree."** They're different normalizations and can
  produce opposite-looking results; both have valid interpretations.

## Reference Implementation

### Standalone

```python
from __future__ import annotations
import numpy as np


def dtf_pdc(
    data: np.ndarray, sfreq: float, order: int = 10, kind: str = "dtf",
    fmin: float = 1.0, fmax: float = 50.0, n_freqs: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """MVAR-based DTF / PDC. Returns (freqs, conn[n_ch, n_ch, n_freqs])."""
    n_ch, n_t = data.shape
    # Fit MVAR via OLS
    rows, targets = [], []
    for t in range(order, n_t):
        rows.append(np.concatenate([data[:, t - k] for k in range(1, order + 1)]))
        targets.append(data[:, t])
    D = np.stack(rows); T = np.stack(targets)
    beta = np.linalg.lstsq(D, T, rcond=None)[0]
    # Reshape A_k matrices: beta shape (order*n_ch, n_ch)
    A = beta.reshape(order, n_ch, n_ch)

    freqs = np.linspace(fmin, fmax, n_freqs)
    conn = np.zeros((n_ch, n_ch, n_freqs), dtype=np.float32)
    for fi, f in enumerate(freqs):
        Af = np.eye(n_ch, dtype=complex)
        for k in range(order):
            Af -= A[k] * np.exp(-1j * 2 * np.pi * f * (k + 1) / sfreq)
        if kind == "dtf":
            H = np.linalg.inv(Af)
            num = np.abs(H) ** 2
            denom = num.sum(axis=1, keepdims=True) + 1e-30
            conn[:, :, fi] = (num / denom).astype(np.float32)
        else:  # pdc
            num = np.abs(Af) ** 2
            denom = num.sum(axis=0, keepdims=True) + 1e-30
            conn[:, :, fi] = (num / denom).astype(np.float32)
    return freqs, conn
```

### EasyBCI-Adapted

```python
from typing import Any, Dict
import time
import numpy as np
from easybci_lib.tools.neural_processing.operator_errors import EasyBCIOperatorError
from easybci_lib.tools.neural_processing.preprocess.step_cache import record_step_elapsed


def operator_dtf_pdc(
    data_dict: Dict[str, Any], *, order: int = 10, kind: str = "dtf",
    fmin: float = 1.0, fmax: float = 50.0, n_freqs: int = 50,
) -> Dict[str, Any]:
    """DTF / PDC frequency-resolved directed connectivity.

    Parameters
    ----------
    data_dict : dict
    order : int
    kind : str
    fmin, fmax : float
    n_freqs : int

    Returns
    -------
    dict — `meta["dtf_pdc"]: (n_ch, n_ch, n_freqs)`.

    Raises
    ------
    EasyBCIOperatorError
        recoverable=False on bad kind / band; recoverable=True if MVAR
        underdetermined.

    Modality coverage
    -----------------
    EEG / MEG / sEEG / ECoG / LFP: yes.

    References
    ----------
    Kaminski & Blinowska 1991; Baccala & Sameshima 2001.
    """
    if kind not in ("dtf", "pdc"):
        raise EasyBCIOperatorError(
            operator="dtf_pdc", reason=f"kind={kind} not in ('dtf','pdc')",
            recoverable=False,
        )
    sfreq = float(data_dict["frequency"])
    if fmax >= sfreq / 2 or fmin >= fmax:
        raise EasyBCIOperatorError(
            operator="dtf_pdc", reason=f"invalid band fmin={fmin} fmax={fmax}",
            recoverable=False,
        )
    data = data_dict["data"]
    n_ch, n_t = data.shape
    if n_ch * (order + 1) > n_t:
        raise EasyBCIOperatorError(
            operator="dtf_pdc", reason="too few samples for MVAR fit",
            recoverable=True, fallback_step=f"dtf_pdc:{order // 2}",
        )

    t0 = time.monotonic()
    rows, targets = [], []
    for t in range(order, n_t):
        rows.append(np.concatenate([data[:, t - k] for k in range(1, order + 1)]))
        targets.append(data[:, t])
    D = np.stack(rows); T = np.stack(targets)
    beta = np.linalg.lstsq(D, T, rcond=None)[0]
    A = beta.reshape(order, n_ch, n_ch)
    freqs = np.linspace(fmin, fmax, n_freqs)
    conn = np.zeros((n_ch, n_ch, n_freqs), dtype=np.float32)
    for fi, f in enumerate(freqs):
        Af = np.eye(n_ch, dtype=complex)
        for k in range(order):
            Af -= A[k] * np.exp(-1j * 2 * np.pi * f * (k + 1) / sfreq)
        if kind == "dtf":
            H = np.linalg.inv(Af)
            num = np.abs(H) ** 2
            denom = num.sum(axis=1, keepdims=True) + 1e-30
            conn[:, :, fi] = (num / denom).astype(np.float32)
        else:
            num = np.abs(Af) ** 2
            denom = num.sum(axis=0, keepdims=True) + 1e-30
            conn[:, :, fi] = (num / denom).astype(np.float32)
    elapsed = time.monotonic() - t0

    out = dict(data_dict)
    out["elapsed_s"] = elapsed
    out["meta"] = {
        **out.get("meta", {}),
        "dtf_pdc": conn, "dtf_pdc_freqs": freqs,
        "dtf_pdc_kind": kind, "dtf_pdc_order": order,
    }
    record_step_elapsed("dtf_pdc", elapsed, (data_dict.get("meta") or {}).get("step_cache_key"))
    return out
```

## Related Skills

- `bci/paradigms/analysis/connectivity/SKILL.md`.

## References

1. Kaminski, M., & Blinowska, K. J. (1991). *A new method of the
   description of the information flow in the brain structures*.
   Biological Cybernetics 65(3): 203–210. doi:10.1007/BF00198091 — DTF.
2. Baccala, L. A., & Sameshima, K. (2001). *Partial directed coherence:
   a new concept in neural structure determination*. Biological
   Cybernetics 84(6): 463–474. doi:10.1007/PL00007990 — PDC.
3. Blinowska, K. J. (2011). *Review of the methods of determination of
   directed connectivity from multichannel data*. Medical & Biological
   Engineering & Computing 49(5): 521–529. doi:10.1007/s11517-011-0739-x.
