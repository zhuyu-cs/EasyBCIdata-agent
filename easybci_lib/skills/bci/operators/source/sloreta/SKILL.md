---
name: sloreta
description: "standardized Low-Resolution Brain Electromagnetic Tomography — distributed source"
layer: L3
group: source
metadata:
  tags: [operator, source, sloreta, distributed, inverse, mne]
  modalities: [eeg, meg]
  step_string: "sloreta"
  analysis_goal_allowed: [source_localization, feature_extraction, clinical_screening, exploratory]
  analysis_goal_forbidden: [online_inference]
---
# sLORETA — standardized Low-Resolution Brain Electromagnetic Tomography

## Function

Distributed source imaging via Pascual-Marqui's sLORETA — assumes
discrete cortical source grid, applies minimum-norm-like inverse, then
standardizes by source-power variance to mitigate depth bias. The
canonical alternative to LCMV for distributed-source analysis.

Input / Output: `(n_channels, n_times)` → `(n_sources, n_times)`.

## Algorithm & Math

```
T_MN = G^T (G G^T + λ²I)^{-1}                    # MN inverse
S_t = ( T_MN · sensor_data )                       # raw sources
S_sLORETA = S_t / sqrt(diag(T_MN · sensor_cov · T_MN^T))   # standardized
```

`λ²` set by signal-to-noise ratio (typically `tr(noise_cov) / SNR`).

## Parameter Format & Defaults

`sloreta:{snr}`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `snr` | float | 3.0 | Inverse regularization SNR. |
| `n_sources` (kw) | int | 1024 | Source-grid size. |

## Modality-Specific Considerations

EEG / MEG only.

## When to Use / NOT to Use

**Use** when: distributed source needed; high-density montage (32+ EEG /
64+ MEG).

**Don't use** when: focal source (use dipole_fit); online inference;
no head model.

## Constraints & Ordering

- Apply on averaged ERP or per-epoch.
- After bandpass / notch / drop_bads.
- Requires forward model + noise covariance.

## Failure Modes & Detection

| Failure | Symptom | Detection |
|---|---|---|
| Missing leadfield | KeyError. | Raise recoverable. |
| Wrong noise cov | Sources show stripes. | Recompute noise cov from pre-stim baseline. |

## Common Issues

- **"Source map is too smeared."** Inherent — sLORETA prioritizes
  spatial smoothness; switch to LCMV for sharper localization.

## Reference Implementation

### Standalone

```python
from __future__ import annotations
import numpy as np


def sloreta(
    data: np.ndarray, leadfield: np.ndarray, noise_cov: np.ndarray, snr: float = 3.0
) -> np.ndarray:
    """sLORETA inverse.

    Parameters
    ----------
    data : (n_channels, n_times)
    leadfield : (n_channels, n_sources)
    noise_cov : (n_channels, n_channels)
    """
    n_ch, n_src = leadfield.shape
    lam2 = float(np.trace(noise_cov)) / (snr * snr) / max(n_ch, 1)
    # Min-norm inverse
    G = leadfield
    inv_kernel = G.T @ np.linalg.inv(G @ G.T + lam2 * np.eye(n_ch))   # (n_src, n_ch)
    sources_mn = inv_kernel @ data                                     # (n_src, n_t)
    # Standardize by sqrt(diag(T · noise_cov · T^T))
    var_per_src = np.einsum("ij,jk,ki->i", inv_kernel, noise_cov, inv_kernel.T)
    return (sources_mn / np.sqrt(np.maximum(var_per_src, 1e-30))[:, None]).astype(np.float32)
```

### EasyBCI-Adapted

```python
from typing import Any, Dict
import time
import numpy as np
from easybci_lib.tools.neural_processing.operator_errors import EasyBCIOperatorError
from easybci_lib.tools.neural_processing.preprocess.step_cache import record_step_elapsed


def operator_sloreta(
    data_dict: Dict[str, Any], *, snr: float = 3.0,
) -> Dict[str, Any]:
    """sLORETA distributed source inverse.

    Parameters
    ----------
    data_dict : dict
        OperatorIO; `meta["leadfield"]` and `meta["noise_cov"]` required.
    snr : float

    Returns
    -------
    dict — `meta["sources"]: (n_src, n_t)`.

    Raises
    ------
    EasyBCIOperatorError
        recoverable=True if leadfield / noise_cov missing.

    Modality coverage
    -----------------
    EEG / MEG: yes. Others: forbidden.

    References
    ----------
    Pascual-Marqui 2002.
    """
    leadfield = (data_dict.get("meta") or {}).get("leadfield")
    noise_cov = (data_dict.get("meta") or {}).get("noise_cov")
    if leadfield is None or noise_cov is None:
        raise EasyBCIOperatorError(
            operator="sloreta", reason="leadfield and noise_cov required in meta",
            recoverable=True, fallback_step="compute leadfield + noise_cov upstream",
        )

    t0 = time.monotonic()
    G = np.asarray(leadfield); C = np.asarray(noise_cov)
    n_ch = G.shape[0]
    lam2 = float(np.trace(C)) / (snr * snr) / max(n_ch, 1)
    inv_kernel = G.T @ np.linalg.inv(G @ G.T + lam2 * np.eye(n_ch))
    sources_mn = inv_kernel @ data_dict["data"]
    var_per_src = np.einsum("ij,jk,ki->i", inv_kernel, C, inv_kernel.T)
    sources = (sources_mn / np.sqrt(np.maximum(var_per_src, 1e-30))[:, None]).astype(np.float32)
    elapsed = time.monotonic() - t0

    out = dict(data_dict)
    out["elapsed_s"] = elapsed
    out["meta"] = {**out.get("meta", {}), "sources": sources, "sloreta_snr": snr}
    record_step_elapsed("sloreta", elapsed, (data_dict.get("meta") or {}).get("step_cache_key"))
    return out
```

## References

1. Pascual-Marqui, R. D. (2002). *Standardized low-resolution brain
   electromagnetic tomography (sLORETA): technical details*. Methods Find
   Exp Clin Pharmacol 24(Suppl D): 5–12.
2. Hämäläinen, M. S., & Ilmoniemi, R. J. (1994). *Interpreting magnetic
   fields of the brain: minimum norm estimates*. Medical & Biological
   Engineering & Computing 32(1): 35–42. doi:10.1007/BF02512476.
