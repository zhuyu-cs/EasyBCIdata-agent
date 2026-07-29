---
name: lcmv_beamformer
description: "Linearly Constrained Minimum Variance beamformer — adaptive distributed source"
layer: L3
group: source
metadata:
  tags: [operator, source, lcmv, beamformer, adaptive]
  modalities: [eeg, meg]
  step_string: "lcmv"
  analysis_goal_allowed: [source_localization, feature_extraction, clinical_screening, exploratory]
  analysis_goal_forbidden: [online_inference]
---
# LCMV Beamformer

## Function

Linearly Constrained Minimum Variance (LCMV, Van Veen 1997) — adaptive
spatial filter that estimates source activity by minimizing total
variance subject to passing through a specified source orientation.
Sharper spatial localization than sLORETA at the cost of requiring a
good data covariance estimate.

Input / Output: `(n_channels, n_times)` → `(n_sources, n_times)`.

## Algorithm & Math

For each source at position `r`:
```
w(r) = (G(r)^T · C^{-1} · G(r))^{-1} · G(r)^T · C^{-1}
```

Where `C` is the data covariance (regularized), `G(r)` the leadfield at
position `r`. Source activity: `s(t, r) = w(r) · data(t)`.

## Parameter Format & Defaults

`lcmv:{reg}`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `reg` | float | 0.05 | Tikhonov regularization on covariance. |
| `n_sources` (kw) | int | 1024 | Source grid size. |
| `orientation` (kw) | str | "max_power" | "max_power" / "free". |

## Modality-Specific Considerations

MEG: standard for resting-state and task source localization.
EEG: viable but more sensitive to noise.

## When to Use / NOT to Use

**Use** when: sharp distributed source needed; long stable recording for
covariance estimation; high-density montage.

**Don't use** when: short recording (covariance unstable); online; no
head model.

## Constraints & Ordering

- Compute data covariance on active interval; noise cov on baseline.
- After bandpass / notch / drop_bads.

## Failure Modes & Detection

| Failure | Symptom | Detection |
|---|---|---|
| Singular cov | Inverse fails. | Raise reg; pre-check rank. |
| Few samples | Covariance overfits. | If `n_t < 5 · n_ch` warn. |

## Common Issues

- **"Source map shows hotspots at boundaries."** Edge artefact in
  leadfield; use a tighter source grid avoiding boundary.

## Reference Implementation

### Standalone

```python
from __future__ import annotations
import numpy as np


def lcmv_beamformer(
    data: np.ndarray, leadfield: np.ndarray, data_cov: np.ndarray, reg: float = 0.05,
) -> np.ndarray:
    """LCMV beamformer."""
    n_ch, n_src = leadfield.shape
    C_reg = data_cov + reg * np.trace(data_cov) / n_ch * np.eye(n_ch)
    C_inv = np.linalg.inv(C_reg)
    sources = np.zeros((n_src, data.shape[1]), dtype=np.float32)
    for s in range(n_src):
        G_s = leadfield[:, s:s+1]
        denom = G_s.T @ C_inv @ G_s
        w = (np.linalg.inv(denom) @ G_s.T @ C_inv).flatten()
        sources[s] = (w @ data).astype(np.float32)
    return sources
```

### EasyBCI-Adapted

```python
from typing import Any, Dict
import time
import numpy as np
from easybci_lib.tools.neural_processing.operator_errors import EasyBCIOperatorError
from easybci_lib.tools.neural_processing.preprocess.step_cache import record_step_elapsed


def operator_lcmv(
    data_dict: Dict[str, Any], *, reg: float = 0.05, orientation: str = "max_power",
) -> Dict[str, Any]:
    """LCMV beamformer.

    Parameters
    ----------
    data_dict : dict
        OperatorIO; `meta["leadfield"]` and `meta["data_cov"]` required.
    reg : float
    orientation : str

    Returns
    -------
    dict — `meta["sources"]: (n_src, n_t)`.

    Raises
    ------
    EasyBCIOperatorError
        recoverable=True if leadfield / data_cov missing.

    Modality coverage
    -----------------
    EEG / MEG: yes. Others: forbidden.

    References
    ----------
    Van Veen et al. 1997; Gross 2001 (DICS frequency-domain beamformer).
    """
    leadfield = (data_dict.get("meta") or {}).get("leadfield")
    data_cov = (data_dict.get("meta") or {}).get("data_cov")
    if leadfield is None or data_cov is None:
        raise EasyBCIOperatorError(
            operator="lcmv", reason="leadfield + data_cov required in meta",
            recoverable=True, fallback_step="compute upstream",
        )

    t0 = time.monotonic()
    G = np.asarray(leadfield); C = np.asarray(data_cov)
    n_ch, n_src = G.shape
    C_reg = C + reg * float(np.trace(C)) / max(n_ch, 1) * np.eye(n_ch)
    C_inv = np.linalg.inv(C_reg)
    sources = np.zeros((n_src, data_dict["data"].shape[1]), dtype=np.float32)
    for s in range(n_src):
        G_s = G[:, s:s+1]
        denom = G_s.T @ C_inv @ G_s
        w = (np.linalg.inv(denom) @ G_s.T @ C_inv).flatten()
        sources[s] = (w @ data_dict["data"]).astype(np.float32)
    elapsed = time.monotonic() - t0

    out = dict(data_dict)
    out["elapsed_s"] = elapsed
    out["meta"] = {**out.get("meta", {}), "sources": sources,
                   "lcmv_meta": {"reg": reg, "orientation": orientation}}
    record_step_elapsed("lcmv", elapsed, (data_dict.get("meta") or {}).get("step_cache_key"))
    return out
```

## References

1. Van Veen, B. D. et al. (1997). *Localization of brain electrical
   activity via linearly constrained minimum variance spatial
   filtering*. IEEE Trans. Biomed. Eng. 44(9): 867–880.
   doi:10.1109/10.623056.
2. Gross, J. et al. (2001). *Dynamic imaging of coherent sources:
   Studying neural interactions in the human brain*. PNAS 98(2):
   694–699. doi:10.1073/pnas.98.2.694.
