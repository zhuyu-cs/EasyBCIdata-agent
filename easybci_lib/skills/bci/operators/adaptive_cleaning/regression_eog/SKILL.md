---
name: regression_eog
description: "Regression-based EOG removal — fast online ocular artifact suppression"
layer: L3
group: adaptive_cleaning
metadata:
  tags: [operator, adaptive_cleaning, regression, eog, gratton, fast, online]
  modalities: [eeg, meg]
  step_string: "regression_eog"
  analysis_goal_allowed: [classification, source_localization, feature_extraction, clinical_screening, exploratory, generic, connectivity, online_inference]
  analysis_goal_forbidden: []
---
# Regression-Based EOG Removal

## Function

Linear regression of EOG channels out of EEG/MEG channels. Faster than
ICA / SSP, no labelled events needed — the de-facto online BCI choice
when an EOG channel is present. Originally Gratton et al. 1983.

Input / Output: `(n_channels, n_times)` → `(n_channels, n_times)` with
EOG-related variance removed.

## Algorithm & Math

```
β_c = argmin || data[c] − β · eog ||²
out[c] = data[c] − β_c · eog
```

Optionally per-band (Croft & Barry 2000): bandpass first, then regress
per band.

## Parameter Format & Defaults

`regression_eog:{eog_ch}`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `eog_ch` | str | "EOG" | EOG channel substring. |
| `band` (kw) | tuple | None | Optional per-band regression. |

## Modality-Specific Considerations

EEG: standard. MEG: viable (cardiac too via separate channel).
sEEG / ECoG / fNIRS / spike: forbidden (no EOG analog).

## When to Use / NOT to Use

**Use** when: online BCI; latency-critical; EOG channel present.

**Don't use** when: no EOG; PAC / connectivity require very precise
preservation (use SSP or ICA).

## Constraints & Ordering

- Apply after bandpass / drop_bads.
- Apply before epoching / decoder.
- EOG channel kept or dropped after (configurable).

## Failure Modes & Detection

| Failure | Symptom | Detection |
|---|---|---|
| EOG missing | KeyError. | Pre-check; raise recoverable. |
| Cross-talk overcorrection | Frontal EEG flattened. | If frontal alpha drop > 30% warn. |

## Common Issues

- **"Frontal EEG looks flat."** EOG-EEG correlation high; the regression
  removed real frontal signal. Use SSP / ICA for higher-quality removal.

## Reference Implementation

### Standalone

```python
from __future__ import annotations
import numpy as np


def regression_eog(
    data: np.ndarray, eog_idx: int,
) -> np.ndarray:
    """Linear regression of EOG out of EEG."""
    eog = data[eog_idx]
    out = data.copy()
    eog_var = float(np.var(eog)) + 1e-30
    for c in range(data.shape[0]):
        if c == eog_idx: continue
        beta = float(np.dot(data[c], eog)) / (eog_var * data.shape[1])
        out[c] = data[c] - beta * eog
    return out.astype(data.dtype, copy=False)
```

### EasyBCI-Adapted

```python
from typing import Any, Dict
import time
import numpy as np
from easybci_lib.tools.neural_processing.operator_errors import EasyBCIOperatorError
from easybci_lib.tools.neural_processing.preprocess.step_cache import record_step_elapsed


def operator_regression_eog(
    data_dict: Dict[str, Any], *, eog_ch: str = "EOG",
) -> Dict[str, Any]:
    """Regression-based EOG removal.

    Parameters
    ----------
    data_dict : dict
    eog_ch : str

    Returns
    -------
    dict — cleaned data.

    Raises
    ------
    EasyBCIOperatorError
        recoverable=True if EOG channel absent.

    Modality coverage
    -----------------
    EEG / MEG: yes. Others: forbidden.

    References
    ----------
    Gratton et al. 1983; Croft & Barry 2000.
    """
    channels = data_dict.get("channels", [])
    eog_idx = next((i for i, c in enumerate(channels) if eog_ch.lower() in c.lower()), None)
    if eog_idx is None:
        raise EasyBCIOperatorError(
            operator="regression_eog", reason=f"no channel matching {eog_ch!r}",
            recoverable=True, fallback_step="ica",
        )

    t0 = time.monotonic()
    data = data_dict["data"]
    eog = data[eog_idx]
    eog_var = float(np.var(eog)) + 1e-30
    new_data = data.copy()
    for c in range(data.shape[0]):
        if c == eog_idx: continue
        beta = float(np.dot(data[c], eog)) / (eog_var * data.shape[1])
        new_data[c] = data[c] - beta * eog
    new_data = new_data.astype(data.dtype, copy=False)
    elapsed = time.monotonic() - t0

    out = dict(data_dict)
    out["data"] = new_data
    out["elapsed_s"] = elapsed
    out["meta"] = {**out.get("meta", {}), "regression_eog": {"eog_ch": eog_ch}}
    record_step_elapsed("regression_eog", elapsed, (data_dict.get("meta") or {}).get("step_cache_key"))
    return out
```

## References

1. Gratton, G. et al. (1983). *A new method for off-line removal of
   ocular artifact*. EEG Clin. Neurophysiol. 55(4): 468–484.
   doi:10.1016/0013-4694(83)90135-9.
2. Croft, R. J., & Barry, R. J. (2000). *Removal of ocular artifact from
   the EEG: a review*. Neurophysiologie Clinique 30(1): 5–19.
   doi:10.1016/S0987-7053(00)00055-1.
