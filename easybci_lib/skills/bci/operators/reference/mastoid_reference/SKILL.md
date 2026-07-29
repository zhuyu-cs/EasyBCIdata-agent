---
name: mastoid_reference
description: "Linked-mastoid (A1/A2) reference — AASM sleep staging standard"
layer: L3
group: reference
metadata:
  tags: [operator, reference, mastoid, linked_mastoid, a1_a2, sleep, aasm]
  modalities: [eeg]
  step_string: "mastoid_reference"
  analysis_goal_allowed: [classification, feature_extraction, clinical_screening, exploratory, generic, online_inference]
  analysis_goal_forbidden: []
---
# Mastoid / Linked-Mastoid Reference

## Function

Re-references EEG to a single mastoid (A1 or A2) or the average of both
(linked-mastoid). Linked-mastoid is the AASM-recommended montage for
**clinical sleep staging** and standard for many event-related potential
labs.

Input / Output: `(n_channels, n_times)` → `(n_channels, n_times)`.

## Algorithm & Math

```
ref(t) = (data[A1, t] + data[A2, t]) / 2      # linked-mastoid
out[c, t] = data[c, t] - ref(t)               # for all c
```

Single-mastoid uses `data[A1]` or `data[A2]` alone.

## Parameter Format & Defaults

`mastoid_reference:{mode}`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `mode` | str | "linked" | "linked" / "a1" / "a2". |
| `a1_ch`, `a2_ch` (kw) | str, str | "A1","A2" | Channel name lookup substrings. |

## Modality-Specific Considerations

EEG only. Standard in clinical sleep (AASM), audiometry, ERP labs.

## When to Use / NOT to Use

**Use** when: AASM sleep staging; clinical screening pipelines.

**Don't use** when: source localization (use REST); connectivity (use
REST); A1/A2 channels absent.

## Constraints & Ordering

- Apply **after** bandpass + drop_bads.
- Apply **before** epoching.
- Mastoid channels must be present.

## Failure Modes & Detection

| Failure | Symptom | Detection |
|---|---|---|
| A1/A2 missing | KeyError. | Raise recoverable; suggest CAR. |
| One mastoid bad | Linked-mastoid biased toward good side. | Pre-check `np.std(data[a1])` vs `data[a2]`; warn if 5× different. |

## Common Issues

- **"My P300 latency shifted."** Mastoid reference does shift potentials
  slightly relative to CAR; document the choice in `plan/reasoning.md`.

## Reference Implementation

### Standalone

```python
from __future__ import annotations
import numpy as np


def mastoid_reference(
    data: np.ndarray, a1_idx: int, a2_idx: int | None = None, mode: str = "linked"
) -> np.ndarray:
    """Mastoid re-reference."""
    if mode == "linked" and a2_idx is None:
        raise ValueError("linked mode requires both A1 and A2 indices")
    if mode == "linked":
        ref = (data[a1_idx] + data[a2_idx]) / 2.0
    elif mode == "a1":
        ref = data[a1_idx]
    else:  # a2
        if a2_idx is None: raise ValueError("a2 mode requires A2 index")
        ref = data[a2_idx]
    return (data - ref[None, :]).astype(data.dtype, copy=False)
```

### EasyBCI-Adapted

```python
from typing import Any, Dict
import time
import numpy as np
from easybci_lib.tools.neural_processing.operator_errors import EasyBCIOperatorError
from easybci_lib.tools.neural_processing.preprocess.step_cache import record_step_elapsed


def operator_mastoid_reference(
    data_dict: Dict[str, Any], *,
    mode: str = "linked", a1_ch: str = "A1", a2_ch: str = "A2",
) -> Dict[str, Any]:
    """Mastoid / linked-mastoid re-reference.

    Parameters
    ----------
    data_dict : dict
    mode : str
        "linked" / "a1" / "a2".
    a1_ch, a2_ch : str
        Channel substrings.

    Returns
    -------
    dict — re-referenced data.

    Raises
    ------
    EasyBCIOperatorError
        recoverable=True if mastoid channel(s) absent.

    Modality coverage
    -----------------
    EEG: yes. Others: forbidden.

    References
    ----------
    AASM sleep staging manual; Berger 1929.
    """
    channels = data_dict.get("channels", [])
    a1_idx = next((i for i, c in enumerate(channels) if a1_ch.lower() in c.lower()), None)
    a2_idx = next((i for i, c in enumerate(channels) if a2_ch.lower() in c.lower()), None)
    if mode in ("linked", "a1") and a1_idx is None:
        raise EasyBCIOperatorError(
            operator="mastoid_reference", reason=f"channel matching {a1_ch!r} not found",
            recoverable=True, fallback_step="car:median",
        )
    if mode in ("linked", "a2") and a2_idx is None:
        raise EasyBCIOperatorError(
            operator="mastoid_reference", reason=f"channel matching {a2_ch!r} not found",
            recoverable=True, fallback_step="car:median",
        )

    t0 = time.monotonic()
    data = data_dict["data"]
    if mode == "linked":
        ref = (data[a1_idx] + data[a2_idx]) / 2.0
    elif mode == "a1":
        ref = data[a1_idx]
    else:
        ref = data[a2_idx]
    new_data = (data - ref[None, :]).astype(data.dtype, copy=False)
    elapsed = time.monotonic() - t0

    out = dict(data_dict)
    out["data"] = new_data
    out["elapsed_s"] = elapsed
    out["meta"] = {**out.get("meta", {}), "mastoid_reference": {"mode": mode}}
    record_step_elapsed("mastoid_reference", elapsed, (data_dict.get("meta") or {}).get("step_cache_key"))
    return out
```

## References

1. American Academy of Sleep Medicine (2020). *The AASM Manual for the
   Scoring of Sleep and Associated Events*. — linked-mastoid is the
   standard reference montage.
2. Berger, H. (1929). *Über das Elektrenkephalogramm des Menschen*.
   Archiv für Psychiatrie und Nervenkrankheiten 87: 527–570 — original
   single-mastoid reference.
3. Yao, D. et al. (2019). *Which reference should we use for EEG and
   ERP practice?* Brain Topography 32(4): 530–549.
   doi:10.1007/s10548-019-00707-x — montage comparison.
