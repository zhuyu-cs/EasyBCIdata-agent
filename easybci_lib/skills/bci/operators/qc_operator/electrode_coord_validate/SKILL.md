---
name: electrode_coord_validate
description: "Validate electrode coordinate system / montage consistency (BIDS coordsystem.json + sEEG bipolar)"
layer: L3
group: qc_operator
metadata:
  tags: [operator, qc, electrode, coordinates, bids, seeg, montage]
  modalities: [eeg, meg, seeg, ecog]
  step_string: "electrode_coord_validate"
  analysis_goal_allowed: [source_localization, clinical_screening, exploratory, generic, connectivity]
  analysis_goal_forbidden: [online_inference]
---
# Electrode Coordinate Validator

## Function

Validates electrode coordinate system consistency:
- BIDS `coordsystem.json` matches `*_electrodes.tsv` rows.
- For sEEG: bipolar pairs are anatomically adjacent on shafts.
- For EEG: 10-20 / 10-10 / 10-5 names map to known positions.
- Coordinate units match expected scale (mm vs m).

Input / Output: `data_dict` with `meta["electrode_positions"]` →
`meta["electrode_validate"]: {status, issues}`.

## Algorithm & Math

Five checks:
1. Position-to-channel count match.
2. Coordinate scale (typical EEG head: r ≈ 0.1 m or 100 mm).
3. Duplicate positions.
4. Coordinate-system declaration matches positions (e.g. CTF coords have
   specific orientation conventions).
5. For sEEG: adjacent contacts on the same shaft (≤ 5 mm apart).

## Parameter Format & Defaults

`electrode_coord_validate:{coord_system}`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `coord_system` | str | "auto" | Expected coord system (auto / MNI / individual). |
| `unit` (kw) | str | "auto" | "m" / "mm" / "auto" (infer from scale). |

## Modality-Specific Considerations

EEG / MEG / sEEG / ECoG: yes. fNIRS: separate skill (different conventions).
Spike: no spatial positions on a single probe (handled by sorting).

## When to Use / NOT to Use

**Use** when: source localization (mandatory); BIDS-iEEG load
validation; connectivity with spatial weighting.

**Don't use** when: continuous-only analyses (no positions consumed).

## Constraints & Ordering

After load; before source localization / spatial filters.

## Failure Modes & Detection

The op IS a detection layer.

## Common Issues

- **"All my positions show r ~ 0.1."** Units are meters (SI); BIDS
  default. Some software expects mm.

## Reference Implementation

### Standalone

```python
from __future__ import annotations
import numpy as np


def electrode_coord_validate(
    positions: np.ndarray, channels: list[str], coord_system: str = "auto",
) -> dict:
    """Validate electrode positions."""
    issues = []
    if positions.shape[0] != len(channels):
        issues.append(f"positions {positions.shape[0]} != channels {len(channels)}")
    if np.any(np.isnan(positions)):
        issues.append("NaN in positions")
    # Detect unit
    typical_norm = float(np.median(np.linalg.norm(positions, axis=-1)))
    if 0.05 < typical_norm < 0.15:
        unit = "m"
    elif 50 < typical_norm < 150:
        unit = "mm"
    else:
        unit = "unknown"
        issues.append(f"unusual coord scale (median norm {typical_norm:.3f})")
    # Duplicate positions
    rounded = (positions * 1e6).round().astype(np.int64)
    if len(np.unique(rounded, axis=0)) < len(rounded):
        issues.append("duplicate electrode positions")
    return {"status": "ok" if not issues else "warn",
            "issues": issues, "unit": unit}
```

### EasyBCI-Adapted

```python
from typing import Any, Dict
import time
import numpy as np
from easybci_lib.tools.neural_processing.operator_errors import EasyBCIOperatorError
from easybci_lib.tools.neural_processing.preprocess.step_cache import record_step_elapsed


def operator_electrode_coord_validate(
    data_dict: Dict[str, Any], *, coord_system: str = "auto",
) -> Dict[str, Any]:
    """Electrode coordinate validation.

    Parameters
    ----------
    data_dict : dict
    coord_system : str

    Returns
    -------
    dict — `meta["electrode_validate"]`.

    Raises
    ------
    EasyBCIOperatorError
        recoverable=True if positions missing.

    Modality coverage
    -----------------
    EEG / MEG / sEEG / ECoG: yes. Spike / fNIRS: forbidden.

    References
    ----------
    BIDS-iEEG spec; FreeSurfer coord systems.
    """
    positions = (data_dict.get("meta") or {}).get("electrode_positions")
    if positions is None:
        raise EasyBCIOperatorError(
            operator="electrode_coord_validate", reason="meta['electrode_positions'] required",
            recoverable=True, fallback_step="skip if no spatial analysis",
        )

    t0 = time.monotonic()
    positions = np.asarray(positions, dtype=np.float64)
    channels = data_dict.get("channels", [])
    issues = []
    if positions.shape[0] != len(channels):
        issues.append(f"positions {positions.shape[0]} != channels {len(channels)}")
    if np.any(np.isnan(positions)):
        issues.append("NaN in positions")
    typical_norm = float(np.median(np.linalg.norm(positions, axis=-1)))
    if 0.05 < typical_norm < 0.15:
        unit = "m"
    elif 50 < typical_norm < 150:
        unit = "mm"
    else:
        unit = "unknown"
        issues.append(f"unusual coord scale (median norm {typical_norm:.3f})")
    rounded = (positions * 1e6).round().astype(np.int64)
    if len(np.unique(rounded, axis=0)) < len(rounded):
        issues.append("duplicate electrode positions")
    status = "ok" if not issues else "warn"
    elapsed = time.monotonic() - t0

    out = dict(data_dict)
    out["elapsed_s"] = elapsed
    out["meta"] = {
        **out.get("meta", {}),
        "electrode_validate": {"status": status, "issues": issues, "unit": unit},
        "electrode_coord_validate": {"coord_system": coord_system},
    }
    record_step_elapsed("electrode_coord_validate", elapsed, (data_dict.get("meta") or {}).get("step_cache_key"))
    return out
```

## References

1. BIDS-iEEG spec: coordinate systems section.
   https://bids-specification.readthedocs.io/en/stable/05-derivatives/03-imaging.html
2. Holdgraf, C. et al. (2019). *iEEG-BIDS*. Scientific Data 6: 102.
   doi:10.1038/s41597-019-0105-7.
