---
name: fill_nan
description: "Replace non-finite values (NaN, Inf) by per-channel linear interpolation or a constant"
layer: L3
group: channel
metadata:
  tags: [operator, nan, missing, cleanup, repair]
  modalities: [eeg, seeg, ecog, meg, spike, fnirs]
  step_string: "fill_nan"
  analysis_goal_allowed: [classification, source_localization, feature_extraction, clinical_screening, exploratory, generic, connectivity, phase_amplitude_coupling, online_inference]
  analysis_goal_forbidden: []
---
# Fill NaN

## Function

Replaces all non-finite values (NaN, +Inf, -Inf) in the data array, either by per-channel linear interpolation between the neighbouring finite samples or by a constant. A data cleaning step for corrupted samples.

## Parameter Format

`fill_nan:{linear|constant|value}`

Examples:
- `fill_nan:linear` — Interpolate each channel linearly across the gap; leading/trailing gaps hold the nearest finite sample; an all-non-finite channel becomes 0
- `fill_nan:0` — Replace with zero
- `fill_nan:-1` — Replace with -1

Default (no parameter, or `constant`): fills with 0. Any other non-numeric parameter is an error.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| method / value | `linear` \| `constant` \| float | 0.0 | `linear` interpolates; a number is the constant replacement value |

## When to Use

- Data contains NaN values from hardware glitches, corrupted samples, or failed sensor readings
- After loading data that has missing value markers
- Before operations that cannot handle NaN (most numpy operations propagate NaN)
- fNIRS data with dropout periods

## When NOT to Use

- Data is clean (no NaN/Inf present) — this step is a no-op in that case
- When NaN represents meaningful gaps that should be handled differently (e.g., epoch rejection)
- After scaling if zero-fill would create outliers (use the scaled mean instead)

## Constraints

- `linear` only bridges gaps from neighbouring samples — long gaps become straight lines; prefer bad-segment marking for them
- Logs the count of replaced values
- If many values are non-finite (> 10%), consider the data quality and whether processing is meaningful

## Ordering

- Apply EARLY — before any filtering or mathematical operations
- NaN propagates through all subsequent steps if not handled
- Typically first or second step if NaN is detected during inspection

## Reference Code

### Implementation

```python
import numpy as np

fill_val = 0.0
data = np.nan_to_num(data, nan=fill_val, posinf=fill_val, neginf=fill_val)
```

### With Logging

```python
import numpy as np

mask = ~np.isfinite(data)
n_bad = np.count_nonzero(mask)
if n_bad > 0:
    print(f"Replacing {n_bad} non-finite values with 0")
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
```

### Key API

- `np.nan_to_num(data, nan=val, posinf=val, neginf=val)`
- `np.isfinite(data)` — mask for detecting non-finite values
