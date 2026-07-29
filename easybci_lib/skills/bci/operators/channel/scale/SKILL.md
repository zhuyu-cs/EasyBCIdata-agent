---
name: scale
description: "Scale/normalize signal amplitude — robust, standard, or numeric factor"
layer: L3
group: channel
metadata:
  tags: [operator, scale, normalize, standardize, robust]
  modalities: [eeg, seeg, ecog, meg, fnirs]
  step_string: "scale"
  analysis_goal_allowed: [classification, feature_extraction, exploratory, generic, online_inference]
  analysis_goal_forbidden: [source_localization]
---
# Scale / Normalize

## Function

Normalizes signal amplitude per channel. Three modes: robust scaling (median + IQR), standard scaling (mean + std), or multiplication by a numeric factor.

## Parameter Format

`scale:{method}`

Examples:
- `scale:robust` — RobustScaler (median-centered, IQR-normalized)
- `scale:standard` — StandardScaler (zero-mean, unit-variance)
- `scale:1e6` — Multiply by 1,000,000 (volts → microvolts)

## Parameters

| Parameter | Type | Options | Description |
|-----------|------|---------|-------------|
| method | string or float | `robust`, `standard`, or numeric | Scaling method |

## Method Details

| Method | Formula | Use Case |
|--------|---------|----------|
| `robust` | `(x - median) / IQR` | Outlier-resistant; best for EEG with artifacts |
| `standard` | `(x - mean) / std` | When Gaussian assumption holds; ML default |
| `{number}` | `x * factor` | Unit conversion (e.g., V → uV) |

Scaling is applied **per channel** (fit across time dimension).

## When to Use

- Before feeding data to machine learning models (normalization required)
- When combining data from different amplifiers (different gain settings)
- Unit conversion for display or compatibility
- After all filtering steps (scaling before filter can cause numerical issues)

## When NOT to Use

- Before ICA (ICA handles scale internally)
- If downstream analysis is scale-invariant (e.g., correlation-based methods)
- For spike rate data (already in meaningful units: spikes/s)

## Ordering

- Apply as one of the **last** steps (after all filtering, resampling)
- Apply AFTER: notch, bandpass, ICA, resample
- Apply BEFORE: clip (clip operates on scaled values)

## Recommended Parameters

| Paradigm | Method | Rationale |
|----------|--------|-----------|
| Motor Imagery (CSP) | robust | CSP sensitive to outliers; robust handles artifact trials |
| P300/ERP | standard | ERP averaging benefits from zero-mean channels |
| Deep learning input | standard | Most DL models expect ~N(0,1) input |
| General | robust | Safer default for EEG data with potential outliers |

## Reference Code

### Robust Scaling

```python
from sklearn.preprocessing import RobustScaler
import numpy as np

# data shape: (n_channels, n_samples)
data = RobustScaler().fit_transform(data.T).T.astype(np.float32)
```

### Standard Scaling

```python
from sklearn.preprocessing import StandardScaler
import numpy as np

data = StandardScaler().fit_transform(data.T).T.astype(np.float32)
```

### Factor Multiplication

```python
# Convert volts to microvolts
data = data * 1e6
```

### Key API

- `RobustScaler()` — uses median and IQR, resistant to outliers
- `StandardScaler()` — uses mean and std, assumes Gaussian
- `.fit_transform(data.T).T` — scale per channel (fit across time axis)
