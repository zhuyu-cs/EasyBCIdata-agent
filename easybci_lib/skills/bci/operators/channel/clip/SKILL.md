---
name: clip
description: "Clamp signal amplitude to a maximum absolute value"
layer: L3
group: channel
metadata:
  tags: [operator, clip, clamp, artifact, threshold]
  modalities: [eeg, seeg, ecog, meg]
  step_string: "clip"
  analysis_goal_allowed: [classification, source_localization, feature_extraction, clinical_screening, exploratory, generic, connectivity, phase_amplitude_coupling, online_inference]
  analysis_goal_forbidden: []
---
# Clip

## Function

Clamps all data values to the range `[-max_abs, +max_abs]`. Values exceeding the threshold are replaced with the threshold value. Simple but effective artifact suppression.

## Parameter Format

`clip:{max_abs}`

Examples:
- `clip:5` — Clamp to [-5, +5] (after standard scaling)
- `clip:100` — Clamp to [-100, +100] (raw microvolts)
- `clip:3` — Aggressive clipping at 3 standard deviations

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| max_abs | float | required | Maximum absolute amplitude |

## When to Use

- After scaling, to suppress remaining outlier samples (> 3-5 SD)
- As a simple artifact rejection: large transients are truncated
- Before ML model input to prevent extreme values from dominating gradients
- When you want to keep all time points (vs. epoch rejection which discards entire segments)

## When NOT to Use

- Before ICA (clipping destroys the linearity ICA relies on)
- When absolute amplitude is meaningful (e.g., analyzing ERD/ERS magnitude)
- For spike data (amplitude carries information)

## Choosing the Threshold

| After scaling method | Recommended clip | Rationale |
|---------------------|-----------------|-----------|
| `scale:robust` | clip:5 | IQR-based; 5 IQRs captures 99.9% of clean data |
| `scale:standard` | clip:3 or clip:5 | 3 SD = 99.7% of Gaussian; 5 SD = very conservative |
| No scaling | clip:100 (uV) | Typical EEG range is ±100 uV |

## Ordering

- Apply AFTER: scale (so threshold is in normalized units)
- Apply as one of the last steps
- Apply BEFORE: final output / ML feature extraction

## Reference Code

### Implementation

```python
import numpy as np

max_abs = 5.0  # threshold in scaled units
data = np.clip(data, -max_abs, max_abs)
```

### Key API

- `np.clip(data, -max_abs, max_abs)` — in-place-safe clamping
