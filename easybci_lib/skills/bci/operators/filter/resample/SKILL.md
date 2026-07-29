---
name: resample
description: "Resample signal to target frequency — downsampling with anti-aliasing"
layer: L3
group: filter
metadata:
  tags: [operator, resample, downsample, frequency]
  modalities: [eeg, seeg, ecog, meg]
  step_string: "resample"
  analysis_goal_allowed: [classification, source_localization, feature_extraction, clinical_screening, exploratory, generic, connectivity, phase_amplitude_coupling, online_inference]
  analysis_goal_forbidden: []
---
# Resample

## Function

Changes the sampling rate of the signal to a target frequency. Uses MNE's polyphase resampling with automatic anti-aliasing filter. Preserves temporal alignment of events/annotations.

## Parameter Format

`resample:{target_hz}`

Examples:
- `resample:256` — Resample to 256 Hz
- `resample:500` — Resample to 500 Hz
- `resample:128` — Aggressive downsample for low-frequency analysis

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| target_hz | float | required | Target sampling rate in Hz |

## When to Use

- Standardize sampling rate across subjects/sessions recorded at different rates
- Reduce data volume for faster processing (fewer samples per second)
- Match the sampling rate expected by downstream models or feature extraction

## When NOT to Use

- Target rate is same as current rate (no-op, returns immediately)
- Need to preserve high-frequency content: don't downsample below 2x your max frequency of interest (Nyquist)
- Spike sorting data (temporal precision matters at sub-millisecond level)

## Constraints

- **Nyquist rule**: Target rate must be >= 2x the highest frequency of interest. Example: if you need 40 Hz content, sample at >= 80 Hz (128 or 256 Hz recommended)
- Anti-aliasing filter is applied automatically before downsampling
- Upsampling (target > current) is valid but uncommon and doesn't add information

## Ordering

- Apply AFTER: bandpass (remove high frequencies first to avoid aliasing), notch, ICA
- Apply BEFORE: scale, clip (these are rate-independent)
- Rationale: filter before downsample ensures clean anti-aliasing

## Recommended Parameters

| Paradigm | Target Hz | Rationale |
|----------|-----------|-----------|
| Motor Imagery (EEG) | 256 | Preserves up to 128 Hz; standard |
| P300/ERP | 256 | Standard; 128 Hz also acceptable |
| SSVEP | 256 | Need precision for stimulus frequency harmonics |
| sEEG/ECoG | 500-1000 | Preserve high-gamma (70-150 Hz) |
| Sleep staging | 128 | Only need up to 35 Hz |
| General EEG | 256 | Good default for most analyses |

## Reference Code

### MNE Chain (on existing Raw object)

```python
raw.resample(256.0, verbose=False)
```

### Standalone Implementation

```python
import mne
import numpy as np

target_sfreq = 256.0
info = mne.create_info(ch_names, sfreq, ch_types='eeg')
raw = mne.io.RawArray(data, info, verbose=False)
raw.resample(target_sfreq, verbose=False)
data = raw.get_data().astype(np.float32)
sfreq = target_sfreq
```

### Key API

- `raw.resample(sfreq, verbose=False)`
- Anti-aliasing filter applied automatically before downsampling
