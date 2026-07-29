---
name: p300_erp
description: 'P300/ERP processing: baseline correction, epoch alignment, trial averaging'
version: 1.0.0
layer: L2
group: paradigm
metadata:
  tags:
  - eeg
  - p300
  - erp
  - speller
  - oddball
  - event_related
  modalities:
  - eeg
  paradigms:
  - p300
  - erp
  - oddball
  - speller
  analysis_goal_allowed:
  - classification
  - feature_extraction
  - clinical_screening
  - exploratory
  - generic
  analysis_goal_forbidden:
  - online_inference
  - phase_amplitude_coupling
---
# P300 / Event-Related Potential Processing

## Overview

P300-based BCIs detect a positive voltage deflection ~300 ms after rare target stimuli (oddball paradigm). The signal is small (5-15 μV) relative to background EEG (~50 μV), requiring careful preprocessing to maximize signal-to-noise ratio through averaging and artifact rejection.

## Recommended Pipeline

```
notch:50 → bandpass:0.1,20 → resample:256 → drop_bads → scale:standard
```

### Step Rationale

1. **notch:50** — Remove line noise. P300 is below 20 Hz, but notch prevents spectral leakage during filtering.
2. **bandpass:0.1,20** — P300 energy is concentrated between 0.5-8 Hz. The wide 0.1 Hz high-pass preserves slow cortical potentials and prevents baseline distortion. 20 Hz low-pass removes EMG/alpha while keeping the P300 waveform intact.
3. **resample:256** — Standard rate. P300's temporal resolution (tens of ms) doesn't benefit from higher sampling.
4. **drop_bads** — Remove noisy channels. Single bad electrode in averaging can mask P300.
5. **scale:standard** — Z-score normalization. ERP analyses often assume Gaussian-like distributions for statistical testing.

## Key Temporal Components

| Component | Latency | Amplitude | Topography |
|-----------|---------|-----------|------------|
| N100 | 80-120 ms | -2 to -5 μV | Fronto-central |
| P200 | 150-250 ms | +2 to +5 μV | Central |
| N200 | 200-300 ms | -2 to -4 μV | Fronto-central |
| **P300** | 250-500 ms | +5 to +15 μV | **Centro-parietal (Pz)** |
| P300a | 250-300 ms | Novelty | Frontal |
| P300b | 300-500 ms | Target detection | Parietal |

## Quality Checks

- **Trial count**: Minimum 30 target trials for reliable P300 estimation (>60 preferred)
- **Artifact rejection**: Reject epochs with |amplitude| > 100 μV (eye blinks, movements)
- **Target/non-target ratio**: Standard oddball uses 15-20% targets
- **Grand average check**: P300 should be visible at Pz in averaged target waveform
- **Baseline stability**: Pre-stimulus baseline (-200 to 0 ms) should be near zero after correction

## Segmentation

- **Epoch window**: -0.2 to 0.8 s relative to stimulus onset
- **Baseline correction**: -200 to 0 ms (subtract mean pre-stimulus voltage)
- **Overlap**: None — epochs are stimulus-locked, non-overlapping
- **Rejection threshold**: ±100 μV (or ±75 μV for stringent cleaning)

## Common Issues

- **No visible P300**: Check stimulus timing accuracy, verify event markers aligned correctly, ensure sufficient target trials
- **Latency jitter**: P300 latency varies with task difficulty. Consider dynamic time warping or single-trial latency estimation
- **Alpha contamination**: Prominent 10 Hz oscillation in single trials — verify bandpass is working, check if subject was drowsy
- **Double-peaked P300**: Common in older adults or complex stimuli — not necessarily an artifact

## Complete Pipeline Example

```python
import mne
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load data
raw = mne.io.read_raw_edf("sub01_p300.edf", preload=True, verbose=False)

# notch:50 → bandpass:0.1,20 → resample:256 → drop_bads → scale:standard
raw.notch_filter(50.0, verbose=False)
raw.filter(l_freq=0.1, h_freq=20, verbose=False)
raw.resample(256.0, verbose=False)

# Drop bad channels
# raw.drop_channels(bad_list)

# Extract and scale
data = raw.get_data().astype(np.float32)
sfreq = raw.info['sfreq']
channels = list(raw.ch_names)
data = StandardScaler().fit_transform(data.T).T.astype(np.float32)

# Epoching for P300 (stimulus-locked)
events = mne.find_events(raw, verbose=False)
epochs = mne.Epochs(raw, events, tmin=-0.2, tmax=0.8,
                    baseline=(-0.2, 0), reject=dict(eeg=100e-6),
                    preload=True, verbose=False)
```
