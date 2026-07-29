---
name: seeg_epilepsy
description: 'sEEG epilepsy processing: bipolar reference, HFO detection, SOZ localization'
version: 1.0.0
layer: L2
group: clinical
metadata:
  tags:
  - seeg
  - ecog
  - epilepsy
  - hfo
  - bipolar
  - soz
  - intracranial
  modalities:
  - seeg
  - ecog
  paradigms:
  - seeg_epilepsy
  - epilepsy
  - intracranial
  analysis_goal_allowed:
  - clinical_screening
  - feature_extraction
  - exploratory
  analysis_goal_forbidden:
  - online_inference
---
# sEEG Epilepsy Monitoring Processing

## Overview

Stereo-EEG (sEEG) recordings from depth electrodes provide direct measurement of neural activity across cortical and subcortical structures. Epilepsy processing focuses on identifying seizure onset zones (SOZ) through ictal/interictal pattern analysis, high-frequency oscillations (HFOs), and connectivity measures.

## Recommended Pipeline

```
bipolar_ref → notch:50 → bandpass:1,500 → scale:robust
```

### Step Rationale

1. **bipolar_ref** — Bipolar montage between adjacent contacts on the same electrode shaft. Eliminates common-mode noise, improves spatial specificity, reveals local field potentials. Essential for sEEG before any frequency analysis.
2. **notch:50** — Remove line noise after bipolar referencing (some residual may remain). Apply to all harmonics up to Nyquist.
3. **bandpass:1,500** — Very wide passband. sEEG contains meaningful activity up to 500 Hz (HFOs: ripples 80-250 Hz, fast ripples 250-500 Hz). Low cutoff at 1 Hz removes DC drift while preserving slow oscillations relevant to seizure dynamics.
4. **scale:robust** — Channels have vastly different amplitudes depending on tissue proximity. Robust scaling normalizes without being distorted by ictal bursts.

## Key Frequency Bands

| Band | Range | Clinical Relevance |
|------|-------|--------------------|
| Delta | 1-4 Hz | Post-ictal slowing, NREM sleep |
| Theta | 4-8 Hz | Temporal lobe seizure onset |
| Alpha | 8-13 Hz | Background rhythm disruption |
| Beta | 13-30 Hz | Desynchronization before seizure |
| Gamma | 30-80 Hz | Ictal fast activity (onset marker) |
| Ripples | 80-250 Hz | **HFOs — interictal SOZ biomarker** |
| Fast Ripples | 250-500 Hz | **Strongest SOZ biomarker** |

## HFO Detection Criteria

- **Minimum duration**: 4 oscillations (or ~20 ms for ripples, ~8 ms for fast ripples)
- **Amplitude**: >3 standard deviations above background in the filtered band
- **Isolation**: Not associated with sharp transients (filter ringing artifacts)
- **Rate threshold**: >5 HFOs/minute in a channel suggests SOZ involvement

## Quality Checks

- **Bipolar channel count**: Verify adjacent contact pairing is anatomically correct
- **Saturation detection**: sEEG amplifiers clip during ictal events — flag channels with flat segments >100 ms
- **Cross-talk**: Adjacent bipolar channels shouldn't be perfectly correlated (indicates broken contact)
- **Noise floor**: Background noise should be <5 μV RMS after bipolar referencing
- **Sampling rate adequacy**: Must be ≥1000 Hz for HFO analysis (≥2000 Hz for fast ripples)

## Segmentation

- **Interictal analysis**: 5-minute segments during NREM sleep (highest HFO rates)
- **Ictal epochs**: -30 s to +60 s relative to seizure onset (marked by clinician)
- **SOZ mapping**: Continuous 1-hour segments with sliding 10-second windows

## Electrode Naming Convention

sEEG electrodes follow anatomical naming:
- Letter(s) = target structure (e.g., A = amygdala, H = hippocampus, TB = temporal basal)
- Number = contact position (1 = deepest/medial, increasing toward lateral)
- Bipolar pairs: A1-A2, A2-A3, etc. (adjacent contacts, ~3.5 mm spacing)

## Common Issues

- **High-frequency artifacts from stimulation**: If cortical stimulation mapping was performed, those segments must be excluded from HFO analysis
- **Volume conduction in monopolar**: If bipolar_ref was not applied, apparent synchrony between distant channels is likely volume conduction, not true connectivity
- **Filter ringing**: Sharp transients (spikes) produce oscillatory artifacts in narrow-band filters that mimic HFOs. Always verify HFO candidates in the raw unfiltered trace
- **Electrode drift**: Chronic recordings may show impedance changes over days — compare channel properties across recording sessions

## Complete Pipeline Example

```python
import mne
import numpy as np

# Load sEEG data
raw = mne.io.read_raw_edf("sub01_seeg.edf", preload=True, verbose=False)
ch_names = list(raw.ch_names)

# bipolar_ref → notch:50 → bandpass:1,500 → scale:robust
# Step 1: Bipolar reference (auto-pair adjacent contacts)
anodes, cathodes = [], []
for a, c in zip(ch_names[:-1], ch_names[1:]):
    prefix_a = ''.join(ch for ch in a if not ch.isdigit())
    prefix_c = ''.join(ch for ch in c if not ch.isdigit())
    if prefix_a == prefix_c:
        anodes.append(a)
        cathodes.append(c)

raw = mne.set_bipolar_reference(raw, anodes, cathodes, verbose='WARNING')

# Step 2: Notch + bandpass
raw.notch_filter(50.0, verbose=False)
raw.filter(l_freq=1, h_freq=500, verbose=False)

# Step 3: Extract and scale
data = raw.get_data().astype(np.float32)
sfreq = raw.info['sfreq']
channels = list(raw.ch_names)

from sklearn.preprocessing import RobustScaler
data = RobustScaler().fit_transform(data.T).T.astype(np.float32)

# HFO detection (ripple band 80-250 Hz)
from scipy.signal import butter, filtfilt
b, a = butter(4, [80, 250], btype='band', fs=sfreq)
ripple_band = filtfilt(b, a, data, axis=1)
# Envelope via Hilbert → threshold at mean + 3*std per channel
```
