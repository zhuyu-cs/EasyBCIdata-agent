---
name: ssvep
description: 'SSVEP frequency detection: FFT peak analysis, CCA, filter bank methods'
version: 1.0.0
layer: L2
group: paradigm
metadata:
  tags:
  - eeg
  - ssvep
  - frequency
  - fft
  - cca
  - filter_bank
  modalities:
  - eeg
  paradigms:
  - ssvep
  - steady_state
  analysis_goal_allowed:
  - classification
  - online_inference
  - feature_extraction
  - exploratory
  - generic
  analysis_goal_forbidden:
  - source_localization
  - clinical_screening
---
# SSVEP (Steady-State Visual Evoked Potential) Processing

## Overview

SSVEP BCIs detect brain responses at the exact frequency of visual flickers. The signal is highly periodic and concentrated at stimulus frequency and its harmonics. Processing focuses on maximizing frequency resolution and SNR in the occipital channels.

## Recommended Pipeline

```
notch:50 → bandpass:3,90 → resample:512 → drop_bads → scale:robust
```

### Step Rationale

1. **notch:50** — Remove line noise. Critical for SSVEP because 50 Hz is a common stimulus frequency and its interference must be eliminated before frequency analysis.
2. **bandpass:3,90** — Wide passband to capture fundamental frequencies (typically 5-45 Hz) AND second harmonics (up to ~90 Hz). SSVEP at 40 Hz has harmonics at 80 Hz.
3. **resample:512** — Higher sampling rate than MI/P300 because SSVEP uses frequencies up to 45+ Hz. 512 Hz gives Nyquist at 256 Hz, adequate for harmonics up to the 3rd.
4. **drop_bads** — Remove channels that don't contribute to occipital patterns.
5. **scale:robust** — Robust scaling handles amplitude variability across sessions.

## Key Parameters

| Parameter | Typical Range | Notes |
|-----------|--------------|-------|
| Stimulus frequencies | 5-45 Hz | Must avoid 50/60 Hz line frequency |
| Frequency resolution | 0.25-0.5 Hz | Requires 2-4 s epochs minimum |
| Key channels | O1, Oz, O2, POz | Occipital region |
| Harmonics used | 1st-3rd | Higher harmonics improve classification |
| Minimum epoch | 1/Δf seconds | Δf = minimum frequency spacing |

## Detection Methods

1. **FFT Peak Detection** — Simplest; compute PSD, find peak at target frequencies
2. **CCA (Canonical Correlation Analysis)** — Reference signals are sine/cosine at each frequency + harmonics. Best single-method accuracy.
3. **Filter Bank CCA (FBCCA)** — Decompose into sub-bands, apply CCA to each, weight by band importance
4. **TRCA (Task-Related Component Analysis)** — Uses training data to find maximally reproducible components

## Quality Checks

- **SNR at target frequency**: Signal power at f₀ should be >3 dB above background (neighbors ±1 Hz)
- **Harmonic presence**: 2nd harmonic (2f₀) should be detectable for good SSVEP responders
- **Occipital dominance**: Power at f₀ should be highest at O1/Oz/O2, not frontal (artifact indicator)
- **Frequency spacing**: Stimuli must be separated by at least the frequency resolution (1/epoch_length)

## Segmentation

- **Epoch window**: 0.5 to end of stimulus (skip first 0.5 s for SSVEP onset transient)
- **Epoch length**: 2-5 s (longer = better frequency resolution)
- **Overlap**: 50% overlap for sliding-window detection (real-time BCI)
- **No baseline correction**: Unlike ERP, SSVEP is ongoing — no pre-stimulus baseline concept

## Common Issues

- **Cannot distinguish close frequencies**: Increase epoch length for better frequency resolution, or use CCA which tolerates shorter windows
- **Line noise at stimulus frequency**: If stimulus is at 50 Hz, use notch carefully or choose different frequency. Interpolation-based notch preferred over sharp IIR
- **Low SSVEP amplitude**: Check if subject is fixating on the stimulus. Peripheral vision produces weaker SSVEP. Consider higher contrast stimuli
- **Harmonic contamination**: 10 Hz stimulus harmonic at 20 Hz may overlap with another target. Design frequency set to avoid harmonic collisions

## Complete Pipeline Example

```python
import mne
import numpy as np
from sklearn.preprocessing import RobustScaler

# Load data
raw = mne.io.read_raw_edf("sub01_ssvep.edf", preload=True, verbose=False)

# notch:50 → bandpass:3,90 → car → resample:512 → scale:robust
raw.notch_filter(50.0, verbose=False)
raw.filter(l_freq=3, h_freq=90, verbose=False)
raw.set_eeg_reference('average', projection=False, verbose=False)
raw.resample(512.0, verbose=False)

# Drop bad channels
# raw.drop_channels(bad_list)

# Extract and scale
data = raw.get_data().astype(np.float32)
sfreq = raw.info['sfreq']
channels = list(raw.ch_names)
data = RobustScaler().fit_transform(data.T).T.astype(np.float32)

# CCA-based frequency detection
from sklearn.cross_decomposition import CCA
stim_freqs = [8.0, 10.0, 12.0, 15.0]
n_harmonics = 3
# Build reference signals: sin/cos at each freq + harmonics
# Apply CCA between EEG segments and reference signals
```
