---
name: sleep_staging
layer: L2
group: clinical
metadata:
  analysis_goal_allowed:
  - classification
  - clinical_screening
  - feature_extraction
  - exploratory
  analysis_goal_forbidden:
  - online_inference
tags:
- eeg
- sleep
- staging
- polysomnography
- psg
- spindle
- k_complex
- slow_wave
- rem
- nrem
- hypnogram
modality: eeg
---
# Sleep EEG — Polysomnography and Sleep Staging

## Signal Characteristics

| Property | Typical Value |
|----------|--------------|
| Sampling rate | 256–512 Hz |
| EEG Channels | 2–6 (clinical PSG: C3, C4, O1, O2, F3, F4) |
| Additional channels | EOG (×2), EMG (chin), ECG, respiratory, SpO2 |
| Recording duration | 6–10 hours (full night) |
| Reference | Contralateral mastoid (M1/M2) |
| Epoch length | 30 seconds (AASM standard) |

## Sleep Stages (AASM Scoring Manual)

| Stage | EEG Features | Duration |
|-------|-------------|----------|
| Wake (W) | Alpha (8–13 Hz) posterior, eye blinks | Variable |
| N1 (Light) | Theta (4–7 Hz), vertex sharp waves | 5% of TST |
| N2 (Light) | Sleep spindles (12–14 Hz), K-complexes | 45–55% of TST |
| N3 (Deep/SWS) | Delta (0.5–2 Hz) > 75 µV, ≥ 20% of epoch | 15–25% of TST |
| REM | Low-voltage mixed, sawtooth waves, rapid eye movements | 20–25% of TST |

## Key Graphoelements

| Feature | Frequency | Duration | Amplitude | Location |
|---------|-----------|----------|-----------|----------|
| Sleep spindle | 11–16 Hz | 0.5–2 s | 25–50 µV | Central (C3/C4) |
| K-complex | 0.5–1.5 Hz | > 0.5 s | > 75 µV | Frontal |
| Slow oscillation | 0.5–1 Hz | 0.8–2 s | > 75 µV | Frontal |
| Sawtooth wave | 2–6 Hz | Trains | 20–50 µV | Central/frontal |
| Vertex sharp wave | N/A | < 0.5 s | < 200 µV | Cz |

## Recommended Pipeline

```yaml
pipeline:
  - notch:50              # Line noise
  - bandpass:0.3,35       # Preserve delta + spindles
  - resample:256          # Sufficient for sleep features
  - drop_bads             # Check for flat/noisy channels
  - scale:standard        # Normalize for classifier input
```

### Notes
- Do NOT high-pass above 0.5 Hz — slow waves (0.5–2 Hz) are critical for staging
- Low-pass at 35 Hz is sufficient (spindles peak at 12–14 Hz, nothing relevant above 35)
- 30-second epochs are the standard unit for staging
- Each epoch gets ONE label from {W, N1, N2, N3, REM}
- Transition rules: must follow AASM adjacency (no W→N3 without intervening stages)
- Multitaper spectral analysis preferred for frequency estimation in short windows
- For spindle detection: bandpass 11–16 Hz → Hilbert envelope → threshold at mean + 1.5×std

## Automated Staging Features

| Feature Set | Description |
|-------------|-------------|
| Time-domain | Amplitude statistics, zero crossings, Hjorth parameters |
| Frequency-domain | Band power (delta, theta, alpha, sigma, beta), spectral edge |
| Time-frequency | Wavelet energy, spindle rate, slow oscillation coupling |
| EOG | REM density (rapid eye movements per minute) |
| EMG | Chin muscle tone (high in Wake, lowest in REM) |

## Quality Metrics

- Recording completeness: > 6 hours total recording time
- Artifact percentage: < 10% of epochs unusable
- Signal quality per channel: no flat/saturated periods > 30 seconds
- Electrode impedance: stable throughout night (< 10 kΩ)
- Total sleep time (TST): > 4 hours for valid staging

## Complete Pipeline Example

```python
import mne
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load PSG data
raw = mne.io.read_raw_edf("sub01_sleep.edf", preload=True, verbose=False)

# notch:50 → bandpass:0.3,35 → resample:256 → drop_bads → scale:standard
raw.notch_filter(50.0, verbose=False)
raw.filter(l_freq=0.3, h_freq=35, verbose=False)
raw.resample(256.0, verbose=False)

# Drop bad channels
# raw.drop_channels(bad_list)

# Extract and scale
data = raw.get_data().astype(np.float32)
sfreq = raw.info['sfreq']
channels = list(raw.ch_names)
data = StandardScaler().fit_transform(data.T).T.astype(np.float32)

# Epoch into 30-second windows (AASM standard)
epoch_samples = int(30.0 * sfreq)
n_epochs = data.shape[1] // epoch_samples
epochs = data[:, :n_epochs * epoch_samples].reshape(len(channels), n_epochs, epoch_samples)
epochs = epochs.transpose(1, 0, 2)  # (n_epochs, n_channels, n_samples)

# Feature extraction per epoch: band power
from scipy.signal import welch
delta_power = []  # 0.5-4 Hz
for ep in epochs:
    freqs, psd = welch(ep, fs=sfreq, nperseg=int(4*sfreq))
    delta_idx = (freqs >= 0.5) & (freqs <= 4)
    delta_power.append(psd[:, delta_idx].mean(axis=1))
```
