---
name: hilbert
description: "Hilbert transform to extract instantaneous amplitude envelope"
layer: L3
group: spectral
metadata:
  tags: [operator, hilbert, envelope, amplitude, instantaneous]
  modalities: [eeg, seeg, ecog, meg]
  step_string: "hilbert"
  analysis_goal_allowed: [feature_extraction, phase_amplitude_coupling, connectivity, exploratory, generic]
  analysis_goal_forbidden: [classification, online_inference]
---
# Hilbert Transform (Envelope)

## Function

Computes the analytic signal via Hilbert transform and returns the amplitude envelope (instantaneous amplitude). Useful for extracting power fluctuations in specific frequency bands.

## Parameter Format

`hilbert` — No parameters.

## How It Works

1. Computes analytic signal: `x_a(t) = x(t) + j * H[x(t)]`
2. Returns envelope: `|x_a(t)|` = instantaneous amplitude

The result is always non-negative and represents the slowly-varying amplitude modulation of the signal.

## When to Use

- Extract amplitude envelope of band-limited signal (e.g., mu/beta power for MI)
- Compute event-related desynchronization/synchronization (ERD/ERS) time courses
- High-gamma envelope for speech/motor decoding (sEEG/ECoG)
- When you need time-resolved power (vs. spectral methods which average over windows)

## When NOT to Use

- On broadband (unfiltered) signals — meaningless envelope, dominated by noise
- When phase information is needed (envelope discards phase)
- For frequency-domain analysis (use FFT/Welch instead)
- On very short segments (edge effects from Hilbert transform)

## Constraints

- **Must bandpass first**: Hilbert envelope is only meaningful on band-limited signals. Always apply `bandpass` before `hilbert`.
- The output has the same sampling rate but fundamentally different meaning (amplitude, not voltage)
- Signal becomes non-negative after this transform

## Ordering

- Apply AFTER: bandpass (mandatory — Hilbert on broadband is meaningless)
- Apply AFTER: notch (clean the band first)
- Apply BEFORE: resample (envelope is smooth, safe to downsample after)
- Apply BEFORE: scale

## Typical Usage Pattern

```
bandpass:8,12 → hilbert → resample:64 → scale:standard
```
This extracts mu-band (8-12 Hz) power envelope, downsampled to 64 Hz (sufficient for slow power fluctuations), then normalized.

## Reference Code

### MNE Chain (on existing Raw object)

```python
# MUST apply bandpass first
raw.apply_hilbert(envelope=True)
```

### Standalone Implementation

```python
import mne
import numpy as np

# Assumes data is already band-limited (e.g. 8-12 Hz)
info = mne.create_info(ch_names, sfreq, ch_types='eeg')
raw = mne.io.RawArray(data, info, verbose=False)
raw.apply_hilbert(envelope=True)
data = raw.get_data().astype(np.float32)
# Result: non-negative amplitude envelope
```

### Key API

- `raw.apply_hilbert(envelope=True)` — returns amplitude envelope (|analytic signal|)
- `envelope=False` would return the complex analytic signal (for phase extraction)
