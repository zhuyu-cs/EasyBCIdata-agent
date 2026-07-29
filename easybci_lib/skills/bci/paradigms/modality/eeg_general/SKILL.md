---
name: eeg_general
layer: L2
group: modality
metadata:
  analysis_goal_allowed:
  - generic
  - exploratory
  - feature_extraction
  - classification
  - source_localization
  analysis_goal_forbidden: []
tags:
- eeg
- resting_state
- alpha
- general
- scalp
- event_related
- frequency
- connectivity
modality: eeg
---
# EEG — Scalp Electroencephalography (General)

## Signal Characteristics

| Property | Typical Value |
|----------|--------------|
| Sampling rate | 250–1000 Hz (commonly 256 or 512 Hz) |
| Channels | 8–256 (research: 32–128) |
| Electrode systems | 10-20, 10-10, 10-5, BioSemi, EGI |
| Signal amplitude | 10–100 µV |
| Reference | Cz, linked mastoids, average, REST |
| Impedance | < 5 kΩ (gel), < 50 kΩ (dry) |

## Frequency Bands

| Band | Range | Cognitive Correlate |
|------|-------|-------------------|
| Delta | 0.5–4 Hz | Deep sleep, pathology |
| Theta | 4–8 Hz | Working memory, drowsiness, meditation |
| Alpha | 8–13 Hz | Relaxed wakefulness, eyes closed, inhibition |
| SMR (Mu) | 8–13 Hz (central) | Sensorimotor idle |
| Beta | 13–30 Hz | Active thinking, motor planning |
| Gamma | 30–100 Hz | Perception, binding (low SNR at scalp) |

## Common Artifacts

| Artifact | Frequency | Removal Strategy |
|----------|-----------|------------------|
| Eye blinks | < 4 Hz (frontal) | ICA (most common approach) |
| Saccades | < 2 Hz (lateral frontal) | ICA |
| Muscle (EMG) | > 20 Hz (temporal) | ICA or low-pass |
| Line noise | 50/60 Hz | Notch filter |
| Sweat/electrode drift | < 0.1 Hz | High-pass 0.5 Hz |
| Heartbeat (ECG) | ~1 Hz + harmonics | ICA |
| Movement | Broadband, discontinuous | Epoch rejection |

## Recommended Pipeline

```yaml
pipeline:
  - notch:50              # Line noise (60 Hz in Americas/Japan)
  - bandpass:0.5,45       # Standard EEG range
  - resample:256          # Sufficient for most paradigms
  - drop_bads             # Reject bad channels
  - scale:robust          # Handle outlier amplitudes
```

### Notes
- Always filter BEFORE epoching (avoids edge artifacts in short segments)
- ICA requires > 20× number of channels in timepoints (e.g., 64ch → 1280+ samples)
- Re-reference choice matters: average reference needs many channels (>32), linked mastoids for fewer
- For connectivity analysis: use surface Laplacian or source-space to reduce volume conduction
- Passband choice depends on paradigm: sleep needs delta (0.5–4), BCI may need only mu/beta (8–30)
- Notch filter harmonics: 100, 150, 200 Hz if sampling rate is high enough

## Paradigms

| Paradigm | Optimal Passband | Key Feature |
|----------|-----------------|-------------|
| Resting state (eyes open/closed) | 0.5–45 Hz | Alpha power, peak frequency |
| P300 BCI | 0.1–20 Hz | See p300_erp.md |
| Motor imagery BCI | 0.5–40 Hz | See motor_imagery.md |
| SSVEP BCI | 3–90 Hz | See ssvep.md |
| Sleep staging | 0.5–35 Hz | Delta/theta/alpha/spindles |
| Emotion recognition | 1–50 Hz | Frontal alpha asymmetry, gamma |
| Attention/meditation | 4–30 Hz | Theta/alpha ratio, frontal midline theta |
| Auditory evoked (MMN) | 0.1–30 Hz | Fronto-central negativity 100–250 ms |
| N400/Language | 0.1–30 Hz | Centro-parietal negativity 300–500 ms |
| Error-related (ERN) | 0.1–15 Hz | Fronto-central negativity at response |

## Quality Metrics

- Channel noise: std within 2× median of all channels
- Flatline detection: std < 0.5 µV over > 5 seconds
- High amplitude: epochs > ±100 µV rejected
- Channel correlation: < 0.4 with neighbors → bad
- Percentage rejected: < 20% epochs acceptable
- Alpha peak: clear individual alpha frequency (IAF) in 7–13 Hz at posterior sites
