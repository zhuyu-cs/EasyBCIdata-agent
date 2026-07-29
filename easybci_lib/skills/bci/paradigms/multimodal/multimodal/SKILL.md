---
name: multimodal
layer: L2
group: multimodal
metadata:
  analysis_goal_allowed:
  - source_localization
  - feature_extraction
  - exploratory
  - generic
  - classification
  - connectivity
  analysis_goal_forbidden: []
tags:
- multimodal
- eeg_fmri
- eeg_fnirs
- simultaneous
- fusion
- coregistration
- hybrid_bci
modality: multimodal
---
# Multi-Modal — Simultaneous Neural Recordings

## Common Combinations

| Combination | Temporal | Spatial | Application |
|-------------|----------|---------|-------------|
| EEG + fMRI | EEG: ms / fMRI: ~s | EEG: cm / fMRI: mm | Source localization, resting state |
| EEG + fNIRS | Both: ms–s | EEG: cm / fNIRS: cm | Portable hybrid BCI |
| EEG + MEG | Both: ms | MEG: mm / EEG: cm | Forward model validation |
| sEEG + fMRI | sEEG: ms / fMRI: s | Both: mm | Network mapping |
| EEG + EMG | Both: ms | N/A | Motor intent + execution |
| EEG + Eye tracking | Both: ms | N/A | Attention, SSVEP gaze |

## EEG–fNIRS Fusion

### Temporal Alignment
- Different sampling rates: EEG (256–512 Hz) vs fNIRS (5–25 Hz)
- Resample to common timebase (usually fNIRS rate for block design)
- Account for hemodynamic delay (4–6s) when correlating

### Feature Fusion Strategies
| Level | Method | Description |
|-------|--------|-------------|
| Early (feature) | Concatenation | Stack EEG + fNIRS features |
| Intermediate | CCA/PLS | Canonical correlation between modalities |
| Late (decision) | Voting/stacking | Separate classifiers → ensemble |
| Deep | Multi-branch CNN | Shared representation learning |

## EEG–fMRI Artifacts

| Artifact | Cause | Removal |
|----------|-------|---------|
| Gradient artifact | MR switching gradients | Template subtraction (AAS) |
| Ballistocardiogram | Pulse in magnetic field | OBS or ICA |
| Helium pump | Cryogenic pump vibration | Notch at pump frequency |
| RF interference | Scanner RF pulses | Filtering at TR harmonics |

## Recommended Pipeline (EEG + fNIRS example)

```yaml
# EEG stream
eeg_pipeline:
  - notch:50
  - bandpass:0.5,40
  - resample:256
  - scale:robust

# fNIRS stream
fnirs_pipeline:
  - bandpass:0.01,0.2
  - scale:standard

# Fusion
fusion:
  - temporal_align: fnirs_rate   # Downsample EEG epochs to fNIRS rate
  - feature_concat              # Concatenate feature vectors
```

### Notes
- Process each modality with its own pipeline FIRST, then fuse
- Temporal alignment is critical: use hardware trigger markers shared across systems
- For BCI: EEG provides fast temporal features, fNIRS provides spatial/hemodynamic context
- Cross-modal artifact: EEG in MRI scanner requires gradient artifact removal BEFORE any EEG processing
- Clock synchronization: LSL (Lab Streaming Layer) or shared trigger channel
- If modalities have different trial counts (dropout), use only overlapping trials

## Hybrid BCI Paradigms

| Paradigm | EEG Feature | Secondary Feature | Benefit |
|----------|-------------|-------------------|---------|
| MI + SSVEP | ERD/ERS | Frequency power | Reduced false positive |
| MI + fNIRS | ERD/ERS | HbO lateralization | Confirmation channel |
| P300 + SSVEP | P300 ERP | SSVEP power | Faster ITR |
| ErrP + MI | Error potential | Motor command | Error correction |

## Quality Metrics

- Temporal sync: cross-correlation of shared trigger channel (lag < 1 sample)
- Modality dropout: percentage of trials with both modalities valid
- Cross-modal SNR: each modality should independently pass its own QC
- Fusion benefit: classification accuracy of fusion > max(individual modalities)
