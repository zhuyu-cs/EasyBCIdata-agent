---
name: meg
layer: L2
group: modality
metadata:
  analysis_goal_allowed:
  - source_localization
  - generic
  - exploratory
  - feature_extraction
  - classification
  - connectivity
  analysis_goal_forbidden: []
tags:
- meg
- magnetoencephalography
- magnetic
- gradiometer
- magnetometer
- source_localization
- beamformer
modality: meg
---
# MEG — Magnetoencephalography

## Signal Characteristics

| Property | Typical Value |
|----------|--------------|
| Sampling rate | 600–2400 Hz (commonly 1000 Hz) |
| Channels | 102–306 (Elekta: 204 grad + 102 mag; CTF: 275 axial grad) |
| Sensor types | Magnetometers (fT), planar gradiometers (fT/cm), axial gradiometers |
| Signal amplitude | 10–1000 fT |
| Noise floor | ~3–5 fT/√Hz (SQUID) |
| Reference channels | Environmental noise cancellation |

## Sensor Types

| Type | Unit | Sensitivity | Depth |
|------|------|-------------|-------|
| Magnetometer | fT | Highest (also most noise) | Deep sources |
| Planar gradiometer | fT/cm | Moderate | Superficial, focal |
| Axial gradiometer | fT | Moderate | Moderate depth |

## Common Artifacts

| Artifact | Cause | Removal Strategy |
|----------|-------|------------------|
| External interference | Power lines, machinery | tSSS/SSS (MaxFilter) |
| Head movement | Subject motion | Movement compensation (cHPI) |
| Cardiac (ECG) | Heart magnetic field | ICA or SSP |
| Ocular (EOG) | Eye movement/blink | ICA or SSP |
| Dental/metal | Ferromagnetic material | Check before recording |
| Muscle (EMG) | High-freq contamination | Low-pass < 100 Hz |

## Recommended Pipeline

```yaml
pipeline:
  - notch:50               # Power line (or 60 Hz in Americas)
  - bandpass:1,100         # Standard MEG range
  - resample:500           # After filtering (from 1000+ Hz)
  - drop_bads              # Bad channel detection
  - scale:robust           # Normalize
```

### Notes
- MaxFilter (tSSS) should be applied BEFORE any other processing (usually done at acquisition site)
- Separate processing for magnetometers vs gradiometers (different scales, different noise)
- SSP (Signal Space Projection) or ICA for artifact removal
- Source localization requires: forward model (BEM) + inverse (MNE/dSPM/LCMV/beamformer)
- Co-registration: MEG sensors → MRI anatomy (fiducials + head digitization)

## Source Localization Methods

| Method | Type | Best For |
|--------|------|----------|
| MNE/dSPM | Distributed | Whole-brain activity maps |
| LCMV beamformer | Spatial filter | Focal sources, time-frequency |
| eLORETA | Distributed | Low spatial resolution, zero error |
| MUSIC/RAP-MUSIC | Scanning | Few dipolar sources |
| Equivalent dipole | Parametric | Single focal source |

## Paradigms

| Paradigm | Frequency Band | Key Feature |
|----------|---------------|-------------|
| Auditory evoked | 1–30 Hz | M100/M200 in superior temporal |
| Visual evoked | 1–40 Hz | M100 in calcarine |
| Somatosensory | 1–100 Hz | SEF in SI/SII |
| Motor (beta) | 13–30 Hz | MRBD/PMBR in motor cortex |
| Resting state | 1–45 Hz | Alpha peak, connectivity |
| Epilepsy (clinical) | 1–80 Hz | Interictal spike localization |

## Quality Metrics

- Noise level: empty room recording baseline (< 5 fT/√Hz)
- Head position: continuous HPI deviation (< 5 mm acceptable)
- Bad channel percentage: < 10% of array
- After tSSS: inner/outer component ratio
