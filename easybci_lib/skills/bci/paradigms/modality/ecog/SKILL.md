---
name: ecog
layer: L2
group: modality
metadata:
  analysis_goal_allowed:
  - clinical_screening
  - feature_extraction
  - exploratory
  - classification
  - source_localization
  analysis_goal_forbidden: []
tags:
- ecog
- electrocorticography
- intracranial
- subdural
- grid
- strip
- high_gamma
- cortical
modality: ecog
---
# ECoG — Electrocorticography

## Signal Characteristics

| Property | Typical Value |
|----------|--------------|
| Sampling rate | 512–2048 Hz (commonly 1000–2048 Hz) |
| Channels | 16–256 (grid: 64, strip: 4–16, HD: 256) |
| Electrode types | Subdural grids, strips, depth electrodes |
| Signal amplitude | 50–500 µV |
| Spatial resolution | 5–10 mm (standard), 1–2 mm (high-density) |
| Frequency range | DC–500 Hz (broadband, including high-gamma) |
| SNR | Excellent (no skull/scalp attenuation) |

## Frequency Bands of Interest

| Band | Range | Significance |
|------|-------|-------------|
| Delta | 1–4 Hz | Sleep, anesthesia, cortical injury |
| Theta | 4–8 Hz | Memory encoding, navigation |
| Alpha/Mu | 8–13 Hz | Sensorimotor idle, visual suppression |
| Beta | 13–30 Hz | Motor planning, status quo |
| Low gamma | 30–70 Hz | Local circuit processing |
| High gamma (HGB) | 70–200 Hz | Population firing rate proxy, BCI feature |
| Ripples | 80–250 Hz | Memory consolidation, epileptic |
| Fast ripples | 250–500 Hz | Epileptogenic zone marker |

## Common Artifacts

| Artifact | Cause | Removal Strategy |
|----------|-------|------------------|
| 50/60 Hz line noise | Power supply | Notch + harmonics |
| Electrode pop | Loose contact | Interpolation or rejection |
| Interictal spikes | Epileptic activity | Epoch rejection or ICA |
| Stimulation artifact | Cortical stimulation mapping | Template subtraction |
| Reference artifact | Bad reference electrode | Re-reference (CAR or bipolar) |
| Broadband noise | Damaged electrode | Channel rejection |

## Recommended Pipeline

```yaml
pipeline:
  - notch:50               # Line noise + harmonics (auto-extends)
  - bandpass:0.5,200       # Preserve high-gamma, remove DC drift
  - resample:512           # If original > 1000 Hz and HGB not needed above 200 Hz
  - drop_bads              # Remove dead/noisy electrodes
  - scale:robust           # Handle amplitude variation across grid
```

### Notes
- High-gamma (70–200 Hz) is THE key BCI feature in ECoG — preserve it
- Common Average Reference (CAR) is standard; bipolar for depth electrodes
- Do NOT aggressively low-pass — ECoG's advantage is broadband access
- If resample, ensure Nyquist covers highest band of interest (HGB → keep ≥ 512 Hz)
- Electrode localization: CT + MRI co-registration for cortical surface mapping
- Epoch rejection threshold should be higher than scalp EEG (larger amplitudes are normal)

## BCI Applications

| Application | Feature | Decoding |
|-------------|---------|----------|
| Motor BCI | High-gamma power in motor cortex | Linear discriminant, Kalman filter |
| Speech BCI | High-gamma in STG/IFG/motor | RNN/Transformer sequence models |
| Cursor control | Beta/high-gamma motor cortex | Online adaptive decoder |
| Handwriting BCI | Neural population activity | RNN character decoder |
| Affective BCI | Amygdala/OFC gamma | SVM classifier |

## Paradigms

| Paradigm | Key Features | Pipeline Notes |
|----------|-------------|----------------|
| Motor mapping | HGB lateralization | Bandpass 70–200 Hz, Hilbert envelope |
| Speech decoding | Temporal/frontal HGB | Preserve full bandwidth, small windows |
| Seizure detection | HFO + spike-wave | Wide bandpass, long recordings |
| Resting state | Cross-frequency coupling | Multiple band extraction |
| Stimulation mapping | Pre/post stim response | Artifact removal critical |

## Quality Metrics

- Impedance: < 5 kΩ per electrode
- Line noise ratio: 50/60 Hz power vs. broadband (< 3x acceptable)
- Bad channel criteria: flat (std < 1 µV), noisy (amplitude > 5× median), correlation < 0.4 with neighbors
- High-gamma SNR: task/baseline ratio > 2
