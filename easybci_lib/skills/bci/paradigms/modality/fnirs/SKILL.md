---
name: fnirs
layer: L2
group: modality
metadata:
  analysis_goal_allowed:
  - classification
  - feature_extraction
  - exploratory
  - generic
  analysis_goal_forbidden:
  - online_inference
  - phase_amplitude_coupling
tags:
- fnirs
- nirs
- hemodynamic
- oxy
- deoxy
- hbo
- hbr
- optical
- cerebral_blood_flow
modality: fnirs
---
# fNIRS — Functional Near-Infrared Spectroscopy

## Signal Characteristics

| Property | Typical Value |
|----------|--------------|
| Sampling rate | 5–50 Hz (most systems 10 Hz) |
| Channels | 20–100+ (source-detector pairs) |
| Wavelengths | 2 (typically 760nm + 850nm) |
| Signal type | Optical density → HbO/HbR concentration changes |
| Hemodynamic delay | 4–6 seconds (neurovascular coupling) |
| Noise sources | Motion artifacts, systemic physiology (heartbeat, respiration, Mayer waves) |

## Modified Beer-Lambert Law (MBLL)

Raw optical density → concentration changes:
- ΔOD = ε × Δc × DPF × d
- Two wavelengths → solve for HbO and HbR simultaneously
- DPF (differential pathlength factor) is age/wavelength dependent

## Common Artifacts

| Artifact | Cause | Removal Strategy |
|----------|-------|------------------|
| Motion spikes | Head movement | TDDR, wavelet, spline interpolation |
| Baseline drift | Slow drift in coupling | High-pass > 0.01 Hz |
| Heart rate | Systemic pulsation | Low-pass < 0.2 Hz or bandpass |
| Respiration | Chest movement | Bandstop 0.15–0.4 Hz |
| Mayer waves | Blood pressure oscillation | ~0.1 Hz, bandstop or regression |
| Scalp hemodynamics | Non-cerebral signal | Short-separation regression |

## Recommended Pipeline

```yaml
pipeline:
  - bandpass:0.01,0.2      # Remove cardiac + slow drift
  - scale:standard          # Normalize concentration values
```

### Notes
- fNIRS data typically arrives already converted (MBLL applied by acquisition software)
- If raw optical density: apply MBLL first, then filter
- Short-separation channels (< 1.5 cm) measure scalp blood flow — regress out from long channels
- TDDR (Temporal Derivative Distribution Repair) is preferred for motion correction
- Block averaging with 15–30s trials (accounts for hemodynamic delay)
- HbO is more sensitive but also more contaminated by systemic physiology

## Paradigms

| Paradigm | Design | Key Feature |
|----------|--------|-------------|
| Motor execution/imagery | Block (15–30s) | Contralateral HbO increase in motor cortex |
| Cognitive load (n-back) | Block/event | PFC HbO amplitude scales with load |
| Language | Block | Left lateralized HbO in Broca/Wernicke |
| Resting state | Continuous (5–10 min) | Functional connectivity (correlation of HbO) |
| Hyperscanning | Two participants | Inter-brain synchrony (WTC) |

## Quality Metrics

- SNR: HbO peak / baseline std (> 3 acceptable)
- Scalp coupling index (SCI): correlation of cardiac oscillation across wavelengths (> 0.75 good)
- Coefficient of variation: std/mean of optical intensity (< 15% good)
- Motion artifact percentage: % of timepoints exceeding threshold
