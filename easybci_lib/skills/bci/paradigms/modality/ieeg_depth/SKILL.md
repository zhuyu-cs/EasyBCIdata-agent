---
name: ieeg_depth
layer: L2
group: modality
metadata:
  analysis_goal_allowed:
  - clinical_screening
  - feature_extraction
  - exploratory
  - phase_amplitude_coupling
  - connectivity
  analysis_goal_forbidden:
  - online_inference
tags:
- ieeg
- intracranial
- depth
- seeg
- stereoelectroencephalography
- lfp
- local_field_potential
- hippocampus
- dbs
modality: seeg
---
# iEEG/sEEG — Intracranial and Depth Electrode Recordings

## Signal Characteristics

| Property | Typical Value |
|----------|--------------|
| Sampling rate | 512–2048 Hz (commonly 1024 Hz) |
| Channels | 64–256 (5–15 electrodes × 8–18 contacts each) |
| Electrode type | Cylindrical depth electrodes (0.8–1.2 mm diameter) |
| Contact spacing | 2–5 mm (standard 3.5 mm) |
| Signal amplitude | 50–500 µV (LFP), up to 1 mV (epileptic) |
| Recording duration | Days to weeks (epilepsy monitoring) |
| Reference | White matter contact or average of unaffected contacts |

## Anatomical Targets (Common Implantation)

| Target | Structure | Clinical/Research Purpose |
|--------|-----------|--------------------------|
| Mesial temporal | Hippocampus, amygdala | Temporal lobe epilepsy, memory |
| Insular | Insula, operculum | Insular seizures, interoception |
| Frontal | OFC, DLPFC, SMA | Frontal epilepsy, executive function |
| Parietal | Precuneus, SPL | Parietal seizures, spatial processing |
| Cingulate | ACC, PCC | Network hubs, decision making |
| Occipital | Calcarine, V1/V2 | Visual cortex epilepsy |

## Frequency Bands

| Band | Range | Significance in sEEG |
|------|-------|---------------------|
| Slow oscillation | 0.1–1 Hz | Sleep, cortical UP/DOWN states |
| Delta | 1–4 Hz | DLOC, seizure onset |
| Theta | 4–8 Hz | Hippocampal navigation, memory |
| Alpha | 8–13 Hz | Thalamo-cortical loops |
| Beta | 13–30 Hz | Motor network, status quo |
| Low gamma | 30–80 Hz | Local computation |
| High gamma | 80–200 Hz | Population spiking (best SNR in sEEG) |
| Ripples | 80–250 Hz | Memory replay, epileptogenic |
| Fast ripples | 250–500 Hz | Epileptogenic zone ONLY |

## Recommended Pipeline

```yaml
pipeline:
  - bipolar_ref            # Adjacent contact pairs (standard for sEEG)
  - notch:50              # Line noise + harmonics
  - bandpass:0.5,200      # Preserve HFO range
  - drop_bads             # Remove epileptic/noisy contacts
  - scale:robust          # Handle amplitude differences across brain regions
```

### Notes
- Bipolar re-referencing is STANDARD for sEEG (removes volume-conducted far-field)
- Adjacent contacts on same electrode are paired (e.g., A1-A2, A2-A3, ...)
- White matter contacts often used as reference if not bipolar
- High-gamma (80–200 Hz) is more reliable in sEEG than scalp EEG
- For epilepsy: annotate seizure onset zone (SOZ) channels separately
- Long recordings (days): segment into epochs, handle impedance drift
- Cross-frequency coupling (theta-gamma) is a key analysis in hippocampal sEEG

## Analysis Approaches

| Analysis | Method | Application |
|----------|--------|-------------|
| Seizure onset detection | Line-length, energy | Real-time monitoring |
| HFO detection | Band-pass + threshold | SOZ localization |
| Phase-amplitude coupling | Modulation index | Memory, theta-gamma |
| Connectivity | PLV, wPLI, Granger | Network analysis |
| Single-trial decoding | Gamma power features | Cognitive BCI |
| Traveling waves | Phase gradient | Cortical dynamics |

## Quality Metrics

- Contact impedance: < 10 kΩ (checked daily during monitoring)
- Interictal spike rate: identify but don't necessarily reject
- White matter contacts: verify with post-implantation CT/MRI fusion
- Artifact from stimulation: template subtraction if cortical stimulation was applied
- Recording gaps: document amplifier disconnections during patient care
