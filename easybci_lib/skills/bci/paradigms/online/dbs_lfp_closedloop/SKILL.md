---
name: dbs_lfp_closedloop
description: 'Closed-loop DBS — real-time β-burst detection triggers stimulation (adaptive DBS / aDBS)'
layer: L2
group: online
metadata:
  tags: [dbs, closed_loop, adbs, beta_burst, parkinson, real_time, safety]
  modalities: [lfp]
  paradigms: [closed_loop_dbs, adaptive_dbs, abci]
  analysis_goal_allowed:
    - classification
    - feature_extraction
    - clinical_screening
    - online_inference
  analysis_goal_forbidden:
    - source_localization
---
# Closed-Loop DBS

## Neuroscience Background

Adaptive DBS (aDBS) — first shown clinically by Little et al. (2013) —
triggers stimulation **only during pathological β bursts** detected in
real-time from the implanted electrode's LFP. Compared to continuous
DBS (cDBS), aDBS halves stimulation duty cycle while delivering
equivalent or better symptom control, reducing side-effects (dyskinesia,
speech impairment) and battery drain.

The closed-loop cycle:

```
LFP  →  bandpass(13-20Hz)  →  Hilbert envelope  →  threshold β
                                                       │
                                                       ▼
                                           if β_env > θ for > 100 ms
                                                       │
                                                       ▼
                                          STIM-ON for ≤ duty_max_ms
                                                       │
                                                       ▼
                                           if β_env < θ_off → STIM-OFF
```

Critical safety constraints:
- **Latency budget per cycle < 50 ms** to follow burst dynamics.
- **Hard duty-cycle cap** (e.g. 50%) to prevent runaway stim.
- **Programmable shutdown channel** for clinician override.

## Architecture

| Component | Latency budget |
|---|---|
| LFP digitize + buffer | < 5 ms |
| Bandpass 13–20 Hz (causal IIR) | < 10 ms |
| Hilbert envelope (sliding) | < 10 ms |
| Threshold + dwell-time check | < 5 ms |
| Stim trigger | < 10 ms |
| Safety check | < 10 ms |

Total: < 50 ms.

## Recommended Pipeline

```
acquire_lfp:1000Hz → bandpass:13,20,IIR_causal → hilbert →
moving_avg:100ms → threshold:patient_specific → debounce →
safety_check → trigger_stim
```

## Detection Parameters (Per-Patient)

Calibration (one-time, ~10 min OFF-med rest recording):

| Parameter | Calibration |
|---|---|
| Peak β frequency | argmax PSD in 13–25 Hz |
| β-envelope threshold θ | 75–85% percentile of envelope at peak β |
| Dwell time | 100–200 ms (avoids false triggers from noise spikes) |
| Hysteresis θ_off | 0.7 · θ (avoid chattering) |

## Safety Guard List

| Guard | Trigger | Action |
|---|---|---|
| Duty-cycle > 50% over 60 s | stim-on time / window | force STIM-OFF for 10 s |
| Stim amplitude out of bounds | per-patient `(min_mA, max_mA)` | clamp + alarm |
| LFP RMS > 5× baseline | broken electrode / artefact | pause; revert to cDBS |
| Clinician override | external GPIO / button | immediate STIM-OFF |
| Communication loss > 1 s | watchdog | revert to cDBS |

## Recording / Pipeline Notes

- Recording inside stim ON cycle requires **artefact blanking** (mask
  ±2 ms around each stim pulse) — pulse at 130 Hz typical.
- Bipolar reference between non-stim contacts to minimize artefact.

## Pitfalls & Failure Modes

- **Bilateral lead interaction.** Stim on one side leaks into the
  contralateral LFP; analyze each side independently.
- **Over-trigger from non-β bursts.** Movement / dystonia can elevate
  β envelope; require dwell time > 100 ms.
- **Patient awake / asleep difference.** Threshold calibrated awake
  may over-fire during sleep; consider state-aware threshold.

## Boundary with Related Paradigms

- **`parkinson_dbs_lfp.md`** — the offline / characterization paradigm.
  This paradigm is its **online / triggering** counterpart.
- **`closed_loop_bci.md`** — the general closed-loop BCI paradigm;
  aDBS is one instance.

## EasyBCI Pipeline Spec

```yaml
modality: lfp
paradigm: dbs_lfp_closedloop
analysis_goal: online_inference
steps:
  - load:streaming,source=acquisition_buffer
  - bandpass:13,20,method=iir,causal=True
  - hilbert
  - moving_average:100
  - threshold:patient_calibrated
  - safety_check:duty_cap=0.5,impedance_check=True
  - trigger_stim
```

## References

1. Little, S. et al. (2013). *Adaptive deep brain stimulation in advanced
   Parkinson disease*. Annals of Neurology 74(3): 449–457.
   doi:10.1002/ana.23951.
2. Rosin, B. et al. (2011). *Closed-loop deep brain stimulation is
   superior in ameliorating parkinsonism*. Neuron 72(2): 370–384.
   doi:10.1016/j.neuron.2011.08.023.
3. Velisar, A. et al. (2019). *Dual threshold neural closed loop deep
   brain stimulation in Parkinson disease patients*. Brain Stimulation
   12(4): 868–876. doi:10.1016/j.brs.2019.02.020.
4. Tinkhauser, G. et al. (2017). *The modulatory effect of adaptive deep
   brain stimulation on beta bursts in Parkinson's disease*. Brain
   140(4): 1053–1067. doi:10.1093/brain/awx010.
