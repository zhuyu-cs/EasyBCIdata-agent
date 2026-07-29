---
name: closed_loop_bci
description: 'Closed-loop BCI — real-time inference + actuation / stimulation feedback with safety guards'
layer: L2
group: online
metadata:
  tags: [closed_loop, abci, stimulation, feedback, safety, latency_budget]
  modalities: [eeg, ecog, lfp, spike]
  paradigms: [closed_loop_bci, neurofeedback_actuator]
  analysis_goal_allowed:
    - classification
    - feature_extraction
    - clinical_screening
    - online_inference
  analysis_goal_forbidden:
    - source_localization
    - phase_amplitude_coupling
---
# Closed-Loop BCI

## Neuroscience Background

Closed-loop BCI extends `online_inference` with a **feedback action**:
the decoded brain state triggers an actuator (robotic arm, cursor,
stimulator, prosthesis). The defining property is that the user's
**brain state at time t+1 depends on the system's action at time t** —
forming a control loop. Examples:

- **Motor BCI**: M1 spike → cursor displacement → visual feedback → M1 update.
- **Adaptive DBS** (see `dbs_lfp_closedloop.md`): STN-LFP β burst →
  stim ON → β envelope drops → stim OFF.
- **Closed-loop seizure interruption**: HFO detection → stim → seizure aborts.
- **Neurofeedback training**: EEG alpha → audio gain → user learns to
  modulate alpha.

The neuroscience implication: the system must respect a **delay budget**
compatible with the dynamics of the feedback loop. Too slow → user
cannot perceive the feedback; too fast / unsafe → unintended stim.

## Architecture

```
acquire → preprocess (causal) → feature → decoder → state estimator
                                                          │
                                                          ▼
                                              ┌─────────────────┐
                                              │  SAFETY GUARDS  │
                                              └─────────────────┘
                                                          │
                                                          ▼
                                      ┌────────────────────────────┐
                                      │  ACTUATOR (cursor / stim)  │
                                      └────────────────────────────┘
                                                          │
                                                          ▼
                                          USER PERCEIVES FEEDBACK
                                                          │
                                                          ▼
                                      brain state updates → loop again
```

## Latency Budget (per actuator cycle)

| Stage | Target |
|---|---|
| Acquire + buffer | < 5 ms |
| Causal filter + ASR | < 10 ms |
| Decoder | < 10 ms |
| Safety check | < 5 ms |
| Actuator latch | < 5 ms |
| **End-to-end** | **< 50 ms** for cursor; < 30 ms for closed-loop stim |

## Safety Guard List

Any closed-loop BCI **must** implement at minimum:

| Guard | Purpose |
|---|---|
| **Duty-cycle cap** | Prevent runaway stim (e.g., < 50% over 60 s window). |
| **Amplitude bounds** | Per-patient (min_mA, max_mA); hard clamp. |
| **Watchdog** | Communication loss > 1 s → revert to safe state. |
| **Impedance check** | Pre-stim impedance; reject if out of nominal range. |
| **Clinician override** | External GPIO / button forces actuator OFF. |
| **Software fail-safe** | Decoder anomaly (e.g. NaN, > 5σ output) → freeze. |

These are **non-optional** for any clinically-deployed closed-loop BCI;
research closed-loops in animals can relax some but should still
implement the watchdog + clinician override.

## Recommended Pipeline

Built on top of `online_inference` upstream + actuator + safety:

```
load:streaming → bandpass:1,40,causal → asr:20 → riemannian_features →
classifier_predict → safety_check → actuator_trigger
```

For closed-loop with stim:

```
... → safety_check (duty cap, impedance, amplitude) → stim_pulse_generator
```

## Detection / Decoder Choice

| Use case | Detector / Decoder |
|---|---|
| Motor (cursor) | Kalman filter over Riemannian features (BrainGate-class). |
| Adaptive DBS | β-envelope threshold + dwell time. |
| Seizure interrupt | HFO detector + classifier. |
| Neurofeedback | Smoothed band-power → audio gain. |

## Calibration Requirements

| Parameter | Calibration |
|---|---|
| Decoder weights | 5–15 min supervised trials. |
| ASR calibration matrix | 1 min baseline. |
| Safety thresholds | Per-patient + per-electrode. |
| Stim amplitude (if applicable) | Threshold titration under clinician supervision. |

## Pitfalls & Failure Modes

- **Loop instability.** Feedback can amplify oscillations or noise.
  Mitigation: smoothing, refractory windows on actuator triggers.
- **User adaptation.** User changes their brain strategy to maximize
  feedback; this may shift away from the trained decoder. Mitigation:
  online recalibration.
- **Stim contamination.** Stim artefact enters the recording. Mitigation:
  ±2 ms blanking + bipolar reference.
- **False positive trigger.** Movement / blink may trigger stim.
  Mitigation: dwell-time threshold + multi-feature voting.

## Boundary with Related Paradigms

| Related paradigm | Boundary |
|---|---|
| **`online_inference.md`** | Inference-only (no actuation); this paradigm adds actuation + safety. |
| **`dbs_lfp_closedloop.md`** | Specific instance: β-burst-triggered DBS. |
| **`utah_array_motor.md`** | BrainGate-class cursor control uses this paradigm. |
| **`neurofeedback.md`** (future 04 task) | Audio / visual feedback loop using brain state. |

## EasyBCI Pipeline Spec

```yaml
modality: <eeg | lfp | spike>
paradigm: closed_loop_bci
analysis_goal: online_inference
steps:
  - load:streaming,source=acquisition_buffer
  - bandpass:1,40,method=iir,causal=True
  - asr:20
  - <feature>:per_task
  - <classifier>:per_task
  - safety_check:duty_cap=0.5,amp_min=...,amp_max=...
  - <actuator>:per_task
```

## References

1. Hochberg, L. R. et al. (2012). *Reach and grasp by people with
   tetraplegia using a neurally controlled robotic arm*. Nature 485:
   372–375. doi:10.1038/nature11076.
2. Little, S. et al. (2013). *Adaptive deep brain stimulation in
   advanced Parkinson disease*. Annals of Neurology 74(3): 449–457.
   doi:10.1002/ana.23951.
3. Sani, O. G. et al. (2018). *Mood variations decoded from multi-site
   intracranial human brain activity*. Nature Biotechnology 36(10):
   954–961. doi:10.1038/nbt.4200 — closed-loop mood research.
4. Brunton, B. W., & Beyeler, M. (2019). *Data-driven models in human
   neuroscience and neuroengineering*. Current Opinion in Neurobiology
   58: 21–29. doi:10.1016/j.conb.2019.06.008 — closed-loop principles.
