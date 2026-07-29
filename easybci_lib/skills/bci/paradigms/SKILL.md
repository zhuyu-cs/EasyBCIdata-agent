---
name: paradigms-index
description: "Index of all L2 domain skills by group + analysis_goal → skill matrix"
layer: L2
group: modality
metadata:
  tags: [index, paradigms, navigation, l2]
  analysis_goal_allowed: [classification, source_localization, feature_extraction, clinical_screening, exploratory, generic, connectivity, phase_amplitude_coupling, online_inference]
  analysis_goal_forbidden: []
---

# L2 Domain Skill Index — BCI Domain Knowledge

> **Navigation.**  
> [L0 IO formats](../neural-io/SKILL.md) → [L1 Orchestration](../pipeline/SKILL.md) →
> **L2 (this page)** → [L3 Operators](../operators/SKILL.md).

This index lists every L2 domain skill, grouped along orthogonal axes —
`paradigm` (experimental task), `modality` (recording method), `clinical`
(diagnostic application), `analysis` (processing goal), `online` (deployment
mode), and `multimodal`. The L1 orchestrator (`pipeline`) loads the matching
skill before generating a pipeline; the matrix at the bottom shows which skills
suit which `analysis_goal`.

Load any skill with `skill_view(name='<skill>')`.

## Signal Processing Principles

### Frequency Bands of Interest

| Band | Range (Hz) | Function | Relevant Paradigms |
|------|-----------|----------|-------------------|
| Delta | 0.5-4 | Deep sleep, pathology | Sleep staging |
| Theta | 4-8 | Memory, attention | Cognitive BCI, emotion |
| Alpha/Mu | 8-13 | Idle sensorimotor, visual | Motor imagery, relaxation |
| Beta | 13-30 | Active motor, attention | Motor imagery, cognitive |
| Low Gamma | 30-70 | Perception, cognition | SSVEP, working memory |
| High Gamma | 70-150 | Local cortical processing | sEEG/ECoG, speech |
| HFO | 150-500 | Epileptogenic zones | Epilepsy monitoring |

### Modality Characteristics

| Modality | Typical Sfreq | Noise Profile | Reference Scheme |
|----------|--------------|---------------|------------------|
| Scalp EEG | 256-1024 Hz | Muscle, eye, line noise | CAR or linked mastoids |
| sEEG | 1000-2048 Hz | Low artifact, high SNR | Bipolar (adjacent contacts) |
| ECoG | 1000-2048 Hz | Minimal muscle, some line noise | Bipolar or CAR |
| MEG | 1000+ Hz | Environmental magnetic, no reference needed | Already reference-free |
| fNIRS | 10-50 Hz | Motion, systemic physiology | Short-channel regression |
| Spike | 30000 Hz | Electrode drift, crosstalk | Local reference |

### Quality Indicators

| Metric | Good | Warning | Fail |
|--------|------|---------|------|
| Channel variance ratio | < 3x median | 3-10x median | > 10x median |
| Flatline duration | 0 | < 1 sec | > 1 sec continuous |
| Line noise SNR (50/60 Hz) | < 3 dB above band | 3-10 dB | > 10 dB |
| NaN ratio | 0% | < 1% | > 1% |
| Amplitude range (EEG) | ±100 µV | ±200 µV | > ±500 µV |

## Domain skills by group

### `paradigm/` — Experimental paradigm — task the participant performs  *(3)*

| Skill | Modalities | analysis_goal_allowed |
|---|---|---|
| [motor_imagery](paradigm/motor_imagery/SKILL.md) | eeg | classification, exploratory, feature_extraction, generic, online_inference, source_localization |
| [p300_erp](paradigm/p300_erp/SKILL.md) | eeg | classification, clinical_screening, exploratory, feature_extraction, generic |
| [ssvep](paradigm/ssvep/SKILL.md) | eeg | classification, exploratory, feature_extraction, generic, online_inference |

### `modality/` — Modality skeletons — recording-method domain knowledge  *(9)*

| Skill | Modalities | analysis_goal_allowed |
|---|---|---|
| [custom_binary](modality/custom_binary/SKILL.md) | unknown | exploratory, generic |
| [ecog](modality/ecog/SKILL.md) | ecog | classification, clinical_screening, exploratory, feature_extraction, source_localization |
| [eeg_general](modality/eeg_general/SKILL.md) | eeg | classification, exploratory, feature_extraction, generic, source_localization |
| [fnirs](modality/fnirs/SKILL.md) | fnirs | classification, exploratory, feature_extraction, generic |
| [ieeg_depth](modality/ieeg_depth/SKILL.md) | seeg | clinical_screening, connectivity, exploratory, feature_extraction, phase_amplitude_coupling |
| [meg](modality/meg/SKILL.md) | meg | classification, connectivity, exploratory, feature_extraction, generic, source_localization |
| [neuropixel_population](modality/neuropixel_population/SKILL.md) | spike, lfp | classification, exploratory, feature_extraction, online_inference |
| [spike_lfp](modality/spike_lfp/SKILL.md) | spike | classification, exploratory, feature_extraction, generic |
| [unknown_modality](modality/unknown_modality/SKILL.md) | unknown | exploratory, generic |

### `clinical/` — Clinical / diagnostic application  *(5)*

| Skill | Modalities | analysis_goal_allowed |
|---|---|---|
| [emotion_recognition](clinical/emotion_recognition/SKILL.md) | eeg | classification, exploratory, feature_extraction, generic |
| [parkinson_dbs_lfp](clinical/parkinson_dbs_lfp/SKILL.md) | lfp | classification, clinical_screening, connectivity, exploratory, feature_extraction, online_inference, phase_amplitude_coupling |
| [seeg_epilepsy](clinical/seeg_epilepsy/SKILL.md) | seeg, ecog | clinical_screening, exploratory, feature_extraction |
| [sleep_staging](clinical/sleep_staging/SKILL.md) | eeg | classification, clinical_screening, exploratory, feature_extraction |
| [utah_array_motor](clinical/utah_array_motor/SKILL.md) | spike | classification, clinical_screening, feature_extraction, online_inference |

### `analysis/` — Analysis goal — connectivity / coupling / source  *(3)*

| Skill | Modalities | analysis_goal_allowed |
|---|---|---|
| [connectivity](analysis/connectivity/SKILL.md) | eeg, meg, seeg, ecog, lfp | connectivity, exploratory, feature_extraction, source_localization |
| [phase_amplitude_coupling](analysis/phase_amplitude_coupling/SKILL.md) | eeg, seeg, ecog, lfp | exploratory, feature_extraction, phase_amplitude_coupling |
| [source_localization_general](analysis/source_localization_general/SKILL.md) | eeg, meg | clinical_screening, exploratory, feature_extraction, source_localization |

### `online/` — Online inference / closed-loop deployment  *(3)*

| Skill | Modalities | analysis_goal_allowed |
|---|---|---|
| [closed_loop_bci](online/closed_loop_bci/SKILL.md) | eeg, ecog, lfp, spike | classification, clinical_screening, feature_extraction, online_inference |
| [dbs_lfp_closedloop](online/dbs_lfp_closedloop/SKILL.md) | lfp | classification, clinical_screening, feature_extraction, online_inference |
| [online_inference](online/online_inference/SKILL.md) | eeg, ecog, lfp | online_inference |

### `multimodal/` — Multi-modal integration  *(1)*

| Skill | Modalities | analysis_goal_allowed |
|---|---|---|
| [multimodal](multimodal/multimodal/SKILL.md) | multimodal | classification, connectivity, exploratory, feature_extraction, generic, source_localization |

## `analysis_goal` → skill matrix

For each REGISTRY goal, skills whose frontmatter *allows* the goal. `(F)` = the goal is in that skill's `analysis_goal_forbidden` list.

| analysis_goal | Recommended skills |
|---|---|
| `classification` | closed_loop_bci, custom_binary (F), dbs_lfp_closedloop, ecog, eeg_general, emotion_recognition, fnirs, meg, motor_imagery, multimodal, neuropixel_population, p300_erp, parkinson_dbs_lfp, phase_amplitude_coupling (F), sleep_staging, spike_lfp, ssvep, unknown_modality (F), utah_array_motor |
| `source_localization` | closed_loop_bci (F), connectivity, custom_binary (F), dbs_lfp_closedloop (F), ecog, eeg_general, emotion_recognition (F), meg, motor_imagery, multimodal, neuropixel_population (F), online_inference (F), parkinson_dbs_lfp (F), source_localization_general, spike_lfp (F), ssvep (F), unknown_modality (F), utah_array_motor (F) |
| `feature_extraction` | closed_loop_bci, connectivity, custom_binary (F), dbs_lfp_closedloop, ecog, eeg_general, emotion_recognition, fnirs, ieeg_depth, meg, motor_imagery, multimodal, neuropixel_population, p300_erp, parkinson_dbs_lfp, phase_amplitude_coupling, seeg_epilepsy, sleep_staging, source_localization_general, spike_lfp, ssvep, unknown_modality (F), utah_array_motor |
| `clinical_screening` | closed_loop_bci, custom_binary (F), dbs_lfp_closedloop, ecog, emotion_recognition (F), ieeg_depth, motor_imagery (F), neuropixel_population (F), p300_erp, parkinson_dbs_lfp, seeg_epilepsy, sleep_staging, source_localization_general, ssvep (F), unknown_modality (F), utah_array_motor |
| `exploratory` | connectivity, custom_binary, ecog, eeg_general, emotion_recognition, fnirs, ieeg_depth, meg, motor_imagery, multimodal, neuropixel_population, p300_erp, parkinson_dbs_lfp, phase_amplitude_coupling, seeg_epilepsy, sleep_staging, source_localization_general, spike_lfp, ssvep, unknown_modality |
| `generic` | custom_binary, eeg_general, emotion_recognition, fnirs, meg, motor_imagery, multimodal, p300_erp, spike_lfp, ssvep, unknown_modality |
| `connectivity` | connectivity, custom_binary (F), ieeg_depth, meg, multimodal, parkinson_dbs_lfp, unknown_modality (F) |
| `phase_amplitude_coupling` | closed_loop_bci (F), custom_binary (F), fnirs (F), ieeg_depth, online_inference (F), p300_erp (F), parkinson_dbs_lfp, phase_amplitude_coupling, unknown_modality (F), utah_array_motor (F) |
| `online_inference` | closed_loop_bci, connectivity (F), custom_binary (F), dbs_lfp_closedloop, fnirs (F), ieeg_depth (F), motor_imagery, neuropixel_population, online_inference, p300_erp (F), parkinson_dbs_lfp, phase_amplitude_coupling (F), seeg_epilepsy (F), sleep_staging (F), source_localization_general (F), spike_lfp (F), ssvep, unknown_modality (F), utah_array_motor |

## Step Selection Heuristics

Given a data fingerprint, use these rules to select steps:

1. **Always start with**: Identify and handle bad data (fill_nan if needed, then drop_bads or interpolate_bads)
2. **Reference**: CAR for scalp EEG (>= 16 ch), bipolar_ref for sEEG/depth
3. **Line noise**: Notch if PSD shows 50/60 Hz peak
4. **Band limiting**: Bandpass to paradigm-relevant range
5. **Artifact removal**: ICA for EEG if eye/cardiac artifacts present
6. **Resample**: If original sfreq >> needed (save computation)
7. **Normalize**: Scale as final step before output

## Decision Framework

When uncertain about a parameter, choose based on:

1. **Preservation over aggression**: Keep more data; less aggressive filtering preserves neural information
2. **Paradigm constraints**: The downstream analysis defines what frequencies/features matter
3. **Data quality drives decisions**: Clean data needs less processing; noisy data needs more
4. **Reproducibility**: Pin ALL randomness to seed 42 — lock numpy/random at script
   top, pass `random_state=42` to every stochastic operator (ICA, split, sampling).

## Layering contract

- **layer**: every skill carries `layer: L2` in frontmatter.
- **group**: matches the parent directory (`bci/paradigms/<group>/<skill>/SKILL.md`).
- **analysis_goal_allowed/forbidden**: at least one must be non-empty; values come
  from the 9-enum REGISTRY in `easybci_lib/tools/neural_processing/preprocess/analysis_goals.py`.

The layer / group enums are defined in
`easybci_lib/tools/neural_processing/skill_layers.py` (single source of
truth). Run `python -m easybci_lib.tools.neural_processing._check_consistency --strict`
to verify the contract before merging new domain skills.
