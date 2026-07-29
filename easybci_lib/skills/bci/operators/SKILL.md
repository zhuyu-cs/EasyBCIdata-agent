---
name: operators-index
description: "Index of all L3 atomic operators, grouped by algorithm family"
layer: L3
group: misc
metadata:
  tags: [index, atomic-operators, navigation]
  analysis_goal_allowed: [classification, source_localization, feature_extraction, clinical_screening, exploratory, generic, connectivity, phase_amplitude_coupling, online_inference]
  analysis_goal_forbidden: []
---

# L3 Atomic Operators — Index

> **Navigation.**  
> [L0 IO formats](../neural-io/SKILL.md) → [L1 Orchestration](../pipeline/SKILL.md) →
> [L2 Paradigms](../paradigms/SKILL.md) → **L3 (this page)**.

This index lists every L3 atomic operator skill, grouped by algorithm family.
Each operator skill documents its parameter format, modality applicability,
ordering constraints, and failure modes; load one with
`skill_view(name='<operator>')` during pipeline codegen.

## Groups

### `filter/` — Frequency-domain filtering  *(3)*

| Operator | step_string | Modalities | Description |
|---|---|---|---|
| [bandpass_filter](filter/bandpass_filter/SKILL.md) | `bandpass` | eeg, seeg, ecog, meg, fnirs | Band-pass filter to isolate frequency range of interest with Nyquist guard |
| [notch_filter](filter/notch_filter/SKILL.md) | `notch` | eeg, seeg, ecog, meg | Notch filter to remove power line interference (50/60 Hz) and harmonics |
| [resample](filter/resample/SKILL.md) | `resample` | eeg, seeg, ecog, meg | Resample signal to target frequency — downsampling with anti-aliasing |

### `channel/` — Channel selection & repair  *(7)*

| Operator | step_string | Modalities | Description |
|---|---|---|---|
| [clip](channel/clip/SKILL.md) | `clip` | eeg, seeg, ecog, meg | Clamp signal amplitude to a maximum absolute value |
| [drop_bads](channel/drop_bads/SKILL.md) | `drop_bads` | eeg, seeg, ecog, meg | Remove channels marked as bad (flat, noisy, or manually flagged) |
| [drop_nondata_channels](channel/drop_nondata_channels/SKILL.md) | `drop_nondata_channels` | eeg, seeg, ecog, meg | Remove non-data channels (markers/triggers, misc, optionally physiological refs) |
| [fill_nan](channel/fill_nan/SKILL.md) | `fill_nan` | eeg, seeg, ecog, meg, spike, fnirs | Replace non-finite values (NaN, Inf) with a specified constant |
| [interpolate_bads](channel/interpolate_bads/SKILL.md) | `interpolate_bads` | eeg, ecog | Spherical spline interpolation to reconstruct bad channel data from neighbors |
| [pick_channels](channel/pick_channels/SKILL.md) | `pick_channels` | eeg, seeg, ecog, meg | Select channels by name list or type (EEG, MEG, etc.) |
| [scale](channel/scale/SKILL.md) | `scale` | eeg, seeg, ecog, meg, fnirs | Scale/normalize signal amplitude — robust, standard, or numeric factor |

### `reference/` — Reference transformations  *(2)*

| Operator | step_string | Modalities | Description |
|---|---|---|---|
| [bipolar_ref](reference/bipolar_ref/SKILL.md) | `bipolar_ref` | seeg, ecog | Bipolar referencing for sEEG/depth electrodes — subtract adjacent contacts |
| [car](reference/car/SKILL.md) | `car` | eeg, ecog | Common Average Reference — re-reference EEG to the mean of all channels |

### `spatial/` — Spatial filters  *(planned)*

_No operators in this group yet — slot reserved for future skills (03 / T2)._

### `spectral/` — Spectral transforms  *(1)*

| Operator | step_string | Modalities | Description |
|---|---|---|---|
| [hilbert](spectral/hilbert/SKILL.md) | `hilbert` | eeg, seeg, ecog, meg | Hilbert transform to extract instantaneous amplitude envelope |

### `feature_time/` — Time-domain features  *(planned)*

_No operators in this group yet — slot reserved for future skills (03 / T2)._

### `connectivity/` — Connectivity operators  *(planned)*

_No operators in this group yet — slot reserved for future skills (03 / T2)._

### `source/` — Source localization  *(planned)*

_No operators in this group yet — slot reserved for future skills (03 / T2)._

### `spike/` — Spike sorting & unit activity  *(1)*

| Operator | step_string | Modalities | Description |
|---|---|---|---|
| [spike_sorting](spike/spike_sorting/SKILL.md) | `spike_sort` | spike | Spike sorting workflow — detection, feature extraction, clustering, and quality assessment of sin... |

### `adaptive_cleaning/` — Adaptive artifact cleaning  *(1)*

| Operator | step_string | Modalities | Description |
|---|---|---|---|
| [ica](adaptive_cleaning/ica/SKILL.md) | `ica` | eeg, meg | ICA artifact removal — automatic detection and exclusion of EOG/ECG components |

### `fnirs/` — fNIRS-specific operators  *(planned)*

_No operators in this group yet — slot reserved for future skills (03 / T2)._

### `qc_operator/` — Operator-level QC  *(planned)*

_No operators in this group yet — slot reserved for future skills (03 / T2)._

### `misc/` — Index / miscellaneous  *(planned)*

_No operators in this group yet — slot reserved for future skills (03 / T2)._

## Layering contract

- **layer**: every operator skill carries `layer: L3` in frontmatter.
- **group**: matches its parent directory (`bci/operators/<group>/<op>/`).
- **analysis_goal_allowed/forbidden**: drawn from the 9-enum REGISTRY in
  `easybci_lib/tools/neural_processing/preprocess/analysis_goals.py`.

The layer / group enums are defined in
`easybci_lib/tools/neural_processing/skill_layers.py` (single source of
truth). Run `python -m easybci_lib.tools.neural_processing._check_consistency --strict`
to verify the contract before merging new operator skills.

## Output-path contract (ABSOLUTE)

**Output paths follow `sub-{subject_id}/ses-{session_id}/` strictly.**
Both ids come from `<work_dir>/middle_process/inputs_routing.json` (multi-input
runs) or the per-file `inspection_report.identity` field (single-input runs).
Operators that touch output paths **MUST NOT** derive `subject_id` from
`Path(raw).stem` — that's the single regression mode that produced the
"1623 file landed in ses-1842 bucket" bug. The same constraint is enforced
at the orchestrator level (`bci/pipeline/SKILL.md`); it's repeated here
because operators may show up in standalone scripts where the orchestrator
context isn't loaded.

Filename normalization is also mandatory: every artefact name uses
`stem_safe = Path(raw).stem.replace(" ", "_")`. A file with a space in its
name landing in a `_preprocessed.pkl` / `_epochs.pkl` bucket is a contract
violation that `verify_layout_strict_multi` will catch at finalize time.
