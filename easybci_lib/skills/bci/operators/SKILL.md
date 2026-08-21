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

### `filter/` — Frequency-domain filtering  *(6)*

| Operator | step_string | Modalities | Description |
|---|---|---|---|
| [bandpass_filter](filter/bandpass_filter/SKILL.md) | `bandpass` | eeg, seeg, ecog, meg, fnirs | Band-pass filter to isolate frequency range of interest with Nyquist guard |
| [notch_filter](filter/notch_filter/SKILL.md) | `notch` | eeg, seeg, ecog, meg | Notch filter to remove power line interference (50/60 Hz) and harmonics |
| [resample](filter/resample/SKILL.md) | `resample` | eeg, seeg, ecog, meg | Resample signal to target frequency — downsampling with anti-aliasing |
| [detrend](filter/detrend/SKILL.md) | `detrend` | eeg, seeg, ecog, meg | Polynomial detrending (DC offset, linear, quadratic) |
| [filter_bank](filter/filter_bank/SKILL.md) | `filter_bank` | eeg, seeg, ecog, meg | Multi-band filter bank — parallel band-pass, adds band dimension |
| [smooth](filter/smooth/SKILL.md) | `smooth` | eeg, seeg, ecog, meg | Temporal smoothing (moving average, Savitzky-Golay, Gaussian) |

### `channel/` — Channel selection & repair  *(13)*

| Operator | step_string | Modalities | Description |
|---|---|---|---|
| [clip](channel/clip/SKILL.md) | `clip` | eeg, seeg, ecog, meg | Clamp signal amplitude to a maximum absolute value |
| [drop_bads](channel/drop_bads/SKILL.md) | `drop_bads` | eeg, seeg, ecog, meg | Remove channels marked as bad (flat, noisy, or manually flagged) |
| [drop_nondata_channels](channel/drop_nondata_channels/SKILL.md) | `drop_nondata_channels` | eeg, seeg, ecog, meg | Remove non-data channels (markers/triggers, misc, optionally physiological refs) |
| [fill_nan](channel/fill_nan/SKILL.md) | `fill_nan` | eeg, seeg, ecog, meg, spike, fnirs | Replace non-finite values (NaN, Inf) with a specified constant |
| [interpolate_bads](channel/interpolate_bads/SKILL.md) | `interpolate_bads` | eeg, ecog | Spherical spline interpolation to reconstruct bad channel data from neighbors |
| [pick_channels](channel/pick_channels/SKILL.md) | `pick_channels` | eeg, seeg, ecog, meg | Select channels by name list or type (EEG, MEG, etc.) |
| [scale](channel/scale/SKILL.md) | `scale` | eeg, seeg, ecog, meg, fnirs | Scale/normalize signal amplitude — robust, standard, or numeric factor |
| [detect_bads](channel/detect_bads/SKILL.md) | `detect_bads` | eeg, seeg, ecog, meg | Automatic bad-channel detection (flat, amplitude, RANSAC, kurtosis) |
| [mark_bads](channel/mark_bads/SKILL.md) | `mark_bads` | eeg, seeg, ecog, meg | Non-destructive bad-channel marking from external list |
| [set_channel_types](channel/set_channel_types/SKILL.md) | `set_channel_types` | eeg, seeg, ecog, meg | Re-type named channels (EOG, ECG, EMG, misc, stim) |
| [set_montage](channel/set_montage/SKILL.md) | `set_montage` | eeg, seeg, ecog, meg | Attach standard or digitised electrode positions |
| [derive_bipolar_channel](channel/derive_bipolar_channel/SKILL.md) | `derive_bipolar_channel` | eeg, seeg, ecog, meg | Create bipolar monitor channels from electrode pairs |
| [minmax_scale](channel/minmax_scale/SKILL.md) | `minmax_scale` | eeg, seeg, ecog, meg | Min-max feature normalisation to fixed range |

### `reference/` — Reference transformations  *(3)*

| Operator | step_string | Modalities | Description |
|---|---|---|---|
| [bipolar_ref](reference/bipolar_ref/SKILL.md) | `bipolar_ref` | seeg, ecog | Bipolar referencing for sEEG/depth electrodes — subtract adjacent contacts |
| [car](reference/car/SKILL.md) | `car` | eeg, ecog | Common Average Reference — re-reference EEG to the mean of all channels |
| [reref_channels](reference/reref_channels/SKILL.md) | `reref_channels` | eeg, seeg, ecog | Re-reference to named channels (vertex, nose, white-matter contact) |

### `spatial/` — Spatial filters  *(1)*

| Operator | step_string | Modalities | Description |
|---|---|---|---|
| [dss](spatial/dss/SKILL.md) | `dss` | eeg, meg | Denoising source separation — bias-filter spatial filtering |

### `spectral/` — Spectral transforms  *(3)*

| Operator | step_string | Modalities | Description |
|---|---|---|---|
| [hilbert](spectral/hilbert/SKILL.md) | `hilbert` | eeg, seeg, ecog, meg | Hilbert transform to extract instantaneous amplitude envelope |
| [aggregate_bands](spectral/aggregate_bands/SKILL.md) | `aggregate_bands` | eeg, seeg, ecog, meg | Cross-band aggregation (geometric/arithmetic mean) |
| [amplitude_modulation](spectral/amplitude_modulation/SKILL.md) | `amplitude_modulation` | eeg, seeg, ecog, meg | Amplitude-modulation spectrum features |

### `epoch/` — Epoching & segmentation  *(4)*

| Operator | step_string | Modalities | Description |
|---|---|---|---|
| [epoch](epoch/epoch/SKILL.md) | `epoch` | eeg, seeg, ecog, meg | Event-locked epoching |
| [segment](epoch/segment/SKILL.md) | `segment` | eeg, seeg, ecog, meg | Fixed / sliding window segmentation |
| [baseline_correct](epoch/baseline_correct/SKILL.md) | `baseline_correct` | eeg, seeg, ecog, meg | Per-epoch baseline correction (mean, median, z-score, ratio) |
| [reject_epochs](epoch/reject_epochs/SKILL.md) | `reject_epochs` | eeg, seeg, ecog, meg | Criterion-based epoch rejection |

### `event/` — Event & trial management  *(5)*

| Operator | step_string | Modalities | Description |
|---|---|---|---|
| [define_events](event/define_events/SKILL.md) | `define_events` | eeg, seeg, ecog, meg | Derive events from trigger/STIM/photodiode channels |
| [import_events](event/import_events/SKILL.md) | `import_events` | eeg, seeg, ecog, meg | Attach external event/annotation table (BIDS TSV, detector output) |
| [repair_events](event/repair_events/SKILL.md) | `repair_events` | eeg, seeg, ecog, meg | Detect and repair event table defects |
| [select_events](event/select_events/SKILL.md) | `select_events` | eeg, seeg, ecog, meg | Filter events by condition/attribute criteria |
| [attach_metadata](event/attach_metadata/SKILL.md) | `attach_metadata` | eeg, seeg, ecog, meg | Join trial-level behavioural/stimulus metadata onto epochs |

### `adaptive_cleaning/` — Adaptive artifact cleaning  *(8)*

| Operator | step_string | Modalities | Description |
|---|---|---|---|
| [ica](adaptive_cleaning/ica/SKILL.md) | `ica` | eeg, meg | ICA artifact removal — automatic detection and exclusion of EOG/ECG components |
| [ic_classify](adaptive_cleaning/ic_classify/SKILL.md) | `ic_classify` | eeg, meg | Automatic IC classification (ICLabel, MEGNet, correlation) |
| [manual_ic_selection](adaptive_cleaning/manual_ic_selection/SKILL.md) | `manual_ic_selection` | eeg, meg | Human-in-the-loop IC selection from reviewed index list |
| [detect_artifact_spans](adaptive_cleaning/detect_artifact_spans/SKILL.md) | `detect_artifact_spans` | eeg, seeg, ecog, meg | Non-destructive artefact-span annotation |
| [interpolate_artifact](adaptive_cleaning/interpolate_artifact/SKILL.md) | `interpolate_artifact` | eeg, seeg, ecog, meg | Time-domain artefact-span interpolation (linear, PCHIP, spline) |
| [reject_bad_segments](adaptive_cleaning/reject_bad_segments/SKILL.md) | `reject_bad_segments` | eeg, seeg, ecog, meg | Remove or omit contaminated continuous spans |
| [wavelet_ica](adaptive_cleaning/wavelet_ica/SKILL.md) | `wavelet_ica` | eeg, meg | Wavelet-threshold ICA component cleaning |
| [overlap_regression](adaptive_cleaning/overlap_regression/SKILL.md) | `overlap_regression` | eeg, meg | Overlapping-response regression (rERP / deconvolution) |

### `meg/` — MEG hardware-specific operators  *(4)*

| Operator | step_string | Modalities | Description |
|---|---|---|---|
| [maxwell_filter](meg/maxwell_filter/SKILL.md) | `maxwell_filter` | meg | Signal-space separation (SSS / tSSS) |
| [ctf_grad_comp](meg/ctf_grad_comp/SKILL.md) | `ctf_grad_comp` | meg | CTF synthetic-gradiometer noise compensation |
| [estimate_head_position](meg/estimate_head_position/SKILL.md) | `estimate_head_position` | meg | Continuous head-position estimation from cHPI |
| [align_head_position](meg/align_head_position/SKILL.md) | `align_head_position` | meg | Cross-run head-position alignment |

### `dataset/` — Dataset-level operations  *(6)*

| Operator | step_string | Modalities | Description |
|---|---|---|---|
| [concatenate](dataset/concatenate/SKILL.md) | `concatenate` | eeg, seeg, ecog, meg | Run / split concatenation along time or trial axis |
| [crop](dataset/crop/SKILL.md) | `crop` | eeg, seeg, ecog, meg | Time-range cropping |
| [split_runs](dataset/split_runs/SKILL.md) | `split_runs` | eeg, seeg, ecog, meg | Split continuous recording into runs |
| [equalize_channels](dataset/equalize_channels/SKILL.md) | `equalize_channels` | eeg, seeg, ecog, meg | Reduce recordings to common channel set |
| [exclude_subjects](dataset/exclude_subjects/SKILL.md) | `exclude_subjects` | eeg, seeg, ecog, meg | Recording / subject exclusion with criterion |
| [sort_epochs](dataset/sort_epochs/SKILL.md) | `sort_epochs` | eeg, seeg, ecog, meg | Epoch reordering by metadata key |

### `connectivity/` — Connectivity operators  *(1)*

| Operator | step_string | Modalities | Description |
|---|---|---|---|
| [graph_metrics](connectivity/graph_metrics/SKILL.md) | `graph_metrics` | eeg, seeg, ecog, meg | Graph-theoretic node metrics from connectivity matrix |

### `misc/` — Miscellaneous  *(1)*

| Operator | step_string | Modalities | Description |
|---|---|---|---|
| [no_op](misc/no_op/SKILL.md) | `no_op` | eeg, seeg, ecog, meg, spike, fnirs | Declared omission — records a deliberate non-action |

### `qc_operator/` — Operator-level QC  *(1)*

| Operator | step_string | Modalities | Description |
|---|---|---|---|
| [sleep_stager](qc_operator/sleep_stager/SKILL.md) | `sleep_stager` | eeg | AASM sleep-stage scoring |

### `spike/` — Spike sorting & unit activity  *(1)*

| Operator | step_string | Modalities | Description |
|---|---|---|---|
| [spike_sorting](spike/spike_sorting/SKILL.md) | `spike_sort` | spike | Spike sorting workflow — detection, feature extraction, clustering, and quality assessment of sin... |

### `psg/` — Polysomnography-specific operators  *(3)*

| Operator | step_string | Modalities | Description |
|---|---|---|---|
| [respiratory_events](psg/respiratory_events/SKILL.md) | `respiratory_events` | eeg | Detect apnea, hypopnea, and SpO2 desaturation events (AASM criteria) |
| [plm_detect](psg/plm_detect/SKILL.md) | `plm_detect` | eeg | Periodic Limb Movement detection from leg EMG (WASM/AASM criteria) |
| [epoch_qc_sleep](psg/epoch_qc_sleep/SKILL.md) | `epoch_qc_sleep` | eeg | 30-second epoch-level multi-channel quality scoring, hypnogram-aware |

### `fnirs/` — fNIRS-specific operators  *(planned)*

_No operators in this group yet — slot reserved for future skills._

### `source/` — Source localization  *(planned)*

_No operators in this group yet — slot reserved for future skills._

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
