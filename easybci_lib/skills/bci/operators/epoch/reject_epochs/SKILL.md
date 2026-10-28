---
name: reject_epochs
description: "Adaptive low-quality trial rejection (robust z per channel, capped)"
layer: L3
group: epoch
metadata:
  tags: [operator, epoching, reject_epochs, trial_qc]
  modalities: [eeg, seeg, ecog, meg]
  step_string: "reject_epochs"
  analysis_goal_allowed: [classification, source_localization, feature_extraction, clinical_screening, exploratory, generic, connectivity, phase_amplitude_coupling]
  analysis_goal_forbidden: [online_inference]
---
# Adaptive low-quality trial rejection

## Function

Delete trials whose signal is an outlier **for their own channels**. Runs by
default whenever the recording has trial events — no step needed. The step
only exists to tune or disable it.

It is an epoch-stage operator: codegen strips it from the continuous step
chain and configures the trial-QC block of `pipeline.py`, which runs after all
continuous steps (filters, re-reference, channel drops).

## Parameter Format

| Step | Meaning |
|------|---------|
| *(absent)* / `reject_epochs` / `reject_epochs:auto` | defaults below |
| `reject_epochs:off` | keep every trial (epoch window still resolved) |
| `reject_epochs:z=5,multi=3.5,min_ch=2,max_frac=0.2,min_trials=10` | tuned |
| `reject_epochs:window=3.5~7.0` (also `window=3.5,7.0`) | set the epoch window here (overrides event duration; an explicit plan_pipeline `epoch_window` still wins) |

| Key | Default | Meaning |
|-----|---------|---------|
| `z` | 5.0 | one channel above this robust z rejects the trial |
| `multi` | 3.5 | ... or at least `min_ch` channels above this |
| `min_ch` | 2 | channel count for the `multi` rule |
| `max_frac` | 0.2 | cap on statistically rejected trials; worst-first when exceeded (`cap_hit`) |
| `min_trials` | 10 | below this only hard checks run |

## Algorithm

1. **Hard rejections** (uncapped): non-finite samples in the window; a channel
   flat in this trial but live in most trials; per-trial artifact / invalid /
   missing flags shipped with the recording.
2. **Features** per trial × channel: peak-to-peak and log-variance.
3. **Robust z** per channel across trials: `(x − median) / (1.4826·MAD)`,
   one-sided. Each channel is normalised by its own distribution, so a channel
   noisy in every trial (e.g. frontal blinks) raises its own baseline instead
   of condemning all trials — fixed µV thresholds fail exactly there.
4. Reject when `max z > z` or `#channels(z > multi) ≥ min_ch`.
5. Cap at `floor(max_frac · n)`; report `cap_hit` + candidate count instead of
   silently deleting more.

## Epoch window

Resolved in the same block: explicit `epoch_window` (plan_pipeline arg, or
the label-dependent span of `dataset_understanding.trial_timeline`) →
`reject_epochs:window=` → shared trial duration (`[0, duration]`) → fallback
`[-0.2, segment_duration]`. Trials falling outside the recording are counted
(`n_out_of_bounds`), never dropped silently. A recording without trial
events (any modality) is left continuous — no window, no trial QC.

## Outputs

- Curated NWB: `trials` table = kept trials (`label`, `trial_index`);
  `invalid_times` = rejected trials with `reason`; scratch `easybci_trials`
  holds the full decision. The continuous signal is not cut.
- `AI_ready/*_epochs.pkl`: kept trials only, plus `trial_index` and
  `epoch_window`.
- `middle_process/trial_rejection__<file_id>.json`, a "Trial rejection"
  section in each QC report, and a `trial_rejection` block in the tool return.

## When to tune

- `cap_hit` → the whole recording is noisy or a bad channel survived: fix
  channels first (`drop_bads:auto,<name>`), then reconsider `max_frac`.
- `class_imbalance_warning` → one condition lost > 50 % of trials; tell the user.
- Very short recordings (< `min_trials`) → only hard checks apply.
