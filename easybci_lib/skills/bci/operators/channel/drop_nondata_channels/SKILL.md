---
name: drop_nondata_channels
description: "Remove non-data channels (markers/triggers, misc, optionally physiological refs)"
layer: L3
group: channel
metadata:
  tags: [operator, channels, marker, trigger, stim, clean, nondata]
  modalities: [eeg, seeg, ecog, meg]
  step_string: "drop_nondata_channels"
  analysis_goal_allowed: [classification, source_localization, feature_extraction, clinical_screening, exploratory, generic, online_inference]
  analysis_goal_forbidden: [connectivity, phase_amplitude_coupling]
---
# Drop Non-Data Channels

## Function

Removes channels that are **not target neural signal** — pure marker/trigger
channels (STI / Trigger / Status / Event / DC), and optionally misc and
physiological-reference channels (EOG/EMG/ECG). Classification comes from
`channel_classifier` (MNE `ch_types` first, channel-name regex as fallback).

This is **different from `drop_bads`**: `drop_bads` removes *broken data
channels* (flat/noisy); this removes *non-data channels* that were never meant
to be analyzed.

## Parameter Format

`drop_nondata_channels:{mode}`

| mode | Drops | Keeps |
|------|-------|-------|
| `markers_only` (default) | marker/trigger channels | data + physio + misc |
| `markers_and_misc` | marker + misc | data + physio |
| `data_only` | marker + misc + physio | data only |

`drop_nondata_channels` with no param == `markers_only`.

## When to Use

- **Almost always early in the pipeline** when `inspect_data`'s
  `channel_summary.must_drop` is non-empty — marker channels pollute CAR, ICA,
  and filtering.
- Use `markers_only` by default.
- Use `markers_and_misc` if misc channels are clearly junk.
- Use `data_only` **only** when you are sure EOG/ECG are not needed for ICA
  artifact rejection later. If ICA-with-EOG-reference is planned, keep physio.

### Goal-conditional output cleanup (Phase 1+)

After ICA, the codegen layer (`_enforce_clean_output` in
`tools/neural_processing/codegen/generator.py`) decides whether to append a
**second** `drop_nondata_channels:data_only` step before save, based on the
`analysis_goal` declared at `propose_pipeline` time. The decision table is the
single source of truth in
[`improved_docs/plans/goal-driven-preprocessing/02-phase1-goal-first.md` §1.3](../../../../improved_docs/plans/goal-driven-preprocessing/02-phase1-goal-first.md):

| `analysis_goal` | Post-ICA `data_only` cleanup | Rationale |
|-----------------|------------------------------|-----------|
| `classification` | ✅ append | downstream classifier wants pure brain channels |
| `feature_extraction` | ✅ append | features should not mix in EOG/ECG |
| `clinical_screening` | ✅ append | screening models trained on data-only montages |
| `generic` | ✅ append | conservative default for unknown downstream |
| `source_localization` | ❌ **skip** (intentional) | source modelling consumes physio refs / full montage |
| `exploratory` | ❌ skip | exploration phase wants the original physical channels visible |

This is a **design choice**, not a bug. The skill writer does NOT need to
hand-add a trailing `drop_nondata_channels:data_only`; codegen handles it.

## When NOT to Use

- Spike/unit data — no channel-type concept; operator is a no-op.
- MEG `ref_meg` channels are classified as data and **never** dropped here.

## Ordering

Place **before** referencing (CAR/bipolar), ICA, and filtering. Recommended:
right after `inspect`/load, before `notch`/`bandpass`.

## Evidence / Reporting

Dropped channel names are recorded in `meta.dropped_channels` and surfaced in
the CONFIRM display and mini-repo reasoning so the user sees exactly what was
removed and why.
