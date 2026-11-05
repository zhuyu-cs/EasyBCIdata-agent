---
name: ocular_artifact
description: "Decision skill — whether and how to remove ocular (blink / saccade) artifacts, and which steps do NOT count as removal"
layer: L3
group: adaptive_cleaning
metadata:
  kind: decision
  tags: [decision, adaptive_cleaning, eog, blink, ocular, ica, regression]
  modalities: [eeg, meg]
  related_operators: [ica, regression_eog]
  analysis_goal_allowed: [classification, source_localization, feature_extraction, clinical_screening, exploratory, generic, connectivity, phase_amplitude_coupling, online_inference]
  analysis_goal_forbidden: []
---
# Ocular Artifact — Decision

This is a **decision skill**, not an operator: it has no `step_string` and is
never a pipeline step. It tells you which operator to put in the proposal
(`ica:eog` or `regression_eog`) and what to tell the user when ocular
artifacts are left in.

`plan_pipeline` (suggest / propose) and `revise_proposal` already return an
`ocular_artifact` block with the evidence, the reference channel, the
recommendation and its reason. Load this skill only when that block has
`must_present: true` and you need the reasoning behind it.

## Evidence

- `deep_inspect` → `artifact_summary.blink_rate_per_min` counts blink
  **events** (0.5–8 Hz band-pass, > 5·MAD of the channel itself, ≥ 50 ms,
  0.3 s refractory — unit-, gain- and sampling-rate-independent) over a
  ≥ 2-minute window; `blink_reference` names the channels: vertical EOG
  first (horizontal EOG mostly carries saccades), then Fp1/Fp2/Fpz, then
  AF7/AF8/AF3/AF4/AFz. Vendor-decorated names (`EEG Fp1-REF`, `Fp1.`,
  `FP1-F7`) and EOG names (`VEOG`, `VEO`/`HEO`, `LOC`/`ROC`, AASM `E1`/`E2`)
  are recognised. Awake adults blink 15–20 times per minute; ≥ 5/min counts
  as detected.
- `blink_rate_per_min: null` means not measured (no reference, or under
  30 s of data) — not "no blinks".
- A report without `blink_reference` is from the old sample-count metric —
  its number is not a blink rate. Treat ocular contamination as unknown.

## Decision table

ICA data sufficiency = recording ≥ 60 s **and** ≥ 25·k² samples at the ICA
step, k = min(effective channels, 20) (the sampling rate after any earlier
`resample` counts). Judging by samples keeps 128 Hz and 1000 Hz recordings
on the same scale.

| Reference available | Condition | Use |
|---|---|---|
| EOG channel | ICA data sufficient and not `online_inference` | `ica:eog` |
| EOG channel | otherwise | `regression_eog` |
| Frontal pole / AF only | ≥ 12 effective channels, ICA data sufficient, not `online_inference` | `ica:eog` — the frontal channels are the blink reference |
| Frontal pole / AF only | otherwise | `regression_eog:auto` — note in the proposal that real frontal EEG is regressed out with the blink |
| Neither, names recognised as 10-20 / 10-10 | — | Ocular artifacts cannot be removed. Say so in the proposal; trial rejection is the only fallback |
| Neither, names not recognised (`montage_recognized: false`, e.g. `E1..E128`, `A1..D32`) | — | Ask the user which electrodes sit near the eyes and name them: `ica:eog,<ch>,...` or `regression_eog:<ch>,...` |

Effective channels = data channels − channels that must be dropped − EOG
channels.

Not required (the block still carries the recommendation, without
`must_present`):

- invasive recordings (sEEG / ECoG / spike);
- `online_inference`;
- paradigms where eye movements are the signal (`sleep`, `psg`, `eye`,
  `saccade`, `gaze`, `oculo` in the paradigm name — REM in sleep staging,
  saccade / gaze studies). For any other paradigm where the user analyses
  eye movements, do not remove them either — say so in the proposal.

## What does NOT count as ocular removal

| Step | What it actually does | Why the blink stays |
|---|---|---|
| `clip` | Caps amplitudes at ± a threshold | The blink waveform below the cap remains, distorted |
| `scale` | Rescales each channel | Changes units, not content |
| `reject_epochs` | Drops whole outlier trials | Trials with ordinary blinks are kept, blink included |
| `reject_bad_segments` | Marks / cuts extreme spans | Same — only the extreme spans go |
| `bandpass` | Removes frequencies outside the band | Blink energy (0.5–8 Hz) sits inside the usual 0.5–40 Hz band |

Proposing one of these **instead of** `ica:eog` / `regression_eog` leaves the
eyes in the data. If the user decides to keep ocular artifacts (e.g. the
classifier must be robust to them), record that decision and its reason in
the proposal rationale.

## Order

After `bandpass` (and `notch`), **before `car`**. The average reference
spreads the frontal blink to every channel; removing it first keeps the
blink confined to the components / channels that carry it.

`ica` fitted after `car` must use at most rank − 1 components; `op_ica`
bounds `n_components` by the effective rank automatically.

## Rank

Removing k ICA components lowers the data rank by k. The order with `car`
matters: `car` after `ica` (the recommended order) caps the rank at n − 1 —
the average reference maps the ICA-reduced subspace one-to-one and absorbs
the ICA drop; `ica` after `car` gives n − 1 − k. The rank is recorded in
`pipeline_status.data_rank`, the NWB scratch
`easybci_data_rank` and the AI_ready pickle (`rank`). Pass it explicitly to
covariance-based methods (CSP, xDAWN, Riemannian) — see the `csp` skill.

## What to write in the proposal

- Evidence: blink rate and reference channel from the `ocular_artifact` block.
- The chosen method, or the explicit statement that ocular artifacts are not removed and why.
- With the frontal-proxy reference: that frontal EEG correlated with blinks is removed too.

## Failure modes

| Symptom | Cause | Fix |
|---|---|---|
| `[ica] ocular artifacts NOT removed` in pipeline warnings | No EOG, frontal-pole or AF channel recognised | If electrodes near the eyes exist under other names, name them (`ica:eog,<ch>`); else keep trial rejection and state it |
| `[ica] EOG channel(s) ... matched no component` | The EOG electrode does not see the eyes (detached / mis-wired; its blink count on deep_inspect is not trustworthy) | The frontal proxy was used instead — check the EOG electrode; with no frontal channel name the reference explicitly |
| `[ica] no component matched the blink reference` | Few blinks, or blinks already removed upstream | Check the blink rate; consider `regression_eog:auto` |
| Frontal power drops far more than expected | Frontal-proxy ICA removed frontal EEG with the blink | Expected trade-off; report it, or use an EOG channel if one exists |
| CSP collapses after ICA + car | Rank reduced by car and ICA, covariance near-singular | Pass the recorded `rank` |
