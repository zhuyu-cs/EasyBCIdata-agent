---
name: clip
description: "Clamp signal amplitude to ± an absolute value (with a unit such as 500uV, or in data units) or ± k·SD per channel"
layer: L3
group: channel
metadata:
  tags: [operator, clip, clamp, artifact, threshold]
  modalities: [eeg, seeg, ecog, meg]
  step_string: "clip"
  analysis_goal_allowed: [classification, source_localization, feature_extraction, clinical_screening, exploratory, generic, connectivity, phase_amplitude_coupling, online_inference]
  analysis_goal_forbidden: []
---
# Clip

## Function

Clamps every sample into a fixed range; values outside are replaced by the
bound. Suppresses residual extreme samples before model input. It does
**not** remove ocular or muscle artifacts: a blink below the bound stays,
a blink above it is flattened, not removed (see `ocular_artifact`).

## Parameter Format

- `clip:{max_abs}{unit}` — clamp to `±max_abs` in a physical unit (`V`, `mV`,
  `uV`/`µV`, `nV`, `T`, `pT`, `fT`), converted to the unit the signal is held
  in (`meta.data_unit`). **Prefer this for unscaled data**: MNE-read files
  (EDF/BDF/FIF/BrainVision/Curry …) are held in **volts**, other loaders may
  keep µV — `clip:500uV` means the same on both.
- `clip:{max_abs}` — a bare number is in **data units** (V for MNE-read EEG,
  scaled units after `scale`).
- `clip:sd={k}` — clamp each channel to `mean ± k·SD` of that channel.

Examples:
- `clip:150uV` — unscaled EEG, clamp at ±150 µV whatever unit the data is held in
- `clip:5` — after `scale:robust`, clamp at ±5 (scaled units)
- `clip:sd=5` — per channel, clamp at mean ± 5 SD

A missing, zero, negative or non-numeric threshold, or an unsupported unit,
is an error. If the data unit is unknown (e.g. after `scale`, or a format that
records none) or of another family (µV against T), the number is applied
unconverted and `unit_status` reports `unit_unknown` / `unit_mismatch`.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| max_abs | float > 0 [+ unit] | required (unless `sd=`) | Absolute bound; with a unit suffix it is converted to the data unit, bare it is in data units |
| sd | float > 0 | — | Per-channel bound in standard deviations |

## Output record

`pipeline_status.clip = {mode: abs|sd, threshold, unit, data_unit, threshold_data_units, unit_status, clipped_frac, warning}`,
also listed in the QC report "Pipeline notes":

- `warning: no_effect` — no sample was clipped; the step did nothing (the
  threshold is above the data range — e.g. `clip:500uV` on ±100 µV EEG, or a
  bare `clip:500` on EEG held in volts).
- `warning: heavy_clipping` — more than 1 % of samples clipped; the
  threshold is distorting the signal.

## When to Use

- After scaling, to suppress remaining outlier samples (> 3–5 SD)
- Before ML model input to keep extreme values from dominating gradients
- When you want to keep all time points (vs. epoch rejection which discards trials)

## When NOT to Use

- As ocular-artifact removal — use `ica:eog` / `regression_eog` (`ocular_artifact` skill)
- Before ICA (clipping destroys the linearity ICA relies on)
- After `car` when downstream methods rely on the exact rank: clipping breaks
  the exact channel dependency (`data_rank` warns `rank_mismatch`)
- When absolute amplitude is meaningful (e.g. ERD/ERS magnitude)
- For spike data (amplitude carries information)

## Choosing the Threshold

| Data state | Recommended | Rationale |
|------------|-------------|-----------|
| After `scale:robust` | `clip:5` | IQR-scaled units; 5 covers > 99.9 % of clean data |
| After `scale:standard` | `clip:3` – `clip:5` | Units are SD |
| Unscaled | `clip:sd=5`, or an absolute bound with a unit (`clip:150uV`) | Typical EEG is within ±100 µV; blinks reach 100–300 µV |

Check `clipped_frac` after the run: 0 means the bound is too high, > 1 % too low.

## Ordering

- Apply AFTER: scale (so the threshold is in normalized units)
- Apply as one of the last steps
- Apply BEFORE: final output / ML feature extraction

## Reference Code

### Implementation

```python
import numpy as np

# clip:<max_abs>[unit]: convert the bound to the data unit first,
# e.g. 500 µV on data held in V -> 500e-6
data = np.clip(data, -max_abs, max_abs)

# clip:sd=<k>
mu = np.nanmean(data, axis=1, keepdims=True)
sd = np.nanstd(data, axis=1, keepdims=True)
data = np.clip(data, mu - k * sd, mu + k * sd)
```

### Key API

- `np.clip(data, lo, hi)` — element-wise clamping; `lo`/`hi` broadcast per channel
