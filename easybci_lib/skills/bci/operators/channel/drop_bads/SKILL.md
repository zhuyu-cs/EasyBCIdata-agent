---
name: drop_bads
description: "Remove channels marked as bad (flat, noisy, or manually flagged)"
layer: L3
group: channel
metadata:
  tags: [operator, channels, bad, reject, quality]
  modalities: [eeg, seeg, ecog, meg]
  step_string: "drop_bads"
  analysis_goal_allowed: [classification, source_localization, feature_extraction, clinical_screening, exploratory, generic, connectivity, phase_amplitude_coupling, online_inference]
  analysis_goal_forbidden: []
---
# Drop Bad Channels

## Function

Removes channels from the data array that are marked as bad in `meta.bad_channels`. Reduces channel count. Irreversible — use `interpolate_bads` if you want to preserve channel count.

## Parameter Format

`drop_bads` — No parameters. Operates on pre-identified bad channels.

## How Bad Channels Are Identified

Bad channels are marked during `quality_check` or `inspect_data` and stored in `data_dict["meta"]["bad_channels"]`. Criteria:
- Flat channels (near-zero variance)
- Excessively noisy channels (variance > 5x median)
- Channels with high artifact ratio
- Manually flagged by researcher

## When to Use

- Quality check identified clearly broken channels
- Channels are physically disconnected or have extreme impedance
- You want to reduce dimensionality before spatial filtering (CSP, ICA)
- Channel count is high enough that dropping 1-3 channels is acceptable

## When NOT to Use

- No bad channels identified (no-op — returns data unchanged)
- Need to maintain consistent channel count across subjects (use `interpolate_bads` instead)
- Downstream analysis requires specific channel locations (e.g., source localization)

## Constraints

- If no `meta.bad_channels` list exists, this step does nothing
- Permanently removes channels — cannot be undone without reloading
- Updates `data.shape[0]` and `channels` list accordingly

## Ordering

- Apply BEFORE: `car` (bad channels corrupt the average reference)
- Apply BEFORE: `ica` (bad channels waste ICA components)
- Apply AFTER: quality check has been run to identify bad channels
- Alternative: use `interpolate_bads` instead if channel preservation needed

## Reference Code

### Using MNE (on existing Raw object)

```python
raw.drop_channels(['Fp1', 'Fp2'])
```

### Standalone (numpy, from meta.bad_channels)

```python
import numpy as np

bad_channels = ['Fp1', 'Fp2']  # from meta.bad_channels
keep_idx = [i for i, ch in enumerate(channels) if ch not in bad_channels]
data = data[keep_idx]
channels = [channels[i] for i in keep_idx]
```

### Key API

- `raw.drop_channels(bad_list)` — MNE method
- Operates on `data_dict["meta"]["bad_channels"]` in pipeline context


## `drop_bads:auto` (Phase 1+)

When the param is the literal string `auto`, the operator detects bad channels automatically:

| Criterion | Threshold | Notes |
|-----------|-----------|-------|
| flat | `std < 1e-7` | sensor disconnected / amplifier rail |
| high variance | `std > 3 × median(non-zero σ)` | usually muscle / electrical artifact |
| spike | `max(|x|) > 1e-2` (Volts, EEG-scaled) | electrode pop |

Detected names are written to `meta['bad_channels']` and **dropped** from `data` + `channels` arrays. No interpolation is performed — downstream goals that need every channel filled in must explicitly add `interpolate_bads` after this step.

The trailing `dropped_channels` list in `meta` is what `qc.py` and `reasoning.md` surface to the user.

**Goal-conditional auto-injection** (`codegen/generator.py:_enforce_clean_output`):

| analysis_goal | drop_bads:auto auto-inserted? |
|---|---|
| classification | yes |
| feature_extraction | yes |
| clinical_screening | yes |
| generic | yes |
| source_localization | no |
| exploratory | no |

For opted-out goals, manually add `drop_bads:auto` or `interpolate_bads` if you need them.
