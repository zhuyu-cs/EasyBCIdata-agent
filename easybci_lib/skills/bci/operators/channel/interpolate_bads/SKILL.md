---
name: interpolate_bads
description: "Spherical spline interpolation to reconstruct bad channel data from neighbors"
layer: L3
group: channel
metadata:
  tags: [operator, channels, interpolation, repair, spline]
  modalities: [eeg, ecog]
  step_string: "interpolate_bads"
  analysis_goal_allowed: [classification, source_localization, feature_extraction, clinical_screening, exploratory, generic, connectivity, phase_amplitude_coupling, online_inference]
  analysis_goal_forbidden: []
---
# Interpolate Bad Channels

## Function

Reconstructs data for bad channels using spherical spline interpolation from neighboring good channels. Preserves total channel count while replacing corrupted data with estimated values.

## Parameter Format

`interpolate_bads` — No parameters. Operates on channels marked as bad in metadata.

## How It Works

1. Identifies channels in `meta.bads` or `raw.info["bads"]`
2. Uses spherical spline interpolation (MNE) based on electrode geometry
3. Replaces bad channel data with interpolated values
4. Channels remain in the array with repaired data

## When to Use

- Need consistent channel count across subjects (e.g., for group analysis)
- Downstream analysis requires specific electrode positions (source localization, topographic maps)
- Only 1-3 channels are bad (interpolation quality degrades with many bad channels)
- Better than dropping when spatial patterns matter

## When NOT to Use

- No channel location information available (interpolation needs geometry)
- Too many bad channels (> 15-20% of total) — interpolation becomes unreliable
- sEEG depth electrodes (geometry assumptions don't apply)
- Don't care about maintaining channel count → use `drop_bads` (simpler)

## Constraints

- Requires electrode positions (montage) for accurate interpolation
- Quality degrades if neighboring channels are also bad
- Interpolated channels contain NO new information — they smooth from neighbors
- Should be applied before CAR (reconstructed channels participate in the average)

## Ordering

- Apply BEFORE: `car` (interpolated channels should contribute to average)
- Apply BEFORE: `ica` (prevents wasting components on bad channel noise)
- Apply AFTER: quality check identifies which channels are bad

## Reference Code

### Standalone Implementation

```python
import mne
import numpy as np

info = mne.create_info(ch_names, sfreq, ch_types='eeg')
raw = mne.io.RawArray(data, info, verbose=False)
raw.info['bads'] = bad_channels  # e.g. ['Fp1', 'T8']
raw.interpolate_bads(reset_bads=True, verbose=False)
data = raw.get_data().astype(np.float32)
```

### Key API

- `raw.info['bads'] = [...]` — mark channels as bad
- `raw.interpolate_bads(reset_bads=True, verbose=False)` — spherical spline interpolation
- `reset_bads=True` clears the bads list after interpolation
