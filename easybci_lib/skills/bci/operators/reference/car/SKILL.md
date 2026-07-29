---
name: car
description: "Common Average Reference — re-reference EEG to the mean of all channels"
layer: L3
group: reference
metadata:
  tags: [operator, reference, car, average]
  modalities: [eeg, ecog]
  step_string: "car"
  analysis_goal_allowed: [classification, source_localization, feature_extraction, clinical_screening, exploratory, generic, connectivity, phase_amplitude_coupling, online_inference]
  analysis_goal_forbidden: []
---
# Common Average Reference (CAR)

## Function

Subtracts the mean signal across all channels at each time point. This removes shared noise (common mode) while preserving spatially localized neural activity. Uses MNE's `set_eeg_reference("average")`.

## Parameter Format

`car` — No parameters needed.

## When to Use

- Scalp EEG data with original reference that introduces bias (e.g., single mastoid)
- ECoG grids where a shared reference can amplify correlated noise
- Before spatial filtering (CSP) — CAR provides a neutral reference baseline

## When NOT to Use

- sEEG depth electrodes (use `bipolar_ref` instead — adjacent contacts on same probe)
- Very few channels (< 8) — average reference becomes unstable
- When bad channels are still present (they contaminate the average) — apply `drop_bads` or `interpolate_bads` first

## Constraints

- Applied to ALL channels simultaneously
- Changes the reference scheme — subsequent interpretation of topographic patterns must account for average reference
- Does NOT change channel count (unlike `drop_bads` or `bipolar_ref`)

## Ordering

- Apply AFTER: `drop_bads` or `interpolate_bads` (bad channels corrupt the average)
- Apply AFTER: `pick_channels` (only average over relevant channels)
- Apply BEFORE: `bandpass`, `notch` are acceptable either before or after, but typically CAR is applied after initial filtering

## Recommended Use

| Modality | Use CAR? | Alternative |
|----------|----------|-------------|
| Scalp EEG (>= 16 ch) | Yes | — |
| Scalp EEG (< 8 ch) | No | Keep original reference |
| ECoG grid | Yes | Laplacian for higher spatial resolution |
| sEEG | No | Use `bipolar_ref` |
| MEG | No | MEG is reference-free |

## Reference Code

### MNE Chain (on existing Raw object)

```python
raw.set_eeg_reference('average', projection=False, verbose=False)
```

### Standalone Implementation

```python
import mne
import numpy as np

info = mne.create_info(ch_names, sfreq, ch_types='eeg')
raw = mne.io.RawArray(data, info, verbose=False)
raw.set_eeg_reference('average', projection=False, verbose=False)
data = raw.get_data().astype(np.float32)
```

### Key API

- `raw.set_eeg_reference('average', projection=False, verbose=False)`
- `projection=False` applies the reference directly (no SSP projector)
