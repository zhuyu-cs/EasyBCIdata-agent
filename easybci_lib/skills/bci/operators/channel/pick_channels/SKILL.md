---
name: pick_channels
description: "Select channels by name list or type (EEG, MEG, etc.)"
layer: L3
group: channel
metadata:
  tags: [operator, channels, select, subset, pick]
  modalities: [eeg, seeg, ecog, meg]
  step_string: "pick_channels"
  analysis_goal_allowed: [classification, source_localization, feature_extraction, clinical_screening, exploratory, generic, connectivity, phase_amplitude_coupling, online_inference]
  analysis_goal_forbidden: []
---
# Pick Channels

## Function

Subsets the data to include only specified channels. Can select by explicit channel names or by channel type (EEG, MEG, sEEG, etc.).

## Parameter Format

`pick_channels:{selection}`

Examples:
- `pick_channels:eeg` — Keep only EEG-type channels (remove EOG, EMG, stim, etc.)
- `pick_channels:Fp1,Fp2,F3,F4,C3,C4,P3,P4` — Explicit channel names
- `pick_channels:meg` — Keep only MEG channels
- `pick_channels:seeg` — Keep only sEEG channels

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| selection | string | Channel type (`eeg`, `meg`, `seeg`, `ecog`, `emg`) OR comma-separated channel names |

## Type Selection

When a single recognized type is given, channels are filtered by their `ch_types` metadata:
- `eeg` — scalp EEG electrodes
- `meg` — MEG sensors
- `seeg` — sEEG depth contacts
- `ecog` — ECoG grid/strip contacts
- `emg` — EMG channels (usually removed)
- `data` — all data channels (exclude stim/misc)

## When to Use

- Remove non-neural channels (stim, EOG, EMG, misc) before processing
- Select a spatial subset (e.g., only motor cortex channels C3, Cz, C4)
- Focus on specific channel group for paradigm-specific analysis
- Remove reference channels that shouldn't be processed

## When NOT to Use

- When you need ALL channels for spatial methods (ICA, CAR, source localization)
- Keep EOG/ECG channels if ICA artifact detection needs them (pick channels AFTER ICA)

## Ordering

- Apply BEFORE: ICA if you want to exclude certain channel types from decomposition
- Apply AFTER: ICA if you needed EOG/ECG channels for artifact detection
- Generally early in pipeline — reduces computation for subsequent steps

## Common Patterns

| Goal | Selection | Notes |
|------|-----------|-------|
| Remove stim channels | `pick_channels:eeg` | Keeps only EEG-type |
| Motor cortex subset | `pick_channels:C3,Cz,C4,FC3,FC4,CP3,CP4` | MI BCI |
| Remove EOG after ICA | `pick_channels:eeg` | Apply after ICA step |

## Reference Code

### MNE Chain (on existing Raw object)

```python
raw.pick_channels(['C3', 'Cz', 'C4'])
```

### Standalone (numpy, by name list)

```python
import numpy as np

pick_names = ['C3', 'Cz', 'C4']
pick_idx = [i for i, ch in enumerate(channels) if ch in pick_names]
data = data[pick_idx]
channels = [channels[i] for i in pick_idx]
```

### By Channel Type

```python
# Select only EEG channels from mixed recording
ch_types = data_dict['meta']['ch_types']  # e.g. ['eeg','eeg','eog','stim',...]
indices = [i for i, t in enumerate(ch_types) if t == 'eeg']
data = data[indices]
channels = [channels[i] for i in indices]
```

### Key API

- `raw.pick_channels(names)` — MNE method
- `raw.pick_types(eeg=True)` — alternative for type-based selection
