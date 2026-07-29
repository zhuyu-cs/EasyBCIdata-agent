---
name: bipolar_ref
description: "Bipolar referencing for sEEG/depth electrodes — subtract adjacent contacts"
layer: L3
group: reference
metadata:
  tags: [operator, reference, bipolar, seeg, depth]
  modalities: [seeg, ecog]
  step_string: "bipolar_ref"
  analysis_goal_allowed: [classification, feature_extraction, clinical_screening, exploratory, generic, online_inference]
  analysis_goal_forbidden: [source_localization]
---
# Bipolar Reference

## Function

Creates bipolar montage by subtracting adjacent electrode contacts on the same probe/shaft. Standard for sEEG and depth electrode recordings — improves spatial specificity and reduces volume conduction.

## Parameter Format

`bipolar_ref` or `bipolar_ref:auto` or `bipolar_ref:{pairs}`

Examples:
- `bipolar_ref` — Auto-detect adjacent pairs from channel naming convention
- `bipolar_ref:auto` — Same as above
- `bipolar_ref:A1-A2,A2-A3,B1-B2` — Explicit pairs

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| mode | string | auto | `auto` or explicit pairs `A1-A2,A2-A3,...` |

Auto-detection parses channel names (e.g., `LA1, LA2, LA3` → pairs `LA1-LA2, LA2-LA3`) by grouping channels with the same prefix and consecutive numbering.

## When to Use

- sEEG (stereo-EEG) depth electrode recordings
- ECoG strip/depth hybrid recordings
- When volume conduction obscures local field potentials
- Standard preprocessing for epilepsy monitoring, high-frequency oscillation (HFO) detection

## When NOT to Use

- Scalp EEG (use `car` instead)
- Grid ECoG where Laplacian is more appropriate
- Single-contact electrodes

## Constraints

- Changes the channel set: N channels → N-1 bipolar pairs (per probe)
- Channel names become "A1-A2", "A2-A3" format
- Requires consistent channel naming convention for auto mode
- If no valid pairs are found, returns data unchanged with warning

## Ordering

- Apply EARLY — before filters, ICA, etc.
- Bipolar referencing is typically the **first** step for sEEG data
- Apply BEFORE: notch, bandpass, resample
- Reason: re-referencing should be done on raw data to avoid filter edge effects propagating

## Reference Code

### Standalone with Auto-Pair Detection

```python
import mne
import numpy as np

info = mne.create_info(ch_names, sfreq, ch_types='seeg')
raw = mne.io.RawArray(data, info, verbose=False)

# Auto-derive same-probe pairs from channel naming (e.g. LA1,LA2,LA3 → LA1-LA2,LA2-LA3)
anodes, cathodes = [], []
for a, c in zip(ch_names[:-1], ch_names[1:]):
    prefix_a = ''.join(ch for ch in a if not ch.isdigit())
    prefix_c = ''.join(ch for ch in c if not ch.isdigit())
    if prefix_a == prefix_c:
        anodes.append(a)
        cathodes.append(c)

raw = mne.set_bipolar_reference(raw, anodes, cathodes, verbose='WARNING')
data = raw.get_data().astype(np.float32)
channels = list(raw.ch_names)
```

### MNE Chain (with explicit pairs)

```python
anodes = list(raw.ch_names[:-1])
cathodes = list(raw.ch_names[1:])
raw = mne.set_bipolar_reference(raw, anodes, cathodes, verbose=False)
```

### Key API

- `mne.set_bipolar_reference(raw, anode, cathode, verbose='WARNING')`
- Output channel names: `"anode-cathode"` format
