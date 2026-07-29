---
name: ica
description: "ICA artifact removal — automatic detection and exclusion of EOG/ECG components"
layer: L3
group: adaptive_cleaning
metadata:
  tags: [operator, artifact, ica, eog, ecg, decomposition]
  modalities: [eeg, meg]
  step_string: "ica"
  analysis_goal_allowed: [classification, source_localization, feature_extraction, clinical_screening, exploratory, generic, connectivity]
  analysis_goal_forbidden: [online_inference]
---
# ICA (Independent Component Analysis)

## Function

Decomposes signal into independent components using FastICA, then automatically identifies and removes artifact components (eye blinks, heartbeat) by correlating with EOG/ECG reference channels.

## Parameter Format

`ica` or `ica:{artifact_types}`

Examples:
- `ica` — Default: detect and remove EOG components only
- `ica:eog` — Same as default
- `ica:eog,ecg` — Remove both eye and cardiac artifacts
- `ica:ecg` — Remove cardiac artifacts only

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| artifact_types | comma-separated | eog | Which artifact types to detect: `eog`, `ecg` |

Internal parameters (not user-configurable):
- `n_components`: min(n_channels - 1, 25)
- `method`: FastICA
- `random_state`: 42 (EasyBCI global EASYBCI_SEED — keep fixed for reproducibility)
- `max_iter`: 500

## When to Use

- Frequent eye blinks visible as high-amplitude frontal deflections
- Cardiac artifact (QRS complex) visible in EEG, especially with few channels
- Data has dedicated EOG/ECG reference channels for correlation-based detection
- Need to preserve data dimensionality (unlike `drop_bads` which removes channels)

## When NOT to Use

- Fewer than 3 channels (ICA needs >= 3, practically >= 8 for good decomposition)
- Very short recordings (< 60 seconds) — insufficient data for stable decomposition
- sEEG/ECoG data (minimal eye/muscle artifacts due to intracranial placement)
- Spike data

## Constraints

- Requires `n_channels >= 3` (skipped otherwise with warning)
- Works best with `n_channels >= 16` for meaningful decomposition
- Should be applied AFTER bandpass filtering (avoids slow drifts dominating components)
- Should be applied AFTER notch (line noise creates artificial components)
- Data should NOT be resampled to very low rates before ICA (need temporal resolution)

## Ordering

- Apply AFTER: notch, bandpass
- Apply BEFORE: resample, scale, segmentation
- Reason: ICA needs full temporal resolution and clean frequency content to separate sources

## Common Issues

- **No components excluded**: If no artifact components are detected, the data passes through unchanged. This may mean: no reference channels available, artifacts are mild, or n_components is too low.
- **Too many components excluded**: If > 3 components are excluded, consider whether the data has severe artifacts that ICA cannot cleanly separate. Manual inspection may be needed.
- **Slow execution**: ICA on high-channel-count data (> 64 ch) can be slow. Consider PCA dimensionality reduction first (not yet implemented as a step).

## Reference Code

### Standalone Implementation

```python
import mne
from mne.preprocessing import ICA

info = mne.create_info(ch_names, sfreq, ch_types='eeg')
raw = mne.io.RawArray(data, info, verbose=False)

n_components = min(len(ch_names) - 1, 25)
ica = ICA(n_components=n_components, method='fastica', random_state=42, max_iter=500)
ica.fit(raw, verbose=False)

# Auto-detect EOG artifacts
eog_indices, _ = ica.find_bads_eog(raw, verbose=False)
# Auto-detect ECG artifacts
ecg_indices, _ = ica.find_bads_ecg(raw, verbose=False)

ica.exclude = list(set(eog_indices + ecg_indices))
ica.apply(raw, verbose=False)
data = raw.get_data().astype(np.float32)
```

### Key API

- `ICA(n_components=N, method='fastica', random_state=42, max_iter=500)`
- `ica.fit(raw, verbose=False)`
- `ica.find_bads_eog(raw)` → `(indices, scores)`
- `ica.find_bads_ecg(raw)` → `(indices, scores)`
- `ica.exclude = [...]`; `ica.apply(raw)`
