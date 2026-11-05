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
- `n_components`: min(effective data rank, 20) — rank estimated at float32 precision; an earlier `car` lowers it by one
- `method`: FastICA
- `random_state`: 42 (EasyBCI global EASYBCI_SEED — keep fixed for reproducibility)
- `max_iter`: 500

## EOG reference

The `eog` target is matched against, in order (shared resolver
`resolve_eog_reference`, the same one `regression_eog` and `deep_inspect` use):

1. Explicit names — `ica:eog,E25,E8` (or `ica:E25,E8`): any token other than `eog` / `ecg`
   is a reference channel; `measure="correlation"`, threshold 0.6.
2. EOG channels — channel type eog, or an EOG name (`*EOG*`, `VEO`/`HEO`, `LOC`/`ROC`,
   AASM `E1`/`E2`) — MNE default z-score detection. If no component follows the EOG
   channel (detached or mis-wired electrode), the frontal proxy below is used instead
   and `eog_channel_unmatched` names the EOG channel(s).
3. Frontal proxy — Fp1 / Fp2 / Fpz, else AF7 / AF8 / AF3 / AF4 / AFz; vendor-decorated
   names (`EEG Fp1-REF`, `Fp1.`, `FP1-F7`) included — `measure="correlation"`,
   threshold 0.6. Frontal EEG that correlates with blinks is removed too; a pipeline
   warning says so.
4. Neither — nothing is removed; `pipeline_status.ica` and the QC "Pipeline notes" say
   `ocular artifacts NOT removed` (and whether the channel names were recognised as
   10-20 / 10-10 at all).

The decision (`applied`, `n_components`, `excluded`, `eog_reference`,
`eog_reference_source`) is in `pipeline_status.ica`. Each removed component
lowers the data rank by one; a later `car` does not lower it again (n − 1
channels after `ica` → `car`, n − 1 − k after `car` → `ica`; see
`data_rank`). Whether ICA or regression fits
a recording is decided by the `ocular_artifact` decision skill.

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
- Apply BEFORE: car, resample, scale, segmentation (after car the blink is spread to every channel)
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
