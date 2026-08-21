---
name: set_channel_types
description: "Channel-type assignment"
layer: L3
group: channel
metadata:
  tags: [operator, channels, set_channel_types]
  modalities: [eeg, seeg, ecog, meg]
  step_string: "set_channel_types"
  analysis_goal_allowed: [classification, source_localization, feature_extraction, clinical_screening, exploratory, generic, connectivity, phase_amplitude_coupling, online_inference]
  analysis_goal_forbidden: []
---
# Channel-type assignment

## Function

Re-type named channels (EOG, ECG, EMG, misc, stim) so that downstream artefact operators can find their reference channels.

## Parameter Format

`set_channel_types:{mapping},{match}`

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mapping` | varies | — | {channel_name: type} |
| `match` | varies | — | exact | substring |

## When to Use

Refer to the reference pipelines that use this operator — it fills a gap that no existing registry operator covers.

## Ordering

- Apply BEFORE: `ica`, `detect_bads`, `ssp_eog`, `ssp_ecg`

## Relationship to Existing Operators

**Nearest:** `channel/pick_channels`

pick_channels selects by existing type or name and drops the rest; it cannot change a channel's type. ica, ssp_eog, ssp_ecg, regression_eog and peakdetect_qrs all locate their reference channel by type or name substring, so a recording whose EOG channels are typed as EEG silently defeats every one of them.

## Reference Code

```python
def set_channel_types(d, mapping, match="exact", **_):
    names = _names(d, np.asarray(d["data"]).shape[0]); types = list(d.get("meta", {}).get("ch_types", ["eeg"] * len(names)))
    for i, n in enumerate(names):
        for k, v in mapping.items():
            if (n == k if match == "exact" else k.lower() in n.lower()): types[i] = v
    return _out(d, step="set_channel_types", ch_types=types)
```
