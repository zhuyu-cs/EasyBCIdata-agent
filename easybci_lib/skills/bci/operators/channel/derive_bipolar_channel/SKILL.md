---
name: derive_bipolar_channel
description: "Derived bipolar monitor channel"
layer: L3
group: channel
metadata:
  tags: [operator, channels, derive_bipolar_channel]
  modalities: [eeg, seeg, ecog, meg]
  step_string: "derive_bipolar_channel"
  analysis_goal_allowed: [classification, source_localization, feature_extraction, clinical_screening, exploratory, generic, connectivity, phase_amplitude_coupling, online_inference]
  analysis_goal_forbidden: []
---
# Derived bipolar monitor channel

## Function

Create additional bipolar channels from named electrode pairs (HEOG, VEOG, EMG) while keeping the original channels in place.

## Parameter Format

`derive_bipolar_channel:{pairs},{names},{ch_type}`

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pairs` | varies | — | [[anode, cathode], ...] |
| `names` | varies | — | names of the new channels |
| `ch_type` | varies | — | type assigned to the derived channels |
| `keep_originals` | varies | — | true keeps the source channels |

## When to Use

Refer to the reference pipelines that use this operator — it fills a gap that no existing registry operator covers.

## Ordering

- Apply BEFORE: `ica`, `bandpass`

## Relationship to Existing Operators

**Nearest:** `reference/bipolar_ref`

bipolar_ref re-references the whole data set and returns N-1 replacement channels; it is destructive and montage-wide. Deriving two extra EOG monitor channels from a scalp net is additive and leaves the montage untouched, which bipolar_ref cannot express.

## Reference Code

```python
def derive_bipolar_channel(d, pairs, names=None, ch_type="misc", keep_originals=True, **_):
    x = np.asarray(d["data"]); cn = _names(d, x.shape[0]); names = names or [f"{a}-{b}" for a,b in pairs]
    add = np.stack([x[cn.index(a)] - x[cn.index(b)] for a,b in pairs]); y = np.concatenate([x, add], axis=0) if keep_originals else add
    return _out(d, y, "derive_bipolar_channel", ch_names=(cn + list(names)) if keep_originals else list(names), ch_types=(d.get("meta", {}).get("ch_types", ["eeg"]*len(cn)) + [ch_type]*len(names)))
```
