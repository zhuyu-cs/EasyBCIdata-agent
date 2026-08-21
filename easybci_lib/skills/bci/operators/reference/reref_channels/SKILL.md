---
name: reref_channels
description: "Reference to named channels"
layer: L3
group: reference
metadata:
  tags: [operator, reference, reref_channels]
  modalities: [eeg, seeg, ecog]
  step_string: "reref_channels"
  analysis_goal_allowed: [classification, source_localization, feature_extraction, clinical_screening, exploratory, generic, connectivity, phase_amplitude_coupling, online_inference]
  analysis_goal_forbidden: []
---
# Reference to named channels

## Function

Re-reference to one named electrode or to the average of an arbitrary named set (vertex, nose, a white-matter depth contact, external EXG pair), optionally restoring the old reference as a data channel.

## Parameter Format

`reref_channels:{channels},{mode},{restore_channel}`

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `channels` | varies | — | reference channel name(s) |
| `mode` | varies | — | single | average_of |
| `restore_channel` | varies | — | name of the previous reference to add back as data |
| `drop_reference` | varies | — | whether the reference channels stay in the output |
| `applies_to` | varies | — | which channels the rereferencing is applied to, for recordings that carry several channel types receiving different schemes |
| `block_size` | varies | — | number of channels in the block over which the reference is computed, when the reference is local to a block rather than global |
| `exclude` | varies | — | channels excluded from the computation of the reference but still rereferenced by it |
| `variance_window` | varies | — | time window over which per-channel variance is measured, when the channels forming the reference are chosen by variance |

## When to Use

Refer to the reference pipelines that use this operator — it fills a gap that no existing registry operator covers.

## Ordering

- Apply AFTER: `drop_bads`, `interpolate_bads`
- Apply BEFORE: `ica`, `epoch`

## Relationship to Existing Operators

**Nearest:** `reference/mastoid_reference`

The reference group covers only fixed schemes: car (all channels), laplacian_ref, rest_reference, bipolar_ref (adjacent contacts) and mastoid_reference, whose a1_ch/a2_ch substrings are documented as the A1/A2 mastoid pair for AASM staging. Referencing to Cz, to a nose electrode or to a white-matter depth contact has no home, and no operator can add the recording reference back as a channel.

## Reference Code

```python
def reref_channels(d, channels, mode="average_of", restore_channel=None, drop_reference=False, applies_to=None, **_):
    x = np.asarray(d["data"]); cn = _names(d, x.shape[0]); idx = [cn.index(c) for c in channels]; ref = x[idx[0]] if mode == "single" else x[idx].mean(axis=0); y = x - ref
    if restore_channel: y = np.concatenate([y, ref[None]], axis=0); cn.append(restore_channel)
    if drop_reference:
        keep = [i for i,n in enumerate(cn) if n not in channels]; y, cn = y[keep], [cn[i] for i in keep]
    return _out(d, y, "reref_channels", ch_names=cn, reference_channels=list(channels))
```
