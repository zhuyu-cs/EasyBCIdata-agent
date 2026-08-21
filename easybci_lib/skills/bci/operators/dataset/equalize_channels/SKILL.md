---
name: equalize_channels
description: "Channel-set equalisation"
layer: L3
group: dataset
metadata:
  tags: [operator, dataset, equalize_channels]
  modalities: [eeg, seeg, ecog, meg]
  step_string: "equalize_channels"
  analysis_goal_allowed: [classification, source_localization, feature_extraction, clinical_screening, exploratory, generic, connectivity]
  analysis_goal_forbidden: []
---
# Channel-set equalisation

## Function

Reduce several recordings to their common channel set (or a supplied set) so they can be combined.

## Parameter Format

`equalize_channels:{mode},{scope}`

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | varies | — | intersection | supplied_list |
| `scope` | varies | — | runs | sessions | subjects |

## When to Use

Refer to the reference pipelines that use this operator — it fills a gap that no existing registry operator covers.

## Ordering

- Apply AFTER: `drop_bads`, `interpolate_bads`
- Apply BEFORE: `concatenate`

## Relationship to Existing Operators

**Nearest:** `channel/pick_channels`

pick_channels takes a fixed name list or a channel type; it cannot compute the intersection across several recordings, which is the whole point when different runs have different bad-channel sets.

## Reference Code

```python
def equalize_channels(d, inputs=None, mode="intersection", supplied_list=None, **_):
    alln=[_names(d,np.asarray(d["data"]).shape[0])]+[_names(z,np.asarray(z["data"]).shape[0]) for z in (inputs or [])]; names=list(supplied_list) if mode=="supplied_list" else list(set.intersection(*(set(x) for x in alln))); cn=_names(d,np.asarray(d["data"]).shape[0]); idx=[cn.index(n) for n in names]; return _out(d,np.asarray(d["data"])[idx],"equalize_channels",ch_names=names)
```
