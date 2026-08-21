---
name: concatenate
description: "Run / split concatenation"
layer: L3
group: dataset
metadata:
  tags: [operator, dataset, concatenate]
  modalities: [eeg, seeg, ecog, meg]
  step_string: "concatenate"
  analysis_goal_allowed: [classification, source_localization, feature_extraction, clinical_screening, exploratory, generic, connectivity, phase_amplitude_coupling, online_inference]
  analysis_goal_forbidden: []
---
# Run / split concatenation

## Function

Join split acquisition files, runs or epoch sets into one object along the time or trial axis.

## Parameter Format

`concatenate:{level},{scope},{require_same_channels}`

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `level` | varies | — | raw | epochs |
| `scope` | varies | — | what is being joined (split files, runs, sessions) |
| `require_same_channels` | varies | — | fail if the channel sets differ |

## When to Use

Refer to the reference pipelines that use this operator — it fills a gap that no existing registry operator covers.

## Ordering

- Apply AFTER: `maxwell_filter`, `align_head_position`
- Apply BEFORE: `epoch`, `ica`

## Relationship to Existing Operators

**No near equivalent in the registry.**

Every registry operator maps one recording to one recording. Multi-run designs — and MEGIN split-FIF acquisitions, which are one recording stored as several files — cannot be assembled at all.

## Reference Code

```python
def concatenate(d, inputs=None, level="raw", require_same_channels=True, **_):
    arr=[np.asarray(z["data"] if isinstance(z,Mapping) else z) for z in (inputs or [])]; base=np.asarray(d["data"]); allx=[base]+arr
    if require_same_channels and any(a.shape[:-1]!=base.shape[:-1] for a in arr): raise ValueError("channel shapes differ")
    return _out(d,np.concatenate(allx,axis=-1 if level=="raw" else 0),"concatenate")
```
