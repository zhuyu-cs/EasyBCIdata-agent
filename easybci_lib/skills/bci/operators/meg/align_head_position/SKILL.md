---
name: align_head_position
description: "Cross-run head-position alignment"
layer: L3
group: meg
metadata:
  tags: [operator, meg, align_head_position]
  modalities: [meg]
  step_string: "align_head_position"
  analysis_goal_allowed: [classification, source_localization, feature_extraction, clinical_screening, exploratory, generic, connectivity, phase_amplitude_coupling]
  analysis_goal_forbidden: [online_inference]
---
# Cross-run head-position alignment

## Function

Map each run's sensor data to a common head position so runs can be concatenated, either by Maxwell filtering to a destination or by channel-space field mapping.

## Parameter Format

`align_head_position:{method},{reference_run},{mode}`

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | varies | — | maxwell_destination | channel_space_mapping |
| `reference_run` | varies | — | run whose head position is the target |
| `mode` | varies | — | fast | accurate for the field-mapping variant |
| `origin` | varies | — | sphere origin used by the mapping |

## When to Use

Refer to the reference pipelines that use this operator — it fills a gap that no existing registry operator covers.

## Ordering

- Apply AFTER: `maxwell_filter`, `estimate_head_position`
- Apply BEFORE: `concatenate`

## Relationship to Existing Operators

**No near equivalent in the registry.**

Concatenating MEG runs recorded at different head positions without alignment mixes incompatible forward geometries. No registry operator is aware of head position, and interpolate_bads is restricted to eeg/ecog and works within one recording.

## Reference Code

```python
def align_head_position(d, reference_run=None, **kwargs): return _mne_raw_op(d,"maxwell_filter",destination=reference_run,**kwargs)
```
