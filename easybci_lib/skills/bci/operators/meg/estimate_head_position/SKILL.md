---
name: estimate_head_position
description: "Continuous head-position estimation"
layer: L3
group: meg
metadata:
  tags: [operator, meg, estimate_head_position]
  modalities: [meg]
  step_string: "estimate_head_position"
  analysis_goal_allowed: [source_localization, feature_extraction, exploratory, generic]
  analysis_goal_forbidden: []
---
# Continuous head-position estimation

## Function

Estimate the continuous head position from cHPI coil signals and return the head-position traces used for movement compensation.

## Parameter Format

`estimate_head_position:{method},{t_window},{t_step_min}`

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | varies | — | chpi_amplitudes_and_locs |
| `t_window` | varies | — | estimation window |
| `t_step_min` | varies | — | minimum step between estimates |

## When to Use

Refer to the reference pipelines that use this operator — it fills a gap that no existing registry operator covers.

## Ordering

- Apply BEFORE: `maxwell_filter`, `align_head_position`

## Relationship to Existing Operators

**No near equivalent in the registry.**

There is no head-motion capability anywhere in the registry, so the traces that movement-compensated Maxwell filtering consumes cannot be produced.

## Reference Code

```python
def estimate_head_position(d, **kwargs): return _mne_raw_op(d,"compute_head_pos",**kwargs)
```
