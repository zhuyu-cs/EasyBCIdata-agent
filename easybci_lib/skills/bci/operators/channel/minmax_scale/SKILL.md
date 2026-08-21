---
name: minmax_scale
description: "Min-max feature normalisation"
layer: L3
group: channel
metadata:
  tags: [operator, channels, minmax_scale]
  modalities: [eeg, seeg, ecog, meg]
  step_string: "minmax_scale"
  analysis_goal_allowed: [classification, feature_extraction, online_inference]
  analysis_goal_forbidden: []
---
# Min-max feature normalisation

## Function

Rescale values to a fixed range (typically 0-1), with a selectable axis including across-channels-within-window.

## Parameter Format

`minmax_scale:{feature_range},{axis},{clip}`

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `feature_range` | varies | — | [min, max] of the output |
| `axis` | varies | — | across_channels_within_window | across_time | global |
| `clip` | varies | — | clamp values outside the fitted range |
| `method` | varies | — | which scaling formula is applied - min-max onto the feature range, or a variant such as dividing by the maximum only |

## When to Use

Refer to the reference pipelines that use this operator — it fills a gap that no existing registry operator covers.

## Ordering

- Apply AFTER: `bandpass`, `epoch`, `baseline_correct`

## Relationship to Existing Operators

**Nearest:** `channel/scale`

scale offers only robust, standard and a numeric factor — no min-max — and its documentation states that scaling is applied per channel, fit across the time dimension, which is the opposite axis from normalising across electrodes inside one time window.

## Reference Code

```python
def minmax_scale(d, feature_range=(0, 1), axis="global", clip=False, method="minmax", **_):
    x = np.asarray(d["data"], dtype=float); lo, hi = feature_range
    ax = -1 if axis == "across_time" else (1 if axis == "across_channels_within_window" and x.ndim == 3 else None)
    mn = x.min(axis=ax, keepdims=True) if ax is not None else x.min(); mx = x.max(axis=ax, keepdims=True) if ax is not None else x.max()
    y = lo + (x - mn) * (hi - lo) / (mx - mn + 1e-12); y = np.clip(y, lo, hi) if clip else y
    return _out(d, y.astype(np.asarray(d["data"]).dtype, copy=False), "minmax_scale", scale_axis=axis)
```
