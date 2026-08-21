---
name: baseline_correct
description: "Baseline correction"
layer: L3
group: epoch
metadata:
  tags: [operator, epoching, baseline_correct]
  modalities: [eeg, seeg, ecog, meg]
  step_string: "baseline_correct"
  analysis_goal_allowed: [classification, source_localization, feature_extraction, clinical_screening, exploratory, generic, connectivity, phase_amplitude_coupling]
  analysis_goal_forbidden: []
---
# Baseline correction

## Function

Subtract (or z-score against) a per-epoch statistic computed over a baseline window, per channel.

## Parameter Format

`baseline_correct:{mode},{tmin},{tmax}`

Examples:
- `baseline_correct:mean,-0.2,0` — Subtract mean of -200ms to 0ms
- `baseline_correct:zscore,null,0` — Z-score against epoch start to 0ms

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `window` | varies | — | [tmin, tmax] of the baseline in seconds; null start = epoch start |
| `mode` | varies | — | mean | median | zscore | ratio | percent |
| `level` | varies | — | epoch | condition_average |

## When to Use

Refer to the reference pipelines that use this operator — it fills a gap that no existing registry operator covers.

## Ordering

- Apply AFTER: `epoch`
- Apply BEFORE: `reject_epochs`

## Relationship to Existing Operators

**Nearest:** `channel/scale`

scale normalises whole-channel amplitude (robust / standard / numeric factor) fit across the entire time dimension; it has no baseline window, no per-epoch application and no median or ratio mode. Removing a pre-stimulus offset is a different operation from rescaling a channel.

## Reference Code

```python
def baseline_correct(d, window=None, mode="mean", level="epoch", **_):
    x = np.asarray(d["data"]); sf = _sfreq(d); n = x.shape[-1]
    lo = 0 if not window or window[0] is None else max(0, int(round(window[0] * sf)))
    hi = n if not window or window[1] is None else min(n, int(round(window[1] * sf)))
    base = x[..., lo:hi]
    stat = np.mean(base, axis=-1, keepdims=True) if mode in ("mean", "ratio", "percent") else np.median(base, axis=-1, keepdims=True)
    if mode == "zscore": y = (x - stat) / (np.std(base, axis=-1, keepdims=True) + 1e-12)
    elif mode == "ratio": y = x / (stat + 1e-12)
    elif mode == "percent": y = 100 * (x - stat) / (stat + 1e-12)
    else: y = x - stat
    return _out(d, y, "baseline_correct", baseline_window=window, baseline_mode=mode)
```
