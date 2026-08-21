---
name: aggregate_bands
description: "Cross-band aggregation"
layer: L3
group: spectral
metadata:
  tags: [operator, spectral, aggregate_bands]
  modalities: [eeg, seeg, ecog, meg]
  step_string: "aggregate_bands"
  analysis_goal_allowed: [classification, feature_extraction, clinical_screening, exploratory, generic, connectivity]
  analysis_goal_forbidden: []
---
# Cross-band aggregation

## Function

Combine per-band envelopes or power values into one series with a chosen statistic — notably the geometric mean used to build unbiased broadband high-frequency activity.

## Parameter Format

`aggregate_bands:{method},{bands},{normalise_each_band}`

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | varies | — | geometric_mean | arithmetic_mean | median |
| `bands` | varies | — | bands entering the aggregate |
| `normalise_each_band` | varies | — | divide each band by its own mean before combining |

## When to Use

Refer to the reference pipelines that use this operator — it fills a gap that no existing registry operator covers.

## Ordering

- Apply AFTER: `filter_bank`, `hilbert`

## Relationship to Existing Operators

**Nearest:** `spectral/log_band_power`

log_band_power returns one value per band and never combines them; nothing else reduces a set of band envelopes to a single series. An arithmetic mean over 50-200 Hz is dominated by the lowest band, which is exactly why the reference uses a geometric mean.

## Reference Code

```python
def aggregate_bands(d, method="geometric_mean", bands=None, normalise_each_band=False, **_):
    x=np.asarray(d["data"],float); bands=bands or list(range(x.shape[0])); z=x[bands];
    if normalise_each_band: z=z/(z.mean(axis=-1,keepdims=True)+1e-12)
    y=np.exp(np.mean(np.log(np.maximum(z,1e-12)),axis=0)) if method=="geometric_mean" else (np.median(z,axis=0) if method=="median" else np.mean(z,axis=0))
    return _out(d,y,"aggregate_bands",aggregated_bands=bands)
```
