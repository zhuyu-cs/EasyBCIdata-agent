---
name: detrend
description: "Polynomial detrending"
layer: L3
group: filter
metadata:
  tags: [operator, filter, detrend]
  modalities: [eeg, seeg, ecog, meg]
  step_string: "detrend"
  analysis_goal_allowed: [classification, source_localization, feature_extraction, clinical_screening, exploratory, generic, connectivity, phase_amplitude_coupling, online_inference]
  analysis_goal_forbidden: []
---
# Polynomial detrending

## Function

Fit and subtract a low-order polynomial (constant, linear, quadratic) per channel over the whole recording or per epoch, optionally masking the current trial out of the fit.

## Parameter Format

`detrend:{order},{level},{mask_trial}`

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `order` | varies | — | 0 = DC offset removal, 1 = linear, 2 = quadratic |
| `level` | varies | — | recording | epoch |
| `mask_trial` | varies | — | exclude the trial being corrected from its own trend estimate |
| `pad_s` | varies | — | padding used around the masked span |

## When to Use

Refer to the reference pipelines that use this operator — it fills a gap that no existing registry operator covers.

## Ordering

- Apply BEFORE: `bandpass`, `ica`, `epoch`

## Relationship to Existing Operators

**Nearest:** `filter/high_pass_dc_removal`

hp_dc is an FIR high-pass restricted to 0.01-0.5 Hz; it has a frequency response, a transition band and filter-edge behaviour. Papers that detrend specifically to avoid high-pass-filter artefacts in multivariate analyses are choosing a polynomial fit precisely because it is not a filter, so substituting one inverts the intent.

## Reference Code

```python
def detrend(d, order=1, level="recording", mask_trial=False, pad_s=0.0, **_):
    from scipy.signal import detrend as sp_detrend
    x=np.asarray(d["data"]); y=sp_detrend(x,axis=-1,type="constant" if order==0 else "linear") if order<=1 else x-np.polynomial.polynomial.polyval(np.arange(x.shape[-1]),np.polynomial.polynomial.polyfit(np.arange(x.shape[-1]),x,order))
    return _out(d,y,"detrend",detrend_order=order)
```
