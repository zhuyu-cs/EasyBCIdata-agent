---
name: smooth
description: "Temporal smoothing"
layer: L3
group: filter
metadata:
  tags: [operator, filter, smooth]
  modalities: [eeg, seeg, ecog, meg]
  step_string: "smooth"
  analysis_goal_allowed: [feature_extraction, clinical_screening, exploratory, generic]
  analysis_goal_forbidden: []
---
# Temporal smoothing

## Function

Smooth along time with a moving average, Savitzky-Golay polynomial or Gaussian kernel.

## Parameter Format

`smooth:{method},{window},{order}`

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | varies | — | moving_average | savgol | gaussian |
| `window` | varies | — | window length in samples or seconds |
| `order` | varies | — | polynomial order for savgol |

## When to Use

Refer to the reference pipelines that use this operator — it fills a gap that no existing registry operator covers.

## Ordering

- Apply AFTER: `hilbert`, `filter_bank`

## Relationship to Existing Operators

**Nearest:** `filter/bandpass_filter`

A low-pass arm of bandpass is a zero-phase FIR/IIR design, not a moving average or a Savitzky-Golay fit; the impulse responses, edge behaviour and preserved derivatives differ, and papers that smooth a band-limited envelope over 400 ms are not specifying a cut-off frequency.

## Reference Code

```python
def smooth(d, method="moving_average", window=5, order=2, **_):
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import savgol_filter
    x=np.asarray(d["data"]); w=max(1,int(round(float(window)*_sfreq(d)))) if isinstance(window,float) else max(1,int(window));
    if method == "gaussian": y=gaussian_filter1d(x,max(1,w/2),axis=-1)
    elif method == "savgol": y=savgol_filter(x,w if w%2 else w+1,min(order,w-1),axis=-1)
    else: y=np.apply_along_axis(lambda z: np.convolve(z,np.ones(w)/w,mode="same"),-1,x)
    return _out(d,y,"smooth",smooth_method=method)
```
