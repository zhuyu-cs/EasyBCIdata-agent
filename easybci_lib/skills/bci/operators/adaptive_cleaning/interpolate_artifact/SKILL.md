---
name: interpolate_artifact
description: "Artefact-span interpolation"
layer: L3
group: adaptive_cleaning
metadata:
  tags: [operator, artifact, interpolate_artifact]
  modalities: [eeg, seeg, ecog, meg]
  step_string: "interpolate_artifact"
  analysis_goal_allowed: [classification, source_localization, feature_extraction, clinical_screening, exploratory, generic, connectivity, phase_amplitude_coupling, online_inference]
  analysis_goal_forbidden: []
---
# Artefact-span interpolation

## Function

Replace samples inside flagged spans with an interpolant estimated from the surrounding signal (linear, PCHIP, spline), leaving the rest of the trace untouched.

## Parameter Format

`interpolate_artifact:{spans},{window_ms},{method}`

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `spans` | varies | — | annotation label or mask identifying what to replace |
| `window_ms` | varies | — | [start, end] around each event, e.g. [-5, 10] around a stimulation pulse |
| `method` | varies | — | linear | pchip | spline |
| `dilate_s` | varies | — | grow the mask before interpolating |
| `channels` | varies | — | restrict the interpolation to these channels (e.g. the eye-tracking channels only) |

## When to Use

Refer to the reference pipelines that use this operator — it fills a gap that no existing registry operator covers.

## Ordering

- Apply AFTER: `detect_artifact_spans`
- Apply BEFORE: `ica`, `epoch`

## Relationship to Existing Operators

**Nearest:** `channel/fill_nan`

fill_nan substitutes a constant for non-finite samples and clip truncates amplitude; neither estimates a replacement from neighbouring samples. interpolate_bads interpolates across channels (spatially), not along time, so a 15 ms stimulation transient present on every channel is outside all three.

## Reference Code

```python
def interpolate_artifact(d, spans, method="linear", dilate_s=0.0, channels=None, **_):
    from scipy.interpolate import PchipInterpolator, CubicSpline
    y = np.asarray(d["data"]).copy(); sf = _sfreq(d); cn = _names(d, y.shape[0]); cis = [cn.index(c) for c in channels] if channels else range(y.shape[0])
    for span in spans:
        a,b = span; a=max(0,int(round((a-dilate_s)*sf))); b=min(y.shape[-1],int(round((b+dilate_s)*sf))); left=max(0,a-1); right=min(y.shape[-1]-1,b)
        if left == right: continue
        for c in cis:
            if method == "pchip": f=PchipInterpolator([left,right],[y[c,left],y[c,right]])
            elif method == "spline": f=CubicSpline([left,right],[y[c,left],y[c,right]])
            else: f=lambda q: np.interp(q,[left,right],[y[c,left],y[c,right]])
            y[c,a:b] = f(np.arange(a,b))
    return _out(d, y, "interpolate_artifact")
```
