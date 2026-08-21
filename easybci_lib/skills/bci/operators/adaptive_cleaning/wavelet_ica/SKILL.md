---
name: wavelet_ica
description: "Wavelet-ICA residual cleaning"
layer: L3
group: adaptive_cleaning
metadata:
  tags: [operator, artifact, wavelet_ica]
  modalities: [eeg, meg]
  step_string: "wavelet_ica"
  analysis_goal_allowed: [classification, feature_extraction, clinical_screening, exploratory, generic]
  analysis_goal_forbidden: [online_inference]
---
# Wavelet-ICA residual cleaning

## Function

Wavelet-threshold the component time courses of an ICA decomposition and back-project, suppressing artefact energy inside components rather than discarding whole components.

## Parameter Format

`wavelet_ica:{wavelet},{level},{threshold_rule}`

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `wavelet` | varies | — | mother wavelet |
| `level` | varies | — | decomposition depth |
| `threshold_rule` | varies | — | universal | sure | custom multiplier |
| `ica_method` | varies | — | decomposition used underneath (e.g. infomax) |

## When to Use

Refer to the reference pipelines that use this operator — it fills a gap that no existing registry operator covers.

## Ordering

- Apply AFTER: `ica`

## Relationship to Existing Operators

**Nearest:** `adaptive_cleaning/ica`

ica is all-or-nothing per component: a component is either kept or zeroed. wICA keeps every component and removes only its artefactual wavelet coefficients, which changes what is subtracted from the signal, not just how many components are.

## Reference Code

```python
def wavelet_ica(d, wavelet="db4", level=4, threshold_rule="universal", **_):
    import pywt
    x=np.asarray(d["data"]); ica,s,comps=_ica(x); out=[]
    for c in comps:
        coeff=pywt.wavedec(c,wavelet,level=level); sigma=np.median(np.abs(coeff[-1]))/0.6745+1e-12; thr=sigma*np.sqrt(2*np.log(c.size)); coeff=[pywt.threshold(q,thr,mode="soft") for q in coeff]; out.append(pywt.waverec(coeff,wavelet)[:c.size])
    y=ica.inverse_transform(np.asarray(out).reshape(s.shape).T).T.reshape(x.shape)
    return _out(d,y,"wavelet_ica",wavelet=wavelet,threshold_rule=threshold_rule)
```
