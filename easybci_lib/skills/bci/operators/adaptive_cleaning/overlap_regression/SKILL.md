---
name: overlap_regression
description: "Overlapping-response regression (rERP / deconvolution)"
layer: L3
group: adaptive_cleaning
metadata:
  tags: [operator, artifact, overlap_regression]
  modalities: [eeg, meg]
  step_string: "overlap_regression"
  analysis_goal_allowed: [source_localization, feature_extraction, exploratory, generic]
  analysis_goal_forbidden: []
---
# Overlapping-response regression (rERP / deconvolution)

## Function

Fit a linear model with one regressor per event type over the continuous data and subtract the modelled contribution of the unwanted events, deconvolving overlapping responses.

## Parameter Format

`overlap_regression:{regressors},{subtract},{window}`

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `regressors` | varies | — | event types entering the design matrix |
| `subtract` | varies | — | which modelled responses are removed |
| `window` | varies | — | response window modelled per regressor |
| `exclude` | varies | — | spans withheld from the fit (e.g. chunks over the rejection threshold) |

## When to Use

Refer to the reference pipelines that use this operator — it fills a gap that no existing registry operator covers.

## Ordering

- Apply AFTER: `bandpass`, `epoch`

## Relationship to Existing Operators

**Nearest:** `adaptive_cleaning/regression_eog`

regression_eog regresses one continuous EOG channel out of the data. Overlap correction regresses event-locked design-matrix columns and needs an event table, a per-regressor response window and a choice of which fitted response to subtract — none of which exists in the registry.

## Reference Code

```python
def overlap_regression(d, regressors, subtract=None, window=(-0.2,0.8), exclude=None, **_):
    x=np.asarray(d["data"],float); sf=_sfreq(d); n=x.shape[-1]; ev=_events(d); types=list(regressors or sorted({e.get("condition",e.get("code")) for e in ev})); a,b=int(window[0]*sf),int(window[1]*sf); X=np.zeros((n,len(types)*max(1,b-a)))
    for e in ev:
        k=e.get("condition",e.get("code"));
        if k not in types: continue
        c=types.index(k); s=_event_sample(e); lo=max(0,s+a); hi=min(n,s+b); X[lo:hi,c*max(1,b-a):c*max(1,b-a)+(hi-lo)] += 1
    if exclude:
        for aa,bb in exclude: X[int(aa*sf):int(bb*sf)]=0
    beta=np.linalg.lstsq(X,x.T,rcond=None)[0] if X.any() else np.zeros((X.shape[1],x.shape[0])); y=x-(X@beta).T
    return _out(d,y,"overlap_regression", regressors=types)
```
