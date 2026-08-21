---
name: split_runs
description: "Recording split into runs"
layer: L3
group: dataset
metadata:
  tags: [operator, dataset, split_runs]
  modalities: [eeg, seeg, ecog, meg]
  step_string: "split_runs"
  analysis_goal_allowed: [classification, source_localization, feature_extraction, clinical_screening, exploratory, generic, connectivity, phase_amplitude_coupling]
  analysis_goal_forbidden: []
---
# Recording split into runs

## Function

Divide a continuous recording into run-level pieces so that later steps (referencing, normalisation) are computed per run.

## Parameter Format

`split_runs:{by},{boundaries}`

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `by` | varies | — | annotation | boundary_events | fixed_list |
| `boundaries` | varies | — | explicit split points when given |

## When to Use

Refer to the reference pipelines that use this operator — it fills a gap that no existing registry operator covers.

## Ordering

- Apply BEFORE: `concatenate`, `car`

## Relationship to Existing Operators

**No near equivalent in the registry.**

The inverse of concatenate, and equally absent. It matters because it changes the scope over which a later common-average reference or normalisation is estimated, which is a substantive analytic choice rather than bookkeeping.

## Reference Code

```python
def split_runs(d, by="fixed_list", boundaries=None, **_):
    x=np.asarray(d["data"]); sf=_sfreq(d); cuts=[0]+[int(round(v*sf)) for v in (boundaries or [])]+[x.shape[-1]]; runs=[x[...,a:b] for a,b in zip(cuts[:-1],cuts[1:])]; return _out(d,runs,"split_runs",run_boundaries=cuts)
```
