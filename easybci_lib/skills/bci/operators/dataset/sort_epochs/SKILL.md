---
name: sort_epochs
description: "Epoch reordering"
layer: L3
group: dataset
metadata:
  tags: [operator, dataset, sort_epochs]
  modalities: [eeg, seeg, ecog, meg]
  step_string: "sort_epochs"
  analysis_goal_allowed: [classification, feature_extraction, exploratory, generic]
  analysis_goal_forbidden: []
---
# Epoch reordering

## Function

Reorder the trial axis by a metadata key so that trial index is comparable across recordings or conditions.

## Parameter Format

`sort_epochs:{by},{order}`

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `by` | varies | — | metadata key(s) used for the sort |
| `order` | varies | — | ascending | descending | explicit list |

## When to Use

Refer to the reference pipelines that use this operator — it fills a gap that no existing registry operator covers.

## Ordering

- Apply AFTER: `epoch`, `reject_epochs`

## Relationship to Existing Operators

**No near equivalent in the registry.**

Nothing addresses the trial axis as an orderable index. Analyses that compare trial n across contexts depend on this alignment, so it cannot simply be dropped from the reference.

## Reference Code

```python
def sort_epochs(d, by, order="ascending", **_):
    x=np.asarray(d["data"]); rows=d.get("meta",{}).get("trial_metadata",[]); key=lambda i: tuple(rows[i].get(k) for k in by) if isinstance(by,(list,tuple)) else rows[i].get(by); ix=sorted(range(len(rows)),key=key,reverse=order=="descending") if rows else list(range(len(x))); return _out(d,x[ix],"sort_epochs",sorted_indices=ix)
```
