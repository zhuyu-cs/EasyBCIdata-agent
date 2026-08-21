---
name: manual_ic_selection
description: "Manual IC selection"
layer: L3
group: adaptive_cleaning
metadata:
  tags: [operator, artifact, manual_ic_selection]
  modalities: [eeg, meg]
  step_string: "manual_ic_selection"
  analysis_goal_allowed: [classification, source_localization, feature_extraction, clinical_screening, exploratory, generic, connectivity]
  analysis_goal_forbidden: [online_inference]
---
# Manual IC selection

## Function

Human-in-the-loop selection of ICA components to exclude, optionally confirming or overriding an automatic labelling pass, with multi-rater agreement.

## Parameter Format

`manual_ic_selection:{artifact_classes},{n_raters},{criteria}`

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `artifact_classes` | varies | — | classes the raters were asked to identify |
| `n_raters` | varies | — | number of independent raters |
| `criteria` | varies | — | what the raters looked at (topography, time course, spectrum) |
| `indices` | varies | — | the resulting per-run component indices |
| `reviews` | varies | — | the automatic labelling being confirmed, if any |
| `n_components_removed_mean` | varies | — | the average number of components removed per recording, for papers that report this summary instead of the per-recording component lists |

## When to Use

Refer to the reference pipelines that use this operator — it fills a gap that no existing registry operator covers.

## Ordering

- Apply AFTER: `ica`

## Relationship to Existing Operators

**Nearest:** `adaptive_cleaning/ica`

ica decides internally which components to drop and accepts no externally supplied index list, so a pipeline whose exclusions come from two human raters cannot be expressed. This is a human procedure; it is listed so the step is representable, not so it can be automated.

## Reference Code

```python
def manual_ic_selection(d, indices, action="subtract", **_):
    return ic_classify(d, method="correlation", indices=indices, threshold=np.inf, action=action, **_)
```
