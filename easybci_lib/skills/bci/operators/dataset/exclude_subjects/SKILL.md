---
name: exclude_subjects
description: "Recording / subject exclusion"
layer: L3
group: dataset
metadata:
  tags: [operator, dataset, exclude_subjects]
  modalities: [eeg, seeg, ecog, meg]
  step_string: "exclude_subjects"
  analysis_goal_allowed: [classification, source_localization, feature_extraction, clinical_screening, exploratory, generic, connectivity, phase_amplitude_coupling]
  analysis_goal_forbidden: []
---
# Recording / subject exclusion

## Function

Drop whole subjects, sessions or runs from the analysed set against a stated criterion.

## Parameter Format

`exclude_subjects:{subjects},{criterion},{level}`

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `subjects` | varies | — | identifiers removed |
| `criterion` | varies | — | why they were removed |
| `level` | varies | — | subject | session | run |

## When to Use

Refer to the reference pipelines that use this operator — it fills a gap that no existing registry operator covers.

## Ordering

- No strict ordering constraints.

## Relationship to Existing Operators

**No near equivalent in the registry.**

Operators act inside one recording and have no view of the cohort, so a dataset-level exclusion cannot be expressed. It is recorded because it changes the analysed sample and therefore every group-level number the paper reports.

## Reference Code

```python
def exclude_subjects(d, subjects, criterion=None, level="subject", **_):
    sid=d.get("meta",{}).get("subject_id"); return _out(d,step="exclude_subjects", excluded=bool(sid in set(subjects)), exclusion_criterion=criterion)
```
