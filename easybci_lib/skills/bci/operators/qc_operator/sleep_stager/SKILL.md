---
name: sleep_stager
description: "Sleep-stage scoring"
layer: L3
group: qc_operator
metadata:
  tags: [operator, qc, sleep_stager]
  modalities: [eeg]
  step_string: "sleep_stager"
  analysis_goal_allowed: [clinical_screening, feature_extraction, exploratory, generic]
  analysis_goal_forbidden: []
---
# Sleep-stage scoring

## Function

Assign AASM sleep stages to consecutive epochs and expose the intervals belonging to requested stages.

## Parameter Format

`sleep_stager:{target_stages},{method},{scoring_standard}`

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_stages` | varies | — | stages to return, e.g. ['N3'] |
| `method` | varies | — | manual_scoring | classifier |
| `scoring_standard` | varies | — | AASM | R&K |
| `epoch_length_s` | varies | — | scoring epoch, conventionally 30 s |

## When to Use

Refer to the reference pipelines that use this operator — it fills a gap that no existing registry operator covers.

## Ordering

- Apply AFTER: `bandpass`, `notch`
- Apply BEFORE: `segment`, `epoch`

## Relationship to Existing Operators

**Nearest:** `reference/mastoid_reference`

Nothing in the registry outputs stage labels. log_band_power and sef compute the features a stager would use but return features; mastoid_reference sets up the AASM montage but only re-references. A pipeline that must locate N3 intervals cannot get them from any combination of the 58 operators.

## Reference Code

```python
def sleep_stager(d, target_stages=None, method="manual_scoring", scoring_standard="AASM", epoch_length_s=30, **_):
    labels=d.get("meta",{}).get("sleep_stages");
    if labels is None and method != "manual_scoring":
        try:
            import yasa
            labels=yasa.SleepStaging(d.get("meta",{}).get("raw"), eeg_name=d.get("meta",{}).get("eeg_name")).predict()
        except Exception as e: raise RuntimeError("sleep_stager classifier needs YASA or precomputed labels") from e
    keep=[i for i,s in enumerate(labels or []) if not target_stages or s in target_stages]
    return _out(d,step="sleep_stager", sleep_stages=labels, selected_epochs=keep, provenance="upstream_wrapper", upstream="YASA SleepStaging / AASM manual scoring")
```
