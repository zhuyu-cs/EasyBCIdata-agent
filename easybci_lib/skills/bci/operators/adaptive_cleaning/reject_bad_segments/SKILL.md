---
name: reject_bad_segments
description: "Bad continuous-segment rejection"
layer: L3
group: adaptive_cleaning
metadata:
  tags: [operator, artifact, reject_bad_segments]
  modalities: [eeg, seeg, ecog, meg]
  step_string: "reject_bad_segments"
  analysis_goal_allowed: [classification, source_localization, feature_extraction, clinical_screening, exploratory, generic, connectivity, phase_amplitude_coupling]
  analysis_goal_forbidden: [online_inference]
---
# Bad continuous-segment rejection

## Function

Remove or exclude contaminated spans of continuous data — either dropped outright, or omitted from a downstream fit (as when annotated spans are withheld from ICA).

## Parameter Format

`reject_bad_segments:{method},{threshold},{window_ms}`

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | varies | — | clean_windows | osl_default | amplitude_threshold | manual_visual | annotation |
| `threshold` | varies | — | amplitude or fraction-of-bad-channels criterion |
| `window_ms` | varies | — | window the criterion is evaluated over |
| `action` | varies | — | drop | omit_from_fit |
| `min_break_s` | varies | — | long recording gaps to remove as well |

## When to Use

Refer to the reference pipelines that use this operator — it fills a gap that no existing registry operator covers.

## Ordering

- Apply AFTER: `detect_artifact_spans`
- Apply BEFORE: `ica`, `epoch`

## Relationship to Existing Operators

**Nearest:** `adaptive_cleaning/asr`

asr is the repair half of clean_rawdata: it reconstructs bursts and returns data of the same length. The removal half (clean_windows / OSL bad-segment detection / ERPLAB pre-ICA pruning) shortens the recording, which no registry operator does.

## Reference Code

```python
def reject_bad_segments(d, method="annotation", threshold=None, action="drop", **_):
    spans = d.get("meta", {}).get("artifact_spans", []); x=np.asarray(d["data"]); sf=_sfreq(d); mask=np.ones(x.shape[-1],bool)
    for a,b in spans: mask[int(a*sf):int(b*sf)] = False
    return _out(d, x[...,mask] if action == "drop" else x, "reject_bad_segments", omitted_samples=int((~mask).sum()))
```
