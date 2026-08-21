---
name: ctf_grad_comp
description: "CTF gradient compensation"
layer: L3
group: meg
metadata:
  tags: [operator, meg, ctf_grad_comp]
  modalities: [meg]
  step_string: "ctf_grad_comp"
  analysis_goal_allowed: [classification, source_localization, feature_extraction, clinical_screening, exploratory, generic, connectivity, phase_amplitude_coupling]
  analysis_goal_forbidden: [online_inference]
---
# CTF gradient compensation

## Function

Apply CTF synthetic-gradiometer noise compensation at a given grade using the reference sensor array.

## Parameter Format

`ctf_grad_comp:{grade}`

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `grade` | varies | — | 0-3 compensation grade |

## When to Use

Refer to the reference pipelines that use this operator — it fills a gap that no existing registry operator covers.

## Ordering

- Apply BEFORE: `bandpass`, `maxwell_filter`

## Relationship to Existing Operators

**No near equivalent in the registry.**

CTF reference-array compensation is a system-specific linear projection with no analogue in the registry; no operator reads MEG reference sensors. Leaving it out changes the environmental-noise level of every downstream estimate on CTF data.

## Reference Code

```python
def ctf_grad_comp(d, grade=0, **kwargs): return _mne_raw_op(d,"compensate_to_head",grade=grade,**kwargs)
```
