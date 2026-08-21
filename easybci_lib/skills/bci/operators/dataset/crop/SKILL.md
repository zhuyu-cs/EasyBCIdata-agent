---
name: crop
description: "Time-range cropping"
layer: L3
group: dataset
metadata:
  tags: [operator, dataset, crop]
  modalities: [eeg, seeg, ecog, meg]
  step_string: "crop"
  analysis_goal_allowed: [classification, source_localization, feature_extraction, clinical_screening, exploratory, generic, connectivity, phase_amplitude_coupling, online_inference]
  analysis_goal_forbidden: []
---
# Time-range cropping

## Function

Keep only a time range of a recording or of each epoch, anchored to absolute time, to an event, to the last event plus a pad, or to the centre of a block.

## Parameter Format

`crop:{tmin},{tmax},{anchor}`

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tmin` | varies | — | start of the retained range |
| `tmax` | varies | — | end of the retained range |
| `anchor` | varies | — | recording_start | first_event | last_event | block_center |
| `duration_s` | varies | — | length to retain when the range is given as a duration around the anchor |
| `pad_s` | varies | — | padding added beyond the anchor |
| `level` | varies | — | raw | epochs |

## When to Use

Refer to the reference pipelines that use this operator — it fills a gap that no existing registry operator covers.

## Ordering

- Apply BEFORE: `bandpass`, `epoch`

## Relationship to Existing Operators

**No near equivalent in the registry.**

No operator restricts the time range. Cropping to the task span before preprocessing, or to a post-stimulation analysis window after epoching, has to happen outside the pipeline entirely.

## Reference Code

```python
def crop(d, tmin=0.0, tmax=None, anchor="recording_start", duration_s=None, pad_s=0.0, level="raw", **_):
    x=np.asarray(d["data"]); sf=_sfreq(d); a=max(0,int(round(tmin*sf))); b=x.shape[-1] if tmax is None else min(x.shape[-1],int(round(tmax*sf))); return _out(d,x[...,a:b],"crop",crop_samples=[a,b])
```
