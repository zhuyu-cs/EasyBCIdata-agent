---
name: maxwell_filter
description: "Maxwell filtering (SSS / tSSS)"
layer: L3
group: meg
metadata:
  tags: [operator, meg, maxwell_filter]
  modalities: [meg]
  step_string: "maxwell_filter"
  analysis_goal_allowed: [classification, source_localization, feature_extraction, clinical_screening, exploratory, generic, connectivity, phase_amplitude_coupling]
  analysis_goal_forbidden: [online_inference]
---
# Maxwell filtering (SSS / tSSS)

## Function

Signal-space separation for MEG: project onto the internal multipole basis, optionally with temporal extension, cross-talk correction, fine calibration, movement compensation and alignment to a destination head position.

## Parameter Format

`maxwell_filter:{st_duration},{coord_frame},{cross_talk}`

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `st_duration` | varies | — | tSSS buffer length in seconds; null = plain SSS |
| `coord_frame` | varies | — | head | meg |
| `cross_talk` | varies | — | cross-talk correction file |
| `calibration` | varies | — | fine-calibration file |
| `head_pos` | varies | — | cHPI head-position traces for movement compensation |
| `destination` | varies | — | target head position |
| `origin` | varies | — | expansion origin |

## When to Use

Refer to the reference pipelines that use this operator — it fills a gap that no existing registry operator covers.

## Ordering

- Apply AFTER: `detect_bads`, `mark_bads`
- Apply BEFORE: `bandpass`, `ica`, `epoch`

## Relationship to Existing Operators

**Nearest:** `spatial/ssp_eog`

SSS is the standard first step of every Elekta/MEGIN pipeline and the registry has no MEG hardware-denoising operator at all. The SSP operators project out one physiological artefact each from a measured reference channel; they share neither the multipole basis, the movement compensation, nor the bad-channel reconstruction that SSS provides.

## Reference Code

```python
def maxwell_filter(d, **kwargs): return _mne_raw_op(d,"maxwell_filter",**kwargs)
```
