---
name: detect_artifact_spans
description: "Artefact-span annotation"
layer: L3
group: adaptive_cleaning
metadata:
  tags: [operator, artifact, detect_artifact_spans]
  modalities: [eeg, seeg, ecog, meg]
  step_string: "detect_artifact_spans"
  analysis_goal_allowed: [classification, source_localization, feature_extraction, clinical_screening, exploratory, generic, connectivity, phase_amplitude_coupling]
  analysis_goal_forbidden: [online_inference]
---
# Artefact-span annotation

## Function

Mark contaminated time spans without changing the data: blink events from an EOG channel, muscle bursts by high-frequency z-score, high-amplitude transients by robust amplitude statistics, EMG contamination from a monitor channel envelope.

## Parameter Format

`detect_artifact_spans:{method},{band},{threshold}`

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | varies | — | eog_events | muscle_zscore | amplitude_iqr | emg_envelope |
| `band` | varies | — | band the detector works in (1-10 Hz for blinks, 110-140 Hz for muscle) |
| `threshold` | varies | — | z-score, sigma or IQR multiple |
| `window_s` | varies | — | analysis window the criterion is evaluated over |
| `step_s` | varies | — | hop between successive windows for sliding detectors |
| `duration_s` | varies | — | length of the annotation written around each hit |
| `monitor_channels` | varies | — | channels the detector reads |
| `min_length_good_s` | varies | — | shortest gap kept between annotations |
| `dilate_s` | varies | — | grow every detected span by this many seconds at each end, so that a downstream repair or rejection also covers the artifact's shoulders - the samples that fall below the threshold but are still contaminated |

## When to Use

Refer to the reference pipelines that use this operator — it fills a gap that no existing registry operator covers.

## Ordering

- Apply AFTER: `bandpass`
- Apply BEFORE: `ica`, `reject_bad_segments`, `interpolate_artifact`

## Relationship to Existing Operators

**Nearest:** `adaptive_cleaning/asr`

asr reconstructs contaminated segments in a PCA subspace and autoreject drops epochs; neither produces annotations, and both alter the data. References that annotate first and only later decide whether to omit those spans from an ICA fit or from epoching need the detection to be a separate, non-destructive step.

## Reference Code

```python
def detect_artifact_spans(d, method="amplitude_iqr", band=None, threshold=6.0, window_s=0.5, step_s=0.1, duration_s=None, monitor_channels=None, dilate_s=0.0, **_):
    x=np.asarray(d["data"]); sf=_sfreq(d); cn=_names(d,x.shape[0]); cis=[cn.index(c) for c in monitor_channels] if monitor_channels else range(x.shape[0]); z=x[list(cis)].mean(axis=0); spans=[]; w=max(1,int(window_s*sf)); h=max(1,int(step_s*sf))
    if method in ("muscle_zscore","emg_envelope"):
        from scipy.signal import butter,sosfiltfilt
        lo,hi=band or (110,140); sos=butter(4,[lo/(sf/2),hi/(sf/2)],btype="band",output="sos"); z=np.abs(sosfiltfilt(sos,z))
    med=np.median(z); scale=np.median(np.abs(z-med))+1e-12
    for a in range(0,len(z)-w+1,h):
        score=np.max(np.abs(z[a:a+w]-med))/scale
        if score > threshold: spans.append((max(0,a/sf-dilate_s), min(len(z)/sf,(a+w)/sf+dilate_s)))
    return _out(d, step="detect_artifact_spans", artifact_spans=spans, annotations=[{"onset":a,"duration":b-a,"description":"BAD_artifact"} for a,b in spans])
```
