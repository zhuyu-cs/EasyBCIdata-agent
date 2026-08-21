---
name: define_events
description: "Event definition from hardware channels"
layer: L3
group: event
metadata:
  tags: [operator, events, define_events]
  modalities: [eeg, seeg, ecog, meg]
  step_string: "define_events"
  analysis_goal_allowed: [classification, source_localization, feature_extraction, clinical_screening, exploratory, generic, connectivity, phase_amplitude_coupling, online_inference]
  analysis_goal_forbidden: []
---
# Event definition from hardware channels

## Function

Derive an event table from trigger/STIM channels, photodiode traces, stimulation artefacts or existing annotations, with code selection, code remapping and latency correction.

## Parameter Format

`define_events:{source},{channels},{codes}`

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `source` | varies | — | stim_channel | annotations | photodiode | stim_artifact_peak |
| `channels` | varies | — | trigger / photodiode channel names |
| `codes` | varies | — | codes or code ranges to keep, and their condition mapping |
| `latency_shift_ms` | varies | — | constant correction (projector / audio delay) |
| `detect` | varies | — | peak-detection settings when source = stim_artifact_peak (prominence, min width, polarity) |

## When to Use

Refer to the reference pipelines that use this operator — it fills a gap that no existing registry operator covers.

## Ordering

- Apply BEFORE: `epoch`, `select_events`

## Relationship to Existing Operators

**Nearest:** `qc_operator/marker_integrity_check`

marker_integrity_check only validates that already-existing markers align across streams within tolerance_ms; it produces a QC verdict, not an event table, and has no trigger channel, code map, photodiode or latency-shift surface. peakdetect_qrs detects peaks but is hard-wired to ECG/QRS morphology.

## Reference Code

```python
def define_events(d, source="stim_channel", channels=None, codes=None, latency_shift_ms=0.0, detect=None, **_):
    x = np.asarray(d["data"]); sf = _sfreq(d); names = _names(d, x.shape[0]); events = []
    if source == "annotations": events = _events(d)
    elif source == "stim_channel":
        ci = names.index(channels[0]) if channels and channels[0] in names else -1
        if ci < 0: raise ValueError("stim channel is required")
        tr = x[ci]; on = np.flatnonzero((tr[1:] != tr[:-1]) & (tr[1:] != 0)) + 1
        events = [{"sample": int(i + latency_shift_ms * sf / 1000), "code": int(tr[i])} for i in on]
    elif source in ("photodiode", "stim_artifact_peak"):
        from scipy.signal import find_peaks
        ci = names.index(channels[0]) if channels and channels[0] in names else 0
        p = find_peaks(x[ci], **(detect or {}))[0]
        events = [{"sample": int(i + latency_shift_ms * sf / 1000), "code": 1} for i in p]
    if codes:
        keep = set(codes if isinstance(codes, (list, tuple, set)) else codes.keys()); events = [{**e, "code": codes[e.get("code")] if isinstance(codes, Mapping) and e.get("code") in codes else e.get("code")} for e in events if e.get("code") in keep or isinstance(codes, Mapping)]
    return _out(d, step="define_events", events=events, provenance="upstream_wrapper", upstream="MNE annotations/find_events; SciPy find_peaks")
```
