---
name: epoch
description: "Event-locked epoching"
layer: L3
group: epoch
metadata:
  tags: [operator, epoching, epoch]
  modalities: [eeg, seeg, ecog, meg]
  step_string: "epoch"
  analysis_goal_allowed: [classification, source_localization, feature_extraction, clinical_screening, exploratory, generic, connectivity, phase_amplitude_coupling]
  analysis_goal_forbidden: [online_inference]
---
# Event-locked epoching

## Function

Cut event-locked epochs from continuous data over [tmin, tmax] around each event onset, optionally dropping epochs that overlap bad annotations.

## Parameter Format

`epoch:{events},{tmin},{tmax}`

Examples:
- `epoch:stimulus,-0.2,0.8` — Lock to stimulus events, -200ms to +800ms
- `epoch:target,0,1.0` — Lock to target events, 0 to 1000ms

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `events` | varies | — | event name / code / id map to lock to |
| `tmin` | varies | — | start of the epoch relative to onset (s) |
| `tmax` | varies | — | end of the epoch relative to onset (s) |
| `reject_by_annotation` | varies | — | drop epochs overlapping bad spans |
| `baseline` | varies | — | optional; None keeps epochs unbaselined |

## When to Use

Refer to the reference pipelines that use this operator — it fills a gap that no existing registry operator covers.

## Ordering

- Apply AFTER: `bandpass`, `notch`, `ica`, `drop_bads`, `define_events`, `import_events`
- Apply BEFORE: `baseline_correct`, `reject_epochs`

## Relationship to Existing Operators

**No near equivalent in the registry.**

Same gap as segment, but event-locked: nothing in the registry consumes an event table to produce trials, and segment (the only observed segmentation vocabulary) has no event/tmin/tmax surface. ERP-, decoding- and CCEP-style references cannot be expressed without it.

## Reference Code

```python
def epoch(d, events=None, tmin=0.0, tmax=1.0, reject_by_annotation=True, baseline=None, event_id=None, **_):
    """Event-locked epochs; event parsing follows MNE's sample convention."""
    x = np.asarray(d["data"]); sf = _sfreq(d); events = _events({**d, "events": events} if events is not None else d)
    wanted = set(event_id.values() if isinstance(event_id, Mapping) else (event_id or []))
    a, b = int(round(tmin * sf)), int(round(tmax * sf)); good = []
    anns = d.get("meta", {}).get("annotations", [])
    for ev in events:
        code = ev.get("code", ev.get("value", ev.get("event_id")))
        if wanted and code not in wanted and ev.get("name") not in wanted: continue
        s = _event_sample(ev) + a; e = _event_sample(ev) + b
        if s < 0 or e > x.shape[-1]: continue
        if reject_by_annotation and any(max(s, int(z.get("onset", 0) * sf)) < min(e, int((z.get("onset", 0) + z.get("duration", 0)) * sf)) for z in anns if str(z.get("description", z.get("label", ""))).lower().startswith("bad")): continue
        good.append(x[..., s:e])
    arr = np.stack(good) if good else np.empty((0, *x.shape[:-1], max(0, b-a)), x.dtype)
    return _out(d, arr, "epoch", epochs_events=events, tmin=tmin, tmax=tmax, baseline=baseline)
```
