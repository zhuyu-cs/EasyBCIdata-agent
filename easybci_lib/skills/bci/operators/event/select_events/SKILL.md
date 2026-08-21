---
name: select_events
description: "Event / trial selection"
layer: L3
group: event
metadata:
  tags: [operator, events, select_events]
  modalities: [eeg, seeg, ecog, meg]
  step_string: "select_events"
  analysis_goal_allowed: [classification, source_localization, feature_extraction, clinical_screening, exploratory, generic, connectivity, phase_amplitude_coupling]
  analysis_goal_forbidden: [online_inference]
---
# Event / trial selection

## Function

Filter the event or trial table by experimental attributes (condition, stimulation current, site, response validity, set size) including a minimum-count rule per condition.

## Parameter Format

`select_events:{criteria},{min_events_per_condition},{exclude}`

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `criteria` | varies | — | attribute filters, e.g. {current_mA: [4.0, 6.0], site: limbic} |
| `min_events_per_condition` | varies | — | drop conditions with fewer than N events |
| `exclude` | varies | — | attribute values to drop (invalid responses, one-back repeats) |

## When to Use

Refer to the reference pipelines that use this operator — it fills a gap that no existing registry operator covers.

## Ordering

- Apply AFTER: `define_events`, `import_events`, `repair_events`
- Apply BEFORE: `epoch`

## Relationship to Existing Operators

**No near equivalent in the registry.**

There is no event- or trial-table operator of any kind in the registry, so condition selection has nowhere to live. pick_channels is the only selection operator and it works on the channel axis, not the trial axis.

## Reference Code

```python
def select_events(d, criteria=None, min_events_per_condition=None, exclude=None, **_):
    ev = _events(d); criteria = criteria or {}; exclude = exclude or {}
    def ok(e):
        return all(e.get(k) in (v if isinstance(v, (list, tuple, set)) else [v] ) for k,v in criteria.items()) and all(e.get(k) not in (v if isinstance(v, (list, tuple, set)) else [v]) for k,v in exclude.items())
    ev = [e for e in ev if ok(e)]
    if min_events_per_condition:
        from collections import Counter
        key = lambda e: e.get("condition", e.get("code")); c = Counter(key(e) for e in ev); ev = [e for e in ev if c[key(e)] >= min_events_per_condition]
    return _out(d, step="select_events", events=ev)
```
