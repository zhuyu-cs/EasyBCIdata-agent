---
name: repair_events
description: "Event table repair"
layer: L3
group: event
metadata:
  tags: [operator, events, repair_events]
  modalities: [eeg, seeg, ecog, meg]
  step_string: "repair_events"
  analysis_goal_allowed: [classification, source_localization, feature_extraction, clinical_screening, exploratory, generic, connectivity, phase_amplitude_coupling, online_inference]
  analysis_goal_forbidden: []
---
# Event table repair

## Function

Detect and repair defects in an event table: missing or duplicated markers completed against an expected sequence, and mislabelled condition codes corrected for named subjects/sessions/runs.

## Parameter Format

`repair_events:{strategy},{expected_sequence},{corrections}`

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `strategy` | varies | — | complete_missing | recode | drop_duplicates |
| `expected_sequence` | varies | — | the sequence the stimulus program should have produced |
| `corrections` | varies | — | explicit {scope: {old: new}} overrides |
| `scope` | varies | — | subject / session / run the repair applies to |

## When to Use

Refer to the reference pipelines that use this operator — it fills a gap that no existing registry operator covers.

## Ordering

- Apply AFTER: `define_events`, `import_events`
- Apply BEFORE: `epoch`

## Relationship to Existing Operators

**Nearest:** `qc_operator/marker_integrity_check`

marker_integrity_check detects misalignment but never writes events back — it has no repair strategy, no expected-sequence model and no override table. Repairing the event table is the action that its verdict is supposed to trigger.

## Reference Code

```python
def repair_events(d, strategy="drop_duplicates", expected_sequence=None, corrections=None, **_):
    ev = _events(d)
    if strategy == "drop_duplicates":
        seen = set(); ev = [e for e in ev if (_event_sample(e), e.get("code")) not in seen and not seen.add((_event_sample(e), e.get("code")))]
    elif strategy == "recode":
        for e in ev:
            if corrections and e.get("code") in corrections: e["code"] = corrections[e["code"]]
    elif strategy == "complete_missing" and expected_sequence:
        for i, code in enumerate(expected_sequence):
            if i >= len(ev): ev.append({"sample": i, "code": code})
    return _out(d, step="repair_events", events=ev, strategy=strategy)
```
