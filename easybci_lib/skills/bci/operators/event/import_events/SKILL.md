---
name: import_events
description: "External event / annotation import"
layer: L3
group: event
metadata:
  tags: [operator, events, import_events]
  modalities: [eeg, seeg, ecog, meg]
  step_string: "import_events"
  analysis_goal_allowed: [classification, source_localization, feature_extraction, clinical_screening, exploratory, generic, connectivity, phase_amplitude_coupling, online_inference]
  analysis_goal_forbidden: []
---
# External event / annotation import

## Function

Attach an externally produced event or annotation table (BIDS *_events.tsv, derivative detector output, video-review log, clinical scoring) to the recording.

## Parameter Format

`import_events:{source},{event_classes},{detector}`

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `source` | varies | — | path or BIDS entity of the table being imported |
| `event_classes` | varies | — | which annotation classes to keep |
| `detector` | varies | — | name of the producing detector when the table is a derivative |
| `detector_params` | varies | — | detector settings if reported |

## When to Use

Refer to the reference pipelines that use this operator — it fills a gap that no existing registry operator covers.

## Ordering

- Apply BEFORE: `epoch`, `select_events`

## Relationship to Existing Operators

**Nearest:** `qc_operator/ied_detector`

The registry can only re-detect events from the signal (ied_detector, hfo_detector, peakdetect_qrs); nothing reads an existing annotation file. Re-detection is not equivalent: it changes the event set, and for sub-kilohertz scalp recordings hfo_detector is explicitly out of contract anyway.

## Reference Code

```python
def import_events(d, source, event_classes=None, **_):
    p = Path(source); rows = list(csv.DictReader(p.open(newline="", encoding="utf-8-sig"))) if p.suffix.lower() == ".tsv" else json.loads(p.read_text(encoding="utf-8"))
    if isinstance(rows, Mapping): rows = rows.get("events", rows.get("annotations", []))
    if event_classes: rows = [r for r in rows if r.get("trial_type", r.get("description")) in event_classes]
    return _out(d, step="import_events", events=rows, provenance="independent", upstream="BIDS events.tsv convention")
```
