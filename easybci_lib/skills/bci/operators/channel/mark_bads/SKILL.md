---
name: mark_bads
description: "Bad-channel marking (non-destructive)"
layer: L3
group: channel
metadata:
  tags: [operator, channels, mark_bads]
  modalities: [eeg, seeg, ecog, meg]
  step_string: "mark_bads"
  analysis_goal_allowed: [classification, source_localization, feature_extraction, clinical_screening, exploratory, generic, connectivity, phase_amplitude_coupling, online_inference]
  analysis_goal_forbidden: []
---
# Bad-channel marking (non-destructive)

## Function

Write an explicit, externally supplied channel list into bad-channel metadata — or merge several such lists — without removing or altering any data.

## Parameter Format

`mark_bads:{bads},{action},{scope}`

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `bads` | varies | — | channel names to mark |
| `action` | varies | — | mark | merge | review |
| `scope` | varies | — | subject / session / run the list applies to |
| `detection` | varies | — | how the list was produced (manual visual inspection, prior QC table) |
| `source_file` | varies | — | external file the list is read from (e.g. a per-subject CSV shipped with the dataset) |

## When to Use

Refer to the reference pipelines that use this operator — it fills a gap that no existing registry operator covers.

## Ordering

- Apply BEFORE: `drop_bads`, `interpolate_bads`, `ica`, `car`

## Relationship to Existing Operators

**Nearest:** `channel/drop_bads`

drop_bads permanently removes channels and interpolate_bads overwrites them with a spline reconstruction; both destroy exactly what a marking step is supposed to preserve. References that only record known-bad sensors in the derivative keep the samples on disk.

## Reference Code

```python
def mark_bads(d, bads, action="mark", **_):
    old = set(d.get("meta", {}).get("bad_channels", [])); new = set(bads) if action == "mark" else old | set(bads)
    return _out(d, step="mark_bads", bad_channels=sorted(new), bad_channel_detection="manual/external")
```
