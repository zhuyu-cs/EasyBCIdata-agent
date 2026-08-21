---
name: attach_metadata
description: "Trial-level metadata attachment"
layer: L3
group: event
metadata:
  tags: [operator, events, attach_metadata]
  modalities: [eeg, seeg, ecog, meg]
  step_string: "attach_metadata"
  analysis_goal_allowed: [classification, feature_extraction, exploratory, generic]
  analysis_goal_forbidden: []
---
# Trial-level metadata attachment

## Function

Join externally held trial-level information (behavioural responses, stimulus identity, linguistic units) onto epochs so downstream analyses can index by it.

## Parameter Format

`attach_metadata:{source},{keys},{level}`

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `source` | varies | — | behavioural table / stimulus annotation being joined |
| `keys` | varies | — | columns used to align rows to epochs |
| `level` | varies | — | epoch | run | subject |

## When to Use

Refer to the reference pipelines that use this operator — it fills a gap that no existing registry operator covers.

## Ordering

- Apply AFTER: `epoch`

## Relationship to Existing Operators

**No near equivalent in the registry.**

Every registry operator transforms the signal; none carries labels alongside it. Without this step the epoch tensor exists but the trial labels a decoder needs do not, so the step cannot be represented at all.

## Reference Code

```python
def attach_metadata(d, source, keys=None, level="epoch", **_):
    if isinstance(source, (str, Path)):
        p = Path(source); rows = list(csv.DictReader(p.open(newline="", encoding="utf-8-sig")))
    else: rows = list(source)
    meta = dict(d.get("meta", {})); meta["trial_metadata"] = rows
    return _out({**d, "meta": meta}, step="attach_metadata", metadata_keys=keys, metadata_level=level)
```
