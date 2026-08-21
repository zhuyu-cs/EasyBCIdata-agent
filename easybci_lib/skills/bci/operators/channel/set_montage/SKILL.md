---
name: set_montage
description: "Montage / electrode-position assignment"
layer: L3
group: channel
metadata:
  tags: [operator, channels, set_montage]
  modalities: [eeg, seeg, ecog, meg]
  step_string: "set_montage"
  analysis_goal_allowed: [source_localization, feature_extraction, clinical_screening, exploratory, generic, connectivity]
  analysis_goal_forbidden: []
---
# Montage / electrode-position assignment

## Function

Attach a named standard montage or a digitised position file to the channel set.

## Parameter Format

`set_montage:{montage},{on_missing}`

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `montage` | varies | — | standard name (biosemi64, GSN-HydroCel-129, standard_1020) or file path |
| `on_missing` | varies | — | raise | warn | ignore for channels absent from the montage |

## When to Use

Refer to the reference pipelines that use this operator — it fills a gap that no existing registry operator covers.

## Ordering

- Apply BEFORE: `interpolate_bads`, `source_localization`

## Relationship to Existing Operators

**Nearest:** `qc_operator/electrode_coord_validate`

electrode_coord_validate checks a coordinate system that is already present; it cannot assign one. interpolate_bads, laplacian_ref, rest_reference and every source operator require meta['electrode_positions'], which nothing in the registry sets.

## Reference Code

```python
def set_montage(d, montage, on_missing="warn", **_):
    positions = d.get("meta", {}).get("electrode_positions", {})
    if isinstance(montage, (str, Path)) and Path(str(montage)).exists():
        positions.update(json.loads(Path(montage).read_text(encoding="utf-8")))
    else:
        try:
            import mne
            mo = mne.channels.make_standard_montage(str(montage)); positions.update({k: v.tolist() for k,v in mo.get_positions()["ch_pos"].items()})
        except Exception:
            if on_missing == "raise": raise
            warnings.warn("set_montage: optional MNE or montage file unavailable")
    return _out(d, step="set_montage", electrode_positions=positions, montage=montage, provenance="upstream_wrapper", upstream="MNE make_standard_montage")
```
