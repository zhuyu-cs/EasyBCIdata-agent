---
name: detect_bads
description: "Automatic bad-channel detection"
layer: L3
group: channel
metadata:
  tags: [operator, channels, detect_bads]
  modalities: [eeg, seeg, ecog, meg]
  step_string: "detect_bads"
  analysis_goal_allowed: [classification, source_localization, feature_extraction, clinical_screening, exploratory, generic, connectivity, phase_amplitude_coupling, online_inference]
  analysis_goal_forbidden: []
---
# Automatic bad-channel detection

## Function

Score every channel against one or more statistical criteria (flatness, extreme amplitude, low neighbour correlation, RANSAC predictability, high-frequency noise, kurtosis, non-1/f PSD shape, robust ESD outlier, MEG noisy/flat) and write the resulting list to bad-channel metadata.

## Parameter Format

`detect_bads:{criteria},{thresholds},{n_iterations}`

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `criteria` | varies | — | which detectors to run |
| `thresholds` | varies | — | per-criterion threshold (correlation, z, IQR multiple, kurtosis SD) |
| `n_iterations` | varies | — | repeat until convergence, as PREP does |
| `reference_for_detection` | varies | — | reference used while scoring |
| `action` | varies | — | set the bad-channel list, or merge with a list already present |
| `verification` | varies | — | whether the automatic list was confirmed or edited by a human before use |
| `cross_talk` | varies | — | path to the system's cross-talk correction file, when the detection is carried out through a spatial model that needs it |
| `calibration` | varies | — | path to the system's fine-calibration file, same case as cross_talk |
| `coord_frame` | varies | — | coordinate frame the detection is carried out in - typically head or device |
| `head_pos` | varies | — | continuous head-position estimate to compensate for while detecting, for recordings where the subject moved |
| `origin` | varies | — | centre of the sphere used by the spatial model the detection relies on, or auto to fit it from the sensor or electrode positions |

## When to Use

Refer to the reference pipelines that use this operator — it fills a gap that no existing registry operator covers.

## Ordering

- Apply BEFORE: `drop_bads`, `interpolate_bads`, `ica`, `car`

## Relationship to Existing Operators

**Nearest:** `channel/drop_bads`

drop_bads and interpolate_bads both consume data_dict['meta']['bad_channels'], but no operator in the registry ever populates it — the detection half of the contract is missing, so those two operators are unreachable in an automated pipeline.

## Reference Code

```python
def detect_bads(d, criteria=None, thresholds=None, action="merge", **_):
    x = np.asarray(d["data"]); names = _names(d, x.shape[0]); criteria = criteria or ["flat", "amplitude"] ; thresholds = thresholds or {}; bad = set(d.get("meta", {}).get("bad_channels", []))
    for i, ch in enumerate(names):
        z = x[i];
        if "flat" in criteria and np.std(z) <= thresholds.get("flat", 1e-12): bad.add(ch)
        if "amplitude" in criteria and np.max(np.abs(z)) > thresholds.get("amplitude", np.inf): bad.add(ch)
        if "kurtosis" in criteria:
            from scipy.stats import kurtosis
            if abs(kurtosis(z, fisher=False)) > thresholds.get("kurtosis", 10): bad.add(ch)
    if action == "set": bad = set(names) & bad
    return _out(d, step="detect_bads", bad_channels=sorted(bad), bad_channel_scores={})
```
