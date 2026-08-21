---
name: reject_epochs
description: "Criterion-based epoch rejection"
layer: L3
group: epoch
metadata:
  tags: [operator, epoching, reject_epochs]
  modalities: [eeg, seeg, ecog, meg]
  step_string: "reject_epochs"
  analysis_goal_allowed: [classification, source_localization, feature_extraction, clinical_screening, exploratory, generic, connectivity, phase_amplitude_coupling]
  analysis_goal_forbidden: [online_inference]
---
# Criterion-based epoch rejection

## Function

Drop epochs that violate explicit, user-supplied criteria: absolute amplitude, peak-to-peak within a moving window, per-channel-type thresholds, robust outlier statistics, monitor-channel steps, or an explicit index list.

## Parameter Format

`reject_epochs:{criteria},{method},{indices}`

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `criteria` | varies | — | list of {type, threshold, window_ms, channels/ch_type} |
| `method` | varies | — | fixed_threshold | peak_to_peak | esd_outlier | manual_visual | manual_index |
| `indices` | varies | — | explicit epochs to drop when method = manual_index |
| `individualized` | varies | — | whether the thresholds were tuned per subject rather than fixed across the dataset |
| `monitor_channels` | varies | — | the channels the criterion is evaluated on, when it is a subset of the recording rather than all of it |

## When to Use

Refer to the reference pipelines that use this operator — it fills a gap that no existing registry operator covers.

## Ordering

- Apply AFTER: `epoch`, `baseline_correct`

## Relationship to Existing Operators

**Nearest:** `adaptive_cleaning/autoreject`

autoreject chooses its thresholds itself by cross-validated grid search and interpolates up to n_interpolate channels per epoch; it cannot be given the paper's threshold, cannot express a peak-to-peak-in-200-ms or an EOG-step criterion, and cannot drop a named trial index. It is used in these files only where the paper actually ran AutoReject.

## Reference Code

```python
def reject_epochs(d, criteria=None, method="fixed_threshold", indices=None, **_):
    x = np.asarray(d["data"]); bad = set(int(i) for i in (indices or []))
    if method != "manual_index":
        criteria = criteria or []
        for i, ep in enumerate(x):
            for c in criteria:
                z = ep
                if c.get("channels"):
                    names = _names(d, ep.shape[0]); z = ep[[names.index(k) for k in c["channels"] if k in names]]
                val = np.ptp(z, axis=-1).max() if c.get("type") in ("peak_to_peak", "ptp") else np.abs(z).max()
                if val > float(c.get("threshold", np.inf)): bad.add(i); break
    keep = [i for i in range(len(x)) if i not in bad]
    return _out(d, x[keep], "reject_epochs", dropped_epochs=sorted(bad), kept_indices=keep)
```
