---
name: ic_classify
description: "Automatic IC classification"
layer: L3
group: adaptive_cleaning
metadata:
  tags: [operator, artifact, ic_classify]
  modalities: [eeg, meg]
  step_string: "ic_classify"
  analysis_goal_allowed: [classification, source_localization, feature_extraction, clinical_screening, exploratory, generic, connectivity]
  analysis_goal_forbidden: [online_inference]
---
# Automatic IC classification

## Function

Label ICA components by a trained classifier or by correlation with monitor channels, and exclude those above threshold. Covers ICLabel / MNE-ICALabel, MEGNet, OSL-AFRICA kurtosis+cardiac rules, and correlation with EOG/EMG/EXG channels for classes ica does not handle.

## Parameter Format

`ic_classify:{method},{classes},{threshold}`

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | varies | — | iclabel | icalabel | megnet | africa_kurtosis | correlation |
| `classes` | varies | — | artefact classes to reject (muscle, eye, heart, line_noise, channel_noise, head_movement, other) |
| `threshold` | varies | — | probability or score above which a component is rejected |
| `monitor_channels` | varies | — | channels used when method = correlation |
| `max_components` | varies | — | cap on how many components may be removed |
| `action` | varies | — | subtract the labelled components, or only annotate them for review |
| `disabled_classes` | varies | — | classes the classifier is prevented from assigning, so that components which would have fallen into them are left in the data |

## When to Use

Refer to the reference pipelines that use this operator — it fills a gap that no existing registry operator covers.

## Ordering

- Apply AFTER: `ica`

## Relationship to Existing Operators

**Nearest:** `adaptive_cleaning/ica`

ica only auto-detects the two classes in its artifact_types enum (eog, ecg) and needs dedicated reference channels to do it. It cannot label muscle, head-movement, line-noise, channel-noise or other components, cannot run a classifier, and exposes no probability threshold or component cap.

## Reference Code

```python
def ic_classify(d, method="correlation", classes=None, threshold=0.3, monitor_channels=None, max_components=None, action="subtract", indices=None, **_):
    x = np.asarray(d["data"]); ica, s, comps = _ica(x); bad = list(indices or []); cn = _names(d, x.shape[0])
    if method == "correlation" and monitor_channels:
        mi = [cn.index(c) for c in monitor_channels if c in cn]; ref = x[mi].mean(axis=0).ravel()
        bad = [i for i in range(len(s)) if abs(np.corrcoef(s[:, i], ref)[0,1]) >= threshold]
    if max_components: bad = bad[:max_components]
    y = x
    if action == "subtract" and bad:
        s[:, bad] = 0; y = ica.inverse_transform(s).T.reshape(x.shape)
    return _out(d, y, "ic_classify", ica_excluded=bad, ica_method=method, provenance="independent" if method == "correlation" else "upstream_wrapper", upstream="mne-icalabel/MEGNet/osl-africa" if method != "correlation" else None)
```
