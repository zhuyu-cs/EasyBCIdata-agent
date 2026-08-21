---
name: segment
description: "Fixed / sliding window segmentation"
layer: L3
group: epoch
metadata:
  tags: [operator, epoching, segment]
  modalities: [eeg, seeg, ecog, meg]
  step_string: "segment"
  analysis_goal_allowed: [classification, feature_extraction, clinical_screening, exploratory, generic, connectivity, phase_amplitude_coupling, online_inference]
  analysis_goal_forbidden: []
---
# Fixed / sliding window segmentation

## Function

Cut continuous data into fixed-length windows without reference to events (fixed, sliding with stride, or an explicit interval list). Emits an epoch-shaped array (n_windows, n_channels, n_times).

## Parameter Format

`segment:{method},{duration},{stride}`

Examples:
- `segment:fixed,2.0` — 2-second non-overlapping windows
- `segment:sliding,1.0,0.5` — 1s windows with 500ms stride

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | varies | — | fixed | sliding | interval_list |
| `duration` | varies | — | window length (s) |
| `stride` | varies | — | hop (s); equals duration for non-overlapping |
| `selection` | varies | — | optional label/annotation restricting which spans are cut |

## When to Use

Refer to the reference pipelines that use this operator — it fills a gap that no existing registry operator covers.

## Ordering

- Apply AFTER: `bandpass`, `notch`, `ica`, `drop_bads`

## Relationship to Existing Operators

**No near equivalent in the registry.**

No operator in any registry group turns continuous data into segments. The window kwargs that exist (log_band_power.n_per_seg, coherence.nperseg, stft.window_s) are internal to those estimators and do not emit segmented data, while autoreject and every decoder operator already require meta['epochs'] to exist.

## Reference Code

```python
def segment(d, method="fixed", duration=1.0, stride=None, intervals=None, selection=None, **_):
    """Fixed/sliding segmentation, independent NumPy implementation."""
    x = np.asarray(d["data"]); sf = _sfreq(d); n = x.shape[-1]
    if method == "interval_list":
        spans = intervals or selection or []
        idx = [(int(round(a * sf)), int(round(b * sf))) for a, b in spans]
    else:
        w = max(1, int(round(float(duration) * sf))); hop = w if stride is None else max(1, int(round(float(stride) * sf)))
        idx = [(i, i + w) for i in range(0, max(0, n - w + 1), hop)]
    arr = np.stack([x[..., a:b] for a, b in idx if b <= n], axis=0) if idx else np.empty((0, *x.shape[:-1], 0), x.dtype)
    return _out(d, arr, "segment", intervals=[a / sf for a, _ in idx], sfreq=sf)
```
