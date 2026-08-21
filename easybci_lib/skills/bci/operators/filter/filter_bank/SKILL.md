---
name: filter_bank
description: "Multi-band filter bank"
layer: L3
group: filter
metadata:
  tags: [operator, filter, filter_bank]
  modalities: [eeg, seeg, ecog, meg]
  step_string: "filter_bank"
  analysis_goal_allowed: [classification, feature_extraction, clinical_screening, exploratory, generic, connectivity, phase_amplitude_coupling]
  analysis_goal_forbidden: [online_inference]
---
# Multi-band filter bank

## Function

Apply a set of band-pass filters to the same data and return one output per band, so that downstream steps can act band by band. The bands are given as an explicit list, or as a range with a bandwidth and a step (contiguous, overlapping or half-overlapping), optionally with named bands excluded - line-noise fundamentals and their harmonics being the usual reason.

## Parameter Format

`filter_bank:{bands}`

Examples vary by use case.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `bands` | varies | — | explicit list of [low, high] pairs, or a range specification |
| `bandwidth` | varies | — | width of each band when the bands are generated from a range |
| `step` | varies | — | spacing between successive band centres; equal to bandwidth for contiguous bands, half of it for half-overlapping bands |
| `exclude_bands` | varies | — | bands dropped from the bank, typically those containing the line-noise fundamental or its harmonics |
| `method` | varies | — | fir | iir |
| `order` | varies | — | filter order, or the ripple and attenuation specification the paper reports instead |

## When to Use

Refer to the reference pipelines that use this operator — it fills a gap that no existing registry operator covers.

## Ordering

- Apply AFTER: `notch`, `bandpass`
- Apply BEFORE: `aggregate_bands`

## Relationship to Existing Operators

**Nearest:** `filter/bandpass_filter`

bandpass takes one low and one high edge and returns data of the same shape. A filter bank returns an extra band dimension, which is the whole point: the bands are carried separately so that an envelope can be taken per band and the bands combined afterwards (see aggregate_bands). Writing a bank as N sequential bandpass calls is wrong twice over - sequential band-passes intersect rather than branch, and the registry has no way to express a branch anyway. Any high-frequency-activity or broadband-power pipeline in the iEEG literature needs this, and it is also how line-noise contamination is avoided without notching: the offending bands are simply left out of the bank.

## Reference Code

```python
def filter_bank(d, bands=None, bandwidth=None, step=None, exclude_bands=None, method="fir", order=4, **_):
    from scipy.signal import butter, sosfiltfilt
    x=np.asarray(d["data"]); sf=_sfreq(d)
    if bands is None:
        if bandwidth is None or step is None: raise ValueError("filter_bank needs bands or bandwidth+step")
        hi=sf/2; bands=[]; c=bandwidth/2
        while c+bandwidth/2 < hi: bands.append((c-bandwidth/2,c+bandwidth/2)); c+=step
    bands=[tuple(b) for b in bands if not exclude_bands or not any(b[0]<e[1] and b[1]>e[0] for e in exclude_bands)]
    out=[]
    for lo,hi in bands:
        if method == "iir":
            sos=butter(order,[lo/(sf/2),hi/(sf/2)],btype="band",output="sos"); out.append(sosfiltfilt(sos,x,axis=-1))
        else:
            from scipy.signal import firwin, filtfilt
            taps=firwin(min(x.shape[-1]-1, max(31,int(3*sf/max(lo,1)))),[lo,hi],pass_zero=False,fs=sf); out.append(filtfilt(taps,[1],x,axis=-1))
    return _out(d,np.stack(out,axis=0),"filter_bank",filter_bank_bands=bands)


OPERATORS = {k: globals()[k] for k in ("segment epoch baseline_correct reject_epochs define_events import_events repair_events select_events attach_metadata detect_bads mark_bads set_channel_types set_montage derive_bipolar_channel minmax_scale reref_channels ic_classify manual_ic_selection interpolate_artifact detect_artifact_spans reject_bad_segments wavelet_ica overlap_regression detrend smooth dss aggregate_bands amplitude_modulation graph_metrics sleep_stager maxwell_filter ctf_grad_comp estimate_head_position align_head_position concatenate crop split_runs equalize_channels exclude_subjects sort_epochs no_op filter_bank").split()}


__all__ = ["OPERATORS", *OPERATORS]
```
