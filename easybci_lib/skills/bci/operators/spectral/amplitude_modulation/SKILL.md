---
name: amplitude_modulation
description: "Amplitude-modulation features"
layer: L3
group: spectral
metadata:
  tags: [operator, spectral, amplitude_modulation]
  modalities: [eeg, seeg, ecog, meg]
  step_string: "amplitude_modulation"
  analysis_goal_allowed: [feature_extraction, exploratory, generic, connectivity]
  analysis_goal_forbidden: []
---
# Amplitude-modulation features

## Function

Modulation-spectrum features: the spectrum of the band-limited amplitude envelope, per carrier band and modulation frequency.

## Parameter Format

`amplitude_modulation:{carrier_bands},{modulation_bands},{envelope}`

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `carrier_bands` | varies | — | bands whose envelopes are analysed |
| `modulation_bands` | varies | — | modulation-frequency bins |
| `envelope` | varies | — | how the envelope is obtained (hilbert, rectify+lowpass) |

## When to Use

Refer to the reference pipelines that use this operator — it fills a gap that no existing registry operator covers.

## Ordering

- Apply AFTER: `bandpass`

## Relationship to Existing Operators

**Nearest:** `spectral/hilbert`

hilbert produces the envelope but nothing takes its spectrum; log_band_power and multitaper_psd operate on the carrier signal, not on its modulation. Composing them by hand would still be missing the per-carrier-band modulation matrix that the feature set is defined as.

## Reference Code

```python
def amplitude_modulation(d, carrier_bands, modulation_bands=None, envelope="hilbert", **_):
    from scipy.signal import hilbert, butter, sosfiltfilt, welch
    x=np.asarray(d["data"]); sf=_sfreq(d); feats=[]
    for lo,hi in carrier_bands:
        sos=butter(4,[lo/(sf/2),hi/(sf/2)],btype="band",output="sos"); band=sosfiltfilt(sos,x,axis=-1); env=np.abs(hilbert(band,axis=-1)); f,p=welch(env,sf,axis=-1,nperseg=min(2048,env.shape[-1]));
        sel=np.ones(len(f),bool) if not modulation_bands else np.logical_or.reduce([(f>=a)&(f<=b) for a,b in modulation_bands]); feats.append(p[...,sel])
    return _out(d,np.stack(feats,axis=0),"amplitude_modulation",carrier_bands=carrier_bands,modulation_bands=modulation_bands)
```
