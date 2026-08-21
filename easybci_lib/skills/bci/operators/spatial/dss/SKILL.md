---
name: dss
description: "Denoising source separation"
layer: L3
group: spatial
metadata:
  tags: [operator, spatial, dss]
  modalities: [eeg, meg]
  step_string: "dss"
  analysis_goal_allowed: [classification, source_localization, feature_extraction, exploratory, generic]
  analysis_goal_forbidden: []
---
# Denoising source separation

## Function

Bias-filter based spatial filtering that ranks components by reproducibility with respect to a chosen bias (trial-to-trial phase locking, a frequency bin, a condition contrast) and keeps the leading ones.

## Parameter Format

`dss:{bias},{n_keep},{n_pca}`

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `bias` | varies | — | evoked | frequency_bin | condition |
| `n_keep` | varies | — | components retained |
| `n_pca` | varies | — | PCA rank in each whitening stage |
| `domain` | varies | — | time | frequency |

## When to Use

Refer to the reference pipelines that use this operator — it fills a gap that no existing registry operator covers.

## Ordering

- Apply AFTER: `bandpass`, `epoch`

## Relationship to Existing Operators

**Nearest:** `filter/zapline`

zapline is DSS specialised to one bias — line-frequency energy — and its parameters (f_line, n_remove) only express removal of that component. Generic DSS keeps the reproducible subspace instead of removing a nuisance one, and csp/xdawn are supervised class-contrast filters that need trial labels.

## Reference Code

```python
def dss(d, bias="evoked", n_keep=5, n_pca=None, domain="time", **_):
    x=np.asarray(d["data"]); z=x.reshape(x.shape[0],-1); cov=z@z.T/z.shape[1]; bias_cov=np.cov(z) if bias != "evoked" else np.outer(z.mean(axis=1),z.mean(axis=1)); vals,vec=np.linalg.eigh(np.linalg.pinv(cov+1e-9*np.eye(cov.shape[0]))@bias_cov); w=vec[:,np.argsort(vals)[::-1][:n_keep]]; y=(w.T@z).reshape((w.shape[1],*x.shape[1:]))
    return _out(d,y,"dss",dss_components=w, provenance="independent", upstream="python-meegkit (DSS formulation)")
```
