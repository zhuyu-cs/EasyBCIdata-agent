---
name: eeglab_set
description: "EEGLAB SET (.set / .fdt) — MATLAB EEGLAB native format"
layer: L0
group: consumer_eeg
metadata:
  tags: [io, eeglab, set, fdt, matlab]
  formats: [.set, .fdt]
  modalities: [eeg]
---
# EEGLAB SET

## Format Overview

EEGLAB (UCSD Delorme & Makeig 2004) native format — `.set` for header
(MATLAB struct serialized to v7) + `.fdt` for continuous data binary.

## Loader

Primary: `mne.io.read_raw_eeglab(path)`.

## Common Pitfalls

- **MATLAB struct version drift** — `.set` files saved as MAT v7.3
  (HDF5-based) vs v7 (proprietary MAT) handled differently; sometimes
  needs `mat73` package.
- **ICA decomposition embedded** — `.set` carries `EEG.icaweights`;
  preserve if porting to MNE / MNE-ICALabel.

## EasyBCI Path

```yaml
- load:eeglab_set
```

## References

1. Delorme, A., & Makeig, S. (2004). *EEGLAB: an open source toolbox for
   analysis of single-trial EEG dynamics including independent component
   analysis*. J. Neuroscience Methods 134(1): 9–21.
   doi:10.1016/j.jneumeth.2003.10.009.

## Boundary

- **`mat`**: generic MATLAB; this skill is specifically EEGLAB-flavoured.
- **`edf`** / **`fif`** / **`brainvision`**: other consumer EEG.
