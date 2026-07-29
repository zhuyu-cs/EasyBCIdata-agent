---
name: brainvision
description: "BrainVision (.vhdr / .vmrk / .eeg) — Brain Products vendor format"
layer: L0
group: consumer_eeg
metadata:
  tags: [io, brainvision, vhdr, brain_products]
  formats: [.vhdr, .vmrk, .eeg]
  modalities: [eeg]
---
# BrainVision

## Format Overview

Brain Products / BrainVision Analyzer triplet format: `.vhdr` header,
`.vmrk` markers, `.eeg` data.

## Loader

Primary: `mne.io.read_raw_brainvision(vhdr)`.

## Common Pitfalls

- **Three-file dependency.** Moving the `.vhdr` without `.vmrk` / `.eeg`
  breaks the load.
- **Sub-Hz channels.** Some Brain Products amplifiers ship 0.016 Hz
  hardware high-pass; document for ERP.
- **Marker encoding.** S1, R1, etc. — domain-specific.

## EasyBCI Path

```yaml
- load:brainvision,vhdr=path
```

## References

1. Brain Products GmbH — BrainVision Analyzer documentation.

## Boundary

- **`edf`** / **`fif`** / **`eeglab_set`**: alternative consumer EEG formats.
