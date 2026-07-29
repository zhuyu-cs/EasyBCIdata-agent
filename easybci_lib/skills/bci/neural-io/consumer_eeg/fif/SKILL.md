---
name: fif
description: "Elekta/MNE FIF (.fif) — MEG / EEG native format"
layer: L0
group: consumer_eeg
metadata:
  tags: [io, fif, mne, elekta, neuromag, meg]
  formats: [.fif]
  modalities: [meg, eeg]
---
# FIF — Elekta/MNE Native

## Format Overview

FIF is the Neuromag / Elekta MEG vendor format and the MNE-Python
canonical file format. Most MEG labs and many EEG labs using MNE export
FIF.

## Loader

Primary: `mne.io.read_raw_fif(path, preload=True)`.

## Common Pitfalls

- **Maxfilter applied vs not.** Check `info["proc_history"]`; raw MEG vs
  tSSS-processed are different.
- **File splitting.** Long recordings split into `*.fif`, `*-1.fif`, ...;
  MNE handles via the link annotations.
- **EEG-in-FIF gotcha.** Some labs save EEG-only as FIF; channel naming
  remains "EEG NNN" instead of 10-20.

## EasyBCI Path

```yaml
- load:fif
```

## References

1. Gramfort, A. et al. (2013). *MEG and EEG data analysis with MNE-Python*.
   Frontiers in Neuroscience 7: 267. doi:10.3389/fnins.2013.00267.

## Boundary

- **`edf`**: clinical EEG.
- **`brainvision`**: vendor EEG.
- **`eeglab_set`**: MATLAB EEGLAB.
