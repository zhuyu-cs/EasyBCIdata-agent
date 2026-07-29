---
name: edf
description: "European Data Format (.edf) — clinical EEG / sEEG / PSG standard"
layer: L0
group: consumer_eeg
metadata:
  tags: [io, edf, european_data_format, clinical, psg]
  formats: [.edf, .edf+, .bdf]
  modalities: [eeg, seeg, ecog]
---
# EDF — European Data Format

## Format Overview

EDF (Kemp 1992) is the most widely used clinical EEG format. EDF+ adds
asynchronous events. BDF is the 24-bit BioSemi variant. Sleep / PSG
labs, epilepsy units, and many BIDS-iEEG deposits use EDF as the data file.

## Loader

Primary: `mne.io.read_raw_edf` / `mne.io.read_raw_bdf`.

## Common Pitfalls

- **Mixed sample rates across channels** (rare but allowed by spec) —
  MNE upsamples; verify.
- **Annotations vs events** — EDF+ annotations need separate parsing.
- **Header truncation** — some clinical exports trim subject info.

## EasyBCI Path

```yaml
- load:edf
```

Hands off to MNE's reader; dispatcher handles modality inference from
channel-naming + sample rate. See `bci/pipeline/SKILL.md`
Step 1.

## References

1. Kemp, B. et al. (1992). *A simple format for exchange of digitized
   polygraphic recordings*. EEG Clin. Neurophysiol. 82(5): 391–393.
   doi:10.1016/0013-4694(92)90009-7.
2. Kemp, B., & Olivan, J. (2003). *European Data Format `plus' (EDF+)*.
   Clin. Neurophysiol. 114(9): 1755–1761. doi:10.1016/S1388-2457(03)00123-8.

## Boundary

- **`fif`**: MNE-native; preferred for MEG.
- **`brainvision`**: Brain Products vendor format; comparable consumer EEG.
- **`bids_ieeg`**: BIDS sidecar overlay (EDF is the data file).
