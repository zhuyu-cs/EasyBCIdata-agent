---
name: xdf
description: "XDF (.xdf) — Extensible Data Format from LSL recordings"
layer: L0
group: consumer_eeg
metadata:
  tags: [io, xdf, lsl, recording, multi_stream]
  formats: [.xdf]
  modalities: [eeg, meg, fnirs, spike]
---
# XDF — Extensible Data Format

## Format Overview

XDF is the offline storage format for LSL-recorded streams. Stores
multiple parallel streams (EEG + Markers + Audio + ...) with per-stream
metadata + per-sample timestamps with LSL clock-correction.

## Loader

Primary: `pyxdf.load_xdf(path)`. EEG-only path: `mne.io.read_raw_xdf`
(if installed).

## Common Pitfalls

- **Stream selection.** A single XDF can contain EEG + Markers + Audio +
  ...; pick the right stream by name.
- **Clock correction.** `pyxdf` returns timestamps already clock-corrected
  to a common LSL clock; check that the marker stream's timestamps align.
- **Format zoo.** XDF supports many channel formats; ensure dtype is
  float-castable.

## EasyBCI Path

```yaml
- load:xdf,stream=EEG
```

## References

1. Kothe, C. A. (2014). *Lab Streaming Layer (LSL)*. https://github.com/sccn/labstreaminglayer.
2. pyxdf documentation. https://github.com/xdf-modules/pyxdf.

## Boundary

- **`lsl_stream`**: live-stream version; XDF is the recorded archive.
