---
name: mat
description: "MATLAB .mat — generic MATLAB array container (v5 / v7 / v7.3)"
layer: L0
group: consumer_eeg
metadata:
  tags: [io, mat, matlab, hdf5, generic]
  formats: [.mat]
  modalities: [eeg, meg, seeg, ecog, fnirs, lfp, spike]
---
# MATLAB .mat

## Format Overview

`.mat` files: MATLAB native — v5/v6 (proprietary), v7 (compressed), v7.3
(HDF5-based). Common in legacy EEG / decoding code that started in
MATLAB.

## Loader

- v5/v7: `scipy.io.loadmat(path)`.
- v7.3: `h5py.File(path)` or `mat73` PyPI package.

## Common Pitfalls

- **Version drift.** v7 vs v7.3 reader differs. Detect via `scipy.io.matlab.whosmat`.
- **Cell arrays / structs.** Nested MATLAB structs decode to numpy
  objects; need explicit field unpacking.
- **Char arrays.** MATLAB strings come out as numpy arrays of int16.
- **Variable names.** Each `.mat` is a dict-of-variables; user must
  identify which variable holds the data.

## EasyBCI Path

```yaml
- load:mat,varname=USER_PROVIDED,sfreq=USER_PROVIDED
```

## References

1. MathWorks MATLAB documentation — `save` / `load` v7.3 spec.
2. SciPy I/O documentation — `scipy.io.loadmat`.

## Boundary

- **`eeglab_set`**: EEGLAB-specific `.set` is a structured `.mat`.
- **`csv_npy`**: scientific-Python flat arrays.
