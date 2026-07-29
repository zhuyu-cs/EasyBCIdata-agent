---
name: csv_npy
description: "Plain .csv / .npy / .npz array loaders — last-resort generic data import"
layer: L0
group: consumer_eeg
metadata:
  tags: [io, csv, npy, npz, generic, scientific_python]
  formats: [.csv, .npy, .npz]
  modalities: [eeg, meg, seeg, ecog, fnirs, lfp]
---
# CSV / NPY / NPZ

## Format Overview

Plain scientific-Python arrays:
- `.csv`: text, row-major or column-major.
- `.npy`: numpy save format; one array.
- `.npz`: numpy archive; multiple arrays.

Used as last resort when no domain format applies, or as an
intermediate between pipeline steps.

## Loader

Primary:
- `np.load` for `.npy` / `.npz`.
- `pandas.read_csv` or `np.genfromtxt` for `.csv`.

## Common Pitfalls

- **Orientation.** `(n_channels, n_times)` is EasyBCI convention; CSV
  often is the transpose.
- **No metadata.** Sample rate, channel names, units — must come from
  a sidecar or user declaration.
- **Memory mode.** Large `.npy` should use `mmap_mode="r"`.

## EasyBCI Path

```yaml
- load:csv,sfreq=USER_PROVIDED,layout=channel_x_time
- load:npy,sfreq=USER_PROVIDED
```

## References

1. (none specific — generic format.)

## Boundary

- **`mat`**: MATLAB array variant.
- **`custom_binary`**: raw binary without numpy framing.
