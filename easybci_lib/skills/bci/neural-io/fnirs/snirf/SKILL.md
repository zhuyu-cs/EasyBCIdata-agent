---
name: snirf
description: "SNIRF — fNIRS BIDS-recommended HDF5 format (Society for fNIRS canonical)"
layer: L0
group: fnirs
metadata:
  tags: [io, snirf, fnirs, hdf5, society_fnirs, bids_nirs]
  formats: [.snirf]
  modalities: [fnirs]
---
# SNIRF — Shared Near Infrared Spectroscopy Format

## Format Overview

SNIRF is the **fNIRS canonical HDF5 format** maintained by the Society
for fNIRS. BIDS-NIRS recommends SNIRF as the data file format. Stores
per-channel optical intensity / OD / HbO+HbR (any preprocessing stage)
+ source-detector geometry + probe metadata.

## File Manifest

```
sub-01_session-01.snirf       # HDF5 monolithic
sub-01_session-01.json        # optional BIDS-NIRS sidecar
```

## Metadata Schema

| Group | Contents |
|---|---|
| `/nirs/data1/dataTimeSeries` | (n_t, n_channels) intensity / OD / HbO+HbR. |
| `/nirs/data1/measurementList/{i}` | per-channel source/detector indices + wavelength + data type. |
| `/nirs/probe/sourcePos2D / detectorPos2D` | 2D source-detector geometry. |
| `/nirs/probe/wavelengths` | (n_wavelengths,) — e.g. [730, 850]. |
| `/nirs/data1/time` | (n_t,) — absolute timestamps. |
| `/nirs/stim{i}` | event blocks. |

## Loader Library Map

| Priority | Library |
|---|---|
| Primary | `mne-nirs` (`mne_nirs.io.read_raw_snirf`) |
| Secondary | `pysnirf2` |

`lazy_deps` key `io.snirf` → `mne-nirs==0.6.0`.

## Modality Coverage

fNIRS only.

## Common Pitfalls

- **Intensity vs OD vs HbO/HbR.** `dataTypeIndex` selects; loader must
  read correctly.
- **2D probe geometry.** Source/detector positions in `mm`; verify the
  units field.
- **Multiple data blocks.** Some SNIRF files have multiple `data{i}`;
  pick the right one for your downstream task.

## Reference Implementation

### Standalone

```python
import h5py


def load_snirf(path: str) -> dict:
    with h5py.File(path, "r") as f:
        data = f["nirs/data1/dataTimeSeries"][:].T            # (n_ch, n_t)
        time = f["nirs/data1/time"][:]
        wavelengths = f["nirs/probe/wavelengths"][:]
    return {"data": data, "time": time, "wavelengths": list(wavelengths)}
```

### EasyBCI-Adapted

```python
from typing import Any, Dict


def operator_load_snirf(
    data_dict: Dict[str, Any], *, path: str,
) -> Dict[str, Any]:
    """SNIRF loader.

    Parameters
    ----------
    data_dict : dict
    path : str

    Returns
    -------
    dict — populated OperatorIO.

    Raises
    ------
    EasyBCIOperatorError
        recoverable=False on missing file.

    Modality coverage
    -----------------
    fNIRS: yes.

    References
    ----------
    Society for fNIRS SNIRF spec; mne-nirs docs.

    Notes
    -----
    L0 IO operator (Rule 12 exempt).
    """
    import time
    # easybci-allow: file-io  # L0 IO operator
    from easybci_lib.tools.neural_processing.operator_errors import EasyBCIOperatorError
    if not path:
        raise EasyBCIOperatorError(
            operator="load_snirf", reason="path required", recoverable=False,
        )
    t0 = time.monotonic()
    try:
        import h5py
        import numpy as np
        with h5py.File(path, "r") as f:
            data = f["nirs/data1/dataTimeSeries"][:].T.astype(np.float32)
            time_arr = f["nirs/data1/time"][:]
            wavelengths = f["nirs/probe/wavelengths"][:].tolist()
    except Exception as exc:
        raise EasyBCIOperatorError(
            operator="load_snirf", reason=f"load failed: {exc}", recoverable=False,
        ) from exc
    elapsed = time.monotonic() - t0
    sfreq = 1.0 / float(np.median(np.diff(time_arr))) if len(time_arr) > 1 else 1.0
    n_t = data.shape[1]
    return {
        "data": data,
        "channels": [f"src_det_{i}" for i in range(data.shape[0])],
        "frequency": sfreq,
        "duration": n_t / max(sfreq, 1e-30),
        "elapsed_s": elapsed,
        "meta": {"modality": "fnirs", "wavelengths": wavelengths,
                 "load_snirf": {"path": path}},
    }
```

## References

1. SNIRF Specification. https://github.com/fNIRS/snirf.
2. Tucker, S. R. et al. (2023). *fNIRS-BIDS: a community-defined data
   structure*. NeuroImage 274: 120087. doi:10.1016/j.neuroimage.2023.120087.

## Boundary with Related Formats

- **fNIRS lab proprietary**: NIRSport / Hitachi / NIRx vendor formats;
  most convert to SNIRF.
- **`bids_ieeg`**: BIDS variant for iEEG; SNIRF is BIDS-NIRS.
