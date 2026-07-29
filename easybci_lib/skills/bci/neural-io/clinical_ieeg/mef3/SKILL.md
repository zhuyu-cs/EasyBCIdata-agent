---
name: mef3
description: "Multiscale Electrophysiology Format v3 (Mayo Clinic) — long-duration encrypted sEEG"
layer: L0
group: clinical_ieeg
metadata:
  tags: [io, mef3, mef, mayo, seeg, long_duration, encrypted, clinical]
  formats: [.mefd]
  modalities: [seeg, ecog]
---
# MEF3 — Multiscale Electrophysiology Format

## Format Overview

MEF3 (Mayo Clinic, Brinkmann 2009 / 2018) is the clinical sEEG standard
for **week-long continuous recordings**. Each subject session is a
directory `.mefd/` with one segment per channel. Supports per-segment
AES encryption, lossless compression, and millisecond-precision
timestamps anchored to wall-clock.

## File Manifest

```
session.mefd/
  ChannelA.timd/
    SegmentA_001.tdat
    SegmentA_001.tidx
    ...
  ChannelB.timd/...
  metadata.json  (optional lab-side annotation)
```

## Metadata Schema

Each `.timd` directory header carries:
- `name`, `sampling_frequency`, `units_conversion_factor`.
- `recording_start_time` (microsecond UNIX epoch).
- `segment_durations`, `total_samples`.
- Per-segment encryption key handle (if encrypted).

## Loader Library Map

| Priority | Library |
|---|---|
| Primary | `pymef` (Mayo-provided) |
| Secondary | `mef_tools` |

`lazy_deps` key `io.mef3` → `pymef==1.0.13`.

## Modality Coverage

sEEG, ECoG. Long-duration epilepsy monitoring.

## Common Pitfalls

- **Encryption.** Password / level-0 / level-1 keys required;
  unauthorized read fails.
- **Long sessions.** Hundreds of GB per session; chunked reads only.
- **Wall-clock timestamps.** Microsecond UTC; useful for cross-session
  alignment but easy to misinterpret as session-local seconds.
- **Per-channel sample rates differ.** Read each `.timd` independently.

## Reference Implementation

### Standalone

```python
import pymef


def load_mef3(path: str, password_level: int = 0) -> dict:
    session = pymef.MefSession(path, password=("", "")[:password_level + 1])
    return {
        "channels": session.read_ts_channel_basic_info(),
        "duration": session.session_md["session_metadata"]["total_duration"]
                    / 1e6,                    # μs → s
    }
```

### EasyBCI-Adapted

```python
from typing import Any, Dict


def operator_load_mef3(
    data_dict: Dict[str, Any], *, path: str, password: str = "",
) -> Dict[str, Any]:
    """MEF3 loader.

    Parameters
    ----------
    data_dict : dict
    path : str
    password : str

    Returns
    -------
    dict — populated OperatorIO.

    Raises
    ------
    EasyBCIOperatorError
        recoverable=False on missing / encrypted-without-password.

    Modality coverage
    -----------------
    sEEG / ECoG (long-duration clinical): yes.

    References
    ----------
    Brinkmann et al. 2009; pymef docs.

    Notes
    -----
    L0 IO operator (Rule 12 exempt).
    """
    import time
    # easybci-allow: file-io  # L0 IO operator
    from easybci_lib.tools.neural_processing.operator_errors import EasyBCIOperatorError
    if not path:
        raise EasyBCIOperatorError(
            operator="load_mef3", reason="path required", recoverable=False,
        )
    t0 = time.monotonic()
    try:
        import pymef
        import numpy as np
        session = pymef.MefSession(path, password=password)
        basic = session.read_ts_channel_basic_info()
        sfreqs = [c["fsamp"] for c in basic]
        sfreq = float(min(sfreqs)) if sfreqs else 1000.0
        # Read all channels at common sfreq (best effort)
        data_list = []
        for ch_info in basic:
            ch_data = session.read_ts_channels_sample(ch_info["name"],
                                                     [(0, None)])
            data_list.append(ch_data)
        data = np.stack(data_list).astype(np.float32)
        channels = [c["name"] for c in basic]
    except Exception as exc:
        raise EasyBCIOperatorError(
            operator="load_mef3", reason=f"load failed: {exc}", recoverable=False,
        ) from exc
    elapsed = time.monotonic() - t0
    n_t = data.shape[1]
    return {
        "data": data, "channels": channels,
        "frequency": sfreq, "duration": n_t / sfreq, "elapsed_s": elapsed,
        "meta": {"modality": "seeg", "load_mef3": {"path": path}},
    }
```

## References

1. Brinkmann, B. H. et al. (2009). *Large-scale electrophysiology:
   acquisition, compression, encryption, and storage of big data*.
   J. Neuroscience Methods 180(1): 185–192.
   doi:10.1016/j.jneumeth.2009.03.022.
2. Brinkmann, B. H. et al. (2018). *Multiscale electrophysiology format
   (MEF) 3.0*. Annual meeting AES — slide deck.
3. `pymef`. https://github.com/msel-source/pymef

## Boundary with Related Formats

- **`bids_ieeg`**: complements MEF3 — same data, BIDS sidecar metadata.
- **`edf`**: shorter recordings; EDF supports up to a few days; MEF3 is
  needed for weeks.
- **`nwb`**: research deposit; MEF3 = clinical working format.
