---
name: bids_ieeg
description: "BIDS-iEEG sidecar (*_channels.tsv / *_electrodes.tsv / *_coordsystem.json / *_ieeg.json) — clinical sEEG / ECoG standard"
layer: L0
group: clinical_ieeg
metadata:
  tags: [io, bids, ieeg, seeg, ecog, sidecar, clinical]
  formats: [.tsv, .json]
  modalities: [seeg, ecog]
---
# BIDS-iEEG

## Format Overview

BIDS-iEEG (Holdgraf 2019) is the BIDS-derivative format for
intracranial EEG (sEEG / ECoG). Data files (EDF / BrainVision /
EEGLAB / NWB) are paired with sidecar **TSV/JSON** describing channels,
electrodes, events, and coordinate system. This skill covers the
sidecar layer; data-file loading defers to the corresponding skill
(EDF / FIF / etc.).

## File Manifest

```
sub-01/ses-01/ieeg/
  sub-01_ses-01_task-rest_ieeg.edf          # data
  sub-01_ses-01_task-rest_ieeg.json         # acquisition metadata
  sub-01_ses-01_task-rest_channels.tsv      # per-channel name, type, status
  sub-01_ses-01_task-rest_events.tsv        # event table
  sub-01_ses-01_task-rest_electrodes.tsv    # per-electrode x, y, z
  sub-01_ses-01_task-rest_coordsystem.json  # coordinate frame (MNI / Talairach / individual)
```

## Metadata Schema

| File | Critical fields |
|---|---|
| `*_ieeg.json` | `SamplingFrequency`, `iEEGReference`, `RecordingDuration`. |
| `*_channels.tsv` | `name`, `type` (SEEG / ECOG / ECG / EOG), `status` (good / bad), `units`, `low_cutoff`, `high_cutoff`. |
| `*_electrodes.tsv` | `name`, `x`, `y`, `z` (in coord system units). |
| `*_coordsystem.json` | `iEEGCoordinateSystem` (ACPC / MNI152NLin2009aSym / individual), `iEEGCoordinateUnits` ("m" / "mm"). |
| `*_events.tsv` | `onset` (s), `duration` (s), `trial_type`. |

## Loader Library Map

| Priority | Library |
|---|---|
| Primary | `mne-bids` (`mne_bids.read_raw_bids`) |
| Secondary | `pybids` + manual TSV parse |

## Modality Coverage

sEEG, ECoG.

## Common Pitfalls

- **Coord system mismatch.** A `MNI152` coordsystem but raw individual
  coordinates → source localization breaks. Always validate.
- **Channel status not propagated.** "bad" channels in `channels.tsv`
  must drive `drop_bads`; some loaders ignore.
- **Unit confusion.** "m" vs "mm" — BIDS recommends m (SI).
- **Sidecar JSON missing keys.** `iEEGReference` is required per spec; if
  absent the loader must error.

## Reference Implementation

### Standalone

```python
import json
import pandas as pd


def load_bids_ieeg_sidecar(subject_session_prefix: str) -> dict:
    """Read all BIDS-iEEG sidecar files."""
    out = {}
    out["ieeg_meta"] = json.loads(open(f"{subject_session_prefix}_ieeg.json").read())
    out["channels"] = pd.read_csv(f"{subject_session_prefix}_channels.tsv", sep="\t")
    out["electrodes"] = pd.read_csv(f"{subject_session_prefix}_electrodes.tsv", sep="\t")
    out["coordsystem"] = json.loads(open(f"{subject_session_prefix}_coordsystem.json").read())
    out["events"] = pd.read_csv(f"{subject_session_prefix}_events.tsv", sep="\t")
    return out
```

### EasyBCI-Adapted

```python
from typing import Any, Dict


def operator_load_bids_ieeg(
    data_dict: Dict[str, Any], *, prefix: str,
) -> Dict[str, Any]:
    """BIDS-iEEG sidecar loader.

    Parameters
    ----------
    data_dict : dict
    prefix : str
        Subject-session prefix path.

    Returns
    -------
    dict — augments meta with channels / electrodes / coordsystem / events.

    Raises
    ------
    EasyBCIOperatorError
        recoverable=False on missing sidecar files.

    Modality coverage
    -----------------
    sEEG / ECoG: yes.

    References
    ----------
    Holdgraf et al. 2019; BIDS-iEEG spec.

    Notes
    -----
    L0 IO operator (Rule 12 exempt).
    """
    import time
    # easybci-allow: file-io  # L0 IO operator
    import json
    import pandas as pd
    from easybci_lib.tools.neural_processing.operator_errors import EasyBCIOperatorError
    if not prefix:
        raise EasyBCIOperatorError(
            operator="load_bids_ieeg", reason="prefix required", recoverable=False,
        )

    t0 = time.monotonic()
    try:
        with open(f"{prefix}_ieeg.json", encoding="utf-8") as f:
            ieeg_meta = json.load(f)
        ch_df = pd.read_csv(f"{prefix}_channels.tsv", sep="\t")
        el_df = pd.read_csv(f"{prefix}_electrodes.tsv", sep="\t")
        with open(f"{prefix}_coordsystem.json", encoding="utf-8") as f:
            coord = json.load(f)
        ev_df = pd.read_csv(f"{prefix}_events.tsv", sep="\t")
    except Exception as exc:
        raise EasyBCIOperatorError(
            operator="load_bids_ieeg", reason=f"sidecar read failed: {exc}",
            recoverable=False,
        ) from exc
    elapsed = time.monotonic() - t0

    out = dict(data_dict)
    out["elapsed_s"] = elapsed
    out["meta"] = {
        **out.get("meta", {}),
        "modality": "seeg" if "SEEG" in str(ch_df.get("type", [])).upper() else "ecog",
        "bids_ieeg": ieeg_meta,
        "channels_table": ch_df.to_dict("records"),
        "electrodes_table": el_df.to_dict("records"),
        "coordsystem": coord,
        "events_s": ev_df.get("onset", pd.Series()).to_numpy(),
        "load_bids_ieeg": {"prefix": prefix},
    }
    return out
```

## References

1. Holdgraf, C. et al. (2019). *iEEG-BIDS, extending the Brain Imaging
   Data Structure for human intracranial electrophysiology*. Scientific
   Data 6: 102. doi:10.1038/s41597-019-0105-7.
2. BIDS-iEEG spec.
   https://bids-specification.readthedocs.io/en/stable/04-modality-specific-files/04-intracranial-electroencephalography.html

## Boundary with Related Formats

- **`mef3`**: long-duration clinical sEEG monolithic format; complements BIDS.
- **`nwb`**: NWB can host iEEG but rarely paired with BIDS sidecars.
- **`edf`**: most common BIDS-iEEG data file; this skill provides the metadata layer.
