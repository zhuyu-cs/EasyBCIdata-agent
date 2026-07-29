---
name: openephys
description: "Open Ephys GUI Record Node output (continuous.dat / structure.oebin) — open-source ephys acquisition"
layer: L0
group: spike_ephys
metadata:
  tags: [io, openephys, oebin, continuous, ephys, open_source]
  formats: [.dat, .oebin, .npy, .xml]
  modalities: [spike, lfp, ecog]
---
# Open Ephys

## Format Overview

**Open Ephys** is the leading open-source ephys acquisition stack
(https://open-ephys.org/), running on commodity hardware (Intan RHD2000
amplifiers, Acquisition Board, Neuropixels via the IMEC plugin). It
writes data via a **Record Node** in one of two binary layouts:

- **Binary format (0.6+)** — `continuous.dat` + JSON-based
  `structure.oebin` manifest. **Current default since ~2021.**
- **Open Ephys Format (legacy, < 0.6)** — `.continuous` per channel +
  `messages.events` + `settings.xml`. Still found in archived datasets
  but deprecated.

The binary format is multi-stream — one `continuous/<stream_id>/`
directory per probe / amplifier, mirroring SpikeGLX's split.

## File Manifest

Binary format (0.6+):

```
experiment1/
  recording1/
    structure.oebin                     # JSON manifest (REQUIRED for SpikeInterface readers)
    sync_messages.txt
    continuous/
      Neuropix-PXI-100.ProbeA-AP/
        continuous.dat                  # raw int16, ch-interleaved, 30 kHz AP
        sample_numbers.npy
        timestamps.npy
      Neuropix-PXI-100.ProbeA-LFP/
        continuous.dat                  # 2.5 kHz LFP
        sample_numbers.npy
        timestamps.npy
    events/
      Neuropix-PXI-100.ProbeA-AP/TTL/
        states.npy
        sample_numbers.npy
        full_words.npy
      Message_Center-904/text_messages/
        text.npy
        sample_numbers.npy
```

Legacy format (< 0.6):

```
experiment1/recording1/
  100_CH1.continuous                    # one file per channel
  100_CH2.continuous
  ...
  all_channels.events                   # TTL events
  settings.xml                          # acquisition config
  messages.events                       # text messages
```

## Metadata Schema

The `structure.oebin` JSON manifest is the canonical metadata source.
Required fields:

| JSON path | Type | Meaning |
|---|---|---|
| `continuous[i].folder_name` | str | Subdir under `continuous/` containing `continuous.dat`. |
| `continuous[i].sample_rate` | float | Hz. |
| `continuous[i].num_channels` | int | Channel count (data is ch-interleaved). |
| `continuous[i].channels[j].name` | str | Per-channel label. |
| `continuous[i].channels[j].bit_volts` | float | int16 → volts conversion factor. |
| `continuous[i].channels[j].units` | str | "uV" or "V"; multiply accordingly. |
| `events[i].folder_name` | str | TTL events directory. |
| `events[i].sample_rate` | float | Should match continuous stream. |

`structure.oebin` exists in 0.6+. Legacy `.continuous` per-channel files
carry their own embedded header — see `open-ephys-python-tools.OpenEphys.loadContinuous`.

## Loader Library Map

| Priority | Library | Use case | Lazy install |
|---|---|---|---|
| **Primary** | `SpikeInterface.read_openephys` | Schema-aware; handles 0.6+ binary + legacy; auto-detects stream. | `spikeinterface==0.101.0` via `io.openephys` lazy-deps key. |
| **Secondary** | `open-ephys-python-tools` (PyPI `open-ephys-python-tools`) | Open Ephys' own reader; same coverage as SpikeInterface. | `pip install open-ephys-python-tools==0.1.10`. |
| **Tertiary** | `np.memmap` on `continuous.dat` + manual `.oebin` JSON parse | Headless / minimal-dep environments. | Built-in. |

## Modality Coverage

Spike (AP-band, Intan / Neuropixels-via-IMEC-plugin): yes.
LFP (Intan dedicated LFP board / NPx LF): yes.
ECoG (Intan with ECoG headstages): yes.
EEG / MEG / fNIRS: no.

## Common Pitfalls

- **Schema-version drift (0.5 ↔ 0.6 ↔ 0.6.x).** Field names in the
  legacy XML and current JSON differ; if SpikeInterface throws a
  `KeyError` on read, confirm the GUI version that wrote the recording.
- **Multi-stream confusion.** Binary 0.6+ has parallel
  `continuous/<stream>/` dirs. Calling `read_openephys` without
  `stream_id=...` picks one arbitrarily.
- **Channel `bit_volts` not always 0.195 µV.** Intan-default amplifiers
  are 0.195 µV/bit; Neuropixels-via-IMEC-plugin uses ~2.34 µV/bit at
  default gain. Always read `bit_volts` from `.oebin`.
- **Timestamps vs sample_numbers.** `sample_numbers.npy` is monotonic
  sample indices; `timestamps.npy` is wall-clock (when synchronized
  with the NI-DAQ via the Sync plugin). Use `sample_numbers` for
  intra-stream alignment, `timestamps` only for cross-rig sync.
- **TTL events with `states` array.** `states.npy` is +/− channel IDs:
  positive means rising edge, negative means falling. Reconstruct
  `(channel, rising_or_falling, sample_number)` from the trio.
- **Recording across multiple `recordingX` directories.** Open Ephys
  splits long sessions into 1 GB chunks (default; configurable). Load
  with SpikeInterface which concatenates transparently.

## Reference Implementation

### Standalone (no EasyBCI dependencies)

```python
"""Standalone Open Ephys binary-format loader (0.6+)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np


@dataclass
class OpenEphysLoad:
    data: np.ndarray             # (n_channels, n_times) float32, volts
    sfreq: float
    channels: List[str]
    bit_volts: List[float]
    stream_id: str


def load_openephys_binary(
    recording_dir: str,
    *,
    stream_substr: str = "AP",
) -> OpenEphysLoad:
    """Read Open Ephys binary-format recording (0.6+).

    Parameters
    ----------
    recording_dir : str
        Path to `experimentX/recordingY/` containing `structure.oebin`.
    stream_substr : str
        Substring matched against `continuous[i].folder_name` to pick
        the stream (e.g., "AP" for Neuropixels AP band).

    Returns
    -------
    OpenEphysLoad
    """
    rec_p = Path(recording_dir)
    oebin = json.loads((rec_p / "structure.oebin").read_text(encoding="utf-8"))

    stream = next(
        (c for c in oebin["continuous"] if stream_substr in c["folder_name"]),
        oebin["continuous"][0],
    )
    stream_dir = rec_p / "continuous" / stream["folder_name"]
    n_ch = int(stream["num_channels"])
    sfreq = float(stream["sample_rate"])
    bit_volts = [float(ch["bit_volts"]) for ch in stream["channels"]]
    channels = [ch["channel_name"] for ch in stream["channels"]]

    dat_p = stream_dir / "continuous.dat"
    n_total = dat_p.stat().st_size // (n_ch * 2)
    raw = np.memmap(dat_p, dtype=np.int16, mode="r", shape=(n_total, n_ch))
    arr = np.asarray(raw, dtype=np.float32).T            # (n_ch, n_t)

    # int16 → volts (assume "uV" units → multiply then ÷ 1e6 to land in V)
    conv = np.asarray(bit_volts, dtype=np.float32)[:, None] / 1e6
    arr = arr * conv

    return OpenEphysLoad(
        data=arr, sfreq=sfreq, channels=channels,
        bit_volts=bit_volts, stream_id=stream["folder_name"],
    )
```

### EasyBCI-Adapted Implementation

```python
from typing import Any, Dict


def operator_load_openephys(
    data_dict: Dict[str, Any],
    *,
    path: str,
    stream_substr: str = "AP",
) -> Dict[str, Any]:
    """EasyBCI-adapted Open Ephys binary-format loader.

    Parameters
    ----------
    data_dict : dict
        OperatorIO (typically empty on first load).
    path : str
        Path to `experimentX/recordingY/` directory.
    stream_substr : str
        Stream selector; matches against `folder_name` (default "AP").

    Returns
    -------
    dict
        OperatorIO with `data`/`channels`/`frequency`/`duration` populated.

    Raises
    ------
    EasyBCIOperatorError
        ``recoverable=False`` on missing `structure.oebin` / corrupt manifest.

    Modality coverage
    -----------------
    Spike (AP, sfreq >= 20 kHz): yes. LFP (LF, ~2.5 kHz): yes. ECoG (Intan): yes.

    References
    ----------
    Siegle et al. 2017 (Open Ephys); GUI v0.6 binary format docs.

    Notes
    -----
    L0 IO operator; reads from disk (CODE_STANDARD Rule 12 exempt).
    """
    import time
    # easybci-allow: file-io  # L0 IO operator; reading .dat is the function.
    from easybci_lib.tools.neural_processing.operator_errors import EasyBCIOperatorError

    if not path:
        raise EasyBCIOperatorError(
            operator="load_openephys", reason="path is required", recoverable=False
        )
    t0 = time.monotonic()
    try:
        load = load_openephys_binary(path, stream_substr=stream_substr)
    except Exception as exc:
        raise EasyBCIOperatorError(
            operator="load_openephys", reason=f"load failed: {exc}", recoverable=False
        ) from exc
    elapsed = time.monotonic() - t0

    sfreq = load.sfreq
    n_t = load.data.shape[1]
    duration = n_t / sfreq

    if sfreq >= 20_000:
        modality = "spike"
    elif sfreq >= 1500:
        modality = "lfp"
    else:
        modality = "ecog"

    return {
        "data": load.data,
        "channels": load.channels,
        "frequency": sfreq,
        "duration": duration,
        "elapsed_s": elapsed,
        "meta": {
            "modality": modality,
            "stream_id": load.stream_id,
            "bit_volts": load.bit_volts,
            "load_openephys": {"path": path, "stream_substr": stream_substr},
        },
    }
```

## References

1. Siegle, J. H. et al. (2017). *Open Ephys: an open-source, plugin-based
   platform for multichannel electrophysiology*. J. Neural Eng. 14(4):
   045003. doi:10.1088/1741-2552/aa5eea — Open Ephys software paper.
2. Open Ephys GUI Binary Format Spec.
   https://open-ephys.github.io/gui-docs/User-Manual/Recording-data/Binary-format.html
3. Buccino, A. P. et al. (2020). *SpikeInterface, a unified framework
   for spike sorting*. eLife 9: e61834. doi:10.7554/eLife.61834 —
   `read_openephys` reader.
4. Open Ephys Python Tools. https://github.com/open-ephys/open-ephys-python-tools

## Boundary with Related Formats

| Related format | When to prefer |
|---|---|
| **`spikeglx`** | Use SpikeGLX when the acquisition rig is the IMEC-recommended SpikeGLX software stack. Open Ephys is the alternative open-source stack — both can drive Neuropixels but via different software. |
| **`nwb`** | Use NWB when reading **archived / published** data (Open Ephys recordings published to DANDI are converted to NWB; the working format on the rig was Open Ephys). |
| **`blackrock`** | Blackrock is the clinical UEA acquisition format; Open Ephys is open-source research. Mutually exclusive in practice. |
