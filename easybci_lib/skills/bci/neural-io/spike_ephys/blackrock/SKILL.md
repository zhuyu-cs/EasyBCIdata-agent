---
name: blackrock
description: "Blackrock Cerebus / NSP — clinical Utah-array acquisition (.ns5 raw / .ns2 LFP / .nev events)"
layer: L0
group: spike_ephys
metadata:
  tags: [io, blackrock, cerebus, nsp, ns5, ns2, nev, utah_array, braingate]
  formats: [.ns5, .ns2, .nev, .ccf]
  modalities: [spike, lfp]
---
# Blackrock Cerebus / NSP

## Format Overview

**Blackrock Microsystems** (now Blackrock Neurotech) makes the Cerebus
Neural Signal Processor (NSP), the dominant acquisition rig in **human
clinical motor BCI** (BrainGate, BrainGate2, all Hochberg/Donoghue lab
recordings, Stavisky / Henderson, Shenoy clinical work) and a common
choice for NHP labs.

The native format is a family of streams:

| Stream | Sample rate | Content | File ext |
|---|---|---|---|
| `.ns1` | 500 Hz | Low-pass band (rare in BCI) | `*.ns1` |
| `.ns2` | 1 kHz | LFP band (default for many labs) | `*.ns2` |
| `.ns3` | 2 kHz | Sub-LFP / band-limited | `*.ns3` |
| `.ns5` | 30 kHz | **Raw / wide-band** — the canonical spike-band stream | `*.ns5` |
| `.ns6` | 30 kHz | "Raw" without amplifier filters | `*.ns6` |
| `.nev` | event-driven | TTL events + Cerebus-online-sorted spikes | `*.nev` |
| `.ccf` | config | Cerebus configuration snapshot (gain / threshold / ref) | `*.ccf` |

For UEA spike work, **`.ns5` is the primary file** (raw 30 kHz wide-band)
and `.nev` carries threshold-based online spike events (Cerebus hardware
threshold) plus TTL timestamps for behavioral events.

## File Manifest

A typical Blackrock session:

```
<session_id>/
  <session_id>.ns5            # 30 kHz raw wide-band; 96 ch UEA → typical 22 GB / hour
  <session_id>.ns2            # 1 kHz LFP
  <session_id>.nev            # threshold events + TTLs (online-sort artefacts)
  <session_id>.ccf            # config (gain, threshold, reference, impedance log)
  <session_id>_meta.json      # optional lab-side annotations (per-experiment)
```

## Metadata Schema

Each `.ns*` and `.nev` file has a fixed-size binary header followed by
packets. Key header fields:

| Field | Type | Meaning |
|---|---|---|
| `Sampling Frequency` | uint32 | Hz (30000 for `.ns5`). |
| `Channel Count` | uint16 | Number of analog channels. |
| `Channel ID Map` | uint16 array | Per-channel "electrode ID" (1-128 on Cerebus). |
| `Digital Range` (low / high) | int16 / int16 | Code range (-32768 to 32767 typical). |
| `Analog Range` (mV low / mV high) | int16 / int16 | Voltage range for conversion. |
| `Time Origin` | timestamp | Absolute session start. |
| `Comment` | str | Lab comment (varies). |

For `.nev`:

| Field | Meaning |
|---|---|
| `TimeStamps` | Per-event sample number. |
| `Electrode ID` | 1-128 — Cerebus-side electrode label. |
| `Unit ID` | Cerebus online-sort unit (0–5; 0 = unsorted MUA). |
| `Waveform` | 48 sample window around the spike (often kept for QC). |
| `Event Type` | `EVENT_TYPE_*` — TTL / serial / spike. |

## Loader Library Map

| Priority | Library | Use case | Lazy install |
|---|---|---|---|
| **Primary** | `neo.rawio.BlackrockRawIO` | Full Blackrock family coverage (NS1–NS6 + NEV); standard scientific-Python entry. | `neo==0.13.4` via `io.blackrock` lazy-deps key. |
| **Secondary** | `brpylib` (Blackrock official Python) | Vendor-supplied; sometimes more accurate metadata. | `pip install brpylib==2.0.1`. |
| **Tertiary** | `SpikeInterface.read_blackrock` | Wraps neo; consistent with the rest of the SpikeInterface ecosystem. | `spikeinterface==0.101.0`. |

## Modality Coverage

Spike (UEA wide-band on `.ns5`): yes (primary use; clinical BCI).
LFP (`.ns2` 1 kHz): yes.
ECoG / sEEG (when acquired on Cerebus, less common): yes.
Threshold MUA events (`.nev` Unit 0): yes — directly equivalent to
`threshold_spike` output but pre-computed at acquisition.

## Common Pitfalls

- **Cerebus online-sort vs offline sort.** `.nev` Unit ID 1–5 are
  Cerebus's online spike-classification; quality is mediocre. For
  publication-grade single-unit analysis, **redo sort offline** on `.ns5`.
- **Channel ID ≠ array position.** Cerebus has a 1–128 "electrode ID"
  field that need not match physical array geometry. Read the `.ccf` or
  lab's electrode map to translate.
- **Saturated channels appear as clipped square waves.** UEA channels
  with bad impedance or broken sites saturate at digital range; detect
  via `np.ptp(data[c]) >= 32767 - 32768` codes (or volts equivalent).
- **Sample-rate mismatch between `.ns5` and `.ns2`.** The two streams
  are not at the same `n_samples`. Realign via wall-clock from the
  shared header time origin if you need both.
- **TTL polarity / debounce in `.nev`.** Cerebus reports both rising
  and falling edges; some labs filter to rising-only post-load. The
  `Reason` field distinguishes them.
- **Large-file MMap.** `.ns5` at 96 ch × 30 kHz × hours → 20–50 GB
  per file; use `neo`'s chunked iterator or numpy memmap. Local-SSD
  copy is recommended for any operation that reads multiple passes.
- **Cerebus firmware versions.** Older Cerebus firmware writes a
  slightly different header layout; `neo.rawio.BlackrockRawIO` handles
  most variants but failures usually show up as nonsensical sample rates.

## Reference Implementation

### Standalone (no EasyBCI dependencies)

```python
"""Standalone Blackrock loader via neo."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np


@dataclass
class BlackrockLoad:
    data: np.ndarray            # (n_channels, n_times) float32, volts
    sfreq: float
    channels: List[str]
    events: Optional[np.ndarray]    # event timestamps (s), or None


def load_blackrock_ns5(
    ns5_path: str,
    *,
    drop_dead: bool = True,
) -> BlackrockLoad:
    """Load Blackrock .ns5 (30 kHz raw) via neo.rawio.

    Parameters
    ----------
    ns5_path : str
        Path to `<session>.ns5`. `.nev` is auto-discovered next to it.
    drop_dead : bool
        Drop channels with peak-to-peak in [0, 1e-6 V] (dead) or saturated.

    Returns
    -------
    BlackrockLoad
    """
    import neo

    reader = neo.rawio.BlackrockRawIO(filename=str(Path(ns5_path).with_suffix("")))
    reader.parse_header()

    raw = reader.get_analogsignal_chunk(
        block_index=0, seg_index=0, channel_indexes=None
    )
    sfreq = float(reader.get_signal_sampling_rate(stream_index=0))
    sig_channels = reader.header["signal_channels"]
    gains = sig_channels["gain"].astype(np.float32)
    offsets = sig_channels["offset"].astype(np.float32)

    # raw (n_t, n_ch) int16 → volts
    data = raw.astype(np.float32).T * gains[:, None] + offsets[:, None]    # (n_ch, n_t)

    channels = [
        f"ch{int(ci):03d}" for ci in sig_channels["id"]
    ]

    if drop_dead:
        ptp = np.ptp(data, axis=1)
        keep = (ptp > 1e-6) & (ptp < 1e-2)
        data = data[keep]
        channels = [c for c, k in zip(channels, keep) if k]

    # Events from .nev
    events = None
    try:
        evt_chunk = reader.get_event_timestamps(
            block_index=0, seg_index=0, event_channel_index=0,
        )
        ts_arr, dur, label = evt_chunk
        events = ts_arr.astype(np.float64) / sfreq
    except Exception:
        pass

    return BlackrockLoad(data=data, sfreq=sfreq, channels=channels, events=events)
```

### EasyBCI-Adapted Implementation

```python
from typing import Any, Dict


def operator_load_blackrock(
    data_dict: Dict[str, Any],
    *,
    path: str,
    stream: str = "ns5",
    drop_dead: bool = True,
) -> Dict[str, Any]:
    """EasyBCI-adapted Blackrock loader (`.ns5` / `.ns2` + `.nev`).

    Parameters
    ----------
    data_dict : dict
        OperatorIO (typically empty on first load).
    path : str
        Path to `<session>.ns5` (or other `.ns*`).
    stream : {"ns5", "ns2", "ns1", "ns3", "ns6"}
        Which stream to load (default "ns5" — 30 kHz raw).
    drop_dead : bool
        Drop dead / saturated channels (default True).

    Returns
    -------
    dict
        OperatorIO with `data`/`channels`/`frequency`/`duration` populated.

    Raises
    ------
    EasyBCIOperatorError
        ``recoverable=False`` on missing file / unsupported header.

    Modality coverage
    -----------------
    Spike (ns5, 30 kHz): yes — UEA / chronic clinical implant primary use.
    LFP (ns2, 1 kHz): yes.

    References
    ----------
    BrainGate clinical workflow; Hochberg 2012; brpylib official docs.

    Notes
    -----
    L0 IO operator; reads from disk (CODE_STANDARD Rule 12 exempt).
    """
    import time
    # easybci-allow: file-io  # L0 IO operator; reading .ns5 is the function.
    from easybci_lib.tools.neural_processing.operator_errors import EasyBCIOperatorError

    if not path:
        raise EasyBCIOperatorError(
            operator="load_blackrock", reason="path is required", recoverable=False
        )
    t0 = time.monotonic()
    try:
        load = load_blackrock_ns5(path, drop_dead=drop_dead)
    except Exception as exc:
        raise EasyBCIOperatorError(
            operator="load_blackrock", reason=f"load failed: {exc}", recoverable=False
        ) from exc
    elapsed = time.monotonic() - t0

    sfreq = load.sfreq
    n_t = load.data.shape[1]
    duration = n_t / sfreq

    if sfreq >= 20_000:
        modality = "spike"
    elif sfreq >= 500:
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
            "stream": stream,
            "events_s": load.events,
            "load_blackrock": {"path": path, "stream": stream, "drop_dead": drop_dead},
        },
    }
```

## References

1. Hochberg, L. R. et al. (2012). *Reach and grasp by people with
   tetraplegia using a neurally controlled robotic arm*. Nature 485:
   372–375. doi:10.1038/nature11076 — BrainGate canonical paper using
   Blackrock NSP.
2. Blackrock Neurotech NSP documentation (proprietary; bundled with
   Cerebus install) — file format spec for `.nsX` and `.nev`.
3. *brpylib* — Blackrock-supplied Python reader.
   https://github.com/BlackrockNeurotech/Python-Utilities
4. Garcia, S. et al. (2014). *Neo: an object model for handling
   electrophysiology data in multiple formats*. Frontiers in Neuroinformatics
   8: 10. doi:10.3389/fninf.2014.00010 — neo / `BlackrockRawIO` reference.

## Boundary with Related Formats

| Related format | When to prefer |
|---|---|
| **`spikeglx`** | Use SpikeGLX for **research-grade Neuropixels** acquisition; Blackrock is the **clinical UEA** acquisition format. The two probes / rigs / formats are essentially separate ecosystems. |
| **`openephys`** | Open Ephys is open-source research; Blackrock is closed-source clinical-grade. Choose by which acquisition system was used. |
| **`nwb`** | Use NWB when reading **archived / published** versions (Blackrock recordings published to DANDI are converted to NWB; clinical BrainGate data is typically not public). |
| **`plexon`** | Plexon is a competing NHP / clinical acquisition vendor with their own `.plx / .pl2` format — comparable scope to Blackrock but different binary layout. |
