---
name: spikeglx
description: "SpikeGLX (.bin + .meta) — IMEC Neuropixels vendor acquisition format (AP / LF / nidq streams)"
layer: L0
group: spike_ephys
metadata:
  tags: [io, spikeglx, neuropixels, imec, npx, bin, meta]
  formats: [.bin, .meta]
  modalities: [spike, lfp]
---
# SpikeGLX

## Format Overview

**SpikeGLX** is the IMEC-recommended acquisition software for Neuropixels
probes (https://billkarsh.github.io/SpikeGLX/). The recording is laid
out as a **`.bin + .meta` binary pair**:

- `*.bin` — raw int16 samples, channel-interleaved (sample i: `[ch0, ch1, ..., chN-1]`).
- `*.meta` — ASCII key-value sidecar with sample rate, gain, channel
  map, probe geometry, file size.

Each probe / stream produces an independent pair:

| Stream | Sample rate | Typical content |
|---|---|---|
| `<run>.imec0.ap.bin/meta` | 30 kHz | Neuropixels 1.0/2.0 AP band, 384 ch |
| `<run>.imec0.lf.bin/meta` | 2.5 kHz | Neuropixels 1.0 LF band (NPx 2.0 has no LF) |
| `<run>.nidq.bin/meta` | up to 30 kHz | NI-DAQ analog/digital sync channels (TTLs, audio, behavioral) |

SpikeGLX is the **most common Neuropixels acquisition format**;
converters to NWB exist (`spikeinterface.exporters.NwbExporter`) but raw
SpikeGLX is often kept as the working format alongside NWB deposits.

## File Manifest

A standard SpikeGLX session:

```
<run_name>_g0/
  <run_name>_g0_imec0/
    <run_name>_g0_t0.imec0.ap.bin       # 30 kHz AP, 384 ch interleaved int16
    <run_name>_g0_t0.imec0.ap.meta      # sidecar
    <run_name>_g0_t0.imec0.lf.bin       # 2.5 kHz LF (NPx 1.0 only)
    <run_name>_g0_t0.imec0.lf.meta
  <run_name>_g0_t0.nidq.bin             # auxiliary NI-DAQ stream
  <run_name>_g0_t0.nidq.meta
```

For multi-probe rigs the `imec0`, `imec1`, ... directories scale per probe.

## Metadata Schema

`.meta` is a flat key-value text file. Fields required for downstream
processing:

| Key | Type | Meaning |
|---|---|---|
| `imSampRate` | float | AP / LF sample rate (Hz). |
| `niSampRate` | float | NI-DAQ stream rate (when nidq present). |
| `nSavedChans` | int | Number of channels written. |
| `imAiRangeMax`, `imAiRangeMin` | float | Volt range of int16 codes; convert via `(code / 512) · (max−min) / 65536`. |
| `imRoFile` / `~imroTbl` | str | Per-channel "readout" table: site index, bank, refElec, gainAP, gainLF. |
| `~snsChanMap` | str | Probe geometry — site x,y,z. |
| `~snsShankMap` | str | Per-shank index (NPx 2.0 4-shank only). |
| `fileTimeSecs` | float | Recording duration. |
| `fileSizeBytes` | int | Used by integrity check (file size = `nSavedChans · n_samples · 2`). |

For probe geometry distinguishing NPx 1.0 vs 2.0 single-shank vs 4-shank,
read `imDatPrb_type` (== `0` for NPx 1.0, `21` for NPx 2.0 single-shank,
`24` for 4-shank).

## Loader Library Map

| Priority | Library | Notes | Lazy install |
|---|---|---|---|
| **Primary** | `SpikeInterface.read_spikeglx` | Schema-aware; auto-loads `.meta`; geometry / probe mapping; memory-mapped reads. | `spikeinterface==0.101.0` via `io.spikeglx` lazy-deps key. |
| **Secondary** | `numpy.memmap` on `.bin` + manual `.meta` parse | When SpikeInterface install fails or you need bare-metal access. | Built-in. |
| **CatGT preprocessing wrapper** | Bill Karsh's `CatGT` CLI | Combines AP + LF into a multi-stream cat file; useful for very long sessions. | Out-of-band install. |

## Modality Coverage

Spike (AP-band): yes (primary use). LFP (LF-band): yes (NPx 1.0). NI-DAQ
auxiliary streams (TTLs, audio, force sensors): yes — load via the
nidq side. EEG / MEG / fNIRS: no.

## Common Pitfalls

- **Multi-stream confusion.** A directory like `<run>_imec0` has both
  AP and LF; `SpikeInterface.read_spikeglx(stream_id="imec0.ap")` is
  the explicit form. Bare `read_spikeglx(folder)` picks an arbitrary
  default — confirm with `available_streams`.
- **Int16 → volts conversion.** The volt-scale is in the `.meta`:
  ```
  V = (raw_int16 / 512) * (imAiRangeMax - imAiRangeMin) / 65536
  ```
  Missing the conversion produces values ~6 orders of magnitude off.
  SpikeInterface handles this automatically.
- **NPx 1.0 vs 2.0 geometry differs.** Each has a different per-channel
  x,y mapping; downstream spatial filters that assume 1.0 break on 2.0.
  Read `imDatPrb_type` to confirm.
- **Big file mmap on Windows / NFS.** Some 100–500 GB AP files break
  `np.memmap` on Windows + NTFS or on NFS-mounted storage; fall back
  to chunked reads or copy to local SSD.
- **`fileTimeSecs` vs `fileSizeBytes` mismatch.** When recording is
  interrupted, the meta sometimes reports a duration the bin file
  doesn't have. Always verify `n_samples = filesize / (nSavedChans * 2)`.
- **Probe banks / sub-selections.** NPx 1.0 has 960 sites but only 384
  simultaneously recorded; the `imRoFile` specifies which 384. Don't
  assume contiguous site indices.

## Reference Implementation

### Standalone (no EasyBCI dependencies)

```python
"""Standalone SpikeGLX loader using only numpy + stdlib."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np


@dataclass
class SpikeGLXLoad:
    data: np.ndarray            # (n_channels, n_times) float32, volts
    sfreq: float
    channels: list[str]
    meta: Dict[str, str]


def parse_meta(meta_path: str) -> Dict[str, str]:
    """Parse SpikeGLX .meta into a flat str-keyed dict."""
    out: Dict[str, str] = {}
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def load_spikeglx(
    bin_path: str,
    *,
    stream: str = "ap",
    start_s: float = 0.0,
    stop_s: Optional[float] = None,
) -> SpikeGLXLoad:
    """Load a SpikeGLX .bin via numpy.memmap; convert int16 → volts.

    Parameters
    ----------
    bin_path : str
        Path to `.bin` file. The `.meta` sidecar is auto-located.
    stream : {"ap", "lf", "nidq"}
        Cosmetic only — used for channel-name prefix.
    start_s, stop_s : float
        Optional time range (seconds). stop_s=None reads to end.

    Returns
    -------
    SpikeGLXLoad
    """
    bin_p = Path(bin_path)
    meta_p = bin_p.with_suffix(".meta")
    if not meta_p.exists():
        raise FileNotFoundError(f"Missing meta sidecar: {meta_p}")
    meta = parse_meta(str(meta_p))

    n_ch = int(meta["nSavedChans"])
    sfreq = float(meta.get("imSampRate") or meta.get("niSampRate"))
    if not sfreq:
        raise ValueError("meta missing imSampRate / niSampRate")

    n_total = bin_p.stat().st_size // (n_ch * 2)
    s0 = int(start_s * sfreq)
    s1 = int(stop_s * sfreq) if stop_s is not None else n_total
    if s1 > n_total:
        s1 = n_total
    if s0 >= s1:
        raise ValueError(f"start_s={start_s} >= stop_s={stop_s}")

    raw = np.memmap(bin_p, dtype=np.int16, mode="r", shape=(n_total, n_ch))
    arr = np.asarray(raw[s0:s1], dtype=np.float32).T  # (n_ch, n_t)

    # int16 → volts via meta-declared analog range
    vmax = float(meta.get("imAiRangeMax", "0.6"))     # typical NPx 1.0
    vmin = float(meta.get("imAiRangeMin", "-0.6"))
    arr = (arr / 512.0) * (vmax - vmin) / 65536.0

    channels = [f"{stream}_{i}" for i in range(n_ch)]
    return SpikeGLXLoad(data=arr, sfreq=sfreq, channels=channels, meta=meta)
```

### EasyBCI-Adapted Implementation

```python
from typing import Any, Dict, Optional


def operator_load_spikeglx(
    data_dict: Dict[str, Any],
    *,
    path: str,
    stream: str = "ap",
    start_s: float = 0.0,
    stop_s: Optional[float] = None,
) -> Dict[str, Any]:
    """EasyBCI-adapted SpikeGLX loader.

    Parameters
    ----------
    data_dict : dict
        OperatorIO (typically empty on first load).
    path : str
        Path to `<run>_g0_t0.imec0.<stream>.bin`.
    stream : {"ap", "lf", "nidq"}
        Stream kind (cosmetic; for channel naming + modality inference).
    start_s, stop_s : float
        Optional read window (seconds).

    Returns
    -------
    dict
        OperatorIO with `data`/`channels`/`frequency`/`duration` populated.

    Raises
    ------
    EasyBCIOperatorError
        ``recoverable=False`` on missing .meta sidecar or schema error.

    Modality coverage
    -----------------
    Spike (AP, sfreq >= 20 kHz): yes. LFP (LF, ~2.5 kHz): yes. NI-DAQ aux: yes.

    References
    ----------
    Karsh / IMEC SpikeGLX docs; Steinmetz et al. 2021 (NPx 2.0).

    Notes
    -----
    L0 IO operator; reads from disk (CODE_STANDARD Rule 12 exempt).
    """
    import time
    # easybci-allow: file-io  # L0 IO operator; reading .bin is the function.
    from easybci_lib.tools.neural_processing.operator_errors import EasyBCIOperatorError

    if not path:
        raise EasyBCIOperatorError(
            operator="load_spikeglx", reason="path is required", recoverable=False
        )
    t0 = time.monotonic()
    try:
        load = load_spikeglx(path, stream=stream, start_s=start_s, stop_s=stop_s)
    except Exception as exc:
        raise EasyBCIOperatorError(
            operator="load_spikeglx", reason=f"load failed: {exc}", recoverable=False
        ) from exc
    elapsed = time.monotonic() - t0

    sfreq = load.sfreq
    n_t = load.data.shape[1]
    duration = n_t / sfreq

    if sfreq >= 20_000 and stream.lower() == "ap":
        modality = "spike"
    elif sfreq >= 1500 and stream.lower() == "lf":
        modality = "lfp"
    else:
        modality = "auxiliary"

    return {
        "data": load.data,
        "channels": load.channels,
        "frequency": sfreq,
        "duration": duration,
        "elapsed_s": elapsed,
        "meta": {
            "modality": modality,
            "stream": stream,
            "spikeglx_meta": load.meta,
            "load_spikeglx": {"path": path, "start_s": start_s, "stop_s": stop_s},
        },
    }
```

## References

1. Karsh, B. *SpikeGLX*. https://billkarsh.github.io/SpikeGLX/ — official
   acquisition software + format spec.
2. Buccino, A. P. et al. (2020). *SpikeInterface, a unified framework for
   spike sorting*. eLife 9: e61834. doi:10.7554/eLife.61834 — the
   `read_spikeglx` reader.
3. Steinmetz, N. A. et al. (2021). *Neuropixels 2.0: a miniaturized
   high-density probe for stable, long-term brain recordings*.
   Science 372: eabf4588. doi:10.1126/science.abf4588 — 2.0 probe geometry.
4. International Brain Laboratory (2022). *Reproducibility of in vivo
   electrophysiological measurements in mice*. bioRxiv.
   doi:10.1101/2022.05.09.491042 — IBL pipeline uses SpikeGLX as the
   working format pre-DANDI deposit.

## Boundary with Related Formats

| Related format | When to prefer |
|---|---|
| **`nwb`** | Use when the data has been **archived / published** — DANDI / Allen / IBL public deposits are NWB; SpikeGLX is the working / acquisition format. |
| **`openephys`** | Open Ephys is a different open-source acquisition stack with its own binary format (`continuous.dat`); not interchangeable with SpikeGLX. |
| **`blackrock`** (`.ns5`) | Blackrock is the **clinical UEA** acquisition format; SpikeGLX is the **research Neuropixels** acquisition format. Use the one matching the recording rig. |
