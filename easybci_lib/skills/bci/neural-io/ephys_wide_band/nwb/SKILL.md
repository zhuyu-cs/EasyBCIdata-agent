---
name: nwb
description: "Neurodata Without Borders (.nwb) — the de-facto standard for in-vivo electrophysiology (Neuropixels / AIBS / DANDI / IBL)"
layer: L0
group: ephys_wide_band
metadata:
  tags: [io, nwb, neurodata, pynwb, dandi, ibl, allen, neuropixels, hdf5]
  formats: [.nwb]
  modalities: [spike, lfp, eeg, ecog, seeg]
---
# NWB — Neurodata Without Borders

## Format Overview

NWB (Neurodata Without Borders) is the **de-facto data standard** for
in-vivo extracellular electrophysiology and adjacent neural data
(optical imaging, behavioral video, intracellular). The current schema
(NWB 2.x) is maintained by the NWB consortium and is the official
deposit format for:

- **DANDI Archive** (NIH-funded; every IBL session, Allen Visual Coding,
  most US-funded labs after 2022 deposit here).
- **Allen Institute Brain Observatory** datasets (Visual Coding /
  Visual Behavior — all Neuropixels sessions).
- **International Brain Laboratory (IBL) Brain-Wide Map**.
- **Most rodent/NHP labs publishing Neuropixels data 2020+.**

Internally NWB 2.x is a **structured HDF5** file: tree of groups +
typed datasets + attached schema metadata (the `neurodata_type` attr).
Earlier NWB 1.x is a different layout — `pynwb` reads both transparently.

Versions seen in practice:
- **NWB 2.5+** — current standard; ships with pynwb 2.5+.
- **NWB 2.0–2.4** — older deposits; `pynwb` 2.x reads them.
- **NWB 1.x** — pre-2018 legacy; rare; requires the `pynwb.legacy` shim.

## File Manifest

NWB is **single-file** — no required sidecars. A typical deposit:

```
sub-001_ses-01_ecephys.nwb           # 1–500 GB, monolithic HDF5
sub-001_ses-01_ecephys.json          # optional DANDI-side metadata mirror
sub-001_ses-01_video.mp4             # optional behavioral video, separate file
```

For IBL / Allen-style multi-modal datasets, multiple `*.nwb` per session
are common (one per "stream" — ephys, behavior, imaging). Treat each as
an independent loader call.

## Metadata Schema

Required reads — confirm these populate before downstream processing:

| Group / dataset | Type | Use |
|---|---|---|
| `/general/subject` | group | Species, age, sex, genotype. |
| `/general/devices/*` | group | Probe / amp identifiers. |
| `/general/extracellular_ephys/electrode_groups/*` | group | Per-probe / per-shank channel groupings. |
| `/general/extracellular_ephys/electrodes` (dynamic table) | table | **Per-channel** location / impedance / brain region. Row count must match `data` second axis. |
| `/acquisition/ElectricalSeries*` | NWBSeries | Raw / minimally-processed voltage. **Multiple are common**: `ElectricalSeriesAP*` (AP band) and `ElectricalSeriesLF*` (LFP band) for Neuropixels. |
| `/acquisition/SpikeEventSeries*` | NWBSeries | Threshold-detected event times (when sort is not yet done). |
| `/acquisition/Events*` | EventSeries | Behavioral / stimulus event TTLs. |
| `/intervals/trials` | TimeIntervals | Trial start / stop / outcome. |
| `/intervals/epochs` | TimeIntervals | Session-level state epochs (sleep / wake / task). |
| `/units` | Units dynamic table | **Post-sort** unit table — `spike_times`, `spike_times_index`, `electrodes`. Present iff sorting was already done. |
| `/processing/ecephys/*` | ProcessingModule | Derived signals (filtered copies, spike trains, LFP envelope). **Do NOT re-process these**. |
| `/session_start_time` | datetime | Absolute timestamp for cross-modal alignment. |

Schema-implicit fields:

- `ElectricalSeries.rate` is sampling rate in Hz.
- `ElectricalSeries.conversion` × raw int16 = voltage in volts.
- `ElectricalSeries.electrodes` is a `DynamicTableRegion` referencing the
  `/general/extracellular_ephys/electrodes` table — that's how each
  data column maps to a physical channel.

## Loader Library Map

| Priority | Library | Use case | Lazy install |
|---|---|---|---|
| **Primary** | `pynwb` ≥ 2.5 | Full NWB schema awareness; `units/` table reading; cross-references. | `easybci_lib/tools/lazy_deps.py` key `io.nwb` → `pynwb==3.1.3` + `hdf5plugin==4.4.0`. |
| **Secondary** | `h5py` + `hdf5plugin` | Raw HDF5 reads when you only need one dataset and don't want pynwb's overhead. | Already in EasyBCI base deps via numpy/h5py path. |
| **Avoid** | `mne.io.read_raw_nwb` | Legacy MNE adapter; ignores `units/` table, misses multi-stream `ElectricalSeries`. | — |

When to fall back to h5py instead of pynwb:

- File is corrupted at the NWB-schema level but HDF5 itself is valid
  (rare; `pynwb.NWBHDF5IO` raises a `SchemaError`).
- You only need a single dataset and the file is multi-hundred-GB
  (`pynwb` constructs the full object graph at open; `h5py` is lazy).
- The file uses NWB 1.x with a missing legacy shim.

`hdf5plugin` is **required** to read Neuropixels NWBs that use the GZIP
+ Bitshuffle codec for AP-band compression — without it the open
succeeds but `electrical_series.data[:]` raises a filter error.

## Modality Coverage

| Modality | Stream typically present | Notes |
|---|---|---|
| Spike (Neuropixels) | `acquisition/ElectricalSeriesAP*` 30 kHz | Default for in-vivo extracellular. |
| LFP (Neuropixels) | `acquisition/ElectricalSeriesLF*` 2.5 kHz | Parallel to AP; often co-deposited. |
| LFP (UEA / depth) | `acquisition/ElectricalSeries` 1–2 kHz | Single-band; check `rate`. |
| ECoG (human or NHP) | `acquisition/ElectricalSeries` 1–5 kHz | Often with `electrodes/group` flagged "ECoG". |
| EEG (rare in NWB; usually EDF) | `acquisition/ElectricalSeries` 250–1000 Hz | OK to load but EDF is more common. |
| sEEG (depth) | `acquisition/ElectricalSeries` 2 kHz | Some clinical archives use NWB. |
| Imaging (Ca / 2P) | `acquisition/TwoPhotonSeries` | Out of scope — use `pynwb.image` adapters. |
| Behavior / video | `acquisition/ImageSeries` or `processing/behavior` | Out of scope — load separately. |

## Common Pitfalls

- **Multiple `ElectricalSeries`.** Neuropixels NWBs ship `*AP*` (30 kHz)
  and `*LF*` (2.5 kHz) as separate `ElectricalSeries`. Calling
  `read_nwb_recording` without `electrical_series_name=...` selects an
  arbitrary one — confirm via `inspect_neural` first.
- **`electrodes` table row count ≠ data column count.** Some pipelines
  drop channels from `data` without updating the `electrodes` table.
  Always read `electrodes` shape AND `data.shape[1]`; reconcile.
- **`session_start_time` vs trial timestamps.** Trial / event timestamps
  in NWB are **seconds relative to `session_start_time`**. Aligning to
  wall-clock requires reading `session_start_time` and offsetting.
- **`units/spike_times` exists → don't re-sort.** Pre-sorted NWBs have
  the `/units` table populated. Re-running spike sort wastes hours and
  produces a parallel, inconsistent result. Detect via:
  ```python
  has_units = "/units" in h5py.File(path)
  ```
- **Threshold-only NWBs lack `/units` but have `acquisition/SpikeEventSeries`.**
  In these cases there are spike *event times* per channel but no unit
  labels. Use `threshold_spike → mua_binning` path, not `spike_sort`.
- **`processing/ecephys/Filtered*` is a derived copy.** Do not re-bandpass
  this stream — it has already been filtered.
- **`conversion` factor units.** `data` is often raw int16; multiplying
  by `electrical_series.conversion` gives volts. Forgetting the
  conversion gives data ~6 orders of magnitude off (silent; no error).
- **HDF5 filter plugins.** Without `hdf5plugin` imported, GZIP+BitShuffle
  compressed Neuropixels NWBs throw a filter error on `.data[:]`. The
  fix is `import hdf5plugin` (yes, just importing — no function call —
  it registers the filter with libhdf5).

## Reference Implementation

### Standalone (no EasyBCI dependencies)

```python
"""Standalone NWB → (data, sfreq, channels, events) loader.

Reads the AP-band ElectricalSeries (or first one if no AP stream) and any
present /units table. Suitable for headless / no-pynwb environments via
h5py fallback.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class NWBLoad:
    data: np.ndarray              # (n_channels, n_times) float32
    sfreq: float
    channels: List[str]
    units: Optional[Dict[int, np.ndarray]]   # unit_id → spike_times_s, None if no /units
    events: Optional[np.ndarray]             # (n_events,) seconds, None if no events
    electrode_table: Dict[str, np.ndarray]   # per-column arrays from /general/.../electrodes
    session_start: Optional[str]             # ISO datetime


def load_nwb(
    path: str,
    *,
    electrical_series_name: Optional[str] = None,
    prefer_lib: str = "pynwb",
) -> NWBLoad:
    """Load an NWB file into a flat record.

    Parameters
    ----------
    path : str
        Path to .nwb file.
    electrical_series_name : str or None
        Name of the ElectricalSeries to read. If None, prefers a name
        containing "AP" (Neuropixels AP band) else the first found.
    prefer_lib : {"pynwb", "h5py"}
        Which library to try first. Falls back to the other on schema
        / import error.

    Returns
    -------
    NWBLoad
    """
    if prefer_lib == "pynwb":
        try:
            return _load_via_pynwb(path, electrical_series_name)
        except Exception as e:
            logger.warning("pynwb load failed (%s); falling back to h5py", e)
            return _load_via_h5py(path, electrical_series_name)
    return _load_via_h5py(path, electrical_series_name)


def _load_via_pynwb(path, name):
    try:
        import hdf5plugin  # noqa: F401  — registers GZIP+Bitshuffle codec
    except ImportError:
        logger.warning("hdf5plugin not installed; may fail on compressed Neuropixels NWB")
    from pynwb import NWBHDF5IO

    with NWBHDF5IO(path, "r", load_namespaces=True) as io:
        nwb = io.read()

        # Pick ElectricalSeries
        candidates = list(nwb.acquisition.values())
        es_list = [c for c in candidates if c.neurodata_type == "ElectricalSeries"]
        if not es_list:
            raise ValueError("No ElectricalSeries in /acquisition")
        chosen = None
        if name is not None:
            chosen = next((c for c in es_list if c.name == name), None)
        if chosen is None:
            chosen = next((c for c in es_list if "AP" in c.name), es_list[0])

        data = chosen.data[:].T.astype(np.float32)           # (n_ch, n_t)
        if getattr(chosen, "conversion", 1.0) and chosen.conversion != 1.0:
            data = data * float(chosen.conversion)
        sfreq = float(chosen.rate)

        # Electrode table → channel names + structured columns
        et = nwb.electrodes.to_dataframe() if nwb.electrodes is not None else None
        if et is not None and len(et) == data.shape[0]:
            channels = [f"{loc}_{i}" for i, loc in enumerate(et.get("location", [""] * len(et)))]
            electrode_table = {col: et[col].to_numpy() for col in et.columns}
        else:
            channels = [f"ch{i}" for i in range(data.shape[0])]
            electrode_table = {}

        # Units (post-sort)
        units = None
        if nwb.units is not None and len(nwb.units) > 0:
            units = {}
            df = nwb.units.to_dataframe()
            for uid, row in df.iterrows():
                units[int(uid)] = np.asarray(row["spike_times"], dtype=np.float64)

        # First event TTL series (best-effort)
        events = None
        for series in nwb.acquisition.values():
            if series.neurodata_type in ("EventSeries", "TimeSeries"):
                events = np.asarray(series.timestamps[:], dtype=np.float64)
                break

        session_start = (
            nwb.session_start_time.isoformat()
            if getattr(nwb, "session_start_time", None) is not None
            else None
        )

        return NWBLoad(
            data=data,
            sfreq=sfreq,
            channels=channels,
            units=units,
            events=events,
            electrode_table=electrode_table,
            session_start=session_start,
        )


def _load_via_h5py(path, name):
    try:
        import hdf5plugin  # noqa: F401
    except ImportError:
        pass
    import h5py

    with h5py.File(path, "r") as f:
        # Find ElectricalSeries
        acq = f["acquisition"]
        es_names = []
        for k in acq.keys():
            attr = acq[k].attrs.get("neurodata_type", b"").decode() if isinstance(acq[k].attrs.get("neurodata_type", ""), bytes) else acq[k].attrs.get("neurodata_type", "")
            if attr == "ElectricalSeries":
                es_names.append(k)
        if not es_names:
            raise ValueError("No ElectricalSeries in /acquisition")
        if name is None:
            chosen = next((n for n in es_names if "AP" in n), es_names[0])
        else:
            chosen = name
        es = acq[chosen]

        data = es["data"][:].T.astype(np.float32)
        sfreq = float(es["starting_time"].attrs.get("rate", es.attrs.get("rate", 30000.0)))
        conv = float(es.attrs.get("conversion", 1.0))
        if conv != 1.0:
            data = data * conv

        n_ch = data.shape[0]
        channels = [f"ch{i}" for i in range(n_ch)]

        # Units, if any
        units = None
        if "units" in f and "spike_times" in f["units"]:
            t = f["units/spike_times"][:]
            idx = f["units/spike_times_index"][:]
            units = {}
            prev = 0
            for uid, end in enumerate(idx):
                units[uid] = t[prev:int(end)]
                prev = int(end)

        return NWBLoad(
            data=data, sfreq=sfreq, channels=channels,
            units=units, events=None, electrode_table={}, session_start=None,
        )
```

### EasyBCI-Adapted Implementation

```python
from typing import Any, Dict, Optional

import numpy as np

from easybci_lib.tools.neural_processing.operator_errors import EasyBCIOperatorError


def operator_load_nwb(
    data_dict: Dict[str, Any],
    *,
    path: str,
    electrical_series_name: Optional[str] = None,
) -> Dict[str, Any]:
    """EasyBCI-adapted NWB loader.

    Parameters
    ----------
    data_dict : dict
        OperatorIO (typically empty on first load).
    path : str
        Path to .nwb file.
    electrical_series_name : str or None
        Specific ElectricalSeries name (default: prefer "AP" then first).

    Returns
    -------
    dict
        OperatorIO populated with:
        - ``data``: ndarray (n_channels, n_times) float32, voltage (V)
        - ``channels``: list[str] from electrodes table
        - ``frequency``: float sampling rate (Hz)
        - ``duration``: float seconds
        - ``meta["modality"]``: inferred from sfreq + name ("spike" if AP+>=20kHz else "lfp"/"eeg")
        - ``meta["units"]``: dict[unit_id → spike_times_s] if /units present
        - ``meta["electrode_table"]``: dict of per-column ndarrays
        - ``meta["session_start_time"]``: ISO datetime string or None

    Raises
    ------
    EasyBCIOperatorError
        ``recoverable=False`` on missing ElectricalSeries, schema corruption.

    Modality coverage
    -----------------
    Spike (Neuropixels / Utah-via-Blackrock-converted): yes.
    LFP / ECoG / sEEG / EEG-stored-as-NWB: yes (loads as continuous voltage).
    Optical imaging / behavior video: out of scope (use dedicated loaders).

    References
    ----------
    Rübel 2022 (NWB:N 2.0); Teeters 2015 (original NWB).

    Notes
    -----
    This operator reads from disk and is therefore exempt from CODE_STANDARD
    Rule 12 (no-file-io). See the comment marker below.
    """
    # easybci-allow: file-io  # L0 IO operator; reading raw .nwb is the function.
    if not path:
        raise EasyBCIOperatorError(
            operator="load_nwb",
            reason="path is required",
            recoverable=False,
        )

    import time
    t0 = time.monotonic()
    try:
        load = load_nwb(path, electrical_series_name=electrical_series_name, prefer_lib="pynwb")
    except Exception as exc:
        raise EasyBCIOperatorError(
            operator="load_nwb",
            reason=f"load failed: {exc}",
            recoverable=False,
        ) from exc
    elapsed = time.monotonic() - t0

    sfreq = float(load.sfreq)
    n_t = int(load.data.shape[1])
    duration = n_t / sfreq if sfreq > 0 else 0.0

    # Modality inference
    if sfreq >= 20_000:
        modality = "spike"
    elif sfreq >= 1500:
        modality = "lfp"
    else:
        modality = "eeg"

    return {
        "data": load.data,
        "channels": load.channels,
        "frequency": sfreq,
        "duration": duration,
        "elapsed_s": elapsed,
        "meta": {
            "modality": modality,
            "units": load.units,
            "electrode_table": load.electrode_table,
            "session_start_time": load.session_start,
            "load_nwb": {
                "path": path,
                "electrical_series_name": electrical_series_name,
            },
        },
    }
```

## References

1. Rübel, O. et al. (2022). *The Neurodata Without Borders ecosystem for
   neurophysiological data science*. eLife 11: e78362.
   doi:10.7554/eLife.78362 — the current NWB 2.x reference.
2. Teeters, J. L. et al. (2015). *Neurodata Without Borders: Creating a
   Common Data Format for Neurophysiology*. Neuron 88(4): 629–634.
   doi:10.1016/j.neuron.2015.10.025 — the original NWB design paper.
3. International Brain Laboratory (2022). *Open Neurophysiology
   Environment*. https://int-brain-lab.github.io/ONE/ — IBL public data
   layer; NWB-on-DANDI deposits.
4. NWB Schema (current). https://nwb-schema.readthedocs.io/ — the
   authoritative neurodata_type / field reference.
5. DANDI Archive. https://dandiarchive.org/ — NIH-funded NWB deposit
   portal.

## Boundary with Related Formats

| Related format | When to prefer |
|---|---|
| **`spikeglx`** (`.bin + .meta`) | Use when you have **raw Neuropixels acquisition output** that has not yet been converted to NWB. SpikeGLX is the IMEC vendor's native format; NWB is the archive-canonical format. The conversion is one-way (SpikeGLX → NWB via `spikeinterface.NwbExporter`). |
| **`openephys`** | Open Ephys is an open-source acquisition stack; deposits also flow into DANDI as NWB. Use openephys IO when reading from the GUI's `Record Node` output before conversion. |
| **`blackrock`** (`.ns5`) | Blackrock Cerebus / NSP is the clinical UEA acquisition format. Conversion to NWB exists (`spikeinterface.exporters.NwbExporter`) but is **not** the default in BrainGate-style pipelines — raw `.ns5` is more common. |
| **`bids_ieeg`** | BIDS-iEEG sidecar (`*_ieeg.tsv` etc.) is a research-clinical format. NWB and BIDS-iEEG overlap in scope but rarely coexist on the same dataset. |
| **`mef3`** | MEF3 (Mayo) is for **long-duration clinical sEEG** (weeks-long). NWB can hold that data but rarely does in clinical practice. |
