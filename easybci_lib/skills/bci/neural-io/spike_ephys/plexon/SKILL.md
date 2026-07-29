---
name: plexon
description: "Plexon (.plx / .pl2) — NHP electrophysiology + behavior synchronization"
layer: L0
group: spike_ephys
metadata:
  tags: [io, plexon, plx, pl2, nhp, behavior_sync]
  formats: [.plx, .pl2]
  modalities: [spike, lfp]
---
# Plexon

## Format Overview

Plexon Inc. (now part of NeuralCorp) acquisition hardware writes
`.plx` (legacy) and `.pl2` (current) files. Common in NHP labs running
behavioral experiments with synchronized neural + behavioral channels.

## File Manifest

```
session.plx          # legacy ~32 ch
session.pl2          # current; up to 256 ch + 8 wideband + 16 analog
```

## Metadata Schema

Header carries: per-channel name, gain, filter cut-offs, sample rate (per channel; mixed rates supported), event channel definitions.

## Loader Library Map

| Priority | Library | Lazy install |
|---|---|---|
| Primary | `neo.io.PlexonIO` (or `Pl2RawIO` for pl2) | `neo==0.13.4` via `io.plexon`. |
| Secondary | `pypl2` (Plexon-provided) | `pip install pypl2`. |

## Modality Coverage

Spike (sorted online by Plexon), LFP, NI-DAQ-style analog channels.

## Common Pitfalls

- **`.plx` vs `.pl2` reader path differ in neo.** Use `Pl2RawIO` for
  `.pl2`; `PlexonIO` for `.plx`.
- **Mixed sample rates.** Spike channels at 40 kHz; LFP at 1 kHz; analog
  at variable rates. Verify per-stream.
- **Online-sort spikes embedded.** Like Blackrock NEV, the `.plx`
  carries Plexon Sort Client-classified spikes — re-sort offline for
  publication-grade analysis.

## Reference Implementation

### Standalone

```python
import numpy as np
import neo


def load_plexon(path: str, stream: str = "spike") -> dict:
    reader = neo.io.PlexonIO(filename=path) if path.endswith(".plx") else neo.rawio.PlexonRawIO(filename=path)
    reader.parse_header()
    return {"sfreq": float(reader.get_signal_sampling_rate(0)),
            "n_channels": int(reader.signal_channels_count())}
```

### EasyBCI-Adapted

```python
from typing import Any, Dict


def operator_load_plexon(
    data_dict: Dict[str, Any], *, path: str, stream: str = "spike",
) -> Dict[str, Any]:
    """EasyBCI-adapted Plexon loader.

    Parameters
    ----------
    data_dict : dict
    path : str
    stream : str

    Returns
    -------
    dict — populated OperatorIO.

    Raises
    ------
    EasyBCIOperatorError
        recoverable=False on missing file.

    Modality coverage
    -----------------
    Spike / LFP (NHP): yes.

    References
    ----------
    Plexon DataWarehouse spec; neo.rawio docs.

    Notes
    -----
    L0 IO operator (CODE_STANDARD Rule 12 exempt).
    """
    import time
    # easybci-allow: file-io  # L0 IO operator
    from easybci_lib.tools.neural_processing.operator_errors import EasyBCIOperatorError
    if not path:
        raise EasyBCIOperatorError(
            operator="load_plexon", reason="path required", recoverable=False,
        )
    t0 = time.monotonic()
    try:
        import neo
        reader = neo.rawio.PlexonRawIO(filename=path)
        reader.parse_header()
        sfreq = float(reader.get_signal_sampling_rate(stream_index=0))
        sig = reader.get_analogsignal_chunk(block_index=0, seg_index=0, channel_indexes=None)
        import numpy as np
        gains = reader.header["signal_channels"]["gain"].astype(np.float32)
        offsets = reader.header["signal_channels"]["offset"].astype(np.float32)
        data = sig.astype(np.float32).T * gains[:, None] + offsets[:, None]
    except Exception as exc:
        raise EasyBCIOperatorError(
            operator="load_plexon", reason=f"load failed: {exc}", recoverable=False,
        ) from exc
    elapsed = time.monotonic() - t0
    n_t = data.shape[1]
    return {
        "data": data, "channels": [f"ch{i}" for i in range(data.shape[0])],
        "frequency": sfreq, "duration": n_t / sfreq, "elapsed_s": elapsed,
        "meta": {"modality": "spike" if sfreq >= 20000 else "lfp",
                 "load_plexon": {"path": path, "stream": stream}},
    }
```

## References

1. Garcia, S. et al. (2014). *Neo: an object model for handling
   electrophysiology data*. Frontiers in Neuroinformatics 8: 10.
   doi:10.3389/fninf.2014.00010.
2. Plexon Documentation (proprietary). https://www.plexon.com/

## Boundary with Related Formats

- **`blackrock`**: clinical UEA; Plexon = NHP research.
- **`nwb`**: archived deposit; Plexon = vendor working format.
- **`openephys` / `spikeglx`**: alternative research acquisition stacks.
