---
name: neuralynx
description: "Neuralynx (.ncs / .nse / .nev / .ntt) — tetrode-era ephys with rich event/spike sidecars"
layer: L0
group: spike_ephys
metadata:
  tags: [io, neuralynx, ncs, nse, nev, ntt, tetrode]
  formats: [.ncs, .nse, .nev, .ntt]
  modalities: [spike, lfp]
---
# Neuralynx

## Format Overview

Neuralynx Cheetah acquisition writes one file per channel per data type.
Common in tetrode-era and many modern rodent labs.

## File Manifest

```
session/
  CSC1.ncs CSC2.ncs ...     # one continuous file per channel
  Sc1.nse Sc1.ntt ...       # spike events / tetrode-packed spikes
  Events.nev                # behavioral / TTL events
  VT1.nvt                   # video tracking (optional)
```

## Metadata Schema

Each `.ncs` carries a 16 KB ASCII header (sample rate, channel name, ADC
range) + binary records of 512 samples.

## Loader Library Map

| Priority | Library |
|---|---|
| Primary | `neo.rawio.NeuralynxRawIO` |
| Secondary | `neuralynx-io` PyPI |

`lazy_deps` key `io.neuralynx` → `neo==0.13.4`.

## Modality Coverage

Spike (tetrode + single-channel), LFP, events.

## Common Pitfalls

- **One-file-per-channel.** Loading 96 channels = 96 file handles; large I/O.
- **Tetrode packing in `.ntt`.** Each spike record has 4 waveforms; unpacking
  to per-channel needs care.
- **Inconsistent ADC range across channels.** Always read header per file.

## Reference Implementation

### Standalone

```python
import neo


def load_neuralynx(dir_path: str) -> dict:
    reader = neo.rawio.NeuralynxRawIO(dirname=dir_path)
    reader.parse_header()
    return {"n_channels": reader.signal_channels_count(),
            "sfreq": float(reader.get_signal_sampling_rate(0))}
```

### EasyBCI-Adapted

```python
from typing import Any, Dict


def operator_load_neuralynx(
    data_dict: Dict[str, Any], *, path: str,
) -> Dict[str, Any]:
    """Neuralynx loader.

    Parameters
    ----------
    data_dict : dict
    path : str
        Directory path.

    Returns
    -------
    dict — populated OperatorIO.

    Raises
    ------
    EasyBCIOperatorError
        recoverable=False on missing path / bad headers.

    Modality coverage
    -----------------
    Spike / LFP (tetrode + single-channel rodent ephys): yes.

    References
    ----------
    Neuralynx Cheetah manual; neo.rawio docs.

    Notes
    -----
    L0 IO operator (Rule 12 exempt).
    """
    import time
    # easybci-allow: file-io  # L0 IO operator
    from easybci_lib.tools.neural_processing.operator_errors import EasyBCIOperatorError
    if not path:
        raise EasyBCIOperatorError(
            operator="load_neuralynx", reason="path required", recoverable=False,
        )
    t0 = time.monotonic()
    try:
        import neo
        import numpy as np
        reader = neo.rawio.NeuralynxRawIO(dirname=path)
        reader.parse_header()
        sfreq = float(reader.get_signal_sampling_rate(stream_index=0))
        sig = reader.get_analogsignal_chunk(block_index=0, seg_index=0, channel_indexes=None)
        gains = reader.header["signal_channels"]["gain"].astype(np.float32)
        offsets = reader.header["signal_channels"]["offset"].astype(np.float32)
        data = sig.astype(np.float32).T * gains[:, None] + offsets[:, None]
    except Exception as exc:
        raise EasyBCIOperatorError(
            operator="load_neuralynx", reason=f"load failed: {exc}", recoverable=False,
        ) from exc
    elapsed = time.monotonic() - t0
    n_t = data.shape[1]
    return {
        "data": data,
        "channels": [f"ch{i}" for i in range(data.shape[0])],
        "frequency": sfreq, "duration": n_t / sfreq, "elapsed_s": elapsed,
        "meta": {"modality": "spike" if sfreq >= 20000 else "lfp",
                 "load_neuralynx": {"path": path}},
    }
```

## References

1. Garcia, S. et al. (2014). *Neo: an object model for handling
   electrophysiology data in multiple formats*. Frontiers in
   Neuroinformatics 8: 10. doi:10.3389/fninf.2014.00010.
2. Neuralynx Cheetah documentation (proprietary). https://www.neuralynx.com/.

## Boundary with Related Formats

- **`blackrock`**: clinical UEA; Neuralynx = rodent tetrode legacy.
- **`spikeglx` / `openephys`**: modern Neuropixels rigs.
- **`nwb`**: archive deposit format.
