---
name: lsl_stream
description: "Lab Streaming Layer (LSL) — real-time multi-stream acquisition for closed-loop / online BCI"
layer: L0
group: streaming
metadata:
  tags: [io, lsl, streaming, online, real_time, openviking]
  formats: [stream]
  modalities: [eeg, meg, seeg, ecog, fnirs, spike]
---
# LSL — Lab Streaming Layer

## Format Overview

LSL (Kothe 2014) is a real-time multi-stream framework — UDP-based
discovery + per-stream pull buffer. The de-facto standard for
**online BCI / closed-loop / multi-sensor synchronization**. Each
stream advertises its name, type (`EEG` / `Markers` / `Audio`), sample
rate, and channel format; consumers pull chunks with hardware timestamps.

Streams are **not files** — they're live network broadcasts on the local
machine or LAN.

## File Manifest

LSL itself is streaming; recordings are saved to `.xdf` files (see
`xdf` IO skill) for offline replay.

## Metadata Schema

Per-stream:
- `name`, `type`, `channel_count`, `nominal_srate`.
- `channel_format` ("float32" / "int32" / "string" / "double64").
- Per-channel metadata via XDF (`<info><desc><channels>...`).

## Loader Library Map

| Priority | Library |
|---|---|
| Primary | `pylsl` (PyPI) |
| Secondary | `mne_lsl` (MNE wrapper) |

`lazy_deps` key `io.lsl` → `pylsl==1.16.2`.

## Modality Coverage

Any modality with an LSL emitter (EEG / MEG / sEEG / fNIRS / spike).

## Common Pitfalls

- **Multi-stream sync.** Marker stream + EEG stream have separate
  timebases; align via `pull_chunk(timeout)` + the LSL clock-correction
  step.
- **Stream disappearance.** Producer may stop broadcasting mid-session;
  consumer must detect via timeout.
- **Buffer overflow.** If you don't pull fast enough, samples are dropped
  silently.
- **Multiple producers same name.** Resolves arbitrarily; specify the
  `source_id`.

## Reference Implementation

### Standalone

```python
import time
import pylsl


def lsl_pull_loop(stream_name: str = "EEG", duration_s: float = 10.0) -> dict:
    streams = pylsl.resolve_byprop("name", stream_name, timeout=5.0)
    if not streams:
        raise RuntimeError(f"LSL stream {stream_name!r} not found")
    inlet = pylsl.StreamInlet(streams[0])
    t_end = time.monotonic() + duration_s
    samples, ts = [], []
    while time.monotonic() < t_end:
        chunk, chunk_ts = inlet.pull_chunk(timeout=0.1, max_samples=1024)
        if chunk:
            samples.extend(chunk); ts.extend(chunk_ts)
    return {"samples": samples, "timestamps": ts}
```

### EasyBCI-Adapted

```python
from typing import Any, Dict


def operator_load_lsl(
    data_dict: Dict[str, Any], *,
    stream_name: str = "EEG", duration_s: float = 10.0,
) -> Dict[str, Any]:
    """LSL stream pull.

    Parameters
    ----------
    data_dict : dict
    stream_name : str
    duration_s : float

    Returns
    -------
    dict — populated OperatorIO with `data`/`channels`/`frequency`.

    Raises
    ------
    EasyBCIOperatorError
        recoverable=True if stream not resolved.

    Modality coverage
    -----------------
    Any (per the LSL producer).

    References
    ----------
    Kothe et al. 2014; pylsl docs.

    Notes
    -----
    L0 IO operator (Rule 12 exempt). Network usage explicit in this op.
    """
    import time
    # easybci-allow: file-io  # L0 IO operator (LSL is the equivalent of a file)
    from easybci_lib.tools.neural_processing.operator_errors import EasyBCIOperatorError
    try:
        import pylsl
        import numpy as np
    except ImportError as exc:
        raise EasyBCIOperatorError(
            operator="load_lsl", reason="pylsl not installed",
            recoverable=True, fallback_step="pip install pylsl",
        ) from exc

    streams = pylsl.resolve_byprop("name", stream_name, timeout=5.0)
    if not streams:
        raise EasyBCIOperatorError(
            operator="load_lsl", reason=f"stream {stream_name!r} not resolved",
            recoverable=True, fallback_step="confirm producer is running",
        )

    t0 = time.monotonic()
    inlet = pylsl.StreamInlet(streams[0])
    info = inlet.info()
    n_ch = info.channel_count()
    sfreq = float(info.nominal_srate()) or 1.0
    t_end = time.monotonic() + duration_s
    samples, _ts = [], []
    while time.monotonic() < t_end:
        chunk, chunk_ts = inlet.pull_chunk(timeout=0.1, max_samples=1024)
        if chunk:
            samples.extend(chunk); _ts.extend(chunk_ts)
    data = np.asarray(samples, dtype=np.float32).T if samples else np.zeros((n_ch, 0), dtype=np.float32)
    elapsed = time.monotonic() - t0

    return {
        "data": data,
        "channels": [f"ch{i}" for i in range(n_ch)],
        "frequency": sfreq,
        "duration": data.shape[1] / max(sfreq, 1.0),
        "elapsed_s": elapsed,
        "meta": {"modality": info.type().lower() or "eeg",
                 "load_lsl": {"stream_name": stream_name, "duration_s": duration_s}},
    }
```

## References

1. Kothe, C. A. (2014). *Lab Streaming Layer (LSL)*.
   https://github.com/sccn/labstreaminglayer.
2. Bjareholt, J. (2021). *MNE-LSL*. PyPI / GitHub.

## Boundary with Related Formats

- **`xdf`**: offline format storing recorded LSL streams.
- **`online_inference.md`**: paradigm that consumes this stream skill.
- **`closed_loop_bci.md`**: also consumes LSL for closed-loop.
