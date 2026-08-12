"""Nihon Kohden (EEG-1200A) reader backend.

Ported from a validated gold-standard project's nk_io.py. Parses the NK
multi-file recording (.EEG signal + .21E electrode labels + .LOG events) into
the standard easybci loader dict. Source data is read-only (Rule 5).
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np


SUPPORTED_VERSION = "EEG-1200A V01.00"


@dataclass(frozen=True)
class NkEvent:
    onset_label_time: str
    description: str
    onset_sec: float | None = None  # parsed from HHMMSS; None if unparseable


def _log_time_to_seconds(time_raw: bytes) -> float | None:
    """Decode a NK .LOG time field (6 ASCII digits, HHMMSS) to seconds.

    Returns None when the field is not a valid HHMMSS string — the caller
    must NOT fabricate a time; it counts these so downstream sees that events
    exist but their onset could not be extracted (IO stays honest about the
    limits of its format support).
    """
    try:
        s = time_raw.decode("ascii", errors="ignore").strip()
    except Exception:
        return None
    if len(s) != 6 or not s.isdigit():
        return None
    h, m, sec = int(s[0:2]), int(s[2:4]), int(s[4:6])
    if m >= 60 or sec >= 60:
        return None
    return float(h * 3600 + m * 60 + sec)


@dataclass(frozen=True)
class NkRecording:
    eeg_path: Path
    stem: str
    n_channels: int
    n_signal_channels: int
    sfreq: float
    n_samples: int
    data_offset: int
    phys_indices: list[int]
    channel_names: list[str]
    gains: np.ndarray
    offsets: np.ndarray
    events: list[NkEvent]

    @property
    def duration_sec(self) -> float:
        return self.n_samples / self.sfreq


def _read_u8(f) -> int:
    return struct.unpack("<B", f.read(1))[0]


def _read_u16(f) -> int:
    return struct.unpack("<H", f.read(2))[0]


def _read_u32(f) -> int:
    return struct.unpack("<I", f.read(4))[0]


def default_channel_labels() -> list[str]:
    return [f"CH{i}" for i in range(1, 1097)]


def read_21e_labels(base_path: Path) -> list[str]:
    labels = default_channel_labels()
    path = base_path.with_suffix(".21E")
    if not path.exists():
        return labels

    in_electrode = False
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.strip()
            if line.startswith("["):
                if in_electrode:
                    break
                in_electrode = line == "[ELECTRODE]"
                continue
            if not in_electrode or "=" not in line:
                continue
            idx_str, name = line.split("=", 1)
            try:
                idx = int(idx_str.strip())
            except ValueError:
                continue
            if 0 <= idx < len(labels):
                labels[idx] = name.strip()
    return labels


def compute_gains_and_offsets(phys_indices: list[int], total_channels: int) -> tuple[np.ndarray, np.ndarray]:
    micro_factor = 1e-6 * ((3199.902 + 3200.0) / (32767.0 + 32768.0))
    milli_factor = 1e-3 * ((12002.56 + 12002.9) / (32767.0 + 32768.0))
    table = np.full(1096, milli_factor, dtype=np.float64)
    table[0:42] = micro_factor
    table[74] = micro_factor
    table[75] = micro_factor
    table[78:256] = micro_factor
    table[256:1096] = micro_factor

    gains = [table[i] if 0 <= i < len(table) else micro_factor for i in phys_indices]
    gains.append(1.0)
    offsets = np.full(total_channels, 32768.0, dtype=np.float64)
    return np.asarray(gains, dtype=np.float64), offsets


def read_log_events(base_path: Path) -> list[NkEvent]:
    path = base_path.with_suffix(".LOG")
    if not path.exists():
        return []

    b = path.read_bytes()
    if len(b) < 0x92:
        return []

    events: list[NkEvent] = []
    n_blocks = b[0x91]
    for block_idx in range(n_blocks):
        block_ptr = 0x92 + block_idx * 20
        if block_ptr + 4 > len(b):
            continue
        block_addr = struct.unpack_from("<I", b, block_ptr)[0]
        if block_addr + 0x14 > len(b):
            continue
        n_logs = b[block_addr + 0x12]
        pos = block_addr + 0x14
        for _ in range(n_logs):
            if pos + 45 > len(b):
                break
            desc_raw = b[pos : pos + 20]
            time_raw = b[pos + 20 : pos + 26]
            pos += 45

            desc = ""
            for enc in ("gbk", "cp932", "utf-8", "latin1"):
                try:
                    desc = desc_raw.decode(enc).strip("\x00\r\n ")
                    break
                except UnicodeDecodeError:
                    continue
            label_time = time_raw.decode("ascii", errors="ignore")
            if desc:
                events.append(NkEvent(label_time, desc,
                                      onset_sec=_log_time_to_seconds(time_raw)))
    return events


def load_recording(eeg_path: Path | str) -> NkRecording:
    eeg_path = Path(eeg_path)
    with eeg_path.open("rb") as f:
        version = f.read(16).decode("ascii", errors="ignore").strip("\x00")
        if version != SUPPORTED_VERSION:
            raise ValueError(f"Unsupported Nihon Kohden version in {eeg_path.name}: {version!r}")

        f.seek(0x81)
        ctrl_version = f.read(16).decode("ascii", errors="ignore").strip("\x00")
        if ctrl_version != SUPPORTED_VERSION:
            raise ValueError(f"Invalid control block signature in {eeg_path.name}: {ctrl_version!r}")

        f.seek(0x17FE)
        wave_sig = _read_u8(f)
        if wave_sig != 1:
            raise ValueError(f"Invalid waveform block signature in {eeg_path.name}: {wave_sig}")

        f.seek(0x91)
        n_control_blocks = _read_u8(f)
        if n_control_blocks != 1:
            raise ValueError(f"Multiple control blocks are not supported: {n_control_blocks}")

        f.seek(0x3EE)
        ext_addr = _read_u32(f)
        f.seek(0x92)
        ctl_addr = _read_u32(f)

        f.seek(ctl_addr + 17)
        n_data_blocks = _read_u8(f)
        if n_data_blocks != 1:
            raise ValueError(f"Multiple data blocks are not supported: {n_data_blocks}")

        f.seek(ctl_addr + 18)
        data_addr = _read_u32(f)
        f.seek(data_addr + 0x1A)
        sfreq = float(_read_u16(f) & 0x3FFF)

        f.seek(ext_addr + 18)
        ext2_addr = _read_u32(f)
        f.seek(ext2_addr + 20)
        ext3_addr = _read_u32(f)

        f.seek(ext3_addr + 68)
        n_phys = _read_u16(f)
        phys_indices = []
        for i in range(n_phys):
            f.seek(ext3_addr + 72 + i * 10)
            phys_indices.append(_read_u16(f))

    total_channels = n_phys + 1
    data_offset = ext3_addr + 72 + n_phys * 10
    file_size = eeg_path.stat().st_size
    n_samples = (file_size - data_offset) // (total_channels * 2)

    all_labels = read_21e_labels(eeg_path)
    channel_names = [all_labels[i] if 0 <= i < len(all_labels) else f"CH{i}" for i in phys_indices]
    channel_names.append("Events")
    gains, offsets = compute_gains_and_offsets(phys_indices, total_channels)

    return NkRecording(
        eeg_path=eeg_path,
        stem=eeg_path.stem,
        n_channels=total_channels,
        n_signal_channels=total_channels - 1,
        sfreq=sfreq,
        n_samples=int(n_samples),
        data_offset=int(data_offset),
        phys_indices=phys_indices,
        channel_names=channel_names,
        gains=gains,
        offsets=offsets,
        events=read_log_events(eeg_path),
    )


def read_data_window_uV(rec: NkRecording, start: int, stop: int, include_events: bool = False) -> np.ndarray:
    if start < 0 or stop <= start or stop > rec.n_samples:
        raise ValueError(f"Invalid sample window: start={start}, stop={stop}, n_samples={rec.n_samples}")

    n_samples = stop - start
    n_channels = rec.n_channels
    byte_offset = rec.data_offset + start * n_channels * 2
    count = n_samples * n_channels

    with rec.eeg_path.open("rb") as f:
        f.seek(byte_offset)
        raw = np.fromfile(f, dtype="<u2", count=count)

    if raw.size != count:
        raise IOError(f"Expected {count} uint16 samples, got {raw.size}")

    raw = raw.reshape(n_samples, n_channels)
    n_out = n_channels if include_events else rec.n_signal_channels
    values = raw[:, :n_out].T.astype(np.float32, copy=False)
    values -= rec.offsets[:n_out, None].astype(np.float32)
    values *= rec.gains[:n_out, None].astype(np.float32)
    values *= np.float32(1e6)
    return values


def _read_decimated_uV(rec: NkRecording, stop: int, factor: int,
                       window_s: float = 300.0) -> np.ndarray:
    """Read [0, stop) in windows, decimating each by ``factor`` on the fly.

    A full-resolution 261ch/2000Hz/4h recording up-casts to ~30 GB in float32,
    and the NK reader's intermediate copies push peak RSS to ~56 GB — enough to
    OOM a 62 GB host and impossible to run two in parallel. Reading in ~5-min
    windows and decimating each window before concatenating keeps peak RSS to
    ~14 GB (measured), which is the only way multi-hour sEEG fits in memory.
    Mirrors the reference nk_io.py's windowed ``read_data_window_uV`` approach.
    """
    from scipy.signal import decimate

    win = max(int(rec.sfreq * window_s), factor * 8)
    # decimate needs each window to be a whole multiple of factor so segment
    # boundaries line up after downsampling (no fractional-sample drift).
    win -= win % factor
    segments: list[np.ndarray] = []
    start = 0
    while start < stop:
        seg_stop = min(stop, start + win)
        n = seg_stop - start
        n -= n % factor  # drop a sub-factor tail so decimate is exact
        if n < factor:
            break
        seg = read_data_window_uV(rec, start, start + n, include_events=False)
        segments.append(decimate(seg, factor, axis=1, ftype="fir").astype(np.float32))
        del seg
        start += n
    if not segments:
        return np.zeros((rec.n_signal_channels, 0), dtype=np.float32)
    return np.concatenate(segments, axis=1)


def load_nk(filepath: str, inspect_only: bool = False, modality: str = "auto",
            max_duration: float | None = None, target_hz: float | None = None) -> dict:
    """Load a Nihon Kohden recording into the standard easybci loader dict.

    Accepts the .EEG signal file OR any sibling (.21E/.LOG/.PNT); resolves to
    the .EEG. When inspect_only, loads just the first 1s window for stats.
    When max_duration is set (seconds), only the first N seconds are read into
    memory — the .EEG file is int16 and up-casts ~4x, so this is the guard that
    keeps a multi-hour recording from OOM-ing the host.

    When ``target_hz`` is set and strictly below the native rate, the signal is
    read in windows and **decimated on the fly** to the nearest integer-factor
    rate at or below ``target_hz`` — peak RSS stays bounded (see
    :func:`_read_decimated_uV`) so multi-hour high-channel sEEG can be loaded at
    all. The returned ``frequency`` reflects the actual decimated rate.
    """
    p = Path(filepath)
    eeg_path = p if p.suffix.upper() == ".EEG" else p.with_suffix(".EEG")
    if not eeg_path.exists():
        raise FileNotFoundError(f"NK signal file not found next to {p.name}: {eeg_path}")

    rec = load_recording(eeg_path)

    if inspect_only:
        stop = min(rec.n_samples, int(rec.sfreq))  # first 1 second
    elif max_duration is not None and max_duration > 0:
        stop = min(rec.n_samples, max(1, int(rec.sfreq * max_duration)))
    else:
        stop = rec.n_samples

    # Decide integer decimation factor. Only decimate when not inspecting and a
    # target below native was requested; round DOWN in rate (factor rounds UP)
    # so we never alias above the requested Nyquist.
    decim_factor = 1
    if not inspect_only and target_hz and target_hz > 0 and target_hz < rec.sfreq:
        decim_factor = max(1, int(rec.sfreq // target_hz))

    if decim_factor > 1:
        data = _read_decimated_uV(rec, stop, decim_factor)
        out_sfreq = rec.sfreq / decim_factor
    else:
        data = read_data_window_uV(rec, 0, stop, include_events=False).astype(np.float32)
        out_sfreq = rec.sfreq

    loaded_duration_s = stop / rec.sfreq if rec.sfreq else 0.0

    # Build annotations from .LOG events so reject_by_labels (and segmenting)
    # can act on seizure/stim windows. Events whose HHMMSS time didn't parse
    # keep their description but are counted separately — IO surfaces "events
    # exist but time not extracted" instead of silently dropping them.
    ann_onset: list[float] = []
    ann_desc: list[str] = []
    n_unparsed_time = 0
    for ev in rec.events:
        if ev.onset_sec is None:
            n_unparsed_time += 1
            continue
        ann_onset.append(float(ev.onset_sec))
        ann_desc.append(ev.description)
    annotations = {
        "onset": ann_onset,
        "duration": [0.0] * len(ann_onset),
        "description": ann_desc,
    }

    return {
        "data": data,
        "frequency": float(out_sfreq),
        "channels": list(rec.channel_names[: rec.n_signal_channels]),
        "duration": float(rec.duration_sec),
        "meta": {
            "format": "nihon_kohden",
            "source_file": str(eeg_path),
            "data_unit": "uV",
            "modality": "seeg",
            "n_channels": rec.n_signal_channels,
            "n_samples_total": rec.n_samples,
            "n_samples_loaded": int(data.shape[1]),
            "loaded_duration_s": round(float(loaded_duration_s), 3),
            "cropped": bool(stop < rec.n_samples),
            "native_sfreq": float(rec.sfreq),
            "decimation_factor": int(decim_factor),
            "load_time_decimated": bool(decim_factor > 1),
            "n_events": len(rec.events),
            "annotations": annotations,
            "events_unparsed_time": n_unparsed_time,
            "inspect_only": inspect_only,
        },
    }
