"""Multi-stream time alignment — synchronize data from different sources/clocks.

Provides methods to align auxiliary streams (eye-tracking, EMG, behavioral data)
to a master timebase (typically EEG). Handles:
- LSL ClockOffset correction (from XDF metadata)
- Shared-event alignment (cross-correlation on common triggers)
- Interpolation-based resampling to master timebase
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StreamData:
    """A single data stream with its own timebase."""
    name: str
    data: np.ndarray  # (n_channels, n_samples) or (n_samples,)
    frequency: float
    timestamps: Optional[np.ndarray] = None  # absolute timestamps per sample
    clock_offset: float = 0.0  # offset applied by LSL clock sync
    stream_type: str = "unknown"  # eeg, emg, eye, marker, etc.


@dataclass
class AlignedStream:
    """A stream after alignment to master timebase."""
    name: str
    data: np.ndarray  # resampled to master frequency
    original_frequency: float
    offset_applied: float  # time offset applied during alignment (seconds)
    method: str  # alignment method used
    quality: float  # alignment quality score (0-1)


@dataclass
class AlignedBundle:
    """Result of multi-stream alignment."""
    master: StreamData
    aligned_streams: List[AlignedStream] = field(default_factory=list)
    alignment_report: Dict[str, Any] = field(default_factory=dict)


def align_to_master(
    streams: List[StreamData],
    master_idx: int = 0,
    method: str = "auto",
    target_freq: Optional[float] = None,
) -> AlignedBundle:
    """Align multiple streams to a master timebase.

    Parameters
    ----------
    streams : list of StreamData
        All data streams to align. Must have at least one.
    master_idx : int
        Index of the master stream (others align to it).
    method : str
        Alignment method:
        - "auto": pick best method based on available metadata
        - "clock_offset": use LSL-style clock offsets
        - "shared_event": find common events and cross-correlate
        - "interpolate": resample all to master frequency, assume aligned timestamps
    target_freq : float, optional
        Target frequency for output. If None, uses master stream's frequency.

    Returns
    -------
    AlignedBundle with master stream and all aligned auxiliary streams.
    """
    if not streams:
        logger.warning("No streams provided for alignment — returning empty AlignedBundle")
        return AlignedBundle(
            master=StreamData(name="empty", data=np.array([]), frequency=0.0),
            aligned_streams=[],
            alignment_report={"error": "no streams provided"},
        )
    if master_idx >= len(streams):
        logger.warning("master_idx %d out of range for %d streams — clamping to last index", master_idx, len(streams))
        master_idx = len(streams) - 1

    master = streams[master_idx]
    aux_streams = [s for i, s in enumerate(streams) if i != master_idx]
    output_freq = target_freq or master.frequency

    if method == "auto":
        method = _select_method(master, aux_streams)

    aligned = []
    report: Dict[str, Any] = {
        "master_stream": master.name,
        "master_frequency": master.frequency,
        "output_frequency": output_freq,
        "method": method,
        "n_streams_aligned": len(aux_streams),
        "per_stream": [],
    }

    for aux in aux_streams:
        if method == "clock_offset":
            result = _align_clock_offset(master, aux, output_freq)
        elif method == "shared_event":
            result = _align_shared_event(master, aux, output_freq)
        elif method == "interpolate":
            result = _align_interpolate(master, aux, output_freq)
        else:
            logger.warning("Unknown alignment method: %s — falling back to 'interpolate'", method)
            result = _align_interpolate(master, aux, output_freq)

        aligned.append(result)
        report["per_stream"].append({
            "name": aux.name,
            "original_freq": aux.frequency,
            "offset_applied_s": result.offset_applied,
            "method": result.method,
            "quality": result.quality,
        })

    return AlignedBundle(
        master=master,
        aligned_streams=aligned,
        alignment_report=report,
    )


def _select_method(master: StreamData, aux_streams: List[StreamData]) -> str:
    """Auto-select alignment method based on available data."""
    # If timestamps with clock offsets are available → use clock_offset
    has_offsets = any(s.clock_offset != 0.0 for s in aux_streams)
    if has_offsets:
        return "clock_offset"

    # If all streams have absolute timestamps → interpolate
    all_have_timestamps = all(s.timestamps is not None for s in aux_streams)
    if all_have_timestamps and master.timestamps is not None:
        return "interpolate"

    # Default to interpolate (assumes streams start at same time)
    return "interpolate"


def _align_clock_offset(
    master: StreamData,
    aux: StreamData,
    output_freq: float,
) -> AlignedStream:
    """Align using LSL ClockOffset correction.

    XDF files store per-sample timestamps that have already been corrected
    by dejitter and clock offset. This method uses the residual offset.
    """
    offset = aux.clock_offset - master.clock_offset

    # Resample aux to master frequency
    data = aux.data
    if data.ndim == 1:
        data = data[np.newaxis, :]

    n_master_samples = master.data.shape[-1] if master.data.ndim > 1 else len(master.data)
    target_n = int(n_master_samples * output_freq / master.frequency)

    resampled = _resample_to_target(data, aux.frequency, output_freq, target_n, offset)

    quality = 1.0 if abs(offset) < 0.01 else max(0.5, 1.0 - abs(offset) * 10)

    return AlignedStream(
        name=aux.name,
        data=resampled,
        original_frequency=aux.frequency,
        offset_applied=offset,
        method="clock_offset",
        quality=quality,
    )


def _align_shared_event(
    master: StreamData,
    aux: StreamData,
    output_freq: float,
) -> AlignedStream:
    """Align by finding shared events via cross-correlation.

    Finds the time offset that maximizes correlation between the two streams
    (or their envelopes). Best when both streams contain a common reference signal.
    """
    master_data = master.data
    aux_data = aux.data

    if master_data.ndim > 1:
        master_signal = np.mean(master_data, axis=0)
    else:
        master_signal = master_data

    if aux_data.ndim > 1:
        aux_signal = np.mean(aux_data, axis=0)
    else:
        aux_signal = aux_data

    # Resample aux to master frequency for cross-correlation
    if abs(aux.frequency - master.frequency) > 0.1:
        from scipy.signal import resample
        target_len = int(len(aux_signal) * master.frequency / aux.frequency)
        aux_signal_resampled = resample(aux_signal, target_len)
    else:
        aux_signal_resampled = aux_signal

    # Cross-correlation to find optimal lag
    max_lag_samples = int(master.frequency * 2)  # max 2 seconds offset
    master_segment = master_signal[:min(len(master_signal), int(master.frequency * 30))]
    aux_segment = aux_signal_resampled[:min(len(aux_signal_resampled), int(master.frequency * 30))]

    min_len = min(len(master_segment), len(aux_segment))
    if min_len < 10:
        # Not enough data for cross-correlation
        return _align_interpolate(master, aux, output_freq)

    master_segment = master_segment[:min_len]
    aux_segment = aux_segment[:min_len]

    # Normalized cross-correlation
    master_norm = master_segment - np.mean(master_segment)
    aux_norm = aux_segment - np.mean(aux_segment)

    correlation = np.correlate(master_norm, aux_norm, mode="full")
    mid = len(correlation) // 2
    search_start = max(0, mid - max_lag_samples)
    search_end = min(len(correlation), mid + max_lag_samples)
    search_region = correlation[search_start:search_end]

    if len(search_region) == 0:
        return _align_interpolate(master, aux, output_freq)

    best_lag_idx = np.argmax(np.abs(search_region))
    best_lag = best_lag_idx - (mid - search_start)
    offset_seconds = best_lag / master.frequency

    # Quality from correlation peak strength
    peak_val = abs(search_region[best_lag_idx])
    norm_factor = np.sqrt(np.sum(master_norm**2) * np.sum(aux_norm**2))
    quality = float(peak_val / max(norm_factor, 1e-10))
    quality = min(1.0, quality)

    # Resample with offset
    n_master_samples = master.data.shape[-1] if master.data.ndim > 1 else len(master.data)
    target_n = int(n_master_samples * output_freq / master.frequency)

    data = aux.data if aux.data.ndim > 1 else aux.data[np.newaxis, :]
    resampled = _resample_to_target(data, aux.frequency, output_freq, target_n, offset_seconds)

    return AlignedStream(
        name=aux.name,
        data=resampled,
        original_frequency=aux.frequency,
        offset_applied=offset_seconds,
        method="shared_event",
        quality=quality,
    )


def _align_interpolate(
    master: StreamData,
    aux: StreamData,
    output_freq: float,
) -> AlignedStream:
    """Align by resampling aux to master frequency (assumes same start time)."""
    data = aux.data
    if data.ndim == 1:
        data = data[np.newaxis, :]

    n_master_samples = master.data.shape[-1] if master.data.ndim > 1 else len(master.data)
    target_n = int(n_master_samples * output_freq / master.frequency)

    offset = 0.0
    # If both have timestamps, compute actual offset
    if master.timestamps is not None and aux.timestamps is not None:
        if len(master.timestamps) > 0 and len(aux.timestamps) > 0:
            offset = float(aux.timestamps[0] - master.timestamps[0])

    resampled = _resample_to_target(data, aux.frequency, output_freq, target_n, offset)

    quality = 0.7 if offset == 0 else 0.8

    return AlignedStream(
        name=aux.name,
        data=resampled,
        original_frequency=aux.frequency,
        offset_applied=offset,
        method="interpolate",
        quality=quality,
    )


def _resample_to_target(
    data: np.ndarray,
    src_freq: float,
    tgt_freq: float,
    target_n: int,
    time_offset: float = 0.0,
) -> np.ndarray:
    """Resample data to target length with optional time offset.

    Parameters
    ----------
    data : ndarray shape (n_channels, n_src_samples)
    src_freq : float — source sampling rate
    tgt_freq : float — target sampling rate
    target_n : int — desired number of output samples
    time_offset : float — offset in seconds (positive = aux starts later than master)
    """
    n_channels, n_src = data.shape

    # Source and target time vectors
    src_times = np.arange(n_src) / src_freq + time_offset
    tgt_times = np.arange(target_n) / tgt_freq

    result = np.zeros((n_channels, target_n), dtype=np.float32)

    for ch in range(n_channels):
        # Only interpolate within the valid source time range
        valid_mask = (tgt_times >= src_times[0]) & (tgt_times <= src_times[-1])
        if np.any(valid_mask):
            result[ch, valid_mask] = np.interp(
                tgt_times[valid_mask], src_times, data[ch]
            )

    return result


# --- Auto-trigger for multi-stream alignment ---

def detect_multistream_and_align(
    inspect_result: Dict[str, Any],
    data_path: str = "",
) -> Optional[Dict[str, Any]]:
    """Auto-detect multi-stream data and perform alignment if needed.

    Called during orchestration Step 1.5 when sidecar/inspect detects
    data_type == "multi-stream". Returns alignment report or None if
    not applicable.

    Parameters
    ----------
    inspect_result : dict
        Result from _handle_inspect_data() containing data_type, streams, etc.
    data_path : str
        Path to the data file (for loading streams).

    Returns
    -------
    Dict with alignment_report, or None if not multi-stream.
    """
    data_type = inspect_result.get("data_type", "")
    if data_type != "multi-stream":
        return None

    streams_info = inspect_result.get("streams", [])
    if len(streams_info) < 2:
        return None

    # Load streams from XDF or multi-file source
    loaded_streams = _load_streams_for_alignment(data_path, streams_info)
    if not loaded_streams or len(loaded_streams) < 2:
        return {
            "aligned": False,
            "reason": "Could not load multiple streams for alignment",
            "n_streams_detected": len(streams_info),
        }

    # Identify master stream (highest frequency, or EEG-type)
    master_idx = _select_master_stream(loaded_streams)

    # Perform alignment
    bundle = align_to_master(
        streams=loaded_streams,
        master_idx=master_idx,
        method="auto",
    )

    report = bundle.alignment_report
    report["aligned"] = True
    report["n_streams_total"] = len(loaded_streams)

    return report


def build_alignment_step_config(
    inspect_result: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Build configuration for an ALIGN step to insert into the pipeline.

    Returns step config dict if multi-stream detected, None otherwise.
    Used by the orchestrator to insert Step 1.5 ALIGN.
    """
    data_type = inspect_result.get("data_type", "")
    if data_type != "multi-stream":
        return None

    streams_info = inspect_result.get("streams", [])
    if len(streams_info) < 2:
        return None

    # Select master by stream type priority
    master_name = ""
    type_priority = ["eeg", "seeg", "ecog", "meg", "neural", "emg", "eog"]
    for priority_type in type_priority:
        for stream in streams_info:
            stream_type = str(stream.get("type", stream.get("stream_type", ""))).lower()
            if priority_type in stream_type:
                master_name = stream.get("name", stream.get("stream_name", ""))
                break
        if master_name:
            break

    if not master_name and streams_info:
        # Fallback: pick stream with highest frequency
        streams_with_freq = [
            (s.get("frequency", s.get("nominal_srate", 0)), s.get("name", f"stream_{i}"))
            for i, s in enumerate(streams_info)
        ]
        streams_with_freq.sort(reverse=True)
        master_name = streams_with_freq[0][1]

    return {
        "step": "align",
        "master_stream": master_name,
        "method": "auto",
        "n_streams": len(streams_info),
        "streams": [
            {
                "name": s.get("name", s.get("stream_name", f"stream_{i}")),
                "type": s.get("type", s.get("stream_type", "unknown")),
                "frequency": s.get("frequency", s.get("nominal_srate", 0)),
            }
            for i, s in enumerate(streams_info)
        ],
        "reason": (
            f"Multi-stream data detected ({len(streams_info)} streams). "
            f"Auto-inserting alignment step with master='{master_name}'. "
            f"Streams will be synchronized to master timebase before preprocessing."
        ),
    }


def _load_streams_for_alignment(
    data_path: str,
    streams_info: List[Dict[str, Any]],
) -> List[StreamData]:
    """Load stream data for alignment from an XDF or multi-file source."""
    if not data_path:
        return []

    from pathlib import Path
    ext = Path(data_path).suffix.lower()

    if ext in (".xdf", ".xdfz"):
        return _load_xdf_streams(data_path, streams_info)

    # For non-XDF multi-stream, try loading each stream path separately
    loaded = []
    for info in streams_info:
        stream_path = info.get("path", "")
        if stream_path and Path(stream_path).exists():
            try:
                from easybci_lib.tools.neural_processing.io.loader import load_neural
                result = load_neural(stream_path, inspect_only=False)
                data = result.get("data")
                if data is not None and hasattr(data, "shape"):
                    loaded.append(StreamData(
                        name=info.get("name", Path(stream_path).stem),
                        data=data if data.ndim > 1 else data[np.newaxis, :],
                        frequency=float(result.get("frequency", 0)),
                        stream_type=info.get("type", "unknown"),
                    ))
            except Exception as exc:
                logger.debug("Failed to load stream %s: %s", stream_path, exc)

    return loaded


def _load_xdf_streams(
    xdf_path: str,
    streams_info: List[Dict[str, Any]],
) -> List[StreamData]:
    """Load streams from an XDF file."""
    loaded = []
    try:
        import pyxdf
        streams_raw, _ = pyxdf.load_xdf(xdf_path)
    except (ImportError, Exception) as exc:
        logger.debug("Cannot load XDF for alignment: %s", exc)
        return []

    for raw_stream in streams_raw:
        info = raw_stream.get("info", {})
        name = ""
        if isinstance(info, dict):
            name = info.get("name", [""])[0] if isinstance(info.get("name"), list) else str(info.get("name", ""))
        elif hasattr(info, "__getitem__"):
            try:
                name = info["name"][0]
            except (KeyError, IndexError, TypeError):
                name = ""

        time_series = raw_stream.get("time_series", np.array([]))
        timestamps = raw_stream.get("time_stamps", np.array([]))

        if not isinstance(time_series, np.ndarray):
            time_series = np.array(time_series, dtype=np.float32)
        if not isinstance(timestamps, np.ndarray):
            timestamps = np.array(timestamps, dtype=np.float64)

        if time_series.size == 0:
            continue

        # Derive frequency from timestamps
        freq = 0.0
        if len(timestamps) > 1:
            dt = np.median(np.diff(timestamps))
            if dt > 0:
                freq = 1.0 / dt

        # Clock offset from XDF metadata
        clock_offset = 0.0
        clock_offsets = raw_stream.get("clock_offsets", [])
        if clock_offsets:
            try:
                clock_offset = float(clock_offsets[-1].get("value", 0))
            except (AttributeError, TypeError, IndexError, ValueError):
                pass

        data = time_series.T if time_series.ndim == 2 else time_series[np.newaxis, :]

        stream_type = "unknown"
        try:
            stream_type = info.get("type", ["unknown"])[0] if isinstance(info.get("type"), list) else str(info.get("type", "unknown"))
        except (TypeError, AttributeError):
            pass

        loaded.append(StreamData(
            name=name or f"stream_{len(loaded)}",
            data=data.astype(np.float32),
            frequency=freq,
            timestamps=timestamps,
            clock_offset=clock_offset,
            stream_type=stream_type.lower(),
        ))

    return loaded


def _select_master_stream(streams: List[StreamData]) -> int:
    """Select the master stream index by priority: EEG > highest freq > first."""
    # Prefer EEG/neural type
    type_priority = ["eeg", "seeg", "ecog", "meg", "neural"]
    for priority in type_priority:
        for i, s in enumerate(streams):
            if priority in s.stream_type.lower():
                return i

    # Fallback: highest frequency
    max_freq = 0.0
    max_idx = 0
    for i, s in enumerate(streams):
        if s.frequency > max_freq:
            max_freq = s.frequency
            max_idx = i

    return max_idx
