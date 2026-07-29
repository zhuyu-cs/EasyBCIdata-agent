"""Chunked processing — memory-aware large file processing.

For recordings that exceed available RAM (24h sleep studies, multi-hour
Neuropixels), this module provides:
- Memory estimation before loading
- Automatic chunk size calculation based on available memory
- Overlap-add strategy for filter continuity across chunk boundaries
- Sequential chunk processing with final concatenation
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_MEMORY_SAFETY_FACTOR = 0.8  # Use at most 80% of available RAM
_MIN_CHUNK_SAMPLES = 10000  # Minimum chunk size (avoid tiny chunks)
_OVERLAP_SECONDS = 2.0  # Overlap between chunks for filter continuity


def estimate_memory_requirements(
    file_path: str,
    n_channels: int = 0,
    frequency: float = 0.0,
    duration_s: float = 0.0,
    dtype_bytes: int = 8,
) -> Dict[str, Any]:
    """Estimate memory needed to load and process a neural data file.

    Parameters
    ----------
    file_path : str
        Path to the data file.
    n_channels : int
        Number of channels (if known from metadata).
    frequency : float
        Sampling rate Hz (if known).
    duration_s : float
        Recording duration seconds (if known).
    dtype_bytes : int
        Bytes per sample (default 8 for float64).

    Returns
    -------
    dict with estimated memory usage and chunking recommendation.
    """
    file_size = 0
    try:
        file_size = os.path.getsize(file_path)
    except OSError:
        pass

    # Estimate from file size if no metadata available
    if n_channels == 0 or duration_s == 0:
        # Heuristic: most neural formats have ~3x overhead in memory vs disk
        estimated_memory_mb = (file_size * 3) / (1024 * 1024)
    else:
        n_samples = int(frequency * duration_s)
        raw_bytes = n_channels * n_samples * dtype_bytes
        # Processing overhead: ~3x (original + working copy + intermediate)
        estimated_memory_mb = (raw_bytes * 3) / (1024 * 1024)

    available_mb = _get_available_memory_mb()
    budget_mb = available_mb * _MEMORY_SAFETY_FACTOR

    needs_chunking = estimated_memory_mb > budget_mb

    result = {
        "file_size_mb": round(file_size / (1024 * 1024), 1),
        "estimated_memory_mb": round(estimated_memory_mb, 1),
        "available_memory_mb": round(available_mb, 1),
        "memory_budget_mb": round(budget_mb, 1),
        "needs_chunking": needs_chunking,
    }

    if needs_chunking and n_channels > 0 and frequency > 0:
        chunk_info = calculate_chunk_params(
            n_channels, frequency, duration_s, budget_mb, dtype_bytes
        )
        result["chunk_params"] = chunk_info

    return result


def calculate_chunk_params(
    n_channels: int,
    frequency: float,
    total_duration_s: float,
    budget_mb: float,
    dtype_bytes: int = 8,
) -> Dict[str, Any]:
    """Calculate optimal chunk size given memory constraints.

    Parameters
    ----------
    n_channels : int
        Number of channels.
    frequency : float
        Sampling rate.
    total_duration_s : float
        Total recording duration.
    budget_mb : float
        Available memory budget in MB.
    dtype_bytes : int
        Bytes per sample.

    Returns
    -------
    dict with chunk_duration_s, n_chunks, overlap_samples.
    """
    bytes_per_second = n_channels * frequency * dtype_bytes
    # Processing needs ~3x memory (raw + copy + intermediate)
    memory_per_second_mb = (bytes_per_second * 3) / (1024 * 1024)

    if memory_per_second_mb <= 0:
        return {"chunk_duration_s": total_duration_s, "n_chunks": 1, "overlap_samples": 0}

    max_duration_per_chunk = budget_mb / memory_per_second_mb
    # Ensure minimum chunk size
    min_duration = _MIN_CHUNK_SAMPLES / frequency
    chunk_duration = max(min_duration, min(max_duration_per_chunk, total_duration_s))

    overlap_samples = int(_OVERLAP_SECONDS * frequency)
    n_chunks = max(1, int(np.ceil(total_duration_s / chunk_duration)))

    return {
        "chunk_duration_s": round(chunk_duration, 1),
        "n_chunks": n_chunks,
        "overlap_samples": overlap_samples,
        "overlap_seconds": _OVERLAP_SECONDS,
        "memory_per_chunk_mb": round(chunk_duration * memory_per_second_mb, 1),
    }


def process_chunked(
    data: np.ndarray,
    frequency: float,
    steps: List[str],
    channels: Optional[List[str]] = None,
    chunk_duration_s: float = 60.0,
    overlap_s: float = _OVERLAP_SECONDS,
    progress_callback=None,
) -> Dict[str, Any]:
    """Process large data in chunks with overlap-add for filter continuity.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_samples)
        Full continuous data array.
    frequency : float
        Sampling rate.
    steps : list of str
        Pipeline steps to apply.
    channels : list of str, optional
        Channel names.
    chunk_duration_s : float
        Duration of each processing chunk.
    overlap_s : float
        Overlap between chunks (for filter edge effects).
    progress_callback : callable, optional
        Called with (chunk_idx, n_chunks) for progress tracking.

    Returns
    -------
    dict with processed "data", "frequency", "channels", "meta".
    """
    from easybci_lib.tools.neural_processing.preprocess.pipeline import preprocess

    n_channels, n_samples = data.shape
    chunk_samples = int(chunk_duration_s * frequency)
    overlap_samples = int(overlap_s * frequency)

    # Separate filter steps (need overlap-add) from non-filter steps
    filter_steps, other_steps = _classify_steps(steps)

    # For non-filter steps that don't need overlap, process full data
    # For filter steps, use overlap-add
    if not filter_steps:
        # No filter steps — process normally
        data_dict = {
            "data": data,
            "frequency": frequency,
            "channels": channels or [f"Ch{i}" for i in range(n_channels)],
            "meta": {},
        }
        return preprocess(data_dict, steps=steps)

    # --- Chunked processing with overlap-add ---
    output_chunks = []
    chunk_start = 0
    chunk_idx = 0
    n_chunks = max(1, int(np.ceil(n_samples / chunk_samples)))

    while chunk_start < n_samples:
        chunk_end = min(chunk_start + chunk_samples, n_samples)

        # Extended chunk with overlap on both sides
        ext_start = max(0, chunk_start - overlap_samples)
        ext_end = min(n_samples, chunk_end + overlap_samples)
        chunk_data = data[:, ext_start:ext_end]

        # Process this chunk through filter steps
        chunk_dict = {
            "data": chunk_data.copy(),
            "frequency": frequency,
            "channels": channels or [f"Ch{i}" for i in range(n_channels)],
            "meta": {},
        }
        result = preprocess(chunk_dict, steps=filter_steps)
        processed_chunk = result["data"]

        # Trim overlap regions (keep only the valid center portion)
        trim_start = chunk_start - ext_start
        trim_end = trim_start + (chunk_end - chunk_start)
        valid_chunk = processed_chunk[:, trim_start:trim_end]
        output_chunks.append(valid_chunk)

        if progress_callback:
            progress_callback(chunk_idx, n_chunks)

        chunk_start = chunk_end
        chunk_idx += 1

    # Concatenate processed chunks
    processed_data = np.concatenate(output_chunks, axis=1)

    # Apply non-filter steps on the full processed result
    out_dict = {
        "data": processed_data,
        "frequency": frequency,
        "channels": channels or [f"Ch{i}" for i in range(n_channels)],
        "meta": {"chunked_processing": True, "n_chunks": chunk_idx},
    }

    if other_steps:
        out_dict = preprocess(out_dict, steps=other_steps)

    out_dict.setdefault("meta", {})["processing_mode"] = "chunked"
    out_dict["meta"]["chunk_duration_s"] = chunk_duration_s
    out_dict["meta"]["n_chunks_processed"] = chunk_idx

    return out_dict


_FILTER_STEP_NAMES = {"notch", "bandpass", "hilbert"}


def _classify_steps(steps: List[str]) -> Tuple[List[str], List[str]]:
    """Split steps into filter steps (need overlap-add) and others."""
    filter_steps = []
    other_steps = []
    for step in steps:
        name = step.split(":")[0]
        if name in _FILTER_STEP_NAMES:
            filter_steps.append(step)
        else:
            other_steps.append(step)
    return filter_steps, other_steps


def _get_available_memory_mb() -> float:
    """Get available system memory in MB, respecting cgroup limits."""
    available = None
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    available = int(line.split()[1]) / 1024  # kB → MB
                    break
    except (OSError, ValueError, IndexError):
        pass

    # Check cgroup limit (Docker / Kubernetes / systemd)
    cgroup_limit_mb = None
    for cg_path in (
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
        "/sys/fs/cgroup/memory.max",
    ):
        try:
            with open(cg_path, encoding="utf-8") as f:
                val = f.read().strip()
                if val != "max" and val.isdigit():
                    cgroup_limit_mb = int(val) / (1024 * 1024)
                    break
        except (OSError, ValueError):
            continue

    if available is None:
        available = 8000.0  # fallback: assume 8GB

    # Effective available is the lesser of system available and cgroup cap
    if cgroup_limit_mb and cgroup_limit_mb < available:
        available = cgroup_limit_mb

    # Also respect EASYBCI_MEMORY_BUDGET_MB if set by executor
    env_budget = os.environ.get("EASYBCI_MEMORY_BUDGET_MB")
    if env_budget:
        try:
            budget = float(env_budget)
            if budget > 0:
                return budget
        except ValueError:
            pass

    return available
