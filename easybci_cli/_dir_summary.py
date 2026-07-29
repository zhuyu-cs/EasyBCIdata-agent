"""Directory summary helper for the dashboard Source panel.

Returns a compact "at-a-glance" payload (file count, total size, ext histogram,
optional neural metadata) so the WebUI does not have to render the full tree by
default. Used by the ``GET /api/files/summary`` endpoint.

The neural metadata block is best-effort: if MNE / loader fails on the first
candidate file, the ``neural`` key is omitted (the call still returns 200).
"""

from __future__ import annotations

import logging
import os
from collections import OrderedDict
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# File extensions the underlying MNE / EasyBCI loader can probe header-only.
_NEURAL_EXTS = (
    ".edf", ".bdf", ".fif", ".set", ".vhdr", ".cnt", ".gdf",
    ".eeg", ".nwb", ".mat", ".xdf", ".cdt",
)

_CHANNEL_NAME_PREVIEW = 10
_LRU_MAX = 64

_cache: "OrderedDict[Tuple[str, int], Dict[str, Any]]" = OrderedDict()
_cache_lock = Lock()


def _cache_get(key: Tuple[str, int]) -> Optional[Dict[str, Any]]:
    with _cache_lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]
    return None


def _cache_put(key: Tuple[str, int], value: Dict[str, Any]) -> None:
    with _cache_lock:
        _cache[key] = value
        _cache.move_to_end(key)
        while len(_cache) > _LRU_MAX:
            _cache.popitem(last=False)


def _try_load_neural(candidate: Path) -> Optional[Dict[str, Any]]:
    """Try to read header-only metadata via the EasyBCI loader. Returns None on failure."""
    try:
        from easybci_lib.tools.neural_processing.io.loader import load_neural  # lazy import
    except Exception as exc:  # noqa: BLE001
        logger.debug("neural loader unavailable: %s", exc)
        return None

    try:
        result = load_neural(str(candidate), inspect_only=True)
    except Exception as exc:  # noqa: BLE001 — loader may raise anything
        logger.debug("inspect_only load failed for %s: %s", candidate, exc)
        return None

    channels = result.get("channels") or []
    frequency = result.get("frequency") or 0.0
    duration = result.get("duration") or 0.0
    meta = result.get("meta") or {}
    if meta.get("load_error"):
        return None

    modality = meta.get("modality") or meta.get("format") or "unknown"

    return {
        "modality_guess": str(modality),
        "n_channels": int(len(channels)),
        "duration_sec": float(duration),
        "sample_rate_hz": float(frequency),
        "channel_names_preview": [str(c) for c in channels[:_CHANNEL_NAME_PREVIEW]],
    }


def _scan_dir(resolved: Path) -> Dict[str, Any]:
    file_count = 0
    total_size = 0
    ext_hist: Dict[str, int] = {}
    neural_candidate: Optional[Path] = None

    try:
        with os.scandir(resolved) as it:
            for entry in it:
                if entry.name.startswith("."):
                    continue
                try:
                    if not entry.is_file(follow_symlinks=False):
                        continue
                    st = entry.stat(follow_symlinks=False)
                except OSError:
                    continue
                file_count += 1
                total_size += int(st.st_size)
                ext = Path(entry.name).suffix.lower()
                if ext:
                    ext_hist[ext] = ext_hist.get(ext, 0) + 1
                if neural_candidate is None and ext in _NEURAL_EXTS:
                    neural_candidate = Path(entry.path)
    except (PermissionError, FileNotFoundError) as exc:
        return {
            "path": str(resolved),
            "available": False,
            "file_count": 0,
            "total_size_bytes": 0,
            "ext_histogram": {},
            "reason": f"{type(exc).__name__}: {exc}",
        }

    summary: Dict[str, Any] = {
        "path": str(resolved),
        "available": True,
        "file_count": file_count,
        "total_size_bytes": total_size,
        "ext_histogram": ext_hist,
    }

    if neural_candidate is not None:
        neural = _try_load_neural(neural_candidate)
        if neural is not None:
            summary["neural"] = neural

    return summary


def summarize_directory(resolved: Path) -> Dict[str, Any]:
    """Return a cached summary for *resolved* (must already be a real directory)."""
    try:
        mtime = resolved.stat().st_mtime_ns
    except OSError:
        return {
            "path": str(resolved),
            "available": False,
            "file_count": 0,
            "total_size_bytes": 0,
            "ext_histogram": {},
            "reason": "stat failed",
        }
    key = (str(resolved), mtime)
    cached = _cache_get(key)
    if cached is not None:
        return cached
    payload = _scan_dir(resolved)
    _cache_put(key, payload)
    return payload
