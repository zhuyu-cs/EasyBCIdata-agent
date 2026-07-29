"""Pipeline step cache — per-step checkpointing for resume-from-failure.

When enabled, each pipeline step writes a snapshot to disk after completion.
If a later step fails, the pipeline can resume from the last cached state
rather than re-executing all steps from scratch.

Cache keys are derived from: input data content hash + step string + step index.
This ensures cache invalidation when data or parameters change.

Design:
- Cache directory: {work_dir}/.pipeline_cache/ (alongside the mini-repo)
- Format: compressed numpy (.npz) for speed
- LRU eviction: total cache size capped (default 2 GB)
- Thread-safe: file-based locking via atomic rename
"""

import datetime
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from easybci_lib.constants import get_easybci_home
from easybci_lib.tools.neural_processing.progress.fingerprint import coarse_fingerprint
from easybci_lib.tools.neural_processing.progress.history import (
    ProgressHistoryEntry,
    ProgressHistoryStore,
)

logger = logging.getLogger(__name__)

_DEFAULT_MAX_CACHE_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB


class StepCache:
    """Manages per-step pipeline checkpoints on disk."""

    def __init__(
        self,
        cache_dir: str | Path,
        max_bytes: int = _DEFAULT_MAX_CACHE_BYTES,
    ):
        self.cache_dir = Path(cache_dir)
        self.max_bytes = max_bytes
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def compute_data_hash(
        self,
        data: np.ndarray,
        n_sample: int = 65536,
        *,
        frequency: float | None = None,
        channels: Optional[List[str]] = None,
    ) -> str:
        """Content hash identifying a dataset for cache keying.

        Folds in array shape, sampling frequency, and channel identity so two
        recordings that happen to coincide at the sampled positions — but differ
        in sfreq, channel set, or length — do NOT collide and serve each other's
        cached results (HIGH-2). The sample budget is large (64k points) and the
        shape/sfreq/channel prefix makes same-sample collisions astronomically
        unlikely without hashing the full array on every call.
        """
        if data.size == 0:
            return "empty"
        flat = data.ravel()
        sample_indices = np.linspace(0, len(flat) - 1, min(n_sample, len(flat)), dtype=int)
        sample = flat[sample_indices]
        h = hashlib.sha256()
        # Identity prefix: shape + dtype + sfreq + channel names.
        h.update(repr(data.shape).encode())
        h.update(str(data.dtype).encode())
        h.update(repr(frequency).encode())
        if channels:
            h.update("\x00".join(map(str, channels)).encode())
        h.update(sample.tobytes())
        return h.hexdigest()[:16]

    def make_key(
        self,
        data_hash: str,
        steps: List[str],
        step_index: int,
        *,
        output_format: Optional[str] = None,
    ) -> str:
        """Generate a cache key for a specific step state.

        The key encodes: data identity + all steps up to and including step_index
        + (optionally) the final output format. ``output_format`` is folded in
        only when explicitly provided, so existing cache entries written before
        Phase NWB stay reachable. Passing ``"pkl"`` and ``"nwb"`` produce
        distinct keys — preventing a previously cached pkl run from being served
        as an nwb run's resume point and vice versa.
        """
        steps_prefix = json.dumps(steps[:step_index + 1], sort_keys=True)
        if output_format:
            combined = f"{data_hash}|{steps_prefix}|fmt={output_format}"
        else:
            combined = f"{data_hash}|{steps_prefix}"
        return hashlib.sha256(combined.encode()).hexdigest()[:24]

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve a cached step result, or None if not cached."""
        path = self.cache_dir / f"{key}.npz"
        meta_path = self.cache_dir / f"{key}.meta.json"

        if not path.exists() or not meta_path.exists():
            return None

        try:
            npz = np.load(path, allow_pickle=False)
            data = npz["data"]

            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)

            # Touch for LRU tracking
            path.touch()
            meta_path.touch()

            result = {
                "data": data,
                "channels": meta.get("channels", []),
                "frequency": meta.get("frequency", 0),
                "duration": meta.get("duration", 0),
                "meta": meta.get("data_meta", {}),
                "elapsed_s": meta.get("elapsed_s"),  # None for pre-Phase-1 entries
            }
            logger.debug("Cache HIT for key %s", key)
            return result

        except Exception as exc:
            logger.debug("Cache read failed for %s: %s", key, exc)
            return None

    def put(self, key: str, data_dict: Dict[str, Any]) -> None:
        """Store a step result in cache."""
        path = self.cache_dir / f"{key}.npz"
        meta_path = self.cache_dir / f"{key}.meta.json"

        try:
            data = data_dict.get("data")
            if data is None or not isinstance(data, np.ndarray):
                return

            # Write atomically via temp file + rename
            tmp_path = path.with_suffix(".tmp.npz")
            np.savez_compressed(tmp_path, data=data)
            tmp_path.rename(path)

            meta = {
                "channels": data_dict.get("channels", []),
                "frequency": data_dict.get("frequency", 0),
                "duration": data_dict.get("duration", 0),
                "data_meta": {
                    k: v for k, v in data_dict.get("meta", {}).items()
                    if k not in ("step_states",)  # exclude large nested objects
                    and isinstance(v, (str, int, float, bool, list, dict, type(None)))
                },
                "cached_at": time.time(),
                "elapsed_s": data_dict.get("elapsed_s"),  # None when absent (legacy entries)
            }
            tmp_meta = meta_path.with_suffix(".tmp.json")
            with open(tmp_meta, "w", encoding="utf-8") as f:
                json.dump(meta, f)
            tmp_meta.rename(meta_path)

            logger.debug("Cache PUT for key %s (%d bytes)", key, path.stat().st_size)

            # Evict if over budget
            self._evict_if_needed()

        except Exception as exc:
            logger.debug("Cache write failed for %s: %s", key, exc)

    def find_resume_point(
        self,
        data_hash: str,
        steps: List[str],
        *,
        output_format: Optional[str] = None,
    ) -> int:
        """Find the latest cached step index for a given pipeline.

        Returns the index of the last step that has a valid cache entry,
        or -1 if no cached state exists.
        """
        latest = -1
        for i in range(len(steps)):
            key = self.make_key(data_hash, steps, i, output_format=output_format)
            path = self.cache_dir / f"{key}.npz"
            if path.exists():
                latest = i
            else:
                break  # Cache chain broken — can't skip ahead
        return latest

    def get_stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        files = list(self.cache_dir.glob("*.npz"))
        total_bytes = sum(f.stat().st_size for f in files)
        return {
            "n_entries": len(files),
            "total_bytes": total_bytes,
            "total_mb": round(total_bytes / (1024 * 1024), 1),
            "max_mb": round(self.max_bytes / (1024 * 1024), 1),
            "utilization": round(total_bytes / self.max_bytes, 2) if self.max_bytes > 0 else 0,
        }

    def clear(self) -> int:
        """Remove all cache entries. Returns number of files removed."""
        count = 0
        for f in self.cache_dir.iterdir():
            if f.suffix in (".npz", ".json"):
                f.unlink(missing_ok=True)
                count += 1
        return count

    def _evict_if_needed(self) -> None:
        """Remove oldest entries until total size is within budget."""
        files = list(self.cache_dir.glob("*.npz"))
        total_bytes = sum(f.stat().st_size for f in files)

        if total_bytes <= self.max_bytes:
            return

        # Sort by access time (oldest first) for LRU
        files.sort(key=lambda f: f.stat().st_mtime)

        evicted = 0
        for f in files:
            if total_bytes <= self.max_bytes * 0.8:  # evict down to 80%
                break
            size = f.stat().st_size
            f.unlink(missing_ok=True)
            # Also remove companion meta file
            meta_f = f.with_suffix("").with_suffix(".meta.json")
            meta_f.unlink(missing_ok=True)
            total_bytes -= size
            evicted += 1

        if evicted:
            logger.info("Cache eviction: removed %d entries, freed to %d MB", evicted, total_bytes // (1024 * 1024))


def preprocess_with_cache(
    data_dict: Dict[str, Any],
    steps: List[str],
    cache_dir: str | Path,
    record_states: bool = False,
    *,
    output_format: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Run preprocessing pipeline with step-level caching.

    Checks cache for each step. If a contiguous prefix of steps is cached,
    resumes from the last cached state. New results are cached for future use.

    Parameters
    ----------
    data_dict : dict
        Loaded neural data (from load_neural).
    steps : list of str
        Pipeline steps to apply.
    cache_dir : str or Path
        Directory for cache storage.
    record_states : bool
        Whether to record before/after states for each step.
    output_format : str, optional
        When provided ("pkl" or "nwb"), folded into the cache key so pkl/nwb
        runs over identical data + steps maintain SEPARATE cache lines.
        Defaults to None for backward compat with caches written pre-NWB.

    Returns
    -------
    Processed data_dict, same as preprocess().
    """
    from easybci_lib.tools.neural_processing.preprocess.pipeline import preprocess

    cache = StepCache(cache_dir)
    data = data_dict.get("data")
    if data is None or not isinstance(data, np.ndarray):
        return preprocess(data_dict, steps=steps, record_states=record_states, **kwargs)

    data_hash = cache.compute_data_hash(
        data,
        frequency=data_dict.get("frequency"),
        channels=data_dict.get("channels"),
    )
    resume_idx = cache.find_resume_point(data_hash, steps, output_format=output_format)

    if resume_idx >= 0:
        # Load cached state
        key = cache.make_key(data_hash, steps, resume_idx, output_format=output_format)
        cached = cache.get(key)
        if cached is not None:
            logger.info(
                "Resuming pipeline from step %d/%d (cache hit: '%s')",
                resume_idx + 1, len(steps), steps[resume_idx],
            )
            data_dict = cached
            remaining_steps = steps[resume_idx + 1:]
        else:
            remaining_steps = steps
    else:
        remaining_steps = steps

    if not remaining_steps:
        data_dict.setdefault("meta", {})["preprocessing"] = steps
        data_dict["meta"]["cache_status"] = "full_hit"
        return data_dict

    # Run remaining steps
    result = preprocess(data_dict, steps=remaining_steps, record_states=record_states, **kwargs)

    # Cache the final complete result ONLY, keyed by the last step index.
    #
    # Earlier this also wrote `result` under make_key(..., start_idx) — the index
    # of the FIRST remaining step. But `result` is the state after ALL remaining
    # steps, so that key claimed "state after step start_idx" while holding the
    # fully-processed array. A later resume that picked start_idx as its resume
    # point would load fully-processed data and then re-apply steps start_idx+1..end
    # on top of it — silent double-processing (HIGH-1). Only the final-index key
    # is truthful for a whole-tail run; per-step checkpointing would need each
    # intermediate state captured individually (future work).
    final_key = cache.make_key(data_hash, steps, len(steps) - 1, output_format=output_format)
    cache.put(final_key, result)

    result.setdefault("meta", {})["cache_status"] = (
        f"resumed_from_step_{resume_idx + 1}" if resume_idx >= 0 else "miss"
    )

    return result


def record_step_elapsed(
    *,
    operator: str,
    elapsed_s: float,
    modality: str,
    n_channels: int,
    frequency_hz: float,
    duration_s: float,
) -> None:
    """Mirror an elapsed sample into ~/.easybci/progress_history.jsonl.

    Failures are non-fatal — this is observability, not correctness.
    """
    try:
        store = ProgressHistoryStore(path=get_easybci_home() / "progress_history.jsonl")
        fp = coarse_fingerprint(
            modality=modality, n_channels=n_channels,
            frequency_hz=frequency_hz, duration_s=duration_s,
        )
        store.append(ProgressHistoryEntry(
            stage="preprocess",
            operator=operator,
            fingerprint_hash=fp,
            elapsed_s=elapsed_s,
            n_channels=n_channels,
            duration_s=duration_s,
            timestamp=datetime.datetime.utcnow().isoformat() + "Z",
        ))
    except Exception:  # noqa: BLE001
        logger.debug("record_step_elapsed failed", exc_info=True)
