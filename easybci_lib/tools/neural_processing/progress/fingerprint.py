"""Coarse-grained data-shape fingerprinting for cross-session reuse.

These buckets must be coarse enough that two recordings from "similar enough"
recording conditions share a bucket — otherwise progress_history reuse
collapses to per-recording singletons. Cohort matching and federation reuse
this bucketing — schema must remain stable across consumers.
"""
from __future__ import annotations

import hashlib
from typing import Dict


def _bucket_channels(n: int) -> str:
    if n <= 0:
        return "ch-unknown"
    if n < 8:
        return "ch-1-7"
    if n < 32:
        return "ch-8-31"
    if n < 128:
        return "ch-32-127"
    if n < 512:
        return "ch-128-511"
    return "ch-512+"


def _bucket_frequency(hz: float) -> str:
    if hz <= 0:
        return "freq-unknown"
    if hz < 256:
        return "freq-<256"
    if hz < 1024:
        return "freq-256-1023"
    if hz < 4096:
        return "freq-1024-4095"
    return "freq-4096+"


def _bucket_duration(s: float) -> str:
    if s <= 0:
        return "dur-unknown"
    if s < 60:
        return "dur-<1m"
    if s < 600:
        return "dur-1-10m"
    if s < 3600:
        return "dur-10m-1h"
    return "dur-1h+"


def fingerprint_bucket(*, modality: str, n_channels: int, frequency_hz: float, duration_s: float) -> Dict[str, str]:
    """Tuple-shaped bucket dict — human readable, stable across releases."""
    return {
        "modality": modality.lower(),
        "channels": _bucket_channels(n_channels),
        "frequency": _bucket_frequency(frequency_hz),
        "duration": _bucket_duration(duration_s),
    }


def coarse_fingerprint(*, modality: str, n_channels: int, frequency_hz: float, duration_s: float) -> str:
    """SHA256 of the bucket tuple, truncated to 8 bytes (16 hex chars)."""
    bucket = fingerprint_bucket(
        modality=modality,
        n_channels=n_channels,
        frequency_hz=frequency_hz,
        duration_s=duration_s,
    )
    raw = "|".join(f"{k}={v}" for k, v in sorted(bucket.items()))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
