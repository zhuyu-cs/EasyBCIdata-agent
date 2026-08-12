"""Directory-size + human-readable formatting helpers for run summaries.

Shared by the mini-repo README/summary builders so the post-run report can
show a raw-vs-preprocessed footprint (e.g. ``raw 5.6 TB → preprocessed 98 GB``).
Pure/stdlib-only — safe to import from the import-cheap export layer.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Union

_PathLike = Union[str, "os.PathLike[str]"]


def dir_size_bytes(path: _PathLike) -> int:
    """Best-effort recursive size in bytes of ``path``. Returns 0 on error.

    Mirrors ``checkpoint_manager._dir_size_bytes`` — per-file ``stat`` so a
    single unreadable file never aborts the walk.
    """
    total = 0
    try:
        for p in Path(path).rglob("*"):
            try:
                if p.is_file():
                    total += p.stat().st_size
            except OSError:
                continue
    except OSError:
        pass
    return total


def paths_size_bytes(paths: Iterable[_PathLike]) -> int:
    """Sum of ``st_size`` over the given file paths (best-effort, skips errors)."""
    total = 0
    for p in paths:
        try:
            total += os.stat(p).st_size
        except OSError:
            continue
    return total


def format_size(nbytes: float) -> str:
    """Human-readable size (B/KB/MB/GB/TB), 1 decimal place above bytes."""
    n = float(nbytes)
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{int(n)} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"
