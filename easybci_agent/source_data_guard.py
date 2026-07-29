"""Source Data Immutability Guard — unconditional protection for raw research data.

This module is the single source of truth for the source-data-immutability
invariant: original neural data files provided for processing are NEVER writable,
deletable, or moveable by the agent, regardless of user request.

Any path registered via ``register_source_path()`` becomes permanently protected
for the lifetime of the process. The protection is checked by:

- ``agent/file_safety.py``  → blocks write_file / patch / delete_file / move_file
- ``tools/approval.py``     → blocks shell commands targeting source data
- ``tools/neural_processing/executor/runner.py`` → post-execution mtime audit

Thread-safe: gateway runs concurrent sessions with shared state.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import FrozenSet, Optional

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_protected_paths: set[str] = set()
_protected_dirs: set[str] = set()

_DENY_REGISTER_AS_DIR: frozenset[str] = frozenset({
    "/", "/home", "/root", "/tmp", "/var", "/usr", "/opt",
    "/etc", "/mnt", "/media", "/data", "/dev", "/sys", "/proc",
    "/Users",
})


def _is_overly_broad_dir(resolved: str) -> bool:
    """True if resolved path is too broad to register as a protected dir."""
    if not resolved:
        return True
    if resolved in _DENY_REGISTER_AS_DIR:
        return True
    try:
        home = os.path.realpath(os.path.expanduser("~"))
    except (OSError, ValueError):
        home = None
    if home and resolved == home:
        return True
    parts = Path(resolved).parts
    if len(parts) <= 2:
        return True
    return False


class SourceDataViolation(Exception):
    """Raised when an operation would modify or delete protected source data."""

    def __init__(self, path: str, operation: str = "write"):
        self.path = path
        self.operation = operation
        try:
            from easybci_agent.i18n import t
            msg = t(
                "source_guard.blocked_file_write",
                path=path, operation=operation,
            )
        except Exception:
            msg = (
                f"BLOCKED: Source data is immutable. Cannot {operation} '{path}'. "
                "This path is registered as original research data and cannot be "
                "modified or deleted under any circumstances."
            )
        super().__init__(msg)


def register_source_path(path: str) -> None:
    """Register a file path as protected source data.

    Once registered, the path cannot be written to, deleted, moved, or
    modified by any agent tool. Registration is permanent for the lifetime
    of the process — there is no unregister.

    Resolves the path through symlinks to prevent bypass via indirection.
    """
    if not path:
        return
    try:
        resolved = os.path.realpath(os.path.expanduser(path))
    except (OSError, ValueError):
        resolved = path

    with _lock:
        if resolved not in _protected_paths:
            _protected_paths.add(resolved)
            logger.info("Source data registered as immutable: %s", resolved)

    parent = os.path.dirname(resolved)
    if parent:
        register_source_directory(parent)


def register_source_directory(dir_path: str) -> None:
    """Register a directory as source-data-protected.

    Once registered, any write/create/delete inside this directory (or its
    descendants) is rejected by file_safety / approval / executor layers.
    Skips registration with a warning if dir_path is on the deny list
    (overly broad: /, $HOME root, /tmp, /var, paths with depth < 2).
    """
    if not dir_path:
        return
    try:
        resolved = os.path.realpath(os.path.expanduser(dir_path))
    except (OSError, ValueError):
        resolved = dir_path

    if _is_overly_broad_dir(resolved):
        logger.warning(
            "Source directory '%s' is too broad to protect. "
            "Protecting only registered files inside it.",
            resolved,
        )
        return

    with _lock:
        if resolved in _protected_dirs:
            return
        _protected_dirs.add(resolved)
        logger.info("Source directory registered as immutable: %s", resolved)


def is_inside_protected_dir(path: str) -> bool:
    """True if path resolves into any registered protected directory.

    Does NOT require the path to exist — works for pre-check of would-be writes.
    """
    if not path:
        return False
    try:
        resolved = os.path.realpath(os.path.expanduser(path))
    except (OSError, ValueError):
        resolved = path

    with _lock:
        for protected in _protected_dirs:
            if resolved == protected:
                return True
            if resolved.startswith(protected + os.sep):
                return True
    return False


def get_protected_dirs() -> FrozenSet[str]:
    """Return snapshot of all currently registered protected directories."""
    with _lock:
        return frozenset(_protected_dirs)


def is_source_data(path: str) -> bool:
    """Return True if the given path resolves to a registered source data file.

    Checks:
    1. Exact match after realpath resolution
    2. Path is inside the same file (e.g. writing to a source file with a
       different trailing component like .bak still resolves to the same inode
       — but we check exact resolved path here, not inode)
    """
    if not path:
        return False
    try:
        resolved = os.path.realpath(os.path.expanduser(path))
    except (OSError, ValueError):
        resolved = path

    with _lock:
        return resolved in _protected_paths


def is_source_data_or_parent(path: str) -> bool:
    """Return True if path IS source data, or if a write to path could
    destroy source data (e.g. truncating the parent directory, or writing
    to the exact source file under a different access path).

    This is stricter than ``is_source_data`` — use it for operations like
    ``rm -rf`` on directories that might contain source data.
    """
    if not path:
        return False
    try:
        resolved = os.path.realpath(os.path.expanduser(path))
    except (OSError, ValueError):
        resolved = path

    with _lock:
        if resolved in _protected_paths:
            return True
        resolved_with_sep = resolved + os.sep
        for protected in _protected_paths:
            if protected.startswith(resolved_with_sep):
                return True
    return False


def assert_not_source_data(path: str, operation: str = "write") -> None:
    """Raise SourceDataViolation if path is protected source data OR inside a
    protected source directory.

    Call this before any mutating operation on a path.
    """
    if is_source_data(path):
        raise SourceDataViolation(path, operation)
    if is_inside_protected_dir(path):
        raise SourceDataViolation(path, operation)


def check_output_path(output_path: str) -> Optional[str]:
    """Validate that an output path does not overlap with source data.

    Returns an error message if the output would overwrite source data
    (either a registered file or any location inside a protected directory),
    or None if safe.
    """
    if not output_path:
        return None
    try:
        resolved = os.path.realpath(os.path.expanduser(output_path))
    except (OSError, ValueError):
        resolved = output_path

    with _lock:
        if resolved in _protected_paths:
            return (
                f"BLOCKED: Output path '{output_path}' resolves to registered "
                "source data. Choose a different output path — source data "
                "is immutable and must never be overwritten."
            )
        for protected_dir in _protected_dirs:
            if resolved == protected_dir or resolved.startswith(protected_dir + os.sep):
                return (
                    f"BLOCKED: Output path '{output_path}' is inside protected "
                    f"source directory '{protected_dir}'. Choose an output "
                    "path outside the source data tree."
                )
    return None


def get_protected_paths() -> FrozenSet[str]:
    """Return snapshot of all currently registered source data paths."""
    with _lock:
        return frozenset(_protected_paths)


def snapshot_source_files() -> dict[str, tuple[float, int]]:
    """Take a snapshot of (mtime, size) for all registered source files.

    Used by the executor to verify files are unchanged after code execution.
    Returns {resolved_path: (mtime, size_bytes)}.
    Missing files are included with (-1, -1).
    """
    with _lock:
        paths = set(_protected_paths)

    snapshots: dict[str, tuple[float, int]] = {}
    for p in paths:
        try:
            st = os.stat(p)
            snapshots[p] = (st.st_mtime, st.st_size)
        except OSError:
            snapshots[p] = (-1.0, -1)
    return snapshots


def verify_source_integrity(
    before: dict[str, tuple[float, int]],
) -> list[str]:
    """Compare current source file state against a pre-execution snapshot.

    Returns list of violation descriptions (empty = all intact).
    """
    violations: list[str] = []
    for path, (orig_mtime, orig_size) in before.items():
        try:
            st = os.stat(path)
            if st.st_mtime != orig_mtime or st.st_size != orig_size:
                violations.append(
                    f"SOURCE DATA MODIFIED: '{path}' "
                    f"(mtime {orig_mtime} → {st.st_mtime}, "
                    f"size {orig_size} → {st.st_size})"
                )
        except FileNotFoundError:
            if orig_mtime != -1.0:
                violations.append(f"SOURCE DATA DELETED: '{path}'")
        except OSError as exc:
            violations.append(f"SOURCE DATA INACCESSIBLE: '{path}' ({exc})")
    return violations


def _reset_for_testing() -> None:
    """Clear all registered paths. ONLY for use in test fixtures."""
    with _lock:
        _protected_paths.clear()


def _reset_protected_dirs_for_testing() -> None:
    """Clear all registered protected directories. ONLY for test fixtures."""
    with _lock:
        _protected_dirs.clear()
