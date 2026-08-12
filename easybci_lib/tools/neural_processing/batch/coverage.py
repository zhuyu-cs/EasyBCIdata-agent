"""Coverage-gap detection for batch enumeration.

``batch_process_adaptive`` enumerates inputs with ``glob.glob(pattern)``. A
non-recursive pattern (``.../EEG2100/*.EEG``) silently ignores same-extension
files in sibling subtrees. This module recomputes, under the pattern's anchor
directory, the full recursive set of same-extension files and reports whatever
the glob missed, so the batch handler can surface a loud warning instead of
reporting success over a partial run.

Pure/stdlib-only: safe to import from the import-cheap tool handler.
"""
from __future__ import annotations

import glob as _glob
import os
from typing import Dict, List, Sequence


def _anchor_dir(pattern: str) -> str:
    """The deepest fixed (wildcard-free) directory prefix of a glob pattern."""
    # Split on separator; keep leading components until the first one that
    # contains a glob magic char. That fixed prefix is where recursion starts.
    parts = pattern.replace("\\", "/").split("/")
    fixed: List[str] = []
    for part in parts:
        if _glob.has_magic(part):
            break
        fixed.append(part)
    anchor = "/".join(fixed)
    if not anchor:
        return "."
    # If the anchor names a file (pattern had no magic at all), use its dir.
    if os.path.isfile(anchor):
        return os.path.dirname(anchor) or "."
    return anchor


def _extensions(paths: Sequence[str]) -> List[str]:
    """Lower-cased distinct extensions present in ``paths`` (e.g. ['.eeg'])."""
    exts = {os.path.splitext(p)[1].lower() for p in paths if os.path.splitext(p)[1]}
    return sorted(exts)


def pattern_extensions(pattern: str) -> List[str]:
    """Lower-cased extension(s) implied by a glob pattern's trailing component.

    ``.../EEG2100/*.EEG`` → ``['.eeg']``; ``.../*`` or a directory pattern with
    no extension → ``[]`` (caller must not guess a recursive walk in that case).
    """
    tail = pattern.replace("\\", "/").rstrip("/").split("/")[-1]
    ext = os.path.splitext(tail)[1]
    # A bare ``.EEG`` (no stem) or ``*.EEG`` both yield ".eeg"; ``*`` yields "".
    return [ext.lower()] if ext and "*" not in ext and "?" not in ext else []


def _is_strict_ancestor(ancestor: str, descendant: str) -> bool:
    """True when ``ancestor`` is a proper parent directory of ``descendant``.

    Both are resolved to absolute paths first. Equal paths return False (the
    caller only widens when the dataset root sits strictly *above* the initial
    scan root). Off-tree paths (e.g. output on a different disk) return False,
    so the caller falls back to the un-widened scan root.
    """
    a = os.path.abspath(ancestor)
    d = os.path.abspath(descendant)
    if a == d:
        return False
    try:
        return os.path.commonpath([a, d]) == a
    except ValueError:
        # Different drives / mixed absolute-relative → no common ancestor.
        return False


def _dir_is_under(path: str, roots: Sequence[str]) -> bool:
    """True when ``path`` is one of, or nested inside any of, ``roots``."""
    ap = os.path.abspath(path)
    for r in roots:
        if ap == r or _is_strict_ancestor(r, ap):
            return True
    return False


def dataset_root_from_output_dir(output_dir: str) -> "str | None":
    """Recover the dataset root from the batch output directory, mechanically.

    The output convention places the work_dir *inside* the dataset folder — e.g.
    ``/media/x/SEEG_ZHU/SEEG_ZHU_preprocess_work_dirs``. The
    ``_preprocess_work_dir`` suffix is the system's canonical work_dir signal
    (see ``skill_compliance_guard.detect_work_dir`` / ``resolve_work_dir``); the
    plural ``_preprocess_work_dirs`` is the batch container form. When either is
    found among ``output_dir`` and its parents, the dataset root is that
    directory's parent. Returns an absolute path, or None when no such marker is
    present (caller then keeps today's behavior).

    Pure stdlib — safe to import from the import-cheap tool handler.
    """
    if not output_dir:
        return None
    from pathlib import Path
    try:
        p = Path(output_dir).expanduser().resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        return None
    for node in (p, *p.parents):
        name = node.name
        if name.endswith("_preprocess_work_dir") or name.endswith("_preprocess_work_dirs"):
            parent = str(node.parent)
            return parent or None
    return None


def enumerate_signal_inputs(
    scan_root: str, exts: Sequence[str], exclude_under: Sequence[str] = (),
) -> List[str]:
    """Recursively enumerate signal files under ``scan_root`` for ``exts``.

    This is the source-root-driven enumerator: every file whose lower-cased
    extension is in ``exts`` is collected (case-insensitive on disk), then run
    through :func:`filter_signal_files` so per-recording companions/sidecars
    (NK ``.21E``/``.LOG``, BrainVision ``.vhdr``/``.vmrk``, …) never route as
    standalone inputs. Returns a sorted, de-duplicated list of absolute paths.

    ``exclude_under`` names directories whose subtrees are pruned from the walk
    (matching dirs are dropped from ``os.walk``'s dirnames in-place so large
    output/archive trees are never descended). Used to keep a widened walk from
    re-ingesting the work_dir's own outputs / ``_runN`` archives.

    When ``exts`` is empty the scan cannot be safely scoped, so an empty list is
    returned — the caller should fall back to plain ``glob.glob(pattern)``.
    """
    exts_norm = {e.lower() for e in exts}
    if not exts_norm:
        return []
    excl = [os.path.abspath(e) for e in exclude_under if e]
    found: List[str] = []
    for dirpath, dirs, files in os.walk(scan_root):
        # Prune excluded subtrees so we never descend big output/archive trees.
        if excl:
            dirs[:] = [
                d for d in dirs
                if not _dir_is_under(os.path.join(dirpath, d), excl)
            ]
        for name in files:
            if os.path.splitext(name)[1].lower() in exts_norm:
                found.append(os.path.abspath(os.path.join(dirpath, name)))
    # Drop companions/sidecars — same discipline as the coverage gap filter.
    # Lazy import keeps this module's top-level import cheap (loader pulls numpy).
    from easybci_lib.tools.neural_processing.io.loader import filter_signal_files
    return sorted(set(filter_signal_files(found)))


def detect_uncovered_inputs(
    pattern: str, matched: Sequence[str], scan_root: str | None = None
) -> Dict[str, object]:
    """Report same-extension files under a scan root that ``matched`` missed.

    Parameters
    ----------
    pattern : str
        The glob pattern passed to the batch tool.
    matched : sequence of str
        Paths the glob actually returned.
    scan_root : str, optional
        Directory to scan recursively for same-extension files. When given
        (typically the source directory the user pointed at), this catches
        *sibling-subtree* gaps — e.g. a ``A/NKT/EEG2100/*.EEG`` pattern that
        silently ignores ``A/SEEG/NKT/EEG2100/*.EEG``. When omitted, defaults
        to the pattern's fixed anchor directory, which catches only
        *same-tree-deeper* gaps (a non-recursive ``dir/*.EEG`` missing
        ``dir/sub/*.EEG``).

    Returns
    -------
    dict with keys:
        anchor       — the recursive scan root actually used
        extensions   — the extensions considered (derived from ``matched``)
        matched_count — len(matched)
        uncovered    — sorted list of same-extension files not in ``matched``
        gap_count    — len(uncovered)

    When ``matched`` has no discernible extension (can't scope the scan safely),
    returns a zero-gap report rather than guessing.
    """
    exts = _extensions(matched)
    matched_norm = {os.path.abspath(p) for p in matched}
    if not exts:
        return {
            "anchor": scan_root or _anchor_dir(pattern), "extensions": [],
            "matched_count": len(matched), "uncovered": [], "gap_count": 0,
        }

    anchor = scan_root if scan_root else _anchor_dir(pattern)
    uncovered: List[str] = []
    for dirpath, _dirs, files in os.walk(anchor):
        for name in files:
            if os.path.splitext(name)[1].lower() not in exts:
                continue
            full = os.path.join(dirpath, name)
            if os.path.abspath(full) not in matched_norm:
                uncovered.append(full)

    # Companion/sidecar files (e.g. NK .21E next to the .EEG signal) are not
    # independent inputs — drop them so a coverage "gap" never re-introduces a
    # sidecar the batch router would choke on. Lazy import keeps this module's
    # top-level import cheap (loader.py pulls numpy).
    from easybci_lib.tools.neural_processing.io.loader import filter_signal_files
    uncovered = [u for u in filter_signal_files(uncovered)
                 if os.path.abspath(u) not in matched_norm]

    uncovered.sort()
    return {
        "anchor": anchor, "extensions": exts,
        "matched_count": len(matched), "uncovered": uncovered,
        "gap_count": len(uncovered),
    }
