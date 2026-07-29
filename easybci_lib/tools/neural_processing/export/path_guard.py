"""Path guard for preprocess work directories.

When the LLM (or a user-issued shell command) tries to write a file at the
top level of a ``*_preprocess_work_dir`` directory, we redirect the target
into the canonical sub-directory (``code/`` for scripts, ``preprocessed_output/``
for binary outputs) and emit a human-readable warning.

This file intentionally has no third-party deps so it can be imported from
any tool handler without import-cycle risk.

Configuration (read via ``easybci_cli.config.load_config``):

    tools:
      codegen:
        enforce: warn   # warn | hard | off

``warn`` (default) normalises the path and writes; ``hard`` refuses the
write entirely; ``off`` disables guard logic.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Tuple, Union

# Filenames in this set (and matching the version regex) all collapse to
# ``pipeline.py``. They are common scatter targets we have seen LLMs invent.
_PIPELINE_VERSION_RE = re.compile(
    r"^pipeline_(v\d+|final|new|backup|old|copy)\.py$", re.IGNORECASE
)

# File extensions that, at work_dir root, belong in ``preprocessed_output/``.
_DATA_EXTS = {".pkl", ".npz", ".nwb"}
# Compound extensions handled separately (we test with .endswith on lowercase
# basename to catch ``.meta.json``).
_DATA_COMPOUND_SUFFIXES = (".meta.json",)


def _find_work_dir_ancestor(p: Path) -> Path | None:
    """Walk parents until we find one whose name ends with ``_preprocess_work_dir``.

    Returns the work_dir Path, or None if not inside one.
    """
    for ancestor in [p, *p.parents]:
        if ancestor.name.endswith("_preprocess_work_dir"):
            return ancestor
    return None


def _get_enforce_mode() -> str:
    """Read ``tools.codegen.enforce`` from config; default 'warn'."""
    try:
        from easybci_cli.config import load_config
        cfg = load_config() or {}
        tools = cfg.get("tools") or {}
        codegen = tools.get("codegen") or {}
        mode = codegen.get("enforce", "warn")
        if isinstance(mode, str):
            mode = mode.strip().lower()
        if mode in {"warn", "hard", "off"}:
            return mode
    except Exception:
        pass
    return "warn"


def normalize_target_path(
    path: Union[str, Path],
) -> Tuple[Path, str | None]:
    """Normalise a write target inside a preprocess work_dir.

    Returns ``(normalized_path, warning_or_None)``.

    Rules (only fire when target is inside some ``*_preprocess_work_dir``):

    * A ``.py`` file written directly at work_dir root → redirected to
      ``<work_dir>/code/<basename>``.
    * A basename matching ``pipeline_(v\\d+|final|new|backup|old|copy)\\.py``
      (case-insensitive) → renamed to ``pipeline.py`` regardless of where it
      lives inside the work_dir.
    * A top-level ``.pkl`` / ``.npz`` / ``*.meta.json`` → redirected to
      ``<work_dir>/preprocessed_output/<basename>``.

    Paths outside any work_dir are returned unchanged with no warning.
    """
    if isinstance(path, str):
        original_path = Path(path)
    else:
        original_path = path

    try:
        absolute = original_path.expanduser().resolve(strict=False)
    except Exception:
        absolute = original_path

    work_dir = _find_work_dir_ancestor(absolute)
    if work_dir is None:
        return original_path, None

    # Don't normalise the work_dir itself.
    if absolute == work_dir:
        return original_path, None

    basename = absolute.name
    basename_lower = basename.lower()
    parent = absolute.parent
    at_root = parent == work_dir

    # 1) Version-numbered pipeline filenames collapse to pipeline.py.
    #    This also catches them when nested under code/, which is desirable.
    pipeline_match = _PIPELINE_VERSION_RE.match(basename)
    if pipeline_match:
        new_basename = "pipeline.py"
        # If the file was at work_dir root, also move it into code/.
        if at_root:
            target = work_dir / "code" / new_basename
        else:
            target = parent / new_basename
        warning = (
            f"path_guard: '{basename}' is a version-numbered pipeline filename; "
            f"normalised to '{target}'. Always use a single canonical "
            f"'<work_dir>/code/pipeline.py' — the tool layer archives prior "
            f"versions to code/middle_process/ automatically."
        )
        return target, warning

    # 2) Top-level .py → code/
    if at_root and basename_lower.endswith(".py"):
        target = work_dir / "code" / basename
        warning = (
            f"path_guard: '{basename}' was about to be written at the "
            f"work_dir root; redirected to '{target}'. All scripts belong in "
            f"'<work_dir>/code/'."
        )
        return target, warning

    # 3) Top-level binary outputs → preprocessed_output/
    if at_root and (
        any(basename_lower.endswith(ext) for ext in _DATA_EXTS)
        or any(basename_lower.endswith(s) for s in _DATA_COMPOUND_SUFFIXES)
    ):
        target = work_dir / "preprocessed_output" / basename
        warning = (
            f"path_guard: '{basename}' was about to be written at the "
            f"work_dir root; redirected to '{target}'. Binary outputs belong "
            f"in '<work_dir>/preprocessed_output/'."
        )
        return target, warning

    return original_path, None


def get_enforce_mode() -> str:
    """Public accessor for the configured enforcement mode."""
    return _get_enforce_mode()


# ---------------------------------------------------------------------------
# In-file self-check (no pytest). Run with:  python -m tools.neural_processing.export.path_guard
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cases = []

    p, w = normalize_target_path("/x/y/z_preprocess_work_dir/pipeline_v3.py")
    cases.append((
        "version-numbered pipeline at root",
        str(p) == "/x/y/z_preprocess_work_dir/code/pipeline.py" and w is not None,
        f"got={p!s} warn={'yes' if w else 'no'}",
    ))

    p, w = normalize_target_path("/x/y/z_preprocess_work_dir/code/pipeline.py")
    cases.append((
        "canonical pipeline path untouched",
        str(p) == "/x/y/z_preprocess_work_dir/code/pipeline.py" and w is None,
        f"got={p!s} warn={'yes' if w else 'no'}",
    ))

    p, w = normalize_target_path("/x/y/normal_dir/foo.py")
    cases.append((
        "path outside any work_dir untouched",
        str(p) == "/x/y/normal_dir/foo.py" and w is None,
        f"got={p!s} warn={'yes' if w else 'no'}",
    ))

    p, w = normalize_target_path("/x/y/z_preprocess_work_dir/output.pkl")
    cases.append((
        "top-level .pkl redirected to preprocessed_output/",
        str(p) == "/x/y/z_preprocess_work_dir/preprocessed_output/output.pkl"
        and w is not None,
        f"got={p!s} warn={'yes' if w else 'no'}",
    ))

    passed = sum(1 for _, ok, _ in cases if ok)
    print(f"path_guard self-check: {passed}/{len(cases)} passed")
    for name, ok, detail in cases:
        print(f"  [{'OK' if ok else 'FAIL'}] {name} — {detail}")

    raise SystemExit(0 if passed == len(cases) else 1)
