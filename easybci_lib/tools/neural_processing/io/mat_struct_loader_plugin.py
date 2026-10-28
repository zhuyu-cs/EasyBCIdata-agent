"""Standalone struct-container ``.mat`` io_loader plugin — source + provisioning.

Same pattern as ``compumedics_loader_plugin.py``: the built-in
``mat_struct_loader`` depends on numpy + scipy + stdlib only and already
exposes ``matches``/``load``, so the plugin text is its source verbatim.

This module runs in-process (MAY import easybci); only its *output text*
must be self-contained.
"""

from __future__ import annotations

import inspect
from pathlib import Path

from easybci_lib.tools.neural_processing.io import mat_struct_loader
from easybci_lib.tools.neural_processing.io.loader_registry import (
    LOADER_MARKER,
    io_loaders_dir,
)

PLUGIN_NAME = "mat_struct"
PLUGIN_FILENAME = "mat_struct.py"


def plugin_source() -> str:
    """Return the full standalone plugin text (marker + loader source)."""
    return LOADER_MARKER + "\n" + inspect.getsource(mat_struct_loader)


def _write_if_changed(target: Path, text: str) -> Path:
    """Idempotent write: skip if the file already holds identical bytes."""
    try:
        if target.is_file() and target.read_text(encoding="utf-8") == text:
            return target
    except OSError:
        pass
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target


def ensure_global_plugin() -> Path:
    """Provision the plugin at ``~/.easybci/io_loaders/mat_struct.py``."""
    return _write_if_changed(io_loaders_dir() / PLUGIN_FILENAME, plugin_source())


def ensure_repo_plugin(code_dir: Path | str) -> Path:
    """Provision a repo-local copy at ``<code_dir>/io_loaders/mat_struct.py``."""
    return _write_if_changed(
        Path(code_dir) / "io_loaders" / PLUGIN_FILENAME, plugin_source()
    )
