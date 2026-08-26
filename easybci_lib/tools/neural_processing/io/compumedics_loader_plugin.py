"""Standalone Compumedics ProFusion .SLP io_loader plugin — source + provisioning.

Follows the same pattern as ``nk_loader_plugin.py``: inline the built-in
loader via ``inspect.getsource`` and append a thin ``matches``/``load``
wrapper. The built-in ``compumedics_loader`` depends on numpy + scipy +
stdlib only, so the result is already plugin-safe.

This module runs in-process (MAY import easybci); only its *output text*
must be self-contained.
"""

from __future__ import annotations

import inspect
from pathlib import Path

from easybci_lib.tools.neural_processing.io import compumedics_loader
from easybci_lib.tools.neural_processing.io.loader_registry import (
    LOADER_MARKER,
    io_loaders_dir,
)

PLUGIN_NAME = "compumedics_slp"
PLUGIN_FILENAME = "compumedics_slp.py"

# The built-in compumedics_loader already defines matches() and load() with the
# correct signatures, so the wrapper only needs to re-export them. We still
# append the wrapper for explicitness and to strip the logger setup that
# references easybci's logging config.
_WRAPPER = '''

# --- io_loader plugin wrapper (auto-generated; do not edit) ----------------
import logging as _logging
logger = _logging.getLogger(__name__)
'''


def plugin_source() -> str:
    """Return the full standalone plugin text (marker + loader source).

    numpy + scipy + stdlib only — verified by the standalone-import test.
    """
    src = inspect.getsource(compumedics_loader)
    # The source declares `logger = logging.getLogger(__name__)` which is fine
    # standalone. Strip the `from __future__` duplicate (already in marker).
    return LOADER_MARKER + "\n" + src


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
    """Provision the plugin at ``~/.easybci/io_loaders/compumedics_slp.py``."""
    return _write_if_changed(io_loaders_dir() / PLUGIN_FILENAME, plugin_source())


def ensure_repo_plugin(code_dir: Path | str) -> Path:
    """Provision a repo-local copy at ``<code_dir>/io_loaders/compumedics_slp.py``.

    Bundling into the mini-repo makes the generated pipeline portable — it reads
    .SLP correctly on any machine, even one without the global plugin registered.
    """
    return _write_if_changed(
        Path(code_dir) / "io_loaders" / PLUGIN_FILENAME, plugin_source()
    )
