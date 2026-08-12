"""Standalone Nihon Kohden (NK) io_loader plugin — source + provisioning.

The built-in :mod:`easybci_lib.tools.neural_processing.io.nk_backend` reads NK
recordings correctly, but a **generated** ``code/pipeline.py`` cannot import it
(CODE_STANDARD Rule 15 — generated scripts stay self-contained). Generated
pipelines instead discover io_loader plugins via ``_discover_io_plugin()``,
which scans ``code/io_loaders/`` (repo-local) and ``~/.easybci/io_loaders/``.

This module is the single source of truth for that plugin's TEXT. Because
``nk_backend`` imports only numpy + stdlib at top level (scipy is a
function-local import inside ``_read_decimated_uV``), its verbatim source is
already "plugin-safe": we inline it via ``inspect.getsource`` and append a thin
``matches``/``load`` wrapper. Zero duplication, zero drift — the bundled plugin
is byte-identical logic to the built-in reader that reads NK correctly.

This module itself runs in-process, so it MAY import easybci; only its *output
text* must be self-contained.
"""

from __future__ import annotations

import inspect
from pathlib import Path

from easybci_lib.tools.neural_processing.io import nk_backend
from easybci_lib.tools.neural_processing.io.loader_registry import (
    LOADER_MARKER,
    io_loaders_dir,
)

PLUGIN_NAME = "nihon_kohden"
PLUGIN_FILENAME = "nihon_kohden.py"

# Appended after the inlined nk_backend source. References load_nk / Path from
# the inlined module body, so the finished file needs no easybci import.
_WRAPPER = '''

# --- io_loader plugin wrapper (auto-generated; do not edit) ----------------
def matches(path):
    from pathlib import Path as _P
    p = _P(str(path))
    suf = p.suffix.upper()
    if suf == ".21E":
        return True
    if suf in (".EEG", ".LOG", ".PNT", ".EVT"):
        # An NK .EEG only — a sibling .21E disambiguates it from BrainVision .eeg.
        return p.with_suffix(".21E").exists()
    return False


def load(path, inspect_only=False, target_hz=None):
    return load_nk(str(path), inspect_only=inspect_only, target_hz=target_hz)
'''


def plugin_source() -> str:
    """Return the full standalone plugin text (marker + nk_backend + wrapper).

    numpy + stdlib only — verified by the standalone-import unit test.
    """
    return LOADER_MARKER + "\n" + inspect.getsource(nk_backend) + "\n" + _WRAPPER


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
    """Provision the plugin at ``~/.easybci/io_loaders/nihon_kohden.py``.

    Written directly (not via loader_registry.register(), which requires a real
    probe_path to dry-run — we have none without the raw NK file). discover()
    validates the matches/load contract at read time, so a plain write is safe.
    """
    return _write_if_changed(io_loaders_dir() / PLUGIN_FILENAME, plugin_source())


def ensure_repo_plugin(code_dir: Path | str) -> Path:
    """Provision a repo-local copy at ``<code_dir>/io_loaders/nihon_kohden.py``.

    Bundling into the mini-repo makes the generated pipeline portable — it reads
    NK correctly on any machine, even one without the global plugin registered.
    """
    return _write_if_changed(
        Path(code_dir) / "io_loaders" / PLUGIN_FILENAME, plugin_source()
    )
