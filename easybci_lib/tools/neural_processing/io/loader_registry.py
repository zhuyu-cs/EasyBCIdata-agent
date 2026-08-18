"""Agent-authored IO loader plugins — registry + auto-probe gate.

When the built-in loaders in ``loader.py`` cannot read a format, the agent can
write a small loader plugin and register it here. Plugins live under
``get_easybci_home()/io_loaders/<name>.py`` (profile-isolated, persisted so the
next session / dataset reuses them — the flywheel).

Design invariants:

- **Built-ins win.** ``find()`` is only consulted at ``load_neural``'s
  ``backend == "unknown"`` boundary, so a plugin can never shadow a format a
  built-in already reads. It fills blanks, it does not override.
- **Auto-probe gate.** ``register()`` writes the file, imports it, asserts the
  ``matches`` / ``load`` contract, then dry-runs ``load(probe, inspect_only=True)``
  and requires ``validate_loaded_data(...).valid``. Any failure removes the file
  and returns a structured error — nothing half-registered lands on disk.
- **Same validation as built-ins.** A plugin's output flows through the exact
  ``validate_loaded_data`` every built-in loader's output does — no back door.
- **Source data is immutable (Rule 5).** Plugins read; they never mutate inputs.

Plugin contract (every ``<name>.py`` must expose)::

    # EASYBCI_IO_LOADER v1
    def matches(path: str) -> bool: ...          # narrow: extension + optional magic
    def load(path: str, inspect_only: bool = False) -> dict: ...
        # -> {data, frequency, channels, duration, meta{format, source_file, data_unit}}
"""

from __future__ import annotations

import importlib.util
import inspect
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from easybci_lib.constants import get_easybci_home

logger = logging.getLogger(__name__)

# Marker the agent puts at the top of a loader file. Not enforced (a plugin
# without it still works if it exposes matches/load), but register() writes it
# and it documents the contract version for future upgrades.
LOADER_MARKER = "# EASYBCI_IO_LOADER v1"

# A plugin module name must be a safe python identifier stem — no path
# traversal, no dotted names, no clobbering stdlib on import.
_SAFE_NAME_RX = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")


def io_loaders_dir() -> Path:
    """Directory holding agent-authored loader plugins under EASYBCI_HOME.

    Never hardcode ``~/.easybci`` — go through get_easybci_home() so profiles
    and the isolated-home test fixture both work.
    """
    return get_easybci_home() / "io_loaders"


@dataclass
class LoaderPlugin:
    """A discovered, contract-valid loader plugin."""
    name: str
    path: Path
    matches: Callable[[str], bool]
    load: Callable[..., dict]


@dataclass
class RegisterResult:
    """Outcome of register(). ``success`` gates whether the file was kept."""
    success: bool
    name: str
    registered_path: Optional[str] = None
    stage: Optional[str] = None      # which step failed: write|import|contract|probe|validate
    error: Optional[str] = None
    fix_hint: Optional[str] = None


def _load_module_from_path(name: str, path: Path):
    """Import a plugin file as an isolated module object (not sys.modules-cached).

    Each call re-imports from disk so a freshly-registered / edited plugin is
    picked up without a process restart.
    """
    spec = importlib.util.spec_from_file_location(f"easybci_io_loader_{name}", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # may raise — caller handles
    return module


def _plugin_from_module(name: str, path: Path, module) -> LoaderPlugin:
    """Validate a module exposes the matches/load contract; build a LoaderPlugin.

    Raises TypeError with a precise message if the contract is not met — the
    message becomes the agent's fix_hint.
    """
    matches = getattr(module, "matches", None)
    load = getattr(module, "load", None)
    if not callable(matches):
        raise TypeError("loader must define a callable `matches(path) -> bool`")
    if not callable(load):
        raise TypeError(
            "loader must define a callable `load(path, inspect_only=False) -> dict`"
        )
    return LoaderPlugin(name=name, path=path, matches=matches, load=load)


def discover() -> list[LoaderPlugin]:
    """Import every ``*.py`` under io_loaders_dir() that meets the contract.

    A file that fails to import or lacks matches/load is skipped with a warning
    — one broken plugin never breaks discovery of the others.
    """
    d = io_loaders_dir()
    if not d.is_dir():
        return []
    plugins: list[LoaderPlugin] = []
    for py in sorted(d.glob("*.py")):
        if py.name.startswith("_"):
            continue
        name = py.stem
        try:
            module = _load_module_from_path(name, py)
            plugins.append(_plugin_from_module(name, py, module))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping IO loader plugin %s: %s", py.name, exc)
    return plugins


def find(path: str) -> Optional[LoaderPlugin]:
    """Return the first registered plugin whose ``matches(path)`` is True.

    Only called from load_neural's unknown-format branch, so a match here means
    no built-in claimed the file. A plugin whose matches() raises is treated as
    a non-match (and logged) rather than propagating into the load path.
    """
    for plugin in discover():
        try:
            if bool(plugin.matches(path)):
                return plugin
        except Exception as exc:  # noqa: BLE001
            logger.warning("IO loader plugin %s matches() raised: %s", plugin.name, exc)
    return None


def call_load(plugin: "LoaderPlugin", path: str, *, inspect_only: bool = False,
              target_hz: Optional[float] = None) -> dict:
    """Invoke ``plugin.load`` passing ``target_hz`` only if its signature accepts
    it — the v1 contract was ``load(path, inspect_only=False)`` and older plugins
    must keep working. A plugin that declares ``target_hz`` (v1.1) gets the
    load-time-decimation request; one that doesn't loads native.
    """
    kwargs: dict = {"inspect_only": inspect_only}
    if target_hz is not None:
        try:
            params = inspect.signature(plugin.load).parameters
            if "target_hz" in params or any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
            ):
                kwargs["target_hz"] = target_hz
        except (ValueError, TypeError):
            pass  # builtins / C-callables without introspectable signatures
    return plugin.load(path, **kwargs)


def _probe(plugin: LoaderPlugin, probe_path: str) -> tuple[bool, Optional[str], Optional[str]]:
    """Dry-run load(probe, inspect_only=True) + validate. -> (ok, stage, error)."""
    # Import here to avoid a module-level import cycle (loader imports us).
    from easybci_lib.tools.neural_processing.io.validators import validate_loaded_data

    try:
        result = plugin.load(probe_path, inspect_only=True)
    except Exception as exc:  # noqa: BLE001
        return False, "probe", f"load(probe_path, inspect_only=True) raised: {type(exc).__name__}: {exc}"

    if not isinstance(result, dict):
        return False, "probe", f"load() must return a dict, got {type(result).__name__}"

    try:
        vr = validate_loaded_data(result)
    except Exception as exc:  # noqa: BLE001
        return False, "validate", f"validate_loaded_data raised: {type(exc).__name__}: {exc}"

    if not vr.valid:
        return False, "validate", "; ".join(vr.issues) or "loaded data failed validation"
    return True, None, None


def register(name: str, source_code: str, probe_path: str) -> RegisterResult:
    """Write, verify, and (only if it passes) keep a loader plugin.

    Steps — any failure removes the freshly-written file so io_loaders/ never
    holds a half-registered plugin:
      1. validate name is a safe module stem
      2. write get_easybci_home()/io_loaders/<name>.py (prepending the marker)
      3. import + assert matches/load contract
      4. dry-run load(probe_path, inspect_only=True)
      5. require validate_loaded_data(result).valid
    """
    if not _SAFE_NAME_RX.match(name or ""):
        return RegisterResult(
            success=False, name=name, stage="write",
            error=f"invalid loader name {name!r}",
            fix_hint="name must be a python identifier: letters/digits/underscore, "
                     "start with a letter, <=64 chars (e.g. 'neuralynx_ncs').",
        )

    probe = Path(probe_path)
    if not probe.exists():
        return RegisterResult(
            success=False, name=name, stage="probe",
            error=f"probe_path does not exist: {probe_path}",
            fix_hint="Pass probe_path = the real file this loader should read, so "
                     "registration can dry-run load() on it.",
        )

    d = io_loaders_dir()
    d.mkdir(parents=True, exist_ok=True)
    target = d / f"{name}.py"

    body = source_code if source_code.lstrip().startswith(LOADER_MARKER) \
        else f"{LOADER_MARKER}\n{source_code}"
    try:
        target.write_text(body, encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        return RegisterResult(
            success=False, name=name, stage="write",
            error=f"cannot write {target}: {type(exc).__name__}: {exc}",
        )

    # Import + contract check.
    try:
        module = _load_module_from_path(name, target)
        plugin = _plugin_from_module(name, target, module)
    except Exception as exc:  # noqa: BLE001
        _unlink_quiet(target)
        stage = "contract" if isinstance(exc, TypeError) else "import"
        return RegisterResult(
            success=False, name=name, stage=stage,
            error=f"{type(exc).__name__}: {exc}",
            fix_hint="The loader must import cleanly and define "
                     "`matches(path) -> bool` and `load(path, inspect_only=False) -> dict`.",
        )

    # Probe + validate gate.
    ok, stage, error = _probe(plugin, str(probe))
    if not ok:
        _unlink_quiet(target)
        return RegisterResult(
            success=False, name=name, stage=stage, error=error,
            fix_hint="load() must return {data:(n_ch,n_samp) float32 ndarray, "
                     "frequency>0, channels (len==n_ch), duration, "
                     "meta{format,source_file,data_unit}}. "
                     "Fix the loader so it passes validate_loaded_data on probe_path.",
        )

    logger.info("Registered IO loader plugin %s -> %s", name, target)
    return RegisterResult(
        success=True, name=name, registered_path=str(target),
    )


def _unlink_quiet(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not remove rejected loader %s: %s", path, exc)
