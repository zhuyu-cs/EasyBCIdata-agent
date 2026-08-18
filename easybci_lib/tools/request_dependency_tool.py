#!/usr/bin/env python3
"""request_dependency — controlled agent dependency extension.

The sanctioned alternative to a raw ``pip install`` in the terminal (which
bypasses the project's exact-pinning, safety-scanning, and reproducibility
guarantees). When the agent finds a feature blocked on a package that is not in
the built-in :data:`lazy_deps.LAZY_DEPS` floor, it calls this tool with an
exact-pinned package spec. The install flows through the same
:func:`lazy_deps.request` → :func:`lazy_deps.ensure` path as every other lazy
backend, so all existing safety boundaries still apply:

* exact-pin only (``==X.Y.Z``) — ranges / urls / git+ / shell injection rejected;
* the ``security.allow_lazy_installs`` config flag (and the
  ``EASYBCI_DISABLE_LAZY_INSTALLS=1`` env override) remain a global kill-switch;
* venv-scoped install via the uv → pip → ensurepip ladder;
* the accepted spec is persisted to ``~/.easybci/runtime_lazy_deps.json`` so it
  becomes part of the recognised floor next session (the flywheel).

This tool is a general capability (not neural-specific): it belongs to the
``dependency`` toolset and is spread into ``_EASYBCI_CORE_TOOLS`` so every
code-capable session can reach it.
"""
from __future__ import annotations

import json
import logging

from easybci_lib.tools.registry import registry

logger = logging.getLogger(__name__)


REQUEST_DEPENDENCY_SCHEMA = {
    "name": "request_dependency",
    "description": (
        "Install a Python package the current session needs but that is not in "
        "the built-in dependency allowlist. PREFER THIS over `pip install` in the "
        "terminal — terminal pip bypasses version-pinning, safety scanning, and "
        "reproducibility. You MUST give an exact release version (pin), not a "
        "range: version='1.2.3', never '>=1.0'. Installs into the active venv and "
        "persists to the runtime allowlist so it's recognised next session too. "
        "Ranges / urls / git+ / shell metacharacters are rejected. Respects the "
        "security.allow_lazy_installs kill-switch. After a successful install, "
        "most pure-Python packages are importable immediately (no restart)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "package": {
                "type": "string",
                "description": "PyPI package name, e.g. 'mne-bids'.",
            },
            "version": {
                "type": "string",
                "description": "Exact release version to pin, e.g. '1.2.3'. "
                               "Ranges (>=, ~=, *) are NOT accepted.",
            },
            "purpose": {
                "type": "string",
                "description": "One line: why this package is needed (recorded in "
                               "the result note for auditing). Optional.",
            },
        },
        "required": ["package", "version"],
    },
}


def _handle_request_dependency(args, **kw):
    """Controlled dependency install for the agent. Never raises into the loop.

    Enforces exact-pin + safe-spec, then delegates to lazy_deps.request(), which
    persists to the runtime allowlist and runs the standard ensure() install
    (allow_lazy_installs kill-switch still applies).
    """
    if not isinstance(args, dict):
        return json.dumps({"success": False, "error": "invalid args"})
    package = (args.get("package") or "").strip()
    version = (args.get("version") or "").strip()
    purpose = (args.get("purpose") or "").strip()
    if not package or not version:
        return json.dumps({
            "success": False, "error": "package and version are both required",
            "fix_hint": "version must be an exact release, e.g. version='1.2.3'",
        })
    spec = f"{package}=={version}"
    try:
        from easybci_lib.tools import lazy_deps as ld
        if not ld._spec_is_exact_pinned(spec):
            return json.dumps({
                "success": False, "error": f"unsafe or imprecise spec {spec!r}",
                "fix_hint": "use an exact pin: package + version==X.Y.Z; "
                            "no ranges / urls / git+ / shell metacharacters",
            })
        if not ld._allow_lazy_installs():
            return json.dumps({
                "success": False,
                "error": "dependency installs are disabled "
                         "(security.allow_lazy_installs=false)",
                "fix_hint": "ask the user to enable installs, or proceed without "
                            "this package",
            })
        feature = f"adhoc.{package.replace('-', '_')}"
        ld.request(feature, (spec,), prompt=False)
    except Exception as exc:  # noqa: BLE001 (FeatureUnavailable included)
        return json.dumps({
            "success": False, "error": f"{type(exc).__name__}: {exc}",
            "fix_hint": "verify the package name + version exist on PyPI, or "
                        "that the network / index is reachable",
        })
    return json.dumps({
        "success": True, "installed": spec, "feature": feature,
        "note": (
            f"Installed {spec} into the active venv"
            + (f" (purpose: {purpose})" if purpose else "")
            + ". Persisted to the runtime allowlist — importable now for most "
              "pure-Python packages; a few native/compiled packages may need a "
              "Python restart."
        ),
    }, default=str)


registry.register(
    name="request_dependency",
    toolset="dependency",
    schema=REQUEST_DEPENDENCY_SCHEMA,
    handler=_handle_request_dependency,
    emoji="\U0001f4e5",  # 📥
)
