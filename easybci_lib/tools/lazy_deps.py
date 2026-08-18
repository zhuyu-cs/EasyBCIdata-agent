"""
Lazy dependency installer for opt-in EasyBCI Agent backends.

Many EasyBCI features (Honcho memory, Bedrock,
Matrix, etc.) require Python packages that not every user needs. The
historical approach was to bundle them all under ``pyproject.toml`` extras
(``easybci-agent[all]``) and install them eagerly at setup time. That has
two problems:

1. **Fragility.** When one extra's transitive dependency becomes
   unavailable on PyPI (quarantined for malware, yanked, broken upload),
   the *entire* ``[all]`` resolve fails and fresh installs silently fall
   back to a stripped tier — losing 10+ unrelated extras at once.

2. **Bloat.** A user who only ever talks to one provider pulls hundreds
   of packages they will never import.

The lazy-install pattern fixes both. Backends call :func:`ensure` at the
top of their first-import path. If the deps are missing, ``ensure`` checks
the ``security.allow_lazy_installs`` config flag (default true) and runs
a venv-scoped pip install. If the user has explicitly disabled lazy
installs, ``ensure`` raises :class:`FeatureUnavailable` with a clear
remediation hint pointing at ``easybci tools`` or the manual pip command.

Security model:

* **Venv-scoped only.** Installs target ``sys.executable`` in the active
  venv. We never touch the system Python.
* **PyPI by package name only.** Specs may be ``"package>=1.0,<2"`` etc.
  We do NOT support ``--index-url`` overrides, ``git+https://``, file:
  paths, or any other input that could be hijacked by a malicious config.
* **Allowlist.** Only specs that appear in :data:`LAZY_DEPS` can be
  installed via this path. A typo in feature name doesn't get the user
  install-anything semantics.
* **Opt-out.** Setting ``security.allow_lazy_installs: false`` in
  ``config.yaml`` disables runtime installs. Users in restricted networks
  or strict security postures can pin themselves to whatever was installed
  at setup time.
* **Offline detection.** If the install fails (offline, mirror down,
  PyPI 404 / quarantine), we surface the failure as
  :class:`FeatureUnavailable` with the actual pip stderr — no silent
  retries, no caching of bad state.

Adding a new backend:

1. Add an entry to :data:`LAZY_DEPS` with the package specs.
2. At the top of the backend module's import path, call
   ``ensure("feature.name")`` inside a try/except that converts
   :class:`FeatureUnavailable` to a useful runtime error.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

_PATCH_STDOUT_ACTIVE = False


# =============================================================================
# Allowlist of lazy-installable backends.
#
# Keys are dot-separated feature names ("namespace.backend"). Values are
# tuples of pip-installable specs that match the corresponding extra in
# pyproject.toml. The framework enforces that only specs from this map
# can flow into the pip install command.
# =============================================================================


LAZY_DEPS: dict[str, tuple[str, ...]] = {
    # ─── Inference providers ───────────────────────────────────────────────
    # Native Anthropic SDK — needed when provider=anthropic (not via
    # OpenRouter / aggregators which use the openai SDK).
    "provider.anthropic": ("anthropic==0.111.0",),
    # AWS Bedrock provider
    "provider.bedrock": ("boto3==1.43.34",),

    # ─── Web search backends ───────────────────────────────────────────────
    "search.exa": ("exa-py==2.14.0",),
    "search.firecrawl": ("firecrawl-py==4.30.2",),
    "search.parallel": ("parallel-web==1.1.0",),

    # ─── Image generation backends ─────────────────────────────────────────
    "image.fal": ("fal-client==0.14.1",),

    # ─── Memory providers ──────────────────────────────────────────────────
    "memory.honcho": ("honcho-ai==2.1.2",),
    "memory.hindsight": ("hindsight-client==0.8.3",),

    # ─── Terminal backends ─────────────────────────────────────────────────
    "terminal.modal": ("modal==1.5.0",),
    "terminal.daytona": ("daytona==0.189.0",),
    "terminal.vercel": ("vercel==0.5.9",),

    # ─── Skills ────────────────────────────────────────────────────────────
    "skill.google_workspace": (
        "google-api-python-client==2.197.0",
        "google-auth-oauthlib==1.4.0",
        "google-auth-httplib2==0.4.0",
    ),
    "skill.youtube": ("youtube-transcript-api==1.2.4",),

    # ─── Neural processing (optional format support) ────────────────────────
    "neural.edflib": ("pyedflib==0.1.42",),
    "neural.pyxdf": ("pyxdf==1.17.5",),
    "neural.pandas": ("pandas==2.3.3",),
    "neural.pynwb": ("pynwb==3.1.3", "hdmf==4.3.1"),
    # Neuroscan Curry .cdt — MNE's Curry reader imports curryreader. Also
    # shipped in the base neural extra; this entry is a first-use safety net so
    # a session without the extra can still read .cdt without the raw-binary
    # fallback.
    "neural.curry": ("curryreader==0.1.2",),
    # MS-Access event DB (Compumedics EVENTS.MDB). pandas-access also needs the
    # system `mdbtools` package (mdb-export) — best-effort; pipeline degrades
    # gracefully when either is absent (see io/psg_annotations.parse_events).
    "neural.mdb": ("pandas-access==0.0.1",),

    # ─── Tools ─────────────────────────────────────────────────────────────
    # ACP adapter (VS Code / Zed / JetBrains integration)
    "tool.acp": ("agent-client-protocol==0.10.1",),
    # Dashboard (`easybci dashboard`)
    "tool.dashboard": (
        "fastapi==0.138.0",
        "uvicorn[standard]==0.49.0",
    ),

    # ─── Cross-lab federation ──────────────────────────────────────────────
    # Ed25519 keys (cryptography) + git-as-transport (gitpython). Lazy-installed
    # on first `easybci federation init / push / pull / status` so a fresh
    # `pip install -e .` works without mandating cryptography for solo users.
    # Mirror entry in `pyproject.toml [project.optional-dependencies].federation`.
    "federation": (
        "cryptography==49.0.0",
        "gitpython==3.1.50",
    ),
}


# =============================================================================
# Runtime allowlist union (floor, not authority).
#
# LAZY_DEPS is the built-in *floor* — it is never enough (shared/00-principles
# §二·五). The agent extends it at runtime via `request()` / the
# request_dependency tool: safe, exact-pinned specs get merged into
# `_RUNTIME_DEPS` (process-local) and persisted to
# ``~/.easybci/runtime_lazy_deps.json`` so the next process treats them as
# "built-in" too. All read paths below (feature_specs / feature_missing /
# ensure / is_available) consult the UNION of LAZY_DEPS ∪ _RUNTIME_DEPS.
# Built-in LAZY_DEPS always wins on key collision.
# =============================================================================


_RUNTIME_DEPS: dict[str, tuple[str, ...]] = {}


def _runtime_deps_path() -> Path:
    from easybci_lib.constants import get_easybci_home
    return get_easybci_home() / "runtime_lazy_deps.json"


def _load_runtime_deps() -> None:
    """Load the persisted runtime allowlist into ``_RUNTIME_DEPS`` (idempotent)."""
    try:
        p = _runtime_deps_path()
    except Exception:  # noqa: BLE001 — home resolution should never hard-fail here
        return
    if not p.exists():
        return
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, (list, tuple)):
                    _RUNTIME_DEPS[str(k)] = tuple(str(s) for s in v)
    except Exception:  # noqa: BLE001 — a corrupt file must not break imports
        pass


def _known_feature(feature: str) -> bool:
    """True if the feature is registered in the built-in floor or runtime union."""
    return feature in LAZY_DEPS or feature in _RUNTIME_DEPS


def _specs_for(feature: str) -> tuple[str, ...]:
    """Return specs for a feature. Built-in LAZY_DEPS wins over the runtime union."""
    if feature in LAZY_DEPS:
        return LAZY_DEPS[feature]
    return _RUNTIME_DEPS.get(feature, ())


# Conservative regex for spec validation — package name plus optional
# version range. Reject anything that looks like a URL, file path, or shell
# metacharacter.
_SAFE_SPEC = re.compile(
    r"^[A-Za-z0-9_][A-Za-z0-9_.\-]*"        # package name
    r"(?:\[[A-Za-z0-9_,\-]+\])?"            # optional [extras]
    r"(?:[<>=!~]=?[A-Za-z0-9_.\-+,*<>=!~]+)?"  # optional version specifier
    r"$"
)


class FeatureUnavailable(RuntimeError):
    """A lazily-installable feature is missing and cannot be made available.

    Either the deps were never installed and the user has disabled lazy
    installs, or the install attempt failed.
    """

    def __init__(self, feature: str, missing: tuple[str, ...], reason: str):
        self.feature = feature
        self.missing = missing
        self.reason = reason
        super().__init__(self._format())

    def _format(self) -> str:
        spec_list = " ".join(repr(s) for s in self.missing)
        return (
            f"Feature {self.feature!r} unavailable: {self.reason}. "
            f"To enable manually: uv pip install {spec_list}  "
            f"(or: pip install {spec_list})."
        )


@dataclass(frozen=True)
class _InstallResult:
    success: bool
    stdout: str
    stderr: str


# =============================================================================
# Internals
# =============================================================================


def _allow_lazy_installs() -> bool:
    """Return the ``security.allow_lazy_installs`` config flag.

    Defaults to True. If config is unreadable we fail open (allow), because
    refusing to install would lock people out of their own backends; the
    decision to block is an explicit user opt-in.
    """
    if os.environ.get("EASYBCI_DISABLE_LAZY_INSTALLS") == "1":
        return False
    try:
        from easybci_cli.config import load_config
        cfg = load_config()
    except Exception:
        return True
    sec = cfg.get("security") or {}
    val = sec.get("allow_lazy_installs", True)
    return bool(val)


def _spec_is_safe(spec: str) -> bool:
    """Reject pip specs that contain URLs, paths, or shell metacharacters."""
    if not spec or len(spec) > 200:
        return False
    if any(ch in spec for ch in (";", "|", "&", "`", "$", "\n", "\r", "\t", "\\")):
        return False
    if spec.startswith(("-", "/", ".")) or "://" in spec or "@" in spec:
        return False
    return bool(_SAFE_SPEC.match(spec))


# Agent-requested deps are held to a STRICTER bar than built-in LAZY_DEPS:
# exact pin only (==X.Y.Z), no ranges/no bare names. This upholds the project's
# exact-pinning hard constraint (shared/00-principles §三) — the runtime
# allowlist must never smuggle in an unpinned/range spec.
_SAFE_PINNED = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.\-]*"
                          r"(?:\[[A-Za-z0-9_,\-]+\])?"   # optional [extras]
                          r"==[A-Za-z0-9_.\-+]+$")


def _spec_is_exact_pinned(spec: str) -> bool:
    """Accept only exact-pinned specs (``==X.Y.Z``); reject ranges/urls/git/shell."""
    return bool(_SAFE_PINNED.match(spec)) and _spec_is_safe(spec)


def _pkg_name_from_spec(spec: str) -> str:
    """Extract the bare package name from a pip spec.

    ``"mautrix[encryption]>=0.20"`` → ``"mautrix"``
    ``"anthropic==0.111.0"`` → ``"anthropic"``
    """
    m = re.match(r"^([A-Za-z0-9_][A-Za-z0-9_.\-]*)", spec)
    return m.group(1) if m else spec


def _specifier_from_spec(spec: str) -> str:
    """Extract just the version-specifier portion of a pip spec.

    ``"honcho-ai==2.1.2"`` → ``"==2.1.2"``
    ``"mautrix[encryption]>=0.20,<1"`` → ``">=0.20,<1"``
    ``"package"`` → ``""`` (no version constraint)
    """
    # Strip the package name + optional [extras] block.
    m = re.match(r"^[A-Za-z0-9_][A-Za-z0-9_.\-]*(?:\[[A-Za-z0-9_,\-]+\])?", spec)
    if not m:
        return ""
    return spec[m.end():]


def _is_satisfied(spec: str) -> bool:
    """Is ``spec`` already satisfied in the current env?

    Checks both presence AND version. If the package is installed at a
    version outside the spec's range, returns False so the caller will
    upgrade/downgrade to the pinned version. This is what makes
    ``easybci update`` propagate pin bumps in :data:`LAZY_DEPS` to already-
    installed backends instead of silently leaving stale versions in place.

    If ``packaging`` is unavailable for any reason (it's a transitive of
    pip so this should never happen), we fall back to a presence-only check
    so we err on the side of "don't churn".
    """
    pkg = _pkg_name_from_spec(spec)
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:
        return False
    try:
        installed = version(pkg)
    except PackageNotFoundError:
        return False
    except Exception:
        return False

    spec_tail = _specifier_from_spec(spec)
    if not spec_tail:
        # Bare ``"package"`` — no version constraint, presence is enough.
        return True

    try:
        from packaging.specifiers import InvalidSpecifier, SpecifierSet
        from packaging.version import InvalidVersion, Version
    except ImportError:
        # packaging unavailable — fall back to "installed counts as satisfied".
        return True

    try:
        return Version(installed) in SpecifierSet(spec_tail)
    except (InvalidSpecifier, InvalidVersion, Exception):
        # Malformed spec or installed version we can't parse — don't churn.
        return True


def _is_present(spec: str) -> bool:
    """Cheap presence-only check (package name installed at any version).

    Used by :func:`active_features` to detect backends the user has
    previously activated, regardless of whether the version pin moved.
    """
    pkg = _pkg_name_from_spec(spec)
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:
        return False
    try:
        version(pkg)
        return True
    except PackageNotFoundError:
        return False
    except Exception:
        return False


def _venv_pip_install(specs: tuple[str, ...], *, timeout: int = 300) -> _InstallResult:
    """Install ``specs`` into the active venv using uv → pip → ensurepip ladder.

    Mirrors the strategy in ``easybci_cli.tools_config._pip_install`` but
    kept independent here so this module has no CLI dependency.
    """
    if not specs:
        return _InstallResult(True, "", "")

    venv_root = Path(sys.executable).parent.parent
    uv_env = {**os.environ, "VIRTUAL_ENV": str(venv_root)}

    # Tier 1: uv (preferred — fast, doesn't need pip in the venv)
    uv_bin = shutil.which("uv")
    if uv_bin:
        try:
            r = subprocess.run(
                [uv_bin, "pip", "install", *specs],
                capture_output=True, text=True, timeout=timeout, env=uv_env,
            )
            if r.returncode == 0:
                return _InstallResult(True, r.stdout or "", r.stderr or "")
            logger.debug("uv pip install failed: %s", r.stderr)
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.debug("uv invocation failed: %s", e)

    # Tier 2: python -m pip (with ensurepip bootstrap if needed)
    pip_cmd = [sys.executable, "-m", "pip"]
    try:
        probe = subprocess.run(
            pip_cmd + ["--version"],
            capture_output=True, text=True, timeout=15,
        )
        if probe.returncode != 0:
            raise FileNotFoundError("pip not in venv")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        try:
            subprocess.run(
                [sys.executable, "-m", "ensurepip", "--upgrade", "--default-pip"],
                capture_output=True, text=True, timeout=120, check=True,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            return _InstallResult(False, "",
                                  f"pip not available and ensurepip failed: {e}")

    try:
        r = subprocess.run(
            pip_cmd + ["install", *specs],
            capture_output=True, text=True, timeout=timeout,
        )
        return _InstallResult(r.returncode == 0, r.stdout or "", r.stderr or "")
    except subprocess.TimeoutExpired as e:
        return _InstallResult(False, "", f"pip install timed out: {e}")
    except Exception as e:
        return _InstallResult(False, "", f"pip install failed: {e}")


# =============================================================================
# Public API
# =============================================================================


def feature_specs(feature: str) -> tuple[str, ...]:
    """Return the registered specs for a feature, or raise KeyError.

    Consults the union LAZY_DEPS ∪ _RUNTIME_DEPS (built-in floor wins).
    """
    if not _known_feature(feature):
        raise KeyError(f"Unknown lazy feature: {feature!r}")
    return _specs_for(feature)


def feature_missing(feature: str) -> tuple[str, ...]:
    """Return the subset of specs for ``feature`` not currently installed."""
    return tuple(s for s in feature_specs(feature) if not _is_satisfied(s))


def ensure(feature: str, *, prompt: bool = True) -> None:
    """Make sure all packages for ``feature`` are importable.

    If they're missing, attempts to install them in the active venv. Raises
    :class:`FeatureUnavailable` if the user has disabled lazy installs or
    if the install attempt fails.

    ``prompt``: when True (default) and stdin is a TTY, asks the user to
    confirm before installing. Non-interactive callers (gateway, 
    batch) get prompt=False and skip the confirmation — config flag is
    the gate in that case.
    """
    if not _known_feature(feature):
        raise FeatureUnavailable(
            feature, (),
            f"feature {feature!r} not in the dependency allowlist "
            "(use the request_dependency tool to add an exact-pinned package, "
            "rather than a raw `pip install`)",
        )

    missing = feature_missing(feature)
    if not missing:
        return

    # Validate every spec against the allowlist + safety regex. Belt and
    # braces — the keys-in-LAZY_DEPS check above already constrains this.
    for spec in missing:
        if not _spec_is_safe(spec):
            raise FeatureUnavailable(
                feature, missing,
                f"refusing to install unsafe spec {spec!r}"
            )

    if not _allow_lazy_installs():
        raise FeatureUnavailable(
            feature, missing,
            "lazy installs disabled (security.allow_lazy_installs=false)"
        )

    if _PATCH_STDOUT_ACTIVE:
        logger.info("Auto-installing %s for feature %r (CLI mode, interactive prompt unavailable)",
                    " ".join(missing), feature)
    elif prompt and sys.stdin.isatty() and sys.stdout.isatty():
        spec_list = ", ".join(missing)
        try:
            answer = input(
                f"\nFeature {feature!r} requires: {spec_list}\n"
                f"Install into the active venv now? [Y/n] "
            ).strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = "n"
        if answer and answer not in ("y", "yes"):
            raise FeatureUnavailable(
                feature, missing, "user declined install at prompt"
            )

    logger.info("Lazy-installing %s for feature %r", " ".join(missing), feature)
    result = _venv_pip_install(missing)
    if not result.success:
        # Surface the actual pip error so the user can debug PyPI-side
        # issues (404 quarantine, network down, etc.).
        snippet = (result.stderr or result.stdout or "").strip()
        if snippet:
            # Clip to a readable size — pip can dump pages of resolution traces.
            snippet = snippet[-2000:]
        raise FeatureUnavailable(
            feature, missing,
            f"pip install failed: {snippet or 'no error output'}"
        )

    # Verify post-install. importlib.metadata caches per-process, so if we
    # just installed something the cache may not see it without a refresh.
    try:
        import importlib.metadata as _md
        if hasattr(_md, "_cache_clear"):
            _md._cache_clear()  # type: ignore[attr-defined]
    except Exception:
        pass

    still_missing = feature_missing(feature)
    if still_missing:
        raise FeatureUnavailable(
            feature, still_missing,
            "install reported success but packages still not importable "
            "(may require Python restart)"
        )

    logger.info("Lazy install complete for feature %r", feature)


def request(feature: str, specs: tuple[str, ...], *, prompt: bool = False) -> None:
    """Agent-controlled dependency add — the sanctioned alternative to raw pip.

    Stricter than the built-in :data:`LAZY_DEPS`: every spec must be
    exact-pinned (``==X.Y.Z``); ranges / urls / git+ / shell metacharacters are
    rejected (upholds the exact-pinning hard constraint, shared/00-principles §三).

    On accept, the feature is merged into the runtime allowlist union
    (:data:`_RUNTIME_DEPS`) and persisted to ``~/.easybci/runtime_lazy_deps.json``
    so it is recognised on the next process too (the floor grows — a flywheel),
    then :func:`ensure` installs it (the ``security.allow_lazy_installs`` global
    kill-switch still applies via ``ensure``).

    Raises :class:`FeatureUnavailable` on bad input or install failure.
    """
    if not feature or not specs:
        raise FeatureUnavailable(feature, tuple(specs or ()),
                                 "feature and specs are both required")
    for spec in specs:
        if not _spec_is_exact_pinned(spec):
            raise FeatureUnavailable(
                feature, tuple(specs),
                f"spec {spec!r} must be exact-pinned (==X.Y.Z); "
                "no ranges / urls / git+ / shell metacharacters",
            )
    specs = tuple(specs)
    # Register into the process-local union BEFORE ensure(), so the
    # membership/spec lookups inside ensure() see it.
    _RUNTIME_DEPS[feature] = specs
    # Persist so the next process treats it as "built-in" (flywheel). Failure to
    # persist must not block the install — the process-local union still works.
    try:
        p = _runtime_deps_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        cur: dict[str, Any] = {}
        if p.exists():
            loaded = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                cur = loaded
        cur[feature] = list(specs)
        p.write_text(json.dumps(cur, indent=2), encoding="utf-8")
    except Exception:  # noqa: BLE001
        logger.debug("failed to persist runtime dep %r (continuing)", feature)
    ensure(feature, prompt=prompt)


def is_available(feature: str) -> bool:
    """Return True if the feature's deps are already satisfied."""
    if not _known_feature(feature):
        return False
    return not feature_missing(feature)


def feature_install_command(feature: str) -> Optional[str]:
    """Return the ``pip install`` command a user could run manually, or None."""
    if not _known_feature(feature):
        return None
    specs = _specs_for(feature)
    return "uv pip install " + " ".join(repr(s) for s in specs)


def active_features() -> list[str]:
    """Return the list of features the user has ever lazy-installed.

    A feature counts as "active" if at least one of its declared packages
    is currently installed in the venv (presence check, ignoring version).
    Features the user has never enabled stay quiet.

    Used by ``easybci update`` to figure out which lazy backends need a
    refresh pass when pins move in :data:`LAZY_DEPS`.
    """
    active = []
    for feature, specs in LAZY_DEPS.items():
        if any(_is_present(s) for s in specs):
            active.append(feature)
    return active


def refresh_active_features(*, prompt: bool = False) -> dict[str, str]:
    """Re-run ``ensure`` for every feature the user has previously activated.

    Returns a ``{feature: status}`` map where status is one of:
        ``"current"``  — pins already satisfied, no install run
        ``"refreshed"`` — pins were stale, reinstall succeeded
        ``"failed: <reason>"`` — install attempt failed; caller decides
                                  whether to surface it (we don't raise)
        ``"skipped: <reason>"`` — gated off (config flag, user decline)

    Intended for ``easybci update``. Never raises; lazy-install failures
    here must not block the rest of the update flow.
    """
    results: dict[str, str] = {}
    for feature in active_features():
        missing = feature_missing(feature)
        if not missing:
            results[feature] = "current"
            continue
        try:
            ensure(feature, prompt=prompt)
            results[feature] = "refreshed"
        except FeatureUnavailable as e:
            # Distinguish "user opted out" from "install failed" so the
            # update command can render the right message.
            if "lazy installs disabled" in str(e) or "declined" in str(e):
                results[feature] = f"skipped: {e.reason}"
            else:
                results[feature] = f"failed: {e.reason}"
        except Exception as e:
            results[feature] = f"failed: {e}"
    return results


def ensure_and_bind(
    feature: str,
    importer: Callable[[], dict[str, Any]],
    target_globals: dict,
    *,
    prompt: bool = False,
) -> bool:
    """Ensure a feature is installed, then rebind names into the caller's globals.

    Combines :func:`ensure` with a post-install import step that rebinds
    module-level names.  This eliminates the error-prone pattern of manually
    listing every global that needs updating after lazy-install.

    ``importer`` is a zero-arg callable that returns a dict of
    ``{name: value}`` for all symbols the caller needs rebound.  It is called
    only after :func:`ensure` succeeds (or if the packages are already
    installed).

    Returns True on success, False if deps couldn't be installed or imported.

    Example usage in a feature adapter::

        def check_anthropic_requirements() -> bool:
            if ANTHROPIC_AVAILABLE:
                return True
            def _import():
                import anthropic
                return {
                    "anthropic": anthropic,
                    "ANTHROPIC_AVAILABLE": True,
                }
            return ensure_and_bind("provider.anthropic", _import, globals(), prompt=False)
    """
    try:
        ensure(feature, prompt=prompt)
    except (FeatureUnavailable, Exception):
        return False

    try:
        bindings = importer()
    except ImportError:
        return False

    target_globals.update(bindings)
    return True


# Load the persisted runtime allowlist once at import so agent-requested deps
# from prior sessions are recognised (the floor grows across runs).
_load_runtime_deps()
