"""Shared file safety rules used by both tools and ACP shims."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Callable, Optional


def _easybci_home_path() -> Path:
    """Resolve the active EASYBCI_HOME (profile-aware) without circular imports."""
    try:
        from easybci_lib.constants import get_easybci_home  # local import to avoid cycles
        return get_easybci_home()
    except Exception:
        return Path(os.path.expanduser("~/.easybci"))


def build_write_denied_paths(home: str) -> set[str]:
    """Return exact sensitive paths that must never be written."""
    easybci_home = _easybci_home_path()
    return {
        os.path.realpath(p)
        for p in [
            os.path.join(home, ".ssh", "authorized_keys"),
            os.path.join(home, ".ssh", "id_rsa"),
            os.path.join(home, ".ssh", "id_ed25519"),
            os.path.join(home, ".ssh", "config"),
            str(easybci_home / ".env"),
            os.path.join(home, ".bashrc"),
            os.path.join(home, ".zshrc"),
            os.path.join(home, ".profile"),
            os.path.join(home, ".bash_profile"),
            os.path.join(home, ".zprofile"),
            os.path.join(home, ".netrc"),
            os.path.join(home, ".pgpass"),
            os.path.join(home, ".npmrc"),
            os.path.join(home, ".pypirc"),
            "/etc/sudoers",
            "/etc/passwd",
            "/etc/shadow",
        ]
    }


def build_write_denied_prefixes(home: str) -> list[str]:
    """Return sensitive directory prefixes that must never be written."""
    return [
        os.path.realpath(p) + os.sep
        for p in [
            os.path.join(home, ".ssh"),
            os.path.join(home, ".aws"),
            os.path.join(home, ".gnupg"),
            os.path.join(home, ".kube"),
            "/etc/sudoers.d",
            "/etc/systemd",
            os.path.join(home, ".docker"),
            os.path.join(home, ".azure"),
            os.path.join(home, ".config", "gh"),
        ]
    ]


def get_safe_write_root() -> Optional[str]:
    """Return the resolved EASYBCI_WRITE_SAFE_ROOT path, or None if unset."""
    root = os.getenv("EASYBCI_WRITE_SAFE_ROOT", "")
    if not root:
        return None
    try:
        return os.path.realpath(os.path.expanduser(root))
    except Exception:
        return None


def is_write_denied(path: str) -> bool:
    """Return True if path is blocked by the write denylist, safe root, or source data guard."""
    home = os.path.realpath(os.path.expanduser("~"))
    resolved = os.path.realpath(os.path.expanduser(str(path)))

    if resolved in build_write_denied_paths(home):
        return True
    for prefix in build_write_denied_prefixes(home):
        if resolved.startswith(prefix):
            return True

    safe_root = get_safe_write_root()
    if safe_root and not (resolved == safe_root or resolved.startswith(safe_root + os.sep)):
        return True

    from easybci_agent.source_data_guard import is_source_data, is_inside_protected_dir
    if is_source_data(path):
        return True
    if is_inside_protected_dir(path):
        return True

    return False


def get_read_block_error(path: str) -> Optional[str]:
    """Return an error message when a read targets internal EasyBCI cache files."""
    resolved = Path(path).expanduser().resolve()
    easybci_home = _easybci_home_path().resolve()
    blocked_dirs = [
        easybci_home / "skills" / ".hub" / "index-cache",
        easybci_home / "skills" / ".hub",
    ]
    for blocked in blocked_dirs:
        try:
            resolved.relative_to(blocked)
        except ValueError:
            continue
        return (
            f"Access denied: {path} is an internal EasyBCI cache file "
            "and cannot be read directly to prevent prompt injection. "
            "Use the skills_list or skill_view tools instead."
        )
    return None


_read_allowlist_hook_lock = threading.Lock()
_read_allowlist_hook: Optional[Callable[[str], Optional[str]]] = None


def set_read_allowlist_hook(hook: Optional[Callable[[str], Optional[str]]]) -> None:
    """Register a callable that decides whether a read is inside the allowlist.

    The hook receives the raw path string as passed to read_file / search_files
    and must return either ``None`` (allow) or a human-readable error message
    (refuse). Pass ``None`` to clear the hook.

    Only one hook is active at a time. The hook is optional: when unset, all
    reads pass this gate. Personas / toolsets that need a scoped read boundary
    (e.g. the algorithm agent) install their own hook at import time via a
    sync-point module.
    """
    global _read_allowlist_hook
    with _read_allowlist_hook_lock:
        _read_allowlist_hook = hook


def get_read_allowlist_error(path: str) -> Optional[str]:
    """Dispatch to the currently-registered read-allowlist hook.

    Returns ``None`` if no hook is registered or the hook allows the path;
    returns the hook's error string if the read is refused.
    """
    with _read_allowlist_hook_lock:
        hook = _read_allowlist_hook
    if hook is None:
        return None
    try:
        return hook(path)
    except Exception:
        return None


_write_allowlist_hook_lock = threading.Lock()
_write_allowlist_hook: Optional[Callable[[str], Optional[str]]] = None


def set_write_allowlist_hook(hook: Optional[Callable[[str], Optional[str]]]) -> None:
    """Register a callable that decides whether a write is inside the allowlist.

    The hook receives the raw path string as passed to write_file / patch and
    must return either ``None`` (allow) or a human-readable error message
    (refuse). Pass ``None`` to clear the hook.

    Only one hook is active at a time. The hook is optional: when unset, all
    writes pass this gate. Personas that need a scoped write boundary install
    their own hook at import time via a sync-point module.
    """
    global _write_allowlist_hook
    with _write_allowlist_hook_lock:
        _write_allowlist_hook = hook


def get_write_allowlist_error(path: str) -> Optional[str]:
    """Dispatch to the currently-registered write-allowlist hook."""
    with _write_allowlist_hook_lock:
        hook = _write_allowlist_hook
    if hook is None:
        return None
    try:
        return hook(path)
    except Exception:
        return None
