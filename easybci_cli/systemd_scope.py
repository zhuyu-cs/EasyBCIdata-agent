"""systemd-run --user --scope launcher for easybci CLI OOM isolation.

Purpose: when opt-in, re-exec easybci under a fresh systemd user-scope cgroup
so kernel OOM (or explicit MemoryMax breach) kills only easybci itself and
its subprocesses -- not the parent tmux/terminal that spawned it. Without
this the terminal is inside tmux-spawn-*.scope and gets collateral-killed
by systemd's cgroup OOM policy.

Fail-open at every step: any preflight failure disables the launcher and
easybci runs normally.

Not applicable on non-Linux (macOS/Windows lack systemd).
"""
from __future__ import annotations

import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

_ENV_ACTIVE_MARKER = "EASYBCI_SYSTEMD_SCOPE_ACTIVE"
_ENV_OPT_IN = "EASYBCI_SYSTEMD_SCOPE"
_SCOPE_UNIT_PREFIX = "easybci-scope-"


def is_active() -> bool:
    """True when the current process is already running inside an easybci
    systemd scope. Set by the launcher via env var + verified via cgroup."""
    if os.environ.get(_ENV_ACTIVE_MARKER) == "1":
        return True
    try:
        cg = Path("/proc/self/cgroup").read_text(encoding="utf-8")
    except OSError:
        return False
    return _SCOPE_UNIT_PREFIX in cg


def preflight_check() -> Tuple[bool, Optional[str]]:
    """Check if systemd-run --user --scope is viable on this host.

    Returns:
        (ok, reason) -- ok=True means "safe to exec"; ok=False means fall
        back to running easybci directly. reason is a short human string
        for the user / doctor report.
    """
    if sys.platform != "linux":
        return False, f"non-Linux platform ({sys.platform}): systemd-run unavailable"

    if is_active():
        return False, "already inside an easybci systemd scope"

    if not shutil.which("systemd-run"):
        return False, "systemd-run binary not found in PATH"

    dbus_env = os.environ.get("DBUS_SESSION_BUS_ADDRESS", "").strip()
    uid = os.getuid()
    dbus_socket = Path(f"/run/user/{uid}/bus")
    if not dbus_env and not dbus_socket.exists():
        return False, (
            "no D-Bus user session detected "
            "(DBUS_SESSION_BUS_ADDRESS unset and /run/user/<uid>/bus absent). "
            "systemd-run --user requires an active user session bus."
        )

    xdg = os.environ.get("XDG_RUNTIME_DIR", "").strip()
    if not xdg or not Path(xdg).is_dir():
        return False, (
            "XDG_RUNTIME_DIR is unset or not a directory -- "
            "user-mode systemd services need a runtime dir"
        )

    if os.environ.get("CI", "").lower() in {"true", "1"} and \
       os.environ.get(_ENV_OPT_IN) != "1":
        return False, (
            "CI environment detected; skipping scope launcher unless "
            "explicitly opt-in via EASYBCI_SYSTEMD_SCOPE=1"
        )

    return True, None


def build_exec_argv(original_argv: List[str], *, python_exe: Optional[str] = None) -> List[str]:
    """Build the argv for os.execvp() to relaunch easybci under systemd-run.

    Args:
        original_argv: sys.argv equivalents (argv[0] is the easybci entry).
        python_exe:    python executable to invoke. Defaults to sys.executable.

    Returns:
        argv list -- argv[0] is 'systemd-run', suitable for os.execvp().
    """
    if python_exe is None:
        python_exe = sys.executable
    unit_name = f"{_SCOPE_UNIT_PREFIX}{os.getpid()}-{int(time.time())}"
    return [
        "systemd-run",
        "--user",
        "--scope",
        f"--unit={unit_name}",
        "--property=MemoryMax=infinity",
        "--property=OOMPolicy=continue",
        "--collect",
        "--same-dir",
        "--",
        python_exe,
        "-m",
        "easybci_cli.main",
        *original_argv[1:],
    ]


def maybe_reexec(argv: List[str], *, opt_in: bool) -> None:
    """When opt_in and preflight passes, replace current process with a
    systemd-run --user --scope wrapper. Otherwise return without side effects.

    This function does NOT return when it re-execs (os.execvp replaces the
    process). Any exception -> fail-open (log warn + return).
    """
    if not opt_in:
        return
    ok, reason = preflight_check()
    if not ok:
        sys.stderr.write(
            f"[easybci] systemd-scope opt-in but not available: {reason}. "
            "Running without scope isolation. Run `easybci doctor` for details.\n"
        )
        return

    exec_argv = build_exec_argv(argv)
    os.environ[_ENV_ACTIVE_MARKER] = "1"
    sys.stderr.write(
        "[easybci] Launching under systemd user scope for OOM isolation "
        "(MemoryMax=infinity, OOMPolicy=continue). Parent tmux/shell will "
        "not be collateral-killed by kernel OOM.\n"
    )
    sys.stderr.flush()
    try:
        os.execvp(exec_argv[0], exec_argv)
    except OSError as exc:
        sys.stderr.write(
            f"[easybci] systemd-run exec failed ({exc}). "
            "Running in-process. Run `easybci doctor` for details.\n"
        )
        os.environ.pop(_ENV_ACTIVE_MARKER, None)
        return


def diagnostic_report() -> dict:
    """Structured status for `easybci doctor` -- all preflight facts, plus
    whether the launcher would be used if requested."""
    ok, reason = preflight_check()
    on_linux = sys.platform == "linux"
    return {
        "platform": sys.platform,
        "systemd_run_present": shutil.which("systemd-run") is not None,
        "dbus_session_bus": os.environ.get("DBUS_SESSION_BUS_ADDRESS") or "",
        "dbus_socket_exists": (
            Path(f"/run/user/{os.getuid()}/bus").exists() if on_linux else False
        ),
        "xdg_runtime_dir": os.environ.get("XDG_RUNTIME_DIR", ""),
        "already_in_scope": is_active(),
        "in_ci": os.environ.get("CI", "").lower() in {"true", "1"},
        "preflight_ok": ok,
        "preflight_reason": reason,
    }
