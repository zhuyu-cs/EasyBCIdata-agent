"""Gateway CLI helpers — runtime snapshot and process management.

The actual gateway lives in gateway/platforms/api_server.py; this module
provides status introspection for `easybci status` and `easybci doctor`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class GatewaySnapshot:
    running: bool = False
    manager: str = "manual"
    gateway_pids: List[int] = field(default_factory=list)
    service_installed: bool = False
    service_running: bool = False
    has_process_service_mismatch: bool = False


def get_gateway_runtime_snapshot() -> GatewaySnapshot:
    """Build a snapshot of gateway runtime state for status display."""
    from services.gateway.status import get_running_pid, _pid_exists

    snap = GatewaySnapshot()
    pid = get_running_pid()
    if pid is not None and _pid_exists(pid):
        snap.running = True
        snap.gateway_pids = [pid]

    if os.path.exists("/etc/systemd/system/easybci-gateway.service"):
        snap.service_installed = True
        snap.manager = "systemd"
    elif os.path.exists(os.path.expanduser("~/Library/LaunchAgents/com.easybci.gateway.plist")):
        snap.service_installed = True
        snap.manager = "launchd"
    else:
        snap.manager = "manual"

    return snap


def _format_gateway_pids(pids: List[int]) -> str:
    return ", ".join(str(p) for p in pids)


def gateway_command(*args, **kwargs):
    print("Gateway management commands have been removed.")


def get_service_name(*args, **kwargs):
    return "easybci-gateway"


def kill_gateway_processes(*args, **kwargs):
    pass


def find_gateway_pids(*args, **kwargs) -> List[int]:
    snap = get_gateway_runtime_snapshot()
    return snap.gateway_pids
