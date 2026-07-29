"""Stub — gateway restart logic (removed)."""

DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT = 30.0
GATEWAY_SERVICE_RESTART_EXIT_CODE = 75


def parse_restart_drain_timeout(value=None) -> float:
    return DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT
