"""Managed-tool gateway helpers — stub.

The managed gateway has been removed.  All functions in this module are
kept as no-op stubs so existing call-sites don't break.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass(frozen=True)
class ManagedToolGatewayConfig:
    vendor: str
    gateway_origin: str
    user_token: str
    managed_mode: bool


def resolve_managed_tool_gateway(
    vendor: str,
    gateway_builder: Optional[Callable[[str], str]] = None,
    token_reader: Optional[Callable[[], Optional[str]]] = None,
) -> Optional[ManagedToolGatewayConfig]:
    """Always returns None — managed gateway removed."""
    return None


def is_managed_tool_gateway_ready(
    vendor: str,
    gateway_builder: Optional[Callable[[str], str]] = None,
    token_reader: Optional[Callable[[], Optional[str]]] = None,
) -> bool:
    """Always returns False — managed gateway removed."""
    return False


def build_vendor_gateway_url(vendor: str) -> str:
    """Stub — returns empty string."""
    return ""


def read_bci_team_access_token() -> Optional[str]:
    """Stub — always returns None."""
    return None
