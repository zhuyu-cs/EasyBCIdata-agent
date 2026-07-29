"""
EasyBCI Gateway — WebUI backend (api_server).

Provides session management and the HTTP/WebSocket server
that the React WebUI connects to.
"""

from .config import GatewayConfig, PlatformConfig, load_gateway_config
from .session import (
    SessionContext,
    SessionStore,
    SessionResetPolicy,
    build_session_context_prompt,
)

__all__ = [
    "GatewayConfig",
    "PlatformConfig",
    "load_gateway_config",
    "SessionContext",
    "SessionStore",
    "SessionResetPolicy",
    "build_session_context_prompt",
]
