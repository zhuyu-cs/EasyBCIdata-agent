"""
Platform adapters for the EasyBCI gateway.

Only the API server adapter is used (serves WebUI backend).
"""

from .base import BasePlatformAdapter, MessageEvent, SendResult

__all__ = [
    "BasePlatformAdapter",
    "MessageEvent",
    "SendResult",
]
