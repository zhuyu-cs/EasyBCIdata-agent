"""Stub — delivery routing for messaging platforms (removed).

Messaging platform delivery has been removed. This stub preserves the
interface so GatewayRunner initialization doesn't crash.
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class DeliveryTarget:
    platform: str = ""
    chat_id: str = ""
    thread_id: Optional[str] = None


class DeliveryRouter:
    def __init__(self, config=None):
        self._config = config
        self.adapters = {}

    def resolve_targets(self, *args, **kwargs) -> List[DeliveryTarget]:
        return []

    async def deliver(self, *args, **kwargs) -> None:
        pass
