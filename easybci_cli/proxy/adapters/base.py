"""Base class for upstream proxy adapters.

Each concrete adapter wraps a provider's OAuth/credential state and exposes
a uniform interface the proxy server can use to forward requests.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Set


@dataclass
class Credential:
    """A bearer credential resolved from the provider."""
    token: str
    base_url: str
    expires_at: Optional[float] = None


class UpstreamAdapter(ABC):
    """Interface for proxy upstream adapters."""

    display_name: str = "unknown"
    allowed_paths: Set[str] = frozenset({"/v1/chat/completions", "/v1/models"})

    @abstractmethod
    def get_credential(self) -> Credential:
        """Resolve a fresh bearer credential for the upstream."""
        ...

    @abstractmethod
    def is_authenticated(self) -> bool:
        """Return True if the adapter has a valid credential."""
        ...
