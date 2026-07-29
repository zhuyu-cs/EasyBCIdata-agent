"""Stub — subscription feature detection removed.

All feature checks return False/empty. Call sites in setup.py use the
return value to decide whether managed gateway tools are available.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class _FeatureStatus:
    managed_by_bci_team: bool = False
    available: bool = False
    current_provider: Optional[str] = None
    direct_override: bool = False


@dataclass
class SubscriptionFeatures:
    web: _FeatureStatus = field(default_factory=_FeatureStatus)
    modal: _FeatureStatus = field(default_factory=_FeatureStatus)
    bci_team_auth_present: bool = False


def get_bci_team_subscription_features(config: Any = None) -> SubscriptionFeatures:
    """Always returns empty features — subscription support removed."""
    return SubscriptionFeatures()


def apply_bci_team_managed_defaults(config: Any = None, **kwargs: Any) -> set:
    """Stub — always returns empty set. Managed defaults removed."""
    return set()
