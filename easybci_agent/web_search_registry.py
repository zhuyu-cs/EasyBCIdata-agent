"""
Web Search Provider Registry
============================

Central map of registered web providers. Populated by plugins at import-time
via :meth:`PluginContext.register_web_search_provider`; consumed by the
``web_search`` and ``web_extract`` tool wrappers in :mod:`tools.web_tools` to
dispatch each call to the active backend.

Active selection
----------------
The active provider is chosen by configuration with this precedence:

1. ``web.search_backend`` / ``web.extract_backend`` / ``web.crawl_backend``
   (per-capability override).
2. ``web.backend`` (shared fallback).
3. If exactly one capability-eligible provider is registered AND available,
   use it.
4. Legacy preference order — ``firecrawl`` → ``parallel`` → ``tavily`` →
   ``exa`` → ``searxng`` → ``brave-free`` — filtered by
   availability. Matches the historic ``tools.web_tools._get_backend()``
   candidate order so installs that never set a config key keep landing
   on the same provider they did before the plugin migration.
5. Otherwise ``None`` — the tool surfaces a helpful error pointing at
   ``easybci tools``.

The capability filter (``supports_search`` / ``supports_extract`` /
``supports_crawl``) is applied at every step so a search-only provider
(``brave-free``) configured as ``web.extract_backend`` correctly falls
through to an extract-capable backend.
"""

from __future__ import annotations

import logging
import threading
from typing import Dict, List, NamedTuple, Optional, Tuple

from easybci_agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)


class WebSearchActivationError(NamedTuple):
    """Per-provider activation diagnostic.

    Carried in the ``errors`` list returned by
    :func:`diagnose_active_provider`. ``reason`` is the one-line root cause
    (e.g. "TAVILY_API_KEY environment variable not set"); ``fix_hint`` is
    a single actionable command/instruction the user can act on. ``underlying``
    is reserved for the original exception repr when one exists.
    """

    provider: str
    capability: str
    reason: str
    fix_hint: str
    underlying: str = ""


_providers: Dict[str, WebSearchProvider] = {}
_lock = threading.Lock()


def register_provider(provider: WebSearchProvider) -> None:
    """Register a web search/extract provider.

    Re-registration (same ``name``) overwrites the previous entry and logs
    a debug message — makes hot-reload scenarios (tests, dev loops) behave
    predictably.
    """
    if not isinstance(provider, WebSearchProvider):
        raise TypeError(
            f"register_provider() expects a WebSearchProvider instance, "
            f"got {type(provider).__name__}"
        )
    name = provider.name
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Web provider .name must be a non-empty string")
    with _lock:
        existing = _providers.get(name)
        _providers[name] = provider
    if existing is not None:
        logger.debug(
            "Web provider '%s' re-registered (was %r)",
            name, type(existing).__name__,
        )
    else:
        logger.debug(
            "Registered web provider '%s' (%s)",
            name, type(provider).__name__,
        )


def list_providers() -> List[WebSearchProvider]:
    """Return all registered providers, sorted by name."""
    with _lock:
        items = list(_providers.values())
    return sorted(items, key=lambda p: p.name)


def get_provider(name: str) -> Optional[WebSearchProvider]:
    """Return the provider registered under *name*, or None."""
    if not isinstance(name, str):
        return None
    with _lock:
        return _providers.get(name.strip())


# ---------------------------------------------------------------------------
# Active-provider resolution
# ---------------------------------------------------------------------------


def _read_config_key(*path: str) -> Optional[str]:
    """Resolve a dotted config key from ``config.yaml``. Returns None on miss."""
    try:
        from easybci_cli.config import load_config

        cfg = load_config()
        cur = cfg
        for segment in path:
            if not isinstance(cur, dict):
                return None
            cur = cur.get(segment)
        if isinstance(cur, str) and cur.strip():
            return cur.strip()
    except Exception as exc:
        logger.debug("Could not read config %s: %s", ".".join(path), exc)
    return None


# Legacy preference order — preserves behaviour for users who set no
# ``web.backend`` / ``web.<capability>_backend`` config key at all. Matches
# the historic candidate order in :func:`tools.web_tools._get_backend`
# (paid providers first so existing paid setups don't get downgraded to
# a free tier on upgrade). Filtered by ``is_available()`` at walk time so
# we don't surface a provider the user has no credentials for.
_LEGACY_PREFERENCE = (
    "firecrawl",
    "parallel",
    "tavily",
    "exa",
    "searxng",
    "brave-free",
)


def _resolve(
    configured: Optional[str],
    *,
    capability: str,
    strict_available: bool = False,
) -> Optional[WebSearchProvider]:
    """Resolve the active provider for a capability ("search" | "extract" | "crawl").

    Resolution rules (in order):

    1. **Explicit config wins, ignoring availability** (default behavior). If
       ``web.{capability}_backend`` or ``web.backend`` names a registered
       provider that supports *capability*, return it even if its
       :meth:`is_available` returns False — the dispatcher will surface a
       precise "X_API_KEY is not set" error to the user instead of silently
       routing somewhere else. Matches legacy
       :func:`tools.web_tools._get_backend` behavior for configured names.

       When ``strict_available=True`` (used by ``research_preprocessing`` so
       G-2's "available means usable" gate is honored), an explicit-but-
       unavailable provider falls through to the legacy preference walk.
       The legacy walk only returns providers whose ``is_available()`` is
       True, so strict mode never returns a candidate the agent will then
       silently fail on.

    2. **Single-provider shortcut.** When only one registered provider
       supports *capability* AND ``is_available()`` reports True, return it.

    3. **Legacy preference walk, filtered by availability.** Walk the
       :data:`_LEGACY_PREFERENCE` order (firecrawl → parallel → tavily →
       exa → searxng → brave-free) looking for a provider whose
       ``supports_<capability>()`` is True AND whose ``is_available()`` is
       True. Matches the historic ``tools.web_tools._get_backend()``
       candidate order so users with credentials but no explicit config
       key keep landing on the same provider as pre-migration. This is
       the path that fires when no config key is set — pick the
       highest-priority backend the user actually has credentials for.

    Returns None when no provider is configured AND no available provider
    matches the legacy preference; the dispatcher then returns a "set up a
    provider" error to the user.
    """
    with _lock:
        snapshot = dict(_providers)

    def _capable(p: WebSearchProvider) -> bool:
        if capability == "search":
            return bool(p.supports_search())
        if capability == "extract":
            return bool(p.supports_extract())
        if capability == "crawl":
            return bool(p.supports_crawl())
        return False

    def _is_available_safe(p: WebSearchProvider) -> bool:
        """Wrap ``is_available()`` so a buggy provider doesn't kill resolution."""
        try:
            return bool(p.is_available())
        except Exception as exc:  # noqa: BLE001
            logger.debug("provider %s.is_available() raised %s", p.name, exc)
            return False

    # 1. Explicit config wins — return regardless of is_available() so the
    #    user gets a precise downstream error message rather than a silent
    #    backend switch. Matches _get_backend() in web_tools.py.
    #    Exception: strict_available=True means we keep walking when the
    #    explicit pick can't actually serve a request — that's the gate
    #    research_preprocessing relies on.
    if configured:
        provider = snapshot.get(configured)
        if provider is not None and _capable(provider):
            if strict_available and not _is_available_safe(provider):
                logger.debug(
                    "explicit backend '%s' configured but not is_available — "
                    "strict_available=True, falling through to legacy preference walk",
                    configured,
                )
            else:
                return provider
        elif provider is None:
            logger.debug(
                "web backend '%s' configured but not registered; falling back",
                configured,
            )
        else:
            logger.debug(
                "web backend '%s' configured but does not support '%s'; falling back",
                configured, capability,
            )

    # 2. + 3. Fallback path — filter by availability so we don't surface
    #    a provider the user has no credentials for. Without this filter,
    #    a registered-but-unconfigured provider could end up "active" on
    #    a fresh install with no API keys at all.
    eligible = [
        p for p in snapshot.values()
        if _capable(p) and _is_available_safe(p)
    ]
    if len(eligible) == 1:
        return eligible[0]

    for legacy in _LEGACY_PREFERENCE:
        provider = snapshot.get(legacy)
        if (
            provider is not None
            and _capable(provider)
            and _is_available_safe(provider)
        ):
            return provider

    return None


def get_active_search_provider() -> Optional[WebSearchProvider]:
    """Resolve the currently-active web search provider.

    Reads ``web.search_backend`` (preferred) or ``web.backend`` (shared
    fallback) from config.yaml; falls back per the module docstring.
    """
    explicit = _read_config_key("web", "search_backend") or _read_config_key("web", "backend")
    return _resolve(explicit, capability="search")


def get_active_search_provider_strict() -> Optional[WebSearchProvider]:
    """Resolve the active search provider with availability strictly enforced.

    Identical to :func:`get_active_search_provider` except that an explicit
    ``web.search_backend`` / ``web.backend`` whose ``is_available()`` is False
    falls through to the legacy preference walk (which only returns
    available providers). This is the gate ``research_preprocessing`` uses
    so G-2's "available means usable" promise holds.

    The legacy non-strict helper above is preserved for ``web_search`` /
    ``web_extract`` direct tool calls — there, returning the explicitly-
    configured provider lets the dispatcher surface "TAVILY_API_KEY is not
    set" to the user verbatim instead of silently routing somewhere else.
    """
    explicit = _read_config_key("web", "search_backend") or _read_config_key("web", "backend")
    return _resolve(explicit, capability="search", strict_available=True)


def _capability_filter(provider: WebSearchProvider, capability: str) -> bool:
    if capability == "search":
        return bool(provider.supports_search())
    if capability == "extract":
        return bool(provider.supports_extract())
    if capability == "crawl":
        return bool(provider.supports_crawl())
    return False


def _provider_diagnostic(provider: WebSearchProvider, capability: str) -> WebSearchActivationError:
    """Build a :class:`WebSearchActivationError` for one unavailable provider.

    Calls ``provider.availability_diagnostic()`` if the provider implements
    it (the ABC ships a default returning ``("not configured", "")`` so
    third-party providers without the override still produce a reasonable
    record).
    """
    reason, hint = ("not configured", "")
    try:
        if hasattr(provider, "availability_diagnostic"):
            r, h = provider.availability_diagnostic()
            if isinstance(r, str):
                reason = r
            if isinstance(h, str):
                hint = h
    except Exception as exc:  # noqa: BLE001 — diagnostic must never raise upward
        logger.debug("availability_diagnostic for %s raised: %s", provider.name, exc)
    return WebSearchActivationError(
        provider=provider.name,
        capability=capability,
        reason=reason,
        fix_hint=hint,
    )


def diagnose_active_provider(
    capability: str,
) -> Tuple[Optional[WebSearchProvider], List[WebSearchActivationError]]:
    """Return ``(provider, errors)`` for *capability*.

    *provider* is whatever the strict resolver picked (or None when no
    available provider supports *capability*).

    *errors* is a list of per-provider activation diagnostics for every
    registered, capability-supporting provider whose ``is_available()`` is
    False. The list is **always populated for unavailable providers**, even
    when *provider* is non-None — this lets callers narrate "we picked
    Exa but Tavily is also available if you set TAVILY_API_KEY".

    The empty list means "no provider is registered for this capability"
    (or every registered provider is already available). Callers that need
    to distinguish those cases should also check whether *provider* is None.
    """
    cap = capability if capability in ("search", "extract", "crawl") else "search"
    if cap == "search":
        provider = get_active_search_provider_strict()
    elif cap == "extract":
        explicit = _read_config_key("web", "extract_backend") or _read_config_key("web", "backend")
        provider = _resolve(explicit, capability="extract", strict_available=True)
    else:
        explicit = _read_config_key("web", "crawl_backend") or _read_config_key("web", "backend")
        provider = _resolve(explicit, capability="crawl", strict_available=True)

    errors: List[WebSearchActivationError] = []
    with _lock:
        snapshot = list(_providers.values())
    for p in snapshot:
        if not _capability_filter(p, cap):
            continue
        try:
            available = bool(p.is_available())
        except Exception:
            available = False
        if not available:
            errors.append(_provider_diagnostic(p, cap))
    errors.sort(key=lambda e: e.provider)
    return provider, errors


def get_active_extract_provider() -> Optional[WebSearchProvider]:
    """Resolve the currently-active web extract provider.

    Reads ``web.extract_backend`` (preferred) or ``web.backend`` (shared
    fallback) from config.yaml; falls back per the module docstring.
    """
    explicit = _read_config_key("web", "extract_backend") or _read_config_key("web", "backend")
    return _resolve(explicit, capability="extract")


def get_active_crawl_provider() -> Optional[WebSearchProvider]:
    """Resolve the currently-active web crawl provider.

    Reads ``web.crawl_backend`` (preferred) or ``web.backend`` (shared
    fallback) from config.yaml; falls back per the module docstring.

    Crawl is a niche capability — among built-in providers only Tavily and
    Firecrawl implement it. Callers should expect ``None`` and fall back to
    a different strategy (e.g. summarize-via-LLM) when neither is
    configured.
    """
    explicit = _read_config_key("web", "crawl_backend") or _read_config_key("web", "backend")
    return _resolve(explicit, capability="crawl")


def _reset_for_tests() -> None:
    """Clear the registry. **Test-only.**"""
    with _lock:
        _providers.clear()
