"""
Canonical model catalogs and lightweight validation helpers.

Add, remove, or reorder entries here — both `easybci setup` and
`easybci` provider-selection will pick up the change automatically.
"""

from __future__ import annotations

import json
import os
import urllib.request
import urllib.error
import time
from difflib import get_close_matches
from pathlib import Path
from typing import Any, NamedTuple, Optional

from easybci_cli import __version__ as _EASYBCI_VERSION

# Identify ourselves so endpoints fronted by Cloudflare's Browser Integrity
# Check (error 1010) don't reject the default ``Python-urllib/*`` signature.
_EASYBCI_USER_AGENT = f"easybci-cli/{_EASYBCI_VERSION}"

OPENROUTER_MODELS: list[tuple[str, str]] = []
VERCEL_AI_GATEWAY_MODELS: list[tuple[str, str]] = []

_openrouter_catalog_cache: list[tuple[str, str]] | None = None
_ai_gateway_catalog_cache: list[tuple[str, str]] | None = None


_PROVIDER_MODELS: dict[str, list[str]] = {}

# ---------------------------------------------------------------------------
# Canonical provider list — single source of truth for provider identity.
# Every code path that lists, displays, or iterates providers derives from
# this list:  easybci model, /model, list_authenticated_providers.
#
# Fields:
#   slug        — internal provider ID (used in config.yaml, --provider flag)
#   label       — short display name
#   tui_desc    — longer description for the `easybci model` interactive picker
# ---------------------------------------------------------------------------

class ProviderEntry(NamedTuple):
    slug: str
    label: str
    tui_desc: str   # detailed description for `easybci model` TUI

CANONICAL_PROVIDERS: list[ProviderEntry] = [
    ProviderEntry("custom",  "Custom API (Relay)",           "OpenAI-compatible relay endpoint (provide your own URL + API key)"),
]

# Auto-extend CANONICAL_PROVIDERS with any provider registered in providers/
# that is not already in the list above.  Adding plugins/model-providers/<name>/
# is sufficient to expose a new provider in the model picker, /model, and all
# downstream consumers — no edits to this file needed.
_canonical_slugs = {p.slug for p in CANONICAL_PROVIDERS}
try:
    from services.providers import list_providers as _list_providers_for_canonical
    for _pp in _list_providers_for_canonical():
        if _pp.name in _canonical_slugs:
            continue
        if _pp.auth_type in {"oauth_device_code", "oauth_external", "external_process", "aws_sdk"}:
            continue  # non-api-key flows need bespoke picker UX; skip auto-inject
        _label = _pp.display_name or _pp.name
        _desc = _pp.description or f"{_label} (direct API)"
        CANONICAL_PROVIDERS.append(ProviderEntry(_pp.name, _label, _desc))
        _canonical_slugs.add(_pp.name)
except Exception:
    pass

# Derived dicts — used throughout the codebase
_PROVIDER_LABELS = {p.slug: p.label for p in CANONICAL_PROVIDERS}
_PROVIDER_LABELS["custom"] = "Custom endpoint"  # special case: not a named provider


_PROVIDER_ALIASES = {
    "ollama": "custom",
    "ollama-cloud": "custom",
    "ollama_cloud": "custom",
    "vllm": "custom",
    "llamacpp": "custom",
    "llama.cpp": "custom",
    "llama-cpp": "custom",
    "lmstudio": "custom",
    "lm-studio": "custom",
    "lm_studio": "custom",
    "local-llm": "custom",
    "local": "custom",
    "custom-api": "custom",
    "relay": "custom",
}


def get_default_model_for_provider(provider: str) -> str:
    """Return the default model for a provider, or empty string if unknown.

    Uses the first entry in _PROVIDER_MODELS as the default.  This is the
    model a user would be offered first in the ``easybci model`` picker.

    Used as a fallback when the user has configured a provider but never
    selected a model (e.g. ``easybci auth add <provider>`` without
    ``easybci model``).
    """
    models = _PROVIDER_MODELS.get(provider, [])
    return models[0] if models else ""


def _openrouter_model_is_free(pricing: Any) -> bool:
    """Return True when both prompt and completion pricing are zero."""
    if not isinstance(pricing, dict):
        return False
    try:
        return float(pricing.get("prompt", "0")) == 0 and float(pricing.get("completion", "0")) == 0
    except (TypeError, ValueError):
        return False


def _openrouter_model_supports_tools(item: Any) -> bool:
    """Return True when the model's ``supported_parameters`` advertise tool calling.

    easybci-agent is tool-calling-first — every provider path assumes the model
    can invoke tools. Models that don't advertise ``tools`` in their
    ``supported_parameters`` (e.g. image-only or completion-only models) cannot
    be driven by the agent loop and would fail at the first tool call.

    **Permissive when the field is missing.** Some OpenRouter-compatible gateways
    (private mirrors, older catalog snapshots) don't populate
    ``supported_parameters`` at all. Treat that as "unknown capability → allow"
    so the picker doesn't silently empty for those users. Only hide models
    whose ``supported_parameters`` is an explicit list that omits ``tools``.

    Ported from Kilo-Org/kilocode#9068.
    """
    if not isinstance(item, dict):
        return True
    params = item.get("supported_parameters")
    if not isinstance(params, list):
        # Field absent / malformed / None — be permissive.
        return True
    return "tools" in params


def fetch_openrouter_models(
    timeout: float = 8.0,
    *,
    force_refresh: bool = False,
) -> list[tuple[str, str]]:
    """Return the curated OpenRouter picker list, refreshed from the live catalog when possible."""
    global _openrouter_catalog_cache

    if _openrouter_catalog_cache is not None and not force_refresh:
        return list(_openrouter_catalog_cache)

    # Prefer the remotely-hosted catalog manifest; fall back to the in-repo
    # snapshot when the manifest is unreachable. Both are curated lists that
    # drive the picker; the OpenRouter live /v1/models filter (tool support,
    # free pricing) is applied on top either way.
    try:
        from easybci_cli.model_catalog import get_curated_openrouter_models
        remote = get_curated_openrouter_models()
    except Exception:
        remote = None
    fallback = list(remote) if remote else list(OPENROUTER_MODELS)
    preferred_ids = [mid for mid, _ in fallback]

    try:
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/models",
            headers={"Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode())
    except Exception:
        return list(_openrouter_catalog_cache or fallback)

    live_items = payload.get("data", [])
    if not isinstance(live_items, list):
        return list(_openrouter_catalog_cache or fallback)

    live_by_id: dict[str, dict[str, Any]] = {}
    for item in live_items:
        if not isinstance(item, dict):
            continue
        mid = str(item.get("id") or "").strip()
        if not mid:
            continue
        live_by_id[mid] = item

    curated: list[tuple[str, str]] = []
    for preferred_id in preferred_ids:
        live_item = live_by_id.get(preferred_id)
        if live_item is None:
            continue
        # Hide models that don't advertise tool-calling support — easybci-agent
        # requires it and surfacing them leads to immediate runtime failures
        # when the user selects them. Ported from Kilo-Org/kilocode#9068.
        if not _openrouter_model_supports_tools(live_item):
            continue
        desc = "free" if _openrouter_model_is_free(live_item.get("pricing")) else ""
        curated.append((preferred_id, desc))

    if not curated:
        return list(_openrouter_catalog_cache or fallback)

    first_id, _ = curated[0]
    curated[0] = (first_id, "recommended")
    _openrouter_catalog_cache = curated
    return list(curated)


def model_ids(*, force_refresh: bool = False) -> list[str]:
    """Return just the OpenRouter model-id strings."""
    return [mid for mid, _ in fetch_openrouter_models(force_refresh=force_refresh)]


def get_curated_bci_team_model_ids() -> list[str]:
    """Stub — BCI Team Portal removed. Returns empty list."""
    return []


def _ai_gateway_model_is_free(pricing: Any) -> bool:
    """Return True if an AI Gateway model has $0 input AND output pricing."""
    if not isinstance(pricing, dict):
        return False
    try:
        return float(pricing.get("input", "0")) == 0 and float(pricing.get("output", "0")) == 0
    except (TypeError, ValueError):
        return False


def fetch_ai_gateway_models(
    timeout: float = 8.0,
    *,
    force_refresh: bool = False,
) -> list[tuple[str, str]]:
    """Return the curated AI Gateway picker list, refreshed from the live catalog when possible."""
    global _ai_gateway_catalog_cache

    if _ai_gateway_catalog_cache is not None and not force_refresh:
        return list(_ai_gateway_catalog_cache)

    from easybci_lib.constants import AI_GATEWAY_BASE_URL

    fallback = list(VERCEL_AI_GATEWAY_MODELS)
    preferred_ids = [mid for mid, _ in fallback]

    try:
        req = urllib.request.Request(
            f"{AI_GATEWAY_BASE_URL.rstrip('/')}/models",
            headers={"Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode())
    except Exception:
        return list(_ai_gateway_catalog_cache or fallback)

    live_items = payload.get("data", [])
    if not isinstance(live_items, list):
        return list(_ai_gateway_catalog_cache or fallback)

    live_by_id: dict[str, dict[str, Any]] = {}
    for item in live_items:
        if not isinstance(item, dict):
            continue
        mid = str(item.get("id") or "").strip()
        if not mid:
            continue
        live_by_id[mid] = item

    curated: list[tuple[str, str]] = []
    for preferred_id in preferred_ids:
        live_item = live_by_id.get(preferred_id)
        if live_item is None:
            continue
        desc = "free" if _ai_gateway_model_is_free(live_item.get("pricing")) else ""
        curated.append((preferred_id, desc))

    if not curated:
        return list(_ai_gateway_catalog_cache or fallback)

    # If the live catalog offers a free Moonshot model, auto-promote it to
    # position #1 as "recommended" — dynamic discovery without a PR.
    free_moonshot = next(
        (
            mid
            for mid, item in live_by_id.items()
            if mid.startswith("moonshotai/")
            and _ai_gateway_model_is_free(item.get("pricing"))
        ),
        None,
    )
    if free_moonshot:
        curated = [(mid, desc) for mid, desc in curated if mid != free_moonshot]
        curated.insert(0, (free_moonshot, "recommended"))
    else:
        first_id, _ = curated[0]
        curated[0] = (first_id, "recommended")

    _ai_gateway_catalog_cache = curated
    return list(curated)


def ai_gateway_model_ids(*, force_refresh: bool = False) -> list[str]:
    """Return just the AI Gateway model-id strings."""
    return [mid for mid, _ in fetch_ai_gateway_models(force_refresh=force_refresh)]




# ---------------------------------------------------------------------------
# Pricing helpers — fetch live pricing from OpenRouter-compatible /v1/models
# ---------------------------------------------------------------------------

# Cache: maps model_id → {"prompt": str, "completion": str} per endpoint
_pricing_cache: dict[str, dict[str, dict[str, str]]] = {}


def _format_price_per_mtok(per_token_str: str) -> str:
    """Convert a per-token price string to a human-friendly $/Mtok string.

    Always uses 2 decimal places so that prices align vertically when
    right-justified in a column (the decimal point stays in the same position).

    Examples:
        "0.000003"   → "$3.00"      (per million tokens)
        "0.00003"    → "$30.00"
        "0.00000015" → "$0.15"
        "0.0000001"  → "$0.10"
        "0.00018"    → "$180.00"
        "0"          → "free"
    """
    try:
        val = float(per_token_str)
    except (TypeError, ValueError):
        return "?"
    if val == 0:
        return "free"
    per_m = val * 1_000_000
    return f"${per_m:.2f}"


def format_model_pricing_table(
    models: list[tuple[str, str]],
    pricing_map: dict[str, dict[str, str]],
    current_model: str = "",
    indent: str = "      ",
) -> list[str]:
    """Build a column-aligned model+pricing table for terminal display.

    Returns a list of pre-formatted lines ready to print.
    *models* is ``[(model_id, description), ...]``.
    """
    if not models:
        return []

    # Build rows: (model_id, input_price, output_price, cache_price, is_current)
    rows: list[tuple[str, str, str, str, bool]] = []
    has_cache = False
    for mid, _desc in models:
        is_cur = mid == current_model
        p = pricing_map.get(mid)
        if p:
            inp = _format_price_per_mtok(p.get("prompt", ""))
            out = _format_price_per_mtok(p.get("completion", ""))
            cache_read = p.get("input_cache_read", "")
            cache = _format_price_per_mtok(cache_read) if cache_read else ""
            if cache:
                has_cache = True
        else:
            inp, out, cache = "", "", ""
        rows.append((mid, inp, out, cache, is_cur))

    name_col = max(len(r[0]) for r in rows) + 2
    # Compute price column widths from the actual data so decimals align
    price_col = max(
        max((len(r[1]) for r in rows if r[1]), default=4),
        max((len(r[2]) for r in rows if r[2]), default=4),
        3,  # minimum: "In" / "Out" header
    )
    cache_col = max(
        max((len(r[3]) for r in rows if r[3]), default=4),
        5,  # minimum: "Cache" header
    ) if has_cache else 0
    lines: list[str] = []

    # Header
    if has_cache:
        lines.append(f"{indent}{'Model':<{name_col}} {'In':>{price_col}}  {'Out':>{price_col}}  {'Cache':>{cache_col}}  /Mtok")
        lines.append(f"{indent}{'-' * name_col} {'-' * price_col}  {'-' * price_col}  {'-' * cache_col}")
    else:
        lines.append(f"{indent}{'Model':<{name_col}} {'In':>{price_col}}  {'Out':>{price_col}}  /Mtok")
        lines.append(f"{indent}{'-' * name_col} {'-' * price_col}  {'-' * price_col}")

    for mid, inp, out, cache, is_cur in rows:
        marker = "  ← current" if is_cur else ""
        if has_cache:
            lines.append(f"{indent}{mid:<{name_col}} {inp:>{price_col}}  {out:>{price_col}}  {cache:>{cache_col}}{marker}")
        else:
            lines.append(f"{indent}{mid:<{name_col}} {inp:>{price_col}}  {out:>{price_col}}{marker}")

    return lines


def fetch_models_with_pricing(
    api_key: str | None = None,
    base_url: str = "https://openrouter.ai/api",
    timeout: float = 8.0,
    *,
    force_refresh: bool = False,
) -> dict[str, dict[str, str]]:
    """Fetch ``/v1/models`` and return ``{model_id: {prompt, completion}}`` pricing.

    Results are cached per *base_url* so repeated calls are free.
    Works with any OpenRouter-compatible endpoint.
    """
    cache_key = (base_url or "").rstrip("/")
    if not force_refresh and cache_key in _pricing_cache:
        return _pricing_cache[cache_key]

    url = cache_key.rstrip("/") + "/v1/models"
    headers: dict[str, str] = {
        "Accept": "application/json",
        "User-Agent": _EASYBCI_USER_AGENT,
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode())
    except Exception:
        _pricing_cache[cache_key] = {}
        return {}

    result: dict[str, dict[str, str]] = {}
    for item in payload.get("data", []):
        mid = item.get("id")
        pricing = item.get("pricing")
        if mid and isinstance(pricing, dict):
            entry: dict[str, str] = {
                "prompt": str(pricing.get("prompt", "")),
                "completion": str(pricing.get("completion", "")),
            }
            if pricing.get("input_cache_read"):
                entry["input_cache_read"] = str(pricing["input_cache_read"])
            if pricing.get("input_cache_write"):
                entry["input_cache_write"] = str(pricing["input_cache_write"])
            result[mid] = entry

    _pricing_cache[cache_key] = result
    return result


def fetch_ai_gateway_pricing(
    timeout: float = 8.0,
    *,
    force_refresh: bool = False,
) -> dict[str, dict[str, str]]:
    """Fetch Vercel AI Gateway /v1/models and return easybci-shaped pricing.

    Vercel uses ``input`` / ``output`` field names; easybci's picker expects
    ``prompt`` / ``completion``. This translates. Cache read/write field names
    already match.
    """
    from easybci_lib.constants import AI_GATEWAY_BASE_URL

    cache_key = AI_GATEWAY_BASE_URL.rstrip("/")
    if not force_refresh and cache_key in _pricing_cache:
        return _pricing_cache[cache_key]

    try:
        req = urllib.request.Request(
            f"{cache_key}/models",
            headers={"Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode())
    except Exception:
        _pricing_cache[cache_key] = {}
        return {}

    result: dict[str, dict[str, str]] = {}
    for item in payload.get("data", []):
        if not isinstance(item, dict):
            continue
        mid = item.get("id")
        pricing = item.get("pricing")
        if not (mid and isinstance(pricing, dict)):
            continue
        entry: dict[str, str] = {
            "prompt": str(pricing.get("input", "")),
            "completion": str(pricing.get("output", "")),
        }
        if pricing.get("input_cache_read"):
            entry["input_cache_read"] = str(pricing["input_cache_read"])
        if pricing.get("input_cache_write"):
            entry["input_cache_write"] = str(pricing["input_cache_write"])
        result[mid] = entry

    _pricing_cache[cache_key] = result
    return result


def _resolve_openrouter_api_key() -> str:
    """Best-effort OpenRouter API key for pricing fetch."""
    return os.getenv("OPENROUTER_API_KEY", "").strip()


_DEFAULT_BCI_TEAM_INFERENCE_BASE = ""


def _resolve_bci_team_pricing_credentials() -> tuple[str, str]:
    """Stub — BCI Team Portal removed."""
    return ("", "")


def get_pricing_for_provider(provider: str, *, force_refresh: bool = False) -> dict[str, dict[str, str]]:
    """Return live pricing for providers that support it (openrouter, ai-gateway, novita)."""
    normalized = normalize_provider(provider)
    if normalized == "openrouter":
        return fetch_models_with_pricing(
            api_key=_resolve_openrouter_api_key(),
            base_url="https://openrouter.ai/api",
            force_refresh=force_refresh,
        )
    if normalized == "ai-gateway":
        return fetch_ai_gateway_pricing(force_refresh=force_refresh)
    if normalized == "novita":
        return _fetch_novita_pricing(force_refresh=force_refresh)
    return {}


def _fetch_novita_pricing(
    timeout: float = 8.0,
    *,
    force_refresh: bool = False,
) -> dict[str, dict[str, str]]:
    """Fetch pricing from NovitaAI /v1/models.

    NovitaAI returns input/output prices per million tokens in units of
    0.0001 USD. Convert them to the per-token strings used by the shared
    pricing formatter.

    Results are cached in ``_pricing_cache`` keyed on the resolved base URL,
    matching the pattern used by ``fetch_ai_gateway_pricing`` — without this,
    every menu render or pricing lookup re-hits the network.
    """
    api_key = os.getenv("NOVITA_API_KEY", "").strip()
    if not api_key:
        return {}

    base_url = os.getenv("NOVITA_BASE_URL", "").strip() or "https://api.novita.ai/openai/v1"
    cache_key = base_url.rstrip("/")
    if not force_refresh and cache_key in _pricing_cache:
        return _pricing_cache[cache_key]

    url = cache_key + "/models"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "User-Agent": _EASYBCI_USER_AGENT,
    }

    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode())
    except Exception:
        _pricing_cache[cache_key] = {}
        return {}

    result: dict[str, dict[str, str]] = {}
    for item in payload.get("data", []):
        if not isinstance(item, dict):
            continue
        mid = item.get("id")
        if not mid:
            continue
        inp = item.get("input_token_price_per_m")
        out = item.get("output_token_price_per_m")
        if inp is None and out is None:
            continue
        result[str(mid)] = {
            "prompt": str(float(inp or 0) / 10_000 / 1_000_000),
            "completion": str(float(out or 0) / 10_000 / 1_000_000),
        }

    _pricing_cache[cache_key] = result
    return result


# All provider IDs and aliases that are valid for the provider:model syntax.
_KNOWN_PROVIDER_NAMES: set[str] = (
    set(_PROVIDER_LABELS.keys())
    | set(_PROVIDER_ALIASES.keys())
    | {"openrouter", "custom"}
)


def list_available_providers() -> list[dict[str, str]]:
    """Return info about all providers the user could use with ``provider:model``.

    Each dict has ``id``, ``label``, and ``aliases``.
    Checks which providers have valid credentials configured.

    Derives the provider list from :data:`CANONICAL_PROVIDERS` (single
    source of truth shared with ``easybci model``, ``/model``, etc.).
    """
    # Derive display order from canonical list + custom
    provider_order = [p.slug for p in CANONICAL_PROVIDERS] + ["custom"]

    # Build reverse alias map
    aliases_for: dict[str, list[str]] = {}
    for alias, canonical in _PROVIDER_ALIASES.items():
        aliases_for.setdefault(canonical, []).append(alias)

    result = []
    for pid in provider_order:
        label = _PROVIDER_LABELS.get(pid, pid)
        alias_list = aliases_for.get(pid, [])
        # Check if this provider has credentials available
        has_creds = False
        try:
            from easybci_cli.auth import get_auth_status, has_usable_secret
            if pid == "custom":
                custom_base_url = _get_custom_base_url() or ""
                has_creds = bool(custom_base_url.strip())
            elif pid == "openrouter":
                has_creds = has_usable_secret(os.getenv("OPENROUTER_API_KEY", ""))
            else:
                status = get_auth_status(pid)
                has_creds = bool(status.get("logged_in") or status.get("configured"))
        except Exception:
            pass
        result.append({
            "id": pid,
            "label": label,
            "aliases": alias_list,
            "authenticated": has_creds,
        })
    return result


def parse_model_input(raw: str, current_provider: str) -> tuple[str, str]:
    """Parse ``/model`` input into ``(provider, model)``.

    Supports ``provider:model`` syntax to switch providers at runtime::

        openrouter:anthropic/claude-sonnet-4.5  →  ("openrouter", "anthropic/claude-sonnet-4.5")
        bci_team:custom-model-3                 →  (current_provider, "custom-model-3")
        anthropic/claude-sonnet-4.5             →  (current_provider, "anthropic/claude-sonnet-4.5")
        gpt-5.4                                 →  (current_provider, "gpt-5.4")

    The colon is only treated as a provider delimiter if the left side is a
    recognized provider name or alias.  This avoids misinterpreting model names
    that happen to contain colons (e.g. ``anthropic/claude-3.5-sonnet:beta``).

    Returns ``(provider, model)`` where *provider* is either the explicit
    provider from the input or *current_provider* if none was specified.
    """
    stripped = raw.strip()
    colon = stripped.find(":")
    if colon > 0:
        provider_part = stripped[:colon].strip().lower()
        model_part = stripped[colon + 1:].strip()
        if provider_part and model_part and provider_part in _KNOWN_PROVIDER_NAMES:
            # Support custom:name:model triple syntax for named custom
            # providers.  ``custom:local:qwen`` → ("custom:local", "qwen").
            # Single colon ``custom:qwen`` → ("custom", "qwen") as before.
            if provider_part == "custom" and ":" in model_part:
                second_colon = model_part.find(":")
                custom_name = model_part[:second_colon].strip()
                actual_model = model_part[second_colon + 1:].strip()
                if custom_name and actual_model:
                    return (f"custom:{custom_name}", actual_model)
            return (normalize_provider(provider_part), model_part)
    return (current_provider, stripped)


def _get_custom_base_url() -> str:
    """Get the custom endpoint base_url from config.yaml."""
    try:
        from easybci_cli.config import load_config
        config = load_config()
        model_cfg = config.get("model", {})
        if isinstance(model_cfg, dict):
            return str(model_cfg.get("base_url", "")).strip()
    except Exception:
        pass
    return ""


def curated_models_for_provider(
    provider: Optional[str],
    *,
    force_refresh: bool = False,
) -> list[tuple[str, str]]:
    """Return ``(model_id, description)`` tuples for a provider's model list.

    Tries to fetch the live model list from the provider's API first,
    falling back to the static ``_PROVIDER_MODELS`` catalog if the API
    is unreachable.
    """
    normalized = normalize_provider(provider)
    if normalized == "openrouter":
        return fetch_openrouter_models(force_refresh=force_refresh)

    # Try live API first
    live = provider_model_ids(normalized)
    if live:
        return [(m, "") for m in live]

    # Fallback to static catalog
    models = _PROVIDER_MODELS.get(normalized, [])
    return [(m, "") for m in models]


def _provider_keys(provider: str) -> set[str]:
    key = (provider or "").strip().lower()
    normalized = normalize_provider(provider)
    return {k for k in (key, normalized) if k}


def _model_in_provider_catalog(name_lower: str, providers: set[str]) -> bool:
    return any(
        name_lower == model.lower()
        for provider in providers
        for model in _PROVIDER_MODELS.get(provider, [])
    )


_AGGREGATOR_PROVIDERS = frozenset(
    {"openrouter", "ai-gateway", "kilocode"}
)


def _resolve_static_model_alias(
    name_lower: str,
    current_keys: set[str],
) -> Optional[tuple[str, str]]:
    """Resolve short aliases (e.g. sonnet/opus) using static catalogs only."""
    try:
        from easybci_cli.model_switch import MODEL_ALIASES
    except Exception:
        return None

    identity = MODEL_ALIASES.get(name_lower)
    if identity is None:
        return None

    vendor = identity.vendor
    family = identity.family

    def _match(provider: str) -> Optional[str]:
        models = _PROVIDER_MODELS.get(provider, [])
        if not models:
            return None
        prefix = (
            f"{vendor}/{family}"
            if provider in _AGGREGATOR_PROVIDERS
            else family
        ).lower()
        for model in models:
            if model.lower().startswith(prefix):
                return model
        return None

    for provider in current_keys:
        if matched := _match(provider):
            return provider, matched

    for provider in _PROVIDER_MODELS:
        if provider in current_keys or provider in _AGGREGATOR_PROVIDERS:
            continue
        if matched := _match(provider):
            return provider, matched

    for provider in _AGGREGATOR_PROVIDERS:
        if provider in current_keys and (matched := _match(provider)):
            return provider, matched

    return None


def detect_static_provider_for_model(
    model_name: str,
    current_provider: str,
) -> Optional[tuple[str, str]]:
    """Auto-detect a provider from static catalogs only.

    Returns ``(provider_id, model_name)``. The model name may be remapped
    when a static alias or bare provider name resolves to a catalog default.
    Returns ``None`` when no confident match is found.
    """
    name = (model_name or "").strip()
    if not name:
        return None

    name_lower = name.lower()
    current_keys = _provider_keys(current_provider)

    alias_match = _resolve_static_model_alias(name_lower, current_keys)
    if alias_match:
        return alias_match

    # --- Step 0: bare provider name typed as model ---
    # If someone types `/model anthropic`, treat it as a
    # provider switch and pick the first model from that provider's catalog.
    # Skip "custom" and "openrouter" — custom has no model catalog, and
    # openrouter requires an explicit model name to be useful.
    resolved_provider = _PROVIDER_ALIASES.get(name_lower, name_lower)
    if resolved_provider not in {"custom", "openrouter"}:
        default_models = _PROVIDER_MODELS.get(resolved_provider, [])
        if (
            resolved_provider in _PROVIDER_LABELS
            and default_models
            and resolved_provider not in current_keys
        ):
            return (resolved_provider, default_models[0])

    # Aggregators list other providers' models — never auto-switch TO them
    # If the model belongs to the current provider's catalog, don't suggest switching
    if _model_in_provider_catalog(name_lower, current_keys):
        return None

    # --- Step 1: check static provider catalogs for a direct match ---
    for pid, models in _PROVIDER_MODELS.items():
        if pid in current_keys or pid in _AGGREGATOR_PROVIDERS:
            continue
        if any(name_lower == m.lower() for m in models):
            return (pid, name)

    return None


def detect_provider_for_model(
    model_name: str,
    current_provider: str,
) -> Optional[tuple[str, str]]:
    """Auto-detect the best provider for a model name.

    Returns ``(provider_id, model_name)`` — the model name may be remapped
    (e.g. bare ``deepseek-chat`` → ``deepseek/deepseek-chat`` for OpenRouter).
    Returns ``None`` when no confident match is found.

    Priority:
    0. Bare provider name → switch to that provider's default model
    1. Direct provider static catalog match
    2. OpenRouter catalog match
    """
    name = (model_name or "").strip()
    if not name:
        return None

    static_match = detect_static_provider_for_model(name, current_provider)
    if static_match:
        return static_match
    if _model_in_provider_catalog(name.lower(), _provider_keys(current_provider)):
        return None

    # --- Step 2: check OpenRouter catalog ---
    # First try exact match (handles provider/model format)
    or_slug = _find_openrouter_slug(name)
    if or_slug:
        if current_provider != "openrouter":
            return ("openrouter", or_slug)
        # Already on openrouter, just return the resolved slug
        if or_slug != name:
            return ("openrouter", or_slug)
        return None  # already on openrouter with matching name

    return None


def _find_openrouter_slug(model_name: str) -> Optional[str]:
    """Find the full OpenRouter model slug for a bare or partial model name.

    Handles:
    - Exact match: ``anthropic/claude-opus-4.6`` → as-is
    - Bare name: ``deepseek-chat`` → ``deepseek/deepseek-chat``
    - Bare name: ``claude-opus-4.6`` → ``anthropic/claude-opus-4.6``
    """
    name_lower = model_name.strip().lower()
    if not name_lower:
        return None

    # Exact match (already has provider/ prefix)
    for mid in model_ids():
        if name_lower == mid.lower():
            return mid

    # Try matching just the model part (after the /)
    for mid in model_ids():
        if "/" in mid:
            _, model_part = mid.split("/", 1)
            if name_lower == model_part.lower():
                return mid

    return None


def normalize_provider(provider: Optional[str]) -> str:
    """Normalize provider aliases to EasyBCI' canonical provider ids.

    Note: ``"auto"`` passes through unchanged — use
    ``easybci_cli.auth.resolve_provider()`` to resolve it to a concrete
    provider based on credentials and environment.
    """
    normalized = (provider or "custom").strip().lower()
    return _PROVIDER_ALIASES.get(normalized, normalized)


def provider_label(provider: Optional[str]) -> str:
    """Return a human-friendly label for a provider id or alias."""
    original = (provider or "custom").strip()
    normalized = original.lower()
    if normalized == "auto":
        return "Auto"
    normalized = normalize_provider(normalized)
    return _PROVIDER_LABELS.get(normalized, original or "Custom")


# Models that support OpenAI Priority Processing (service_tier="priority").
# See https://openai.com/api-priority-processing/ for the canonical list.
#
# Pattern-based matching — any OpenAI flagship model (gpt-*, o1*, o3*, o4*)
# is assumed to support Priority Processing. service_tier=priority is silently
# ignored by non-OpenAI endpoints (OpenRouter/opencode-zen proxies
# strip the field), so false positives are harmless. 
_OPENAI_FAST_MODE_PREFIXES: tuple[str, ...] = (
    "gpt-",
    "o1",
    "o3",
    "o4",
)


def _is_openai_fast_model(model_id: Optional[str]) -> bool:
    """Return True if the model is an OpenAI flagship eligible for Priority Processing."""
    raw = _strip_vendor_prefix(str(model_id or ""))
    base = raw.split(":")[0]
    if not base:
        return False
    return any(base.startswith(prefix) for prefix in _OPENAI_FAST_MODE_PREFIXES)


# Models that support Anthropic Fast Mode (speed="fast").
# See https://platform.claude.com/docs/en/build-with-claude/fast-mode
#
# Pattern-based matching — any claude-* model is eligible. The anthropic
# adapter gates speed=fast on native Anthropic endpoints only (see
# _is_third_party_anthropic_endpoint in agent/anthropic_adapter.py), so
# third-party proxies that would reject the beta header are protected.


def _strip_vendor_prefix(model_id: str) -> str:
    """Strip vendor/ prefix from a model ID (e.g. 'anthropic/claude-opus-4-6' -> 'claude-opus-4-6')."""
    raw = str(model_id or "").strip().lower()
    if "/" in raw:
        raw = raw.split("/", 1)[1]
    return raw


def model_supports_fast_mode(model_id: Optional[str]) -> bool:
    """Return whether EasyBCI should expose the /fast toggle for this model."""
    return _is_anthropic_fast_model(model_id) or _is_openai_fast_model(model_id)


def _is_anthropic_fast_model(model_id: Optional[str]) -> bool:
    """Return True if the model is a Claude model eligible for Anthropic Fast Mode.

    Fast mode is currently supported on Claude Opus 4.6 only. Per Anthropic's
    docs (https://platform.claude.com/docs/en/build-with-claude/fast-mode):
    "Fast mode is currently supported on Opus 4.6 only. Sending speed: fast
    with an unsupported model returns an error." Opus 4.7 explicitly rejects
    the ``speed`` parameter with HTTP 400.
    """
    raw = _strip_vendor_prefix(str(model_id or ""))
    base = raw.split(":")[0]
    if not base.startswith("claude-"):
        return False
    # Only Opus 4.6 supports fast mode at present.
    return "opus-4-6" in base or "opus-4.6" in base


def resolve_fast_mode_overrides(model_id: Optional[str]) -> dict[str, Any] | None:
    """Return request_overrides for fast/priority mode, or None if unsupported.

    Returns provider-appropriate overrides:
    - OpenAI models: ``{"service_tier": "priority"}`` (Priority Processing)
    - Anthropic models: ``{"speed": "fast"}`` (Anthropic Fast Mode beta)

    The overrides are injected into the API request kwargs by
    ``_build_api_kwargs`` in run_agent.py — each API path handles its own
    keys (service_tier for OpenAI, speed for Anthropic Messages).
    """
    if not model_supports_fast_mode(model_id):
        return None
    if _is_anthropic_fast_model(model_id):
        return {"speed": "fast"}
    return {"service_tier": "priority"}



# Providers where models.dev is treated as authoritative: curated static
# lists are kept only as an offline fallback and to capture custom additions
# the registry doesn't publish yet. Adding a provider here causes its
# curated list to be merged with fresh models.dev entries (fresh first, any
# curated-only names appended) for both the CLI and the gateway /model picker.
#
# DELIBERATELY EXCLUDED:
#   - "openrouter": curated list is already a hand-picked agentic subset of
#     OpenRouter's 400+ catalog. Blindly merging would dump everything.
# Also excluded: providers that already have dedicated live-endpoint
# branches below (anthropic, ai-gateway, custom) — those paths handle freshness themselves.
_MODELS_DEV_PREFERRED: frozenset[str] = frozenset({
    "opencode-go",
    "opencode-zen",
    "deepseek",
    "kilocode",
    "fireworks",
    "mistral",
    "togetherai",
    "cohere",
    "perplexity",
    "groq",
    "nvidia",
    "huggingface",
    "zai",
    "gemini",
    "google",
})


def _merge_with_models_dev(provider: str, curated: list[str]) -> list[str]:
    """Merge curated list with fresh models.dev entries for a preferred provider.

    Returns models.dev entries first (in models.dev order), then any
    curated-only entries appended. Preserves case for curated fallbacks
    (e.g. ``MiniMax-M2.7``) while trusting models.dev for newer variants.

    If models.dev is unreachable or returns nothing, the curated list is
    returned unchanged — this is the offline/CI fallback path.
    """
    try:
        from easybci_agent.models_dev import list_agentic_models
        mdev = list_agentic_models(provider)
    except Exception:
        mdev = []

    if not mdev:
        return list(curated)

    # Case-insensitive dedup while preserving order and curated casing.
    seen_lower: set[str] = set()
    merged: list[str] = []
    for mid in mdev:
        key = str(mid).lower()
        if key in seen_lower:
            continue
        seen_lower.add(key)
        merged.append(mid)
    for mid in curated:
        key = str(mid).lower()
        if key in seen_lower:
            continue
        seen_lower.add(key)
        merged.append(mid)
    return merged


def provider_model_ids(provider: Optional[str], *, force_refresh: bool = False) -> list[str]:
    """Return the best known model catalog for a provider.

    Tries live API endpoints for providers that support them,
    falling back to static lists. For providers in ``_MODELS_DEV_PREFERRED``
    (opencode-go/zen, xiaomi, deepseek, smaller inference providers, etc.),
    models.dev entries are merged on top of curated so new models released
    on the platform appear in ``/model`` without a EasyBCI release.
    """
    normalized = normalize_provider(provider)
    if normalized == "openrouter":
        return model_ids(force_refresh=force_refresh)
    if normalized == "stepfun":
        try:
            from easybci_cli.auth import resolve_api_key_provider_credentials

            creds = resolve_api_key_provider_credentials("stepfun")
            api_key = str(creds.get("api_key") or "").strip()
            base_url = str(creds.get("base_url") or "").strip()
            if api_key and base_url:
                live = fetch_api_models(api_key, base_url)
                if live:
                    return live
        except Exception:
            pass
    if normalized == "anthropic":
        live = _fetch_anthropic_models()
        if live:
            return live
    if normalized == "ai-gateway":
        live = _fetch_ai_gateway_models()
        if live:
            return live
    if normalized == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if api_key:
            base_raw = os.getenv("OPENAI_BASE_URL", "").strip().rstrip("/")
            base = base_raw or "https://api.openai.com/v1"
            try:
                live = fetch_api_models(api_key, base)
                if live:
                    return live
            except Exception:
                pass
    if normalized == "gmi":
        try:
            from easybci_cli.auth import resolve_api_key_provider_credentials

            creds = resolve_api_key_provider_credentials("gmi")
            api_key = str(creds.get("api_key") or "").strip()
            base_url = str(creds.get("base_url") or "").strip()
            if api_key and base_url:
                live = fetch_api_models(api_key, base_url)
                if live:
                    return live
        except Exception:
            pass
    if normalized == "custom":
        base_url = _get_custom_base_url()
        if base_url:
            # Try common API key env vars for custom endpoints
            api_key = (
                os.getenv("CUSTOM_API_KEY", "")
                or os.getenv("OPENAI_API_KEY", "")
                or os.getenv("OPENROUTER_API_KEY", "")
            )
            live = fetch_api_models(api_key, base_url)
            if live:
                return live
    # Bedrock uses live discovery keyed by the resolved AWS region so that
    # EU/AP users see eu.*/ap.* model IDs instead of the static us.* list.
    # Note: early return intentionally skips _MODELS_DEV_PREFERRED merge
    # below — bedrock is not expected to appear in that table.
    if normalized == "bedrock":
        try:
            from easybci_agent.bedrock_adapter import bedrock_model_ids_or_none
            ids = bedrock_model_ids_or_none()
            if ids is not None:
                return ids
        except Exception:
            pass

    # ── Profile-based generic live fetch (all simple api-key providers) ──
    # Handles any provider registered in providers/ with auth_type="api_key".
    # Replaces per-provider copy-paste blocks (stepfun, gmi, zai, etc.).
    try:
        from services.providers import get_provider_profile
        from easybci_cli.auth import resolve_api_key_provider_credentials

        _p = get_provider_profile(normalized)
        if _p and _p.auth_type == "api_key" and _p.base_url:
            try:
                creds = resolve_api_key_provider_credentials(normalized)
                api_key = str(creds.get("api_key") or "").strip()
                base_url = str(creds.get("base_url") or "").strip()
            except Exception:
                api_key, base_url = "", _p.base_url
            if not base_url:
                base_url = _p.base_url
            if api_key:
                live = _p.fetch_models(api_key=api_key)
                if live:
                    return live
            # Use profile's fallback_models if defined
            if _p.fallback_models:
                return list(_p.fallback_models)
    except Exception:
        pass

    curated_static = list(_PROVIDER_MODELS.get(normalized, []))
    if normalized in _MODELS_DEV_PREFERRED:
        return _merge_with_models_dev(normalized, curated_static)
    return curated_static


def _fetch_anthropic_models(timeout: float = 5.0) -> Optional[list[str]]:
    """Fetch available models from the Anthropic /v1/models endpoint.

    Uses resolve_anthropic_token() to find credentials (env vars or
    Claude Code auto-discovery).  Returns sorted model IDs or None.
    """
    try:
        from easybci_agent.anthropic_adapter import resolve_anthropic_token, _is_oauth_token
    except ImportError:
        return None

    token = resolve_anthropic_token()
    if not token:
        return None

    headers: dict[str, str] = {"anthropic-version": "2023-06-01"}
    is_oauth = _is_oauth_token(token)
    if is_oauth:
        headers["Authorization"] = f"Bearer {token}"
        from easybci_agent.anthropic_adapter import _COMMON_BETAS, _OAUTH_ONLY_BETAS, _CONTEXT_1M_BETA
        headers["anthropic-beta"] = ",".join(_COMMON_BETAS + _OAUTH_ONLY_BETAS)
    else:
        headers["x-api-key"] = token

    def _do_request(h: dict[str, str]):
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/models",
            headers=h,
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())

    try:
        try:
            data = _do_request(headers)
        except urllib.error.HTTPError as http_err:
            # Reactive recovery for OAuth subscriptions that reject the 1M
            # context beta with 400 "long context beta is not yet available
            # for this subscription". Retry once without the beta; re-raise
            # anything else so the outer except logs it.
            if (
                is_oauth
                and http_err.code == 400
            ):
                try:
                    body_text = http_err.read().decode(errors="ignore").lower()
                except Exception:
                    body_text = ""
                if "long context beta" in body_text and "not yet available" in body_text:
                    headers["anthropic-beta"] = ",".join(
                        [b for b in _COMMON_BETAS if b != _CONTEXT_1M_BETA]
                        + list(_OAUTH_ONLY_BETAS)
                    )
                    data = _do_request(headers)
                else:
                    raise
            else:
                raise
        models = [m["id"] for m in data.get("data", []) if m.get("id")]
        # Sort: latest/largest first (opus > sonnet > haiku, higher version first)
        return sorted(models, key=lambda m: (
            "opus" not in m,      # opus first
            "sonnet" not in m,    # then sonnet
            "haiku" not in m,     # then haiku
            m,                    # alphabetical within tier
        ))
    except Exception as e:
        import logging
        logging.getLogger(__name__).debug("Failed to fetch Anthropic models: %s", e)
        return None




def _lmstudio_server_root(base_url: Optional[str]) -> Optional[str]:
    """Strip ``/v1`` suffix from an LM Studio base URL to get the native API root.

    Returns ``None`` when the base URL is empty/invalid.
    """
    root = (base_url or "").strip().rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3].rstrip("/")
    return root or None


def _lmstudio_request_headers(api_key: Optional[str] = None) -> dict:
    """Build HTTP headers for LM Studio native API requests."""
    headers = {"User-Agent": _EASYBCI_USER_AGENT}
    token = str(api_key or "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _lmstudio_fetch_raw_models(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: float = 5.0,
) -> Optional[list[dict]]:
    """Fetch the raw model list from LM Studio's ``/api/v1/models``.

    Returns the ``models`` list of dicts on success, ``None`` on network
    errors or malformed responses.  Raises ``AuthError`` on HTTP 401/403.
    """
    server_root = _lmstudio_server_root(base_url)
    if not server_root:
        return None

    headers = _lmstudio_request_headers(api_key)
    request = urllib.request.Request(server_root + "/api/v1/models", headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        if exc.code in {401, 403}:
            from easybci_cli.auth import AuthError
            raise AuthError(
                f"LM Studio rejected the request with HTTP {exc.code}.",
                provider="lmstudio",
                code="auth_rejected",
            ) from exc
        import logging
        logging.getLogger(__name__).debug(
            "LM Studio probe at %s failed with HTTP %s", server_root, exc.code,
        )
        return None
    except Exception as exc:
        import logging
        logging.getLogger(__name__).debug(
            "LM Studio probe at %s failed: %s", server_root, exc,
        )
        return None

    raw_models = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(raw_models, list):
        import logging
        logging.getLogger(__name__).debug(
            "LM Studio probe at %s returned malformed payload (no `models` list)",
            server_root,
        )
        return None
    return raw_models


def probe_lmstudio_models(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: float = 5.0,
) -> Optional[list[str]]:
    """Probe LM Studio's model listing.

    Returns chat-capable model keys on success, including the valid empty-list
    case when the server is reachable but has no non-embedding models.
    Returns ``None`` on network errors, malformed responses, or empty/invalid
    base URLs.

    Raises ``AuthError`` on HTTP 401/403 so callers can surface token issues
    separately from reachability problems.
    """
    raw_models = _lmstudio_fetch_raw_models(api_key=api_key, base_url=base_url, timeout=timeout)
    if raw_models is None:
        return None

    keys: list[str] = []
    for raw in raw_models:
        if not isinstance(raw, dict):
            continue
        if str(raw.get("type") or "").strip().lower() == "embedding":
            continue
        key = str(raw.get("key") or raw.get("id") or "").strip()
        if key and key not in keys:
            keys.append(key)
    return keys


def fetch_lmstudio_models(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: float = 5.0,
) -> list[str]:
    """Fetch LM Studio chat-capable model keys from native ``/api/v1/models``.

    Returns a list of model keys (e.g. ``publisher/model-name``) with embedding
    models filtered out. Returns an empty list on network errors, malformed
    responses, or empty/invalid base URLs.

    Raises ``AuthError`` on HTTP 401/403 so callers can distinguish a missing
    or wrong ``LM_API_KEY`` from an unreachable server — the most common
    LM Studio support case once auth-enabled mode is turned on.
    """
    models = probe_lmstudio_models(api_key=api_key, base_url=base_url, timeout=timeout)
    return models or []


def ensure_lmstudio_model_loaded(
    model: str,
    base_url: Optional[str],
    api_key: Optional[str],
    target_context_length: int,
    timeout: float = 120.0,
) -> Optional[int]:
    """Ensure LM Studio has ``model`` loaded with at least ``target_context_length``.

    No-op when an instance is already loaded with sufficient context. Otherwise
    POSTs ``/api/v1/models/load`` to (re)load with the target context, capped
    at the model's ``max_context_length``. Returns the resolved loaded context
    length, or ``None`` when the probe / load failed.
    """
    server_root = _lmstudio_server_root(base_url)
    if not server_root:
        return None

    headers = _lmstudio_request_headers(api_key)

    try:
        raw_models = _lmstudio_fetch_raw_models(api_key=api_key, base_url=base_url, timeout=10)
    except Exception:
        raw_models = None
    if raw_models is None:
        return None

    target_entry = None
    for raw in raw_models:
        if not isinstance(raw, dict):
            continue
        if raw.get("key") == model or raw.get("id") == model:
            target_entry = raw
            break
    if target_entry is None:
        return None

    max_ctx = target_entry.get("max_context_length")
    if isinstance(max_ctx, int) and max_ctx > 0:
        target_context_length = min(target_context_length, max_ctx)

    for inst in target_entry.get("loaded_instances") or []:
        cfg = inst.get("config") if isinstance(inst, dict) else None
        loaded_ctx = cfg.get("context_length") if isinstance(cfg, dict) else None
        if isinstance(loaded_ctx, int) and loaded_ctx >= target_context_length:
            return loaded_ctx

    body = json.dumps({
        "model": model,
        "context_length": target_context_length,
    }).encode()
    load_headers = dict(headers)
    load_headers["Content-Type"] = "application/json"
    try:
        with urllib.request.urlopen(
            urllib.request.Request(
                server_root + "/api/v1/models/load",
                data=body,
                headers=load_headers,
                method="POST",
            ),
            timeout=timeout,
        ) as resp:
            resp.read()
    except Exception:
        return None
    return target_context_length


def lmstudio_model_reasoning_options(
    model: str,
    base_url: Optional[str],
    api_key: Optional[str] = None,
    timeout: float = 5.0,
) -> list[str]:
    """Return the reasoning ``allowed_options`` LM Studio publishes for ``model``.

    Pulls ``capabilities.reasoning.allowed_options`` from ``/api/v1/models``.
    Returns ``[]`` when the model is unknown, the endpoint is unreachable,
    or the model does not declare a reasoning capability.
    """
    try:
        raw_models = _lmstudio_fetch_raw_models(api_key=api_key, base_url=base_url, timeout=timeout)
    except Exception:
        raw_models = None
    if not raw_models:
        return []

    for raw in raw_models:
        if not isinstance(raw, dict):
            continue
        if raw.get("key") != model and raw.get("id") != model:
            continue
        caps = raw.get("capabilities")
        reasoning = caps.get("reasoning") if isinstance(caps, dict) else None
        opts = reasoning.get("allowed_options") if isinstance(reasoning, dict) else None
        if isinstance(opts, list):
            return [str(o).strip().lower() for o in opts if isinstance(o, str)]
        return []
    return []




# Azure Foundry model families that require the Responses API.  Azure
# rejects /chat/completions against these deployments with
# ``400 "The requested operation is unsupported."`` ``gpt-4o-pure`` on
# the same endpoint worked fine.  Keep the patterns broad enough to cover
# vendor-renamed deployments (e.g. ``gpt-5.4``, ``o1-preview``) but tight 
# enough to leave GPT-4 / Grok deployments on chat completions.
_AZURE_FOUNDRY_RESPONSES_PREFIXES = (
    "gpt-5",       # gpt-5, gpt-5.x
    "o1",          # o1, o1-preview, o1-mini
    "o3",          # o3, o3-mini
    "o4",          # o4, o4-mini
)


def azure_foundry_model_api_mode(model_name: Optional[str]) -> Optional[str]:
    """Infer Azure Foundry api_mode from a deployment/model name.

    Returns ``"responses"`` when the model name matches a family that
    only accepts the Responses API on Azure Foundry (GPT-5.x, o1/o3/o4
    reasoning models).  Returns ``None`` otherwise — the caller should fall
    back to the configured/default api_mode (typically ``chat_completions``)
    so GPT-4o, GPT-4 Turbo, Llama, Mistral, etc. keep working.

    Intentionally does NOT return ``anthropic_messages``; Anthropic-style
    Azure endpoints are disambiguated by URL (``/anthropic`` suffix) in
    ``runtime_provider._detect_api_mode_for_url`` and by the user setting
    ``model.api_mode: anthropic_messages`` explicitly.
    """
    raw = str(model_name or "").strip().lower()
    if not raw:
        return None
    # Strip any vendor/ prefix a user may have copied from OpenRouter.
    if "/" in raw:
        raw = raw.rsplit("/", 1)[-1]
    for prefix in _AZURE_FOUNDRY_RESPONSES_PREFIXES:
        if raw.startswith(prefix):
            return "responses"
    return None


def normalize_opencode_model_id(provider_id: Optional[str], model_id: Optional[str]) -> str:
    """Normalize OpenCode config IDs to the bare model slug used in API requests."""
    provider = normalize_provider(provider_id)
    current = str(model_id or "").strip()
    if not current or provider not in {"opencode-zen", "opencode-go"}:
        return current

    prefix = f"{provider}/"
    if current.lower().startswith(prefix):
        return current[len(prefix):]
    return current


def opencode_model_api_mode(provider_id: Optional[str], model_id: Optional[str]) -> str:
    """Determine the API mode for an OpenCode Zen / Go model.

    OpenCode routes different models behind different API surfaces:

    - GPT-5 models on Zen use ``/v1/responses``
    - Claude models on Zen use ``/v1/messages``
    - MiniMax models on Go use ``/v1/messages``
    - GLM / Kimi on Go use ``/v1/chat/completions``
    - Other Zen models (Gemini, GLM, Kimi, MiniMax, Qwen, etc.) use
      ``/v1/chat/completions``

    This follows the published OpenCode docs for Zen and Go endpoints.
    """
    provider = normalize_provider(provider_id)
    normalized = normalize_opencode_model_id(provider_id, model_id).lower()
    if not normalized:
        return "chat_completions"

    if provider == "opencode-go":
        if normalized.startswith("minimax-"):
            return "anthropic_messages"
        return "chat_completions"

    if provider == "opencode-zen":
        if normalized.startswith("claude-"):
            return "anthropic_messages"
        if normalized.startswith("gpt-"):
            return "responses"
        return "chat_completions"

    return "chat_completions"


def probe_api_models(
    api_key: Optional[str],
    base_url: Optional[str],
    timeout: float = 5.0,
    api_mode: Optional[str] = None,
) -> dict[str, Any]:
    """Probe a ``/models`` endpoint with light URL heuristics.

    For ``anthropic_messages`` mode, uses ``x-api-key`` and
    ``anthropic-version`` headers (Anthropic's native auth) instead of
    ``Authorization: Bearer``.  The response shape (``data[].id``) is
    identical, so the same parser works for both.
    """
    normalized = (base_url or "").strip().rstrip("/")
    if not normalized:
        return {
            "models": None,
            "probed_url": None,
            "resolved_base_url": "",
            "suggested_base_url": None,
            "used_fallback": False,
        }

    if normalized.endswith("/v1"):
        alternate_base = normalized[:-3].rstrip("/")
    else:
        alternate_base = normalized + "/v1"

    candidates: list[tuple[str, bool]] = [(normalized, False)]
    if alternate_base and alternate_base != normalized:
        candidates.append((alternate_base, True))

    tried: list[str] = []
    headers: dict[str, str] = {"User-Agent": _EASYBCI_USER_AGENT}
    if api_key and api_mode == "anthropic_messages":
        headers["x-api-key"] = api_key
        headers["anthropic-version"] = "2023-06-01"
    elif api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    for candidate_base, is_fallback in candidates:
        url = candidate_base.rstrip("/") + "/models"
        tried.append(url)
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
                return {
                    "models": [m.get("id", "") for m in data.get("data", [])],
                    "probed_url": url,
                    "resolved_base_url": candidate_base.rstrip("/"),
                    "suggested_base_url": alternate_base if alternate_base != candidate_base else normalized,
                    "used_fallback": is_fallback,
                }
        except Exception:
            continue

    return {
        "models": None,
        "probed_url": tried[0] if tried else normalized.rstrip("/") + "/models",
        "resolved_base_url": normalized,
        "suggested_base_url": alternate_base if alternate_base != normalized else None,
        "used_fallback": False,
    }


def _fetch_ai_gateway_models(timeout: float = 5.0) -> Optional[list[str]]:
    """Fetch available language models with tool-use from AI Gateway."""
    api_key = os.getenv("AI_GATEWAY_API_KEY", "").strip()
    if not api_key:
        return None
    base_url = os.getenv("AI_GATEWAY_BASE_URL", "").strip()
    if not base_url:
        from easybci_lib.constants import AI_GATEWAY_BASE_URL
        base_url = AI_GATEWAY_BASE_URL

    url = base_url.rstrip("/") + "/models"
    headers: dict[str, str] = {
        "Authorization": f"Bearer {api_key}",
        "User-Agent": _EASYBCI_USER_AGENT,
    }
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
            return [
                m["id"]
                for m in data.get("data", [])
                if m.get("id")
                and m.get("type") == "language"
                and "tool-use" in (m.get("tags") or [])
            ]
    except Exception:
        return None


def fetch_api_models(
    api_key: Optional[str],
    base_url: Optional[str],
    timeout: float = 5.0,
    api_mode: Optional[str] = None,
) -> Optional[list[str]]:
    """Fetch the list of available model IDs from the provider's ``/models`` endpoint.

    Returns a list of model ID strings, or ``None`` if the endpoint could not
    be reached (network error, timeout, auth failure, etc.).
    """
    return probe_api_models(api_key, base_url, timeout=timeout, api_mode=api_mode).get("models")


# ---------------------------------------------------------------------------
# Ollama Cloud — merged model discovery with disk cache
# ---------------------------------------------------------------------------



_OLLAMA_CLOUD_CACHE_TTL = 3600  # 1 hour


def _strip_ollama_cloud_suffix(model_id: str) -> str:
    """Strip :cloud / -cloud suffixes that models.dev appends to Ollama Cloud IDs.

    The live API uses clean IDs (e.g. 'kimi-k2.6') while models.dev sometimes
    returns them as 'kimi-k2.6:cloud'. Normalising before the dedup merge
    prevents duplicate entries in the merged model list.
    """
    for suffix in (":cloud", "-cloud"):
        if model_id.endswith(suffix):
            return model_id[: -len(suffix)]
    return model_id


def _ollama_cloud_cache_path() -> Path:
    """Return the path for the Ollama Cloud model cache."""
    from easybci_lib.constants import get_easybci_home
    return get_easybci_home() / "ollama_cloud_models_cache.json"


def _load_ollama_cloud_cache(*, ignore_ttl: bool = False) -> Optional[dict]:
    """Load cached Ollama Cloud models from disk.

    Args:
        ignore_ttl: If True, return data even if the TTL has expired (stale fallback).
    """
    try:
        cache_path = _ollama_cloud_cache_path()
        if not cache_path.exists():
            return None
        with open(cache_path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None
        models = data.get("models")
        if not (isinstance(models, list) and models):
            return None
        if not ignore_ttl:
            cached_at = data.get("cached_at", 0)
            if (time.time() - cached_at) > _OLLAMA_CLOUD_CACHE_TTL:
                return None  # stale
        return data
    except Exception:
        pass
    return None


def _save_ollama_cloud_cache(models: list[str]) -> None:
    """Persist the merged Ollama Cloud model list to disk."""
    try:
        from easybci_lib.utils import atomic_json_write
        cache_path = _ollama_cloud_cache_path()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json_write(cache_path, {"models": models, "cached_at": time.time()}, indent=None)
    except Exception:
        pass


def fetch_ollama_cloud_models(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    *,
    force_refresh: bool = False,
) -> list[str]:
    """Return local model list (stub — no cloud discovery)."""
    return _PROVIDER_MODELS.get("local", [])


def validate_requested_model(
    model_name: str,
    provider: Optional[str],
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    api_mode: Optional[str] = None,
) -> dict[str, Any]:
    """
    Validate a ``/model`` value for the active provider.

    Performs format checks first, then probes the live API to confirm
    the model actually exists.

    Returns a dict with:
      - accepted: whether the CLI should switch to the requested model now
      - persist: whether it is safe to save to config
      - recognized: whether it matched a known provider catalog
      - message: optional warning / guidance for the user
    """
    requested = (model_name or "").strip()
    normalized = normalize_provider(provider)
    if normalized == "openrouter" and base_url and "openrouter.ai" not in base_url:
        normalized = "custom"
    requested_for_lookup = requested

    if not requested:
        return {
            "accepted": False,
            "persist": False,
            "recognized": False,
            "message": "Model name cannot be empty.",
        }

    if any(ch.isspace() for ch in requested):
        return {
            "accepted": False,
            "persist": False,
            "recognized": False,
            "message": "Model names cannot contain spaces.",
        }

    if normalized == "lmstudio":
        from easybci_cli.auth import AuthError
        # Use probe_lmstudio_models so we can distinguish None (unreachable
        # / malformed response) from [] (reachable, but no chat-capable models
        # are loaded). fetch_lmstudio_models collapses both to [].
        try:
            models = probe_lmstudio_models(api_key=api_key, base_url=base_url)
        except AuthError as exc:
            return {
                "accepted": False, "persist": False, "recognized": False,
                "message": (
                    f"{exc} Set `LM_API_KEY` (or update it) to match the server's bearer token."
                ),
            }
        if models is None:
            return {
                "accepted": False, "persist": False, "recognized": False,
                "message": f"Could not reach LM Studio's `/api/v1/models` to validate `{requested}`.",
            }
        if not models:
            return {
                "accepted": False, "persist": False, "recognized": False,
                "message": (
                    f"LM Studio is reachable but no chat-capable models are loaded. "
                    f"Load `{requested}` in LM Studio (Developer tab → Load Model) and try again."
                ),
            }
        if requested_for_lookup in set(models):
            return {"accepted": True, "persist": True, "recognized": True, "message": None}
        return {
            "accepted": False, "persist": False, "recognized": False,
            "message": f"Model `{requested}` was not found in LM Studio's model listing.",
        }

    if normalized == "custom" or normalized.startswith("custom:"):
        # Try probing with correct auth for the api_mode.
        if api_mode == "anthropic_messages":
            probe = probe_api_models(api_key, base_url, api_mode=api_mode)
        else:
            probe = probe_api_models(api_key, base_url)
        api_models = probe.get("models")
        if api_models is not None:
            if requested_for_lookup in set(api_models):
                return {
                    "accepted": True,
                    "persist": True,
                    "recognized": True,
                    "message": None,
                }

            # Auto-correct if the top match is very similar (e.g. typo)
            auto = get_close_matches(requested_for_lookup, api_models, n=1, cutoff=0.9)
            if auto:
                return {
                    "accepted": True,
                    "persist": True,
                    "recognized": True,
                    "corrected_model": auto[0],
                    "message": f"Auto-corrected `{requested}` → `{auto[0]}`",
                }

            suggestions = get_close_matches(requested, api_models, n=3, cutoff=0.5)
            suggestion_text = ""
            if suggestions:
                suggestion_text = "\n  Similar models: " + ", ".join(f"`{s}`" for s in suggestions)

            message = (
                f"Note: `{requested}` was not found in this custom endpoint's model listing "
                f"({probe.get('probed_url')}). It may still work if the server supports hidden or aliased models."
                f"{suggestion_text}"
            )
            if probe.get("used_fallback"):
                message += (
                    f"\n  Endpoint verification succeeded after trying `{probe.get('resolved_base_url')}`. "
                    f"Consider saving that as your base URL."
                )

            return {
                "accepted": True,
                "persist": True,
                "recognized": False,
                "message": message,
            }

        message = (
            f"Note: could not reach this custom endpoint's model listing at `{probe.get('probed_url')}`. "
            f"EasyBCI will still save `{requested}`, but the endpoint should expose `/models` for verification."
        )
        if api_mode == "anthropic_messages":
            message += (
                "\n  Many Anthropic-compatible proxies do not implement the Models API "
                "(GET /v1/models).  The model name has been accepted without verification."
            )
        if probe.get("suggested_base_url"):
            message += f"\n  If this server expects `/v1`, try base URL: `{probe.get('suggested_base_url')}`"

        return {
            "accepted": api_mode == "anthropic_messages",
            "persist": True,
            "recognized": False,
            "message": message,
        }


    # MiniMax providers don't expose a /models endpoint — validate against
    # the static catalog instead.
    if normalized in {"minimax", "minimax-cn"}:
        try:
            catalog_models = provider_model_ids(normalized)
        except Exception:
            catalog_models = []
        if catalog_models:
            # Case-insensitive lookup (catalog uses mixed case like MiniMax-M2.7)
            catalog_lower = {m.lower(): m for m in catalog_models}
            if requested_for_lookup.lower() in catalog_lower:
                return {
                    "accepted": True,
                    "persist": True,
                    "recognized": True,
                    "message": None,
                }
            # Auto-correct close matches (case-insensitive)
            catalog_lower_list = list(catalog_lower.keys())
            auto = get_close_matches(requested_for_lookup.lower(), catalog_lower_list, n=1, cutoff=0.9)
            if auto:
                corrected = catalog_lower[auto[0]]
                return {
                    "accepted": True,
                    "persist": True,
                    "recognized": True,
                    "corrected_model": corrected,
                    "message": f"Auto-corrected `{requested}` → `{corrected}`",
                }
            suggestions = get_close_matches(requested_for_lookup.lower(), catalog_lower_list, n=3, cutoff=0.5)
            suggestion_text = ""
            if suggestions:
                suggestion_text = "\n  Similar models: " + ", ".join(f"`{catalog_lower[s]}`" for s in suggestions)
            return {
                "accepted": True,
                "persist": True,
                "recognized": False,
                "message": (
                    f"Note: `{requested}` was not found in the MiniMax catalog."
                    f"{suggestion_text}"
                    "\n  MiniMax does not expose a /models endpoint, so EasyBCI cannot verify the model name."
                    "\n  The model may still work if it exists on the server."
                ),
            }

    # Native Anthropic provider: /v1/models requires x-api-key (or Bearer for
    # OAuth) plus anthropic-version headers.  The generic OpenAI-style probe
    # below uses plain Bearer auth and 401s against Anthropic, so dispatch to
    # the native fetcher which handles both API keys and Claude-Code OAuth
    # tokens.  (The api_mode=="anthropic_messages" branch below handles the
    # Messages-API transport case separately.)
    if normalized == "anthropic":
        anthropic_models = _fetch_anthropic_models()
        if anthropic_models is not None:
            if requested_for_lookup in set(anthropic_models):
                return {
                    "accepted": True,
                    "persist": True,
                    "recognized": True,
                    "message": None,
                }
            auto = get_close_matches(requested_for_lookup, anthropic_models, n=1, cutoff=0.9)
            if auto:
                return {
                    "accepted": True,
                    "persist": True,
                    "recognized": True,
                    "corrected_model": auto[0],
                    "message": f"Auto-corrected `{requested}` → `{auto[0]}`",
                }
            suggestions = get_close_matches(requested, anthropic_models, n=3, cutoff=0.5)
            suggestion_text = ""
            if suggestions:
                suggestion_text = "\n  Similar models: " + ", ".join(f"`{s}`" for s in suggestions)
            # Accept anyway — Anthropic sometimes gates newer/preview models
            # (e.g. snapshot IDs, early-access releases) behind accounts
            # even though they aren't listed on /v1/models.
            return {
                "accepted": True,
                "persist": True,
                "recognized": False,
                "message": (
                    f"Note: `{requested}` was not found in Anthropic's /v1/models listing. "
                    f"It may still work if you have early-access or snapshot IDs."
                    f"{suggestion_text}"
                ),
            }
        # _fetch_anthropic_models returned None — no token resolvable or
        # network failure.  Fall through to the generic warning below.

    # Anthropic Messages API: many proxies don't implement /v1/models.
    # Try probing with correct auth; if it fails, accept with a warning.
    if api_mode == "anthropic_messages":
        api_models = fetch_api_models(api_key, base_url, api_mode=api_mode)
        if api_models is not None:
            if requested_for_lookup in set(api_models):
                return {
                    "accepted": True,
                    "persist": True,
                    "recognized": True,
                    "message": None,
                }
            auto = get_close_matches(requested_for_lookup, api_models, n=1, cutoff=0.9)
            if auto:
                return {
                    "accepted": True,
                    "persist": True,
                    "recognized": True,
                    "corrected_model": auto[0],
                    "message": f"Auto-corrected `{requested}` → `{auto[0]}`",
                }
        # Probe failed or model not found — accept anyway (proxy likely
        # doesn't implement the Anthropic Models API).
        return {
            "accepted": True,
            "persist": True,
            "recognized": False,
            "message": (
                f"Note: could not verify `{requested}` against this endpoint's "
                f"model listing.  Many Anthropic-compatible proxies do not "
                f"implement GET /v1/models.  The model name has been accepted "
                f"without verification."
            ),
        }

    # Probe the live API to check if the model actually exists
    api_models = fetch_api_models(api_key, base_url)

    if api_models is not None:
        # Gemini's OpenAI-compat /v1beta/openai/models endpoint returns IDs
        # prefixed with "models/" (e.g. "models/gemini-2.5-flash") — native
        # Gemini-API convention.  Our curated list and user input both use
        # the bare ID, so a direct set-membership check drops every known
        # Gemini model.  Strip the prefix before comparison.  See #12532.
        if normalized == "gemini":
            api_models = [
                m[len("models/"):] if isinstance(m, str) and m.startswith("models/") else m
                for m in api_models
            ]
        if requested_for_lookup in set(api_models):
            # API confirmed the model exists
            return {
                "accepted": True,
                "persist": True,
                "recognized": True,
                "message": None,
            }
        else:
            # API responded but model is not listed.  Accept anyway —
            # the user may have access to models not shown in the public
            # listing (e.g. Z.AI Pro/Max plans can use glm-5 on coding
            # endpoints even though it's not in /models).  Warn but allow.

            # Auto-correct if the top match is very similar (e.g. typo)
            auto = get_close_matches(requested_for_lookup, api_models, n=1, cutoff=0.9)
            if auto:
                return {
                    "accepted": True,
                    "persist": True,
                    "recognized": True,
                    "corrected_model": auto[0],
                    "message": f"Auto-corrected `{requested}` → `{auto[0]}`",
                }

            suggestions = get_close_matches(requested, api_models, n=3, cutoff=0.5)
            suggestion_text = ""
            if suggestions:
                suggestion_text = "\n  Similar models: " + ", ".join(f"`{s}`" for s in suggestions)

        return {
            "accepted": False,
            "persist": False,
            "recognized": False,
            "message": (
                f"Model `{requested}` was not found in this provider's model listing."
                f"{suggestion_text}"
            ),
        }

    # api_models is None — couldn't reach API.  Accept and persist,
    # but warn so typos don't silently break things.

    # Bedrock: use our own discovery instead of HTTP /models endpoint.
    # Bedrock's bedrock-runtime URL doesn't support /models — it uses the
    # AWS SDK control plane (ListFoundationModels + ListInferenceProfiles).
    if normalized == "bedrock":
        try:
            from easybci_agent.bedrock_adapter import discover_bedrock_models, resolve_bedrock_region
            region = resolve_bedrock_region()
            discovered = discover_bedrock_models(region)
            discovered_ids = {m["id"] for m in discovered}
            if requested in discovered_ids:
                return {
                    "accepted": True,
                    "persist": True,
                    "recognized": True,
                    "message": None,
                }
            # Not in discovered list — still accept (user may have custom
            # inference profiles or cross-account access), but warn.
            suggestions = get_close_matches(requested, list(discovered_ids), n=3, cutoff=0.4)
            suggestion_text = ""
            if suggestions:
                suggestion_text = "\n  Similar models: " + ", ".join(f"`{s}`" for s in suggestions)
            return {
                "accepted": True,
                "persist": True,
                "recognized": False,
                "message": (
                    f"Note: `{requested}` was not found in Bedrock model discovery for {region}. "
                    f"It may still work with custom inference profiles or cross-account access."
                    f"{suggestion_text}"
                ),
            }
        except Exception:
            pass  # Fall through to generic warning

    # Static-catalog fallback: when the /models probe was unreachable,
    # validate against the curated list from provider_model_ids() — same
    # pattern as the minimax branches above.  This fixes
    # /model switches in the gateway for providers like opencode-go and
    # opencode-zen whose /models endpoint returns 404 against the HTML
    # marketing site.  Without this block, validate_requested_model would
    # reject every model on such providers, switch_model() would return
    # success=False, and the gateway would never write to
    # _session_model_overrides.
    provider_label = _PROVIDER_LABELS.get(normalized, normalized)
    try:
        catalog_models = provider_model_ids(normalized)
    except Exception:
        catalog_models = []

    if catalog_models:
        catalog_lower = {m.lower(): m for m in catalog_models}
        if requested_for_lookup.lower() in catalog_lower:
            return {
                "accepted": True,
                "persist": True,
                "recognized": True,
                "message": None,
            }
        catalog_lower_list = list(catalog_lower.keys())
        auto = get_close_matches(
            requested_for_lookup.lower(), catalog_lower_list, n=1, cutoff=0.9
        )
        if auto:
            corrected = catalog_lower[auto[0]]
            return {
                "accepted": True,
                "persist": True,
                "recognized": True,
                "corrected_model": corrected,
                "message": f"Auto-corrected `{requested}` → `{corrected}`",
            }
        suggestions = get_close_matches(
            requested_for_lookup.lower(), catalog_lower_list, n=3, cutoff=0.5
        )
        suggestion_text = ""
        if suggestions:
            suggestion_text = "\n  Similar models: " + ", ".join(
                f"`{catalog_lower[s]}`" for s in suggestions
            )
        return {
            "accepted": True,
            "persist": True,
            "recognized": False,
            "message": (
                f"Note: `{requested}` was not found in the {provider_label} curated catalog "
                f"and the /models endpoint was unreachable.{suggestion_text}"
                f"\n  The model may still work if it exists on the provider."
            ),
        }

    # No catalog available — accept with a warning, matching the comment's
    # stated intent ("Accept and persist, but warn").
    return {
        "accepted": True,
        "persist": True,
        "recognized": False,
        "message": (
            f"Note: could not reach the {provider_label} API to validate `{requested}`. "
            f"If the service isn't down, this model may not be valid."
        ),
    }
