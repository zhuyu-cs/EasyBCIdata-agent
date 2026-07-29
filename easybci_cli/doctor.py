"""
Doctor command for easybci CLI.

Diagnoses issues with EasyBCI Agent setup.
"""

import importlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from easybci_cli.config import get_project_root, get_easybci_home, get_env_path
from easybci_cli.env_loader import load_easybci_dotenv
from easybci_lib.constants import display_easybci_home
from easybci_lib.tools.neural_processing.preprocess import analysis_goals as _analysis_goals_module

PROJECT_ROOT = get_project_root()
EASYBCI_HOME = get_easybci_home()
_DHH = display_easybci_home()  # user-facing display path (e.g. ~/.easybci or ~/.easybci/profiles/coder)

# Load environment variables from ~/.easybci/.env so API key checks work
_env_path = get_env_path()
load_easybci_dotenv(easybci_home=_env_path.parent, project_env=PROJECT_ROOT / ".env")

from easybci_cli.colors import Colors, color
from easybci_cli.models import _EASYBCI_USER_AGENT
from easybci_lib.constants import OPENROUTER_MODELS_URL
from easybci_lib.utils import base_url_host_matches


_PROVIDER_ENV_HINTS = (
    "OPENROUTER_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_TOKEN",
    "OPENAI_BASE_URL",
    "BCI_TEAM_API_KEY",
    "GLM_API_KEY",
    "ZAI_API_KEY",
    "Z_AI_API_KEY",
    "KIMI_API_KEY",
    "KIMI_CN_API_KEY",
    "GMI_API_KEY",
    "MINIMAX_API_KEY",
    "MINIMAX_CN_API_KEY",
    "KILOCODE_API_KEY",
    "DEEPSEEK_API_KEY",
    "DASHSCOPE_API_KEY",
    "HF_TOKEN",
    "AI_GATEWAY_API_KEY",
    "OPENCODE_ZEN_API_KEY",
    "OPENCODE_GO_API_KEY",
    "XIAOMI_API_KEY",
    "TOKENHUB_API_KEY",
)


from easybci_lib.constants import is_termux as _is_termux


def _python_install_cmd() -> str:
    return "python -m pip install" if _is_termux() else "uv pip install"


def _system_package_install_cmd(pkg: str) -> str:
    if _is_termux():
        return f"pkg install {pkg}"
    if sys.platform == "darwin":
        return f"brew install {pkg}"
    return f"sudo apt install {pkg}"


def _safe_which(cmd: str) -> str | None:
    """shutil.which wrapper resilient to platform monkeypatching in tests."""
    try:
        return shutil.which(cmd)
    except Exception:
        return None




def _termux_install_all_fallback_notes() -> list[str]:
    return [
        "Termux install profile: use .[termux-all] for broad compatibility (installer default on Termux).",
    ]


def _has_provider_env_config(content: str) -> bool:
    """Return True when ~/.easybci/.env contains provider auth/base URL settings."""
    return any(key in content for key in _PROVIDER_ENV_HINTS)


def _honcho_is_configured_for_doctor() -> bool:
    """Return True when Honcho is configured, even if this process has no active session."""
    try:
        from services.plugins.memory.honcho.client import HonchoClientConfig

        cfg = HonchoClientConfig.from_global_config()
        return bool(cfg.enabled and (cfg.api_key or cfg.base_url))
    except Exception:
        return False


def _doctor_tool_availability_detail(toolset: str) -> str:
    """Optional explanatory suffix for toolsets whose doctor status needs context."""
    return ""


def _apply_doctor_tool_availability_overrides(available: list[str], unavailable: list[dict]) -> tuple[list[str], list[dict]]:
    """Adjust runtime-gated tool availability for doctor diagnostics."""
    updated_available = list(available)
    updated_unavailable = []
    for item in unavailable:
        name = item.get("name")
        if name == "honcho" and _honcho_is_configured_for_doctor():
            if "honcho" not in updated_available:
                updated_available.append("honcho")
            continue
        updated_unavailable.append(item)
    return updated_available, updated_unavailable


def check_ok(text: str, detail: str = ""):
    print(f"  {color('✓', Colors.GREEN)} {text}" + (f" {color(detail, Colors.DIM)}" if detail else ""))

def check_warn(text: str, detail: str = ""):
    print(f"  {color('⚠', Colors.YELLOW)} {text}" + (f" {color(detail, Colors.DIM)}" if detail else ""))

def check_fail(text: str, detail: str = ""):
    print(f"  {color('✗', Colors.RED)} {text}" + (f" {color(detail, Colors.DIM)}" if detail else ""))

def check_info(text: str):
    print(f"    {color('→', Colors.CYAN)} {text}")


def _check_gateway_service_linger(issues: list[str]) -> None:
    """Warn when a systemd user gateway service will stop after logout."""
    try:
        from easybci_cli.gateway import (
            get_systemd_linger_status,
            get_systemd_unit_path,
            is_linux,
        )
    except Exception as e:
        check_warn("Gateway service linger", f"(could not import gateway helpers: {e})")
        return

    if not is_linux():
        return

    unit_path = get_systemd_unit_path()
    if not unit_path.exists():
        return

    print()
    print(color("◆ Gateway Service", Colors.CYAN, Colors.BOLD))

    linger_enabled, linger_detail = get_systemd_linger_status()
    if linger_enabled is True:
        check_ok("Systemd linger enabled", "(gateway service survives logout)")
    elif linger_enabled is False:
        check_warn("Systemd linger disabled", "(gateway may stop after logout)")
        check_info("Run: sudo loginctl enable-linger $USER")
        issues.append("Enable linger for the gateway user service: sudo loginctl enable-linger $USER")
    else:
        check_warn("Could not verify systemd linger", f"({linger_detail})")


_APIKEY_PROVIDERS_CACHE: list | None = None


def _build_apikey_providers_list() -> list:
    """Build the API-key provider health-check list once and cache it.

    Tuple format: (name, env_vars, default_url, base_env, supports_models_endpoint)
    Base list augmented with any ProviderProfile with auth_type="api_key" not
    already present — adding plugins/model-providers/<name>/ is sufficient to get into doctor.
    """
    _static = [
        ("Z.AI / GLM",      ("GLM_API_KEY", "ZAI_API_KEY", "Z_AI_API_KEY"), "https://api.z.ai/api/paas/v4/models", "GLM_BASE_URL", True),
        ("Kimi / Moonshot",  ("KIMI_API_KEY",),                              "https://api.moonshot.ai/v1/models",   "KIMI_BASE_URL", True),
        ("StepFun Step Plan", ("STEPFUN_API_KEY",),                          "https://api.stepfun.ai/step_plan/v1/models", "STEPFUN_BASE_URL", True),
        ("Kimi / Moonshot (China)", ("KIMI_CN_API_KEY",),                    "https://api.moonshot.cn/v1/models",   None, True),
        ("Arcee AI",         ("ARCEEAI_API_KEY",),                           "https://api.arcee.ai/api/v1/models",  "ARCEE_BASE_URL", True),
        ("GMI Cloud",        ("GMI_API_KEY",),                               "https://api.gmi-serving.com/v1/models", "GMI_BASE_URL", True),
        ("DeepSeek",         ("DEEPSEEK_API_KEY",),                          "https://api.deepseek.com/v1/models",  "DEEPSEEK_BASE_URL", True),
        ("Hugging Face",     ("HF_TOKEN",),                                  "https://router.huggingface.co/v1/models", "HF_BASE_URL", True),
        ("NVIDIA NIM",       ("NVIDIA_API_KEY",),                            "https://integrate.api.nvidia.com/v1/models", "NVIDIA_BASE_URL", True),
        ("Alibaba/DashScope", ("DASHSCOPE_API_KEY",),                        "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/models", "DASHSCOPE_BASE_URL", True),
        # MiniMax global: /v1 endpoint supports /models.
        ("MiniMax",          ("MINIMAX_API_KEY",),                           "https://api.minimax.io/v1/models",    "MINIMAX_BASE_URL", True),
        # MiniMax CN: /v1 endpoint does NOT support /models (returns 404).
        ("MiniMax (China)",  ("MINIMAX_CN_API_KEY",),                        "https://api.minimaxi.com/v1/models",  "MINIMAX_CN_BASE_URL", False),
        ("Vercel AI Gateway", ("AI_GATEWAY_API_KEY",),                       "https://ai-gateway.vercel.sh/v1/models", "AI_GATEWAY_BASE_URL", True),
        ("Kilo Code",        ("KILOCODE_API_KEY",),                          "https://api.kilo.ai/api/gateway/models", "KILOCODE_BASE_URL", True),
        ("OpenCode Zen",     ("OPENCODE_ZEN_API_KEY",),                      "https://opencode.ai/zen/v1/models",  "OPENCODE_ZEN_BASE_URL", True),
        # OpenCode Go has no shared /models endpoint; skip the health check.
        ("OpenCode Go",      ("OPENCODE_GO_API_KEY",),                       None,                                  "OPENCODE_GO_BASE_URL", False),
    ]
    _known_names = {t[0] for t in _static}
    # Also index by profile canonical name so profiles without display_name
    # don't create duplicate entries for providers already in the static list.
    _known_canonical: set[str] = set()
    _name_to_canonical = {
        "Z.AI / GLM": "zai", "Kimi / Moonshot": "kimi-coding",
        "StepFun Step Plan": "stepfun", "Kimi / Moonshot (China)": "kimi-coding-cn",
        "Arcee AI": "arcee", "GMI Cloud": "gmi", "DeepSeek": "deepseek",
        "Hugging Face": "huggingface", "NVIDIA NIM": "nvidia",
        "Alibaba/DashScope": "alibaba", "MiniMax": "minimax",
        "MiniMax (China)": "minimax-cn", "Vercel AI Gateway": "ai-gateway",
        "Kilo Code": "kilocode", "OpenCode Zen": "opencode-zen",
        "OpenCode Go": "opencode-go",
    }
    for _label, _canonical in _name_to_canonical.items():
        _known_canonical.add(_canonical)
    # Providers that already have a dedicated health check above the generic
    # API-key loop (with custom headers/auth). Skip their pluggable profiles
    # here so the generic Bearer-auth loop doesn't run a duplicate, broken
    # check (e.g. Anthropic native API requires x-api-key, not Bearer).
    _dedicated_canonical = {"anthropic", "openrouter", "bedrock"}
    _known_canonical.update(_dedicated_canonical)
    try:
        from services.providers import list_providers
        from services.providers.base import ProviderProfile as _PP
        try:
            from easybci_cli.providers import normalize_provider as _normalize_provider
        except Exception:  # pragma: no cover - normalization is best-effort
            def _normalize_provider(_name: str) -> str:
                return (_name or "").strip().lower()
        for _pp in list_providers():
            if not isinstance(_pp, _PP) or _pp.auth_type != "api_key" or not _pp.env_vars:
                continue
            _label = _pp.display_name or _pp.name
            if _label in _known_names or _pp.name in _known_canonical:
                continue
            _candidates = {_normalize_provider(_pp.name)}
            for _alias in (_pp.aliases or ()):
                _candidates.add(_normalize_provider(_alias))
            if _candidates & _dedicated_canonical:
                continue
            # Separate API-key vars from base-URL override vars — the health-check
            # loop sends the first found value as Authorization: Bearer, so a URL
            # string must never be picked.
            _key_vars = tuple(
                v for v in _pp.env_vars
                if not v.endswith("_BASE_URL") and not v.endswith("_URL")
            )
            _base_var = next(
                (v for v in _pp.env_vars if v.endswith("_BASE_URL") or v.endswith("_URL")),
                None,
            )
            if not _key_vars:
                continue
            _models_url = (
                (_pp.models_url or (_pp.base_url.rstrip("/") + "/models"))
                if _pp.base_url else None
            )
            _hc = getattr(_pp, "supports_health_check", True)
            _static.append((_label, _key_vars, _models_url, _base_var, _hc))
    except Exception:
        pass
    return _static


def run_doctor(args):
    """Run diagnostic checks."""
    should_fix = getattr(args, 'fix', False)
    ack_target = getattr(args, 'ack', None)

    # Doctor runs from the interactive CLI, so CLI-gated tool availability
    # checks should see the same context as `easybci`.
    os.environ.setdefault("EASYBCI_INTERACTIVE", "1")

    # Handle `easybci doctor --ack <id>` as a fast path. Persist the ack and
    # return without running the rest of the diagnostics — the user has
    # already seen the advisory and just wants to silence it.
    if ack_target:
        from easybci_cli.security_advisories import (
            ADVISORIES,
            ack_advisory,
        )
        valid_ids = {a.id for a in ADVISORIES}
        if ack_target not in valid_ids:
            print(color(
                f"Unknown advisory ID: {ack_target!r}. Known IDs: "
                f"{', '.join(sorted(valid_ids)) or '(none)'}",
                Colors.RED,
            ))
            sys.exit(2)
        if ack_advisory(ack_target):
            print(color(
                f"  ✓ Acknowledged advisory {ack_target}. "
                f"It will no longer trigger startup banners.",
                Colors.GREEN,
            ))
        else:
            print(color(
                f"  ✗ Failed to persist ack for {ack_target}. "
                f"Check ~/.easybci/config.yaml is writable.",
                Colors.RED,
            ))
            sys.exit(1)
        return

    issues = []
    manual_issues = []  # issues that can't be auto-fixed
    fixed_count = 0

    print()
    print(color("┌─────────────────────────────────────────────────────────┐", Colors.CYAN))
    print(color("│                 🩺 EasyBCI Doctor                        │", Colors.CYAN))
    print(color("└─────────────────────────────────────────────────────────┘", Colors.CYAN))

    # =========================================================================
    # Check: Security advisories  (RUNS FIRST — these are the most urgent)
    # =========================================================================
    print()
    print(color("◆ Security Advisories", Colors.CYAN, Colors.BOLD))
    try:
        from easybci_cli.security_advisories import (
            detect_compromised,
            filter_unacked,
            full_remediation_text,
            get_acked_ids,
        )
        all_hits = detect_compromised()
        fresh_hits = filter_unacked(all_hits)
        if fresh_hits:
            for hit in fresh_hits:
                check_fail(
                    f"{hit.advisory.title}",
                    f"({hit.package}=={hit.installed_version})",
                )
                # Print the full remediation block, indented under the
                # check_fail header so it reads as a single section.
                for line in full_remediation_text(hit):
                    if line:
                        print(f"    {color(line, Colors.YELLOW)}")
                    else:
                        print()
                # Funnel into the action list so the summary block surfaces it
                # for users who scroll past the section.
                manual_issues.append(
                    f"Resolve security advisory {hit.advisory.id}: "
                    f"uninstall {hit.package}=={hit.installed_version} and "
                    f"rotate credentials, then run "
                    f"`easybci doctor --ack {hit.advisory.id}`."
                )
            # Acked-but-still-installed: show as informational so the user
            # knows the package is still on disk after the ack.
            acked_ids = get_acked_ids()
            for h in all_hits:
                if h.advisory.id in acked_ids:
                    check_warn(
                        f"{h.package}=={h.installed_version} still installed "
                        f"(advisory {h.advisory.id} acknowledged)",
                    )
        else:
            check_ok("No active security advisories")
    except Exception as e:
        # Never let a bug in the advisory check block the rest of doctor.
        check_warn(f"Security advisory check failed: {e}")
    
    # =========================================================================
    # Check: Python version
    # =========================================================================
    print()
    print(color("◆ Python Environment", Colors.CYAN, Colors.BOLD))
    
    py_version = sys.version_info
    if py_version >= (3, 11):
        check_ok(f"Python {py_version.major}.{py_version.minor}.{py_version.micro}")
    elif py_version >= (3, 10):
        check_ok(f"Python {py_version.major}.{py_version.minor}.{py_version.micro}")
        check_warn("Python 3.11+ recommended for RL Training tools (tinker requires >= 3.11)")
    elif py_version >= (3, 8):
        check_warn(f"Python {py_version.major}.{py_version.minor}.{py_version.micro}", "(3.10+ recommended)")
    else:
        check_fail(f"Python {py_version.major}.{py_version.minor}.{py_version.micro}", "(3.10+ required)")
        issues.append("Upgrade Python to 3.10+")
    
    # Check if in virtual environment
    in_venv = sys.prefix != sys.base_prefix
    if in_venv:
        check_ok("Virtual environment active")
    else:
        check_warn("Not in virtual environment", "(recommended)")
    
    # =========================================================================
    # Check: Required packages
    # =========================================================================
    print()
    print(color("◆ Required Packages", Colors.CYAN, Colors.BOLD))
    
    required_packages = [
        ("openai", "OpenAI SDK"),
        ("rich", "Rich (terminal UI)"),
        ("dotenv", "python-dotenv"),
        ("yaml", "PyYAML"),
        ("httpx", "HTTPX"),
    ]
    
    optional_packages = []
    
    for module, name in required_packages:
        try:
            __import__(module)
            check_ok(name)
        except ImportError:
            check_fail(name, "(missing)")
            issues.append(f"Install {name}: {_python_install_cmd()} {module}")
    
    for module, name in optional_packages:
        try:
            __import__(module)
            check_ok(name, "(optional)")
        except ImportError:
            check_warn(name, "(optional, not installed)")
    
    # =========================================================================
    # Check: Configuration files
    # =========================================================================
    print()
    print(color("◆ Configuration Files", Colors.CYAN, Colors.BOLD))
    
    # Check ~/.easybci/.env (primary location for user config)
    env_path = EASYBCI_HOME / '.env'
    if env_path.exists():
        check_ok(f"{_DHH}/.env file exists")
        
        # Check for common issues. Pin encoding to UTF-8 because .env files are
        # written as UTF-8 everywhere in the codebase, while Path.read_text()
        # defaults to the system locale — which crashes on non-UTF-8 Windows
        # locales (e.g. GBK) as soon as the file contains any non-ASCII byte.
        content = env_path.read_text(encoding="utf-8")
        if _has_provider_env_config(content):
            check_ok("API key or custom endpoint configured")
        else:
            check_warn(f"No API key found in {_DHH}/.env")
            issues.append("Run 'easybci setup' to configure API keys")
    else:
        # Also check project root as fallback
        fallback_env = PROJECT_ROOT / '.env'
        if fallback_env.exists():
            check_ok(".env file exists (in project directory)")
        else:
            check_fail(f"{_DHH}/.env file missing")
            if should_fix:
                env_path.parent.mkdir(parents=True, exist_ok=True)
                env_path.touch()
                check_ok(f"Created empty {_DHH}/.env")
                check_info("Run 'easybci setup' to configure API keys")
                fixed_count += 1
            else:
                check_info("Run 'easybci setup' to create one")
                issues.append("Run 'easybci setup' to create .env")
    
    # Check ~/.easybci/config.yaml (primary) or project cli-config.yaml (fallback)
    config_path = EASYBCI_HOME / 'config.yaml'
    if config_path.exists():
        check_ok(f"{_DHH}/config.yaml exists")

        # Validate model.provider and model.default values
        try:
            import yaml as _yaml
            cfg = _yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            model_section = cfg.get("model") or {}
            provider_raw = (model_section.get("provider") or "").strip()
            provider = provider_raw.lower()
            default_model = (model_section.get("default") or model_section.get("model") or "").strip()

            known_providers: set = set()
            try:
                from easybci_cli.auth import (
                    PROVIDER_REGISTRY,
                    resolve_provider as _resolve_auth_provider,
                )
                known_providers = set(PROVIDER_REGISTRY.keys()) | {"openrouter", "custom", "auto"}
            except Exception:
                _resolve_auth_provider = None
                pass
            try:
                from easybci_cli.config import get_compatible_custom_providers as _compatible_custom_providers
                from easybci_cli.providers import (
                    normalize_provider as _normalize_catalog_provider,
                    resolve_provider_full as _resolve_provider_full,
                )
            except Exception:
                _compatible_custom_providers = None
                _normalize_catalog_provider = None
                _resolve_provider_full = None

            custom_providers = []
            if _compatible_custom_providers is not None:
                try:
                    custom_providers = _compatible_custom_providers(cfg)
                except Exception:
                    custom_providers = []

            user_providers = cfg.get("providers")
            if isinstance(user_providers, dict):
                known_providers.update(str(name).strip().lower() for name in user_providers if str(name).strip())
            for entry in custom_providers:
                if not isinstance(entry, dict):
                    continue
                name = str(entry.get("name") or "").strip()
                if name:
                    known_providers.add("custom:" + name.lower().replace(" ", "-"))

            valid_provider_ids = set(known_providers)
            provider_ids_to_accept = {provider} if provider else set()
            if _normalize_catalog_provider is not None:
                for known_provider in known_providers:
                    try:
                        valid_provider_ids.add(_normalize_catalog_provider(known_provider))
                    except Exception:
                        continue

            runtime_provider = provider
            if (
                provider
                and _resolve_auth_provider is not None
                and provider not in {"auto", "custom"}
            ):
                try:
                    runtime_provider = _resolve_auth_provider(provider)
                    provider_ids_to_accept.add(runtime_provider)
                except Exception:
                    runtime_provider = provider

            catalog_provider = provider
            if (
                provider
                and _resolve_provider_full is not None
                and provider not in {"auto", "custom"}
            ):
                provider_def = _resolve_provider_full(provider, user_providers, custom_providers)
                catalog_provider = provider_def.id if provider_def is not None else None
                if catalog_provider is not None:
                    provider_ids_to_accept.add(catalog_provider)

            if provider and provider != "auto":
                if catalog_provider is None or (
                    known_providers
                    and not (provider_ids_to_accept & valid_provider_ids)
                ):
                    known_list = ", ".join(sorted(known_providers)) if known_providers else "(unavailable)"
                    check_fail(
                        f"model.provider '{provider_raw}' is not a recognised provider",
                        f"(known: {known_list})",
                    )
                    issues.append(
                        f"model.provider '{provider_raw}' is unknown. "
                        f"Valid providers: {known_list}. "
                        f"Fix: run 'easybci config set model.provider <valid_provider>'"
                    )

            # Warn if model is set to a provider-prefixed name on a provider that doesn't use them
            provider_for_policy = runtime_provider or catalog_provider
            providers_accepting_vendor_slugs = {
                "openrouter",
                "custom",
                "auto",
                "ai-gateway",
                "kilocode",
                "opencode-zen",
                "huggingface",
                "lmstudio",
            }
            if (
                default_model
                and "/" in default_model
                and provider_for_policy
                and provider_for_policy not in providers_accepting_vendor_slugs
            ):
                check_warn(
                    f"model.default '{default_model}' uses a vendor/model slug but provider is '{provider_raw}'",
                    "(vendor-prefixed slugs belong to aggregators like openrouter)",
                )
                issues.append(
                    f"model.default '{default_model}' is vendor-prefixed but model.provider is '{provider_raw}'. "
                    "Either set model.provider to 'openrouter', or drop the vendor prefix."
                )

            # Check credentials for the configured provider.
            # Limit to API-key providers in PROVIDER_REGISTRY — other provider
            # types (OAuth, SDK, openrouter/anthropic/custom/auto) have their
            # own env-var checks elsewhere in doctor, and get_auth_status()
            # returns a bare {logged_in: False} for anything it doesn't
            # explicitly dispatch, which would produce false positives.
            if runtime_provider and runtime_provider not in {"auto", "custom", "openrouter"}:
                try:
                    from easybci_cli.auth import PROVIDER_REGISTRY, get_auth_status
                    pconfig = PROVIDER_REGISTRY.get(runtime_provider)
                    if pconfig and getattr(pconfig, "auth_type", "") == "api_key":
                        status = get_auth_status(runtime_provider) or {}
                        configured = bool(
                            status.get("configured")
                            or status.get("logged_in")
                            or status.get("api_key")
                        )
                        if not configured:
                            check_fail(
                                f"model.provider '{runtime_provider}' is set but no API key is configured",
                                "(check ~/.easybci/.env or run 'easybci setup')",
                            )
                            issues.append(
                                f"No credentials found for provider '{runtime_provider}'. "
                                f"Run 'easybci setup' or set the provider's API key in {_DHH}/.env, "
                                f"or switch providers with 'easybci config set model.provider <name>'"
                            )
                except Exception:
                    pass

        except Exception as e:
            check_warn("Could not validate model/provider config", f"({e})")
    else:
        fallback_config = PROJECT_ROOT / 'cli-config.yaml'
        if fallback_config.exists():
            check_ok("cli-config.yaml exists (in project directory)")
        else:
            example_config = PROJECT_ROOT / 'cli-config.yaml.example'
            if should_fix and example_config.exists():
                config_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(example_config), str(config_path))
                check_ok(f"Created {_DHH}/config.yaml from cli-config.yaml.example")
                fixed_count += 1
            elif should_fix:
                check_warn("config.yaml not found and no example to copy from")
                manual_issues.append(f"Create {_DHH}/config.yaml manually")
            else:
                check_warn("config.yaml not found", "(using defaults)")

    # Check config version and stale keys
    config_path = EASYBCI_HOME / 'config.yaml'
    if config_path.exists():
        try:
            from easybci_cli.config import check_config_version, migrate_config
            current_ver, latest_ver = check_config_version()
            if current_ver < latest_ver:
                check_warn(
                    f"Config version outdated (v{current_ver} → v{latest_ver})",
                    "(new settings available)"
                )
                if should_fix:
                    try:
                        migrate_config(interactive=False, quiet=False)
                        check_ok("Config migrated to latest version")
                        fixed_count += 1
                    except Exception as mig_err:
                        check_warn(f"Auto-migration failed: {mig_err}")
                        issues.append("Run 'easybci setup' to migrate config")
                else:
                    issues.append("Run 'easybci doctor --fix' or 'easybci setup' to migrate config")
            else:
                check_ok(f"Config version up to date (v{current_ver})")
        except Exception:
            pass

        # Detect stale root-level model keys (known bug source — PR #4329)
        try:
            import yaml
            with open(config_path, encoding="utf-8") as f:
                raw_config = yaml.safe_load(f) or {}
            stale_root_keys = [k for k in ("provider", "base_url") if k in raw_config and isinstance(raw_config[k], str)]
            if stale_root_keys:
                check_warn(
                    f"Stale root-level config keys: {', '.join(stale_root_keys)}",
                    "(should be under 'model:' section)"
                )
                if should_fix:
                    model_section = raw_config.setdefault("model", {})
                    for k in stale_root_keys:
                        if not model_section.get(k):
                            model_section[k] = raw_config.pop(k)
                        else:
                            raw_config.pop(k)
                    from easybci_lib.utils import atomic_yaml_write
                    atomic_yaml_write(config_path, raw_config)
                    check_ok("Migrated stale root-level keys into model section")
                    fixed_count += 1
                else:
                    issues.append("Stale root-level provider/base_url in config.yaml — run 'easybci doctor --fix'")
        except Exception:
            pass

        # Validate config structure (catches malformed custom_providers, etc.)
        try:
            from easybci_cli.config import validate_config_structure
            config_issues = validate_config_structure()
            if config_issues:
                print()
                print(color("◆ Config Structure", Colors.CYAN, Colors.BOLD))
                for ci in config_issues:
                    if ci.severity == "error":
                        check_fail(ci.message)
                    else:
                        check_warn(ci.message)
                    # Show the hint indented
                    for hint_line in ci.hint.splitlines():
                        check_info(hint_line)
                    issues.append(ci.message)
        except Exception:
            pass

    # =========================================================================
    # Check: Auth providers
    # =========================================================================
    print()
    print(color("◆ Auth Providers", Colors.CYAN, Colors.BOLD))

    try:
        from easybci_cli.auth import (
            get_gemini_oauth_auth_status,
            get_minimax_oauth_auth_status,
        )

        gemini_status = get_gemini_oauth_auth_status()
        if gemini_status.get("logged_in"):
            email = gemini_status.get("email") or ""
            project = gemini_status.get("project_id") or ""
            pieces = []
            if email:
                pieces.append(email)
            if project:
                pieces.append(f"project={project}")
            suffix = f" ({', '.join(pieces)})" if pieces else ""
            check_ok("Google Gemini OAuth", f"(logged in{suffix})")
        else:
            check_warn("Google Gemini OAuth", "(not logged in)")

        minimax_status = get_minimax_oauth_auth_status()
        if minimax_status.get("logged_in"):
            region = minimax_status.get("region", "global")
            check_ok("MiniMax OAuth", f"(logged in, region={region})")
        else:
            check_warn("MiniMax OAuth", "(not logged in)")
    except Exception as e:
        check_warn("Auth provider status", f"(could not check: {e})")

    # =========================================================================
    # Check: Directory structure
    # =========================================================================
    print()
    print(color("◆ Directory Structure", Colors.CYAN, Colors.BOLD))
    
    easybci_home = EASYBCI_HOME
    if easybci_home.exists():
        check_ok(f"{_DHH} directory exists")
    elif should_fix:
        easybci_home.mkdir(parents=True, exist_ok=True)
        check_ok(f"Created {_DHH} directory")
        fixed_count += 1
    else:
        check_warn(f"{_DHH} not found", "(will be created on first use)")
    
    # Check expected subdirectories
    expected_subdirs = ["sessions", "logs", "skills", "memories"]
    for subdir_name in expected_subdirs:
        subdir_path = easybci_home / subdir_name
        if subdir_path.exists():
            check_ok(f"{_DHH}/{subdir_name}/ exists")
        elif should_fix:
            subdir_path.mkdir(parents=True, exist_ok=True)
            check_ok(f"Created {_DHH}/{subdir_name}/")
            fixed_count += 1
        else:
            check_warn(f"{_DHH}/{subdir_name}/ not found", "(will be created on first use)")
    
    # Check for SOUL.md persona file
    soul_path = easybci_home / "SOUL.md"
    if soul_path.exists():
        content = soul_path.read_text(encoding="utf-8").strip()
        # Check if it's just the template comments (no real content)
        lines = [l for l in content.splitlines() if l.strip() and not l.strip().startswith(("<!--", "-->", "#"))]
        if lines:
            check_ok(f"{_DHH}/SOUL.md exists (persona configured)")
        else:
            check_info(f"{_DHH}/SOUL.md exists but is empty — edit it to customize personality")
    else:
        check_warn(f"{_DHH}/SOUL.md not found", "(create it to give EasyBCI a custom personality)")
        if should_fix:
            soul_path.parent.mkdir(parents=True, exist_ok=True)
            soul_path.write_text(
                "# EasyBCI Agent Persona\n\n"
                "<!-- Edit this file to customize how EasyBCI communicates. -->\n\n"
                "You are EasyBCI, a helpful AI assistant.\n",
                encoding="utf-8",
            )
            check_ok(f"Created {_DHH}/SOUL.md with basic template")
            fixed_count += 1
    
    # Check memory directory
    memories_dir = easybci_home / "memories"
    if memories_dir.exists():
        check_ok(f"{_DHH}/memories/ directory exists")
        memory_file = memories_dir / "MEMORY.md"
        user_file = memories_dir / "USER.md"
        if memory_file.exists():
            size = len(memory_file.read_text(encoding="utf-8").strip())
            check_ok(f"MEMORY.md exists ({size} chars)")
        else:
            check_info("MEMORY.md not created yet (will be created when the agent first writes a memory)")
        if user_file.exists():
            size = len(user_file.read_text(encoding="utf-8").strip())
            check_ok(f"USER.md exists ({size} chars)")
        else:
            check_info("USER.md not created yet (will be created when the agent first writes a memory)")
    else:
        check_warn(f"{_DHH}/memories/ not found", "(will be created on first use)")
        if should_fix:
            memories_dir.mkdir(parents=True, exist_ok=True)
            check_ok(f"Created {_DHH}/memories/")
            fixed_count += 1
    
    # Check SQLite session store
    state_db_path = easybci_home / "state.db"
    if state_db_path.exists():
        try:
            import sqlite3
            conn = sqlite3.connect(str(state_db_path))
            cursor = conn.execute("SELECT COUNT(*) FROM sessions")
            count = cursor.fetchone()[0]
            conn.close()
            check_ok(f"{_DHH}/state.db exists ({count} sessions)")
        except Exception as e:
            check_warn(f"{_DHH}/state.db exists but has issues: {e}")
    else:
        check_info(f"{_DHH}/state.db not created yet (will be created on first session)")

    # Check WAL file size (unbounded growth indicates missed checkpoints)
    wal_path = easybci_home / "state.db-wal"
    if wal_path.exists():
        try:
            wal_size = wal_path.stat().st_size
            if wal_size > 50 * 1024 * 1024:  # 50 MB
                check_warn(
                    f"WAL file is large ({wal_size // (1024*1024)} MB)",
                    "(may indicate missed checkpoints)"
                )
                if should_fix:
                    import sqlite3
                    conn = sqlite3.connect(str(state_db_path))
                    conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
                    conn.close()
                    new_size = wal_path.stat().st_size if wal_path.exists() else 0
                    check_ok(f"WAL checkpoint performed ({wal_size // 1024}K → {new_size // 1024}K)")
                    fixed_count += 1
                else:
                    issues.append("Large WAL file — run 'easybci doctor --fix' to checkpoint")
            elif wal_size > 10 * 1024 * 1024:  # 10 MB
                check_info(f"WAL file is {wal_size // (1024*1024)} MB (normal for active sessions)")
        except Exception:
            pass

    _check_gateway_service_linger(issues)

    # =========================================================================
    # Check: Command installation (easybci bin symlink)
    # =========================================================================
    if True:
        print()
        print(color("◆ Command Installation", Colors.CYAN, Colors.BOLD))

        # Determine the venv entry point location
        _venv_bin = None
        for _venv_name in ("venv", ".venv"):
            _candidate = PROJECT_ROOT / _venv_name / "bin" / "easybci"
            if _candidate.exists():
                _venv_bin = _candidate
                break

        # Determine the expected command link directory (mirrors install.sh logic)
        _prefix = os.environ.get("PREFIX", "")
        _is_termux_env = bool(os.environ.get("TERMUX_VERSION")) or "com.termux/files/usr" in _prefix
        if _is_termux_env and _prefix:
            _cmd_link_dir = Path(_prefix) / "bin"
            _cmd_link_display = "$PREFIX/bin"
        else:
            _cmd_link_dir = Path.home() / ".local" / "bin"
            _cmd_link_display = "~/.local/bin"
        _cmd_link = _cmd_link_dir / "easybci"

        if _venv_bin is None:
            check_warn(
                "Venv entry point not found",
                "(easybci not in venv/bin/ or .venv/bin/ — reinstall with pip install -e '.[all]')"
            )
            manual_issues.append(
                f"Reinstall entry point: cd {PROJECT_ROOT} && source venv/bin/activate && pip install -e '.[all]'"
            )
        else:
            check_ok(f"Venv entry point exists ({_venv_bin.relative_to(PROJECT_ROOT)})")

            # Check the symlink at the command link location
            if _cmd_link.is_symlink():
                _target = _cmd_link.resolve()
                _expected = _venv_bin.resolve()
                if _target == _expected:
                    check_ok(f"{_cmd_link_display}/easybci → correct target")
                else:
                    check_warn(
                        f"{_cmd_link_display}/easybci points to wrong target",
                        f"(→ {_target}, expected → {_expected})"
                    )
                    if should_fix:
                        _cmd_link.unlink()
                        _cmd_link.symlink_to(_venv_bin)
                        check_ok(f"Fixed symlink: {_cmd_link_display}/easybci → {_venv_bin}")
                        fixed_count += 1
                    else:
                        issues.append(f"Broken symlink at {_cmd_link_display}/easybci — run 'easybci doctor --fix'")
            elif _cmd_link.exists():
                # It's a regular file, not a symlink — possibly a wrapper script
                check_ok(f"{_cmd_link_display}/easybci exists (non-symlink)")
            else:
                check_fail(
                    f"{_cmd_link_display}/easybci not found",
                    "(easybci command may not work outside the venv)"
                )
                if should_fix:
                    _cmd_link_dir.mkdir(parents=True, exist_ok=True)
                    _cmd_link.symlink_to(_venv_bin)
                    check_ok(f"Created symlink: {_cmd_link_display}/easybci → {_venv_bin}")
                    fixed_count += 1

                    # Check if the link dir is on PATH
                    _path_dirs = os.environ.get("PATH", "").split(os.pathsep)
                    if str(_cmd_link_dir) not in _path_dirs:
                        check_warn(
                            f"{_cmd_link_display} is not on your PATH",
                            "(add it to your shell config: export PATH=\"$HOME/.local/bin:$PATH\")"
                        )
                        manual_issues.append(f"Add {_cmd_link_display} to your PATH")
                else:
                    issues.append(f"Missing {_cmd_link_display}/easybci symlink — run 'easybci doctor --fix'")

    # =========================================================================
    # Check: External tools
    # =========================================================================
    print()
    print(color("◆ External Tools", Colors.CYAN, Colors.BOLD))
    
    # Git
    if _safe_which("git"):
        check_ok("git")
    else:
        check_warn("git not found", "(optional)")
    
    # ripgrep (optional, for faster file search)
    if _safe_which("rg"):
        check_ok("ripgrep (rg)", "(faster file search)")
    else:
        check_warn("ripgrep (rg) not found", "(file search uses grep fallback)")
        check_info(f"Install for faster search: {_system_package_install_cmd('ripgrep')}")
    
    # Docker (optional)
    terminal_env = os.getenv("TERMINAL_ENV", "local")
    if terminal_env == "docker":
        if _safe_which("docker"):
            # Check if docker daemon is running
            try:
                result = subprocess.run(["docker", "info"], capture_output=True, timeout=10)
            except subprocess.TimeoutExpired:
                result = None
            if result is not None and result.returncode == 0:
                check_ok("docker", "(daemon running)")
            else:
                check_fail("docker daemon not running")
                issues.append("Start Docker daemon")
        else:
            check_fail("docker not found", "(required for TERMINAL_ENV=docker)")
            issues.append("Install Docker or change TERMINAL_ENV")
    elif _safe_which("docker"):
        check_ok("docker", "(optional)")
    elif _is_termux():
        check_info("Docker backend is not available inside Termux (expected on Android)")
    else:
        check_warn("docker not found", "(optional)")
    
    # SSH (if using ssh backend)
    if terminal_env == "ssh":
        ssh_host = os.getenv("TERMINAL_SSH_HOST")
        if ssh_host:
            # Try to connect
            try:
                result = subprocess.run(
                    ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes", ssh_host, "echo ok"],
                    capture_output=True,
                    text=True,
                    timeout=15
                )
            except subprocess.TimeoutExpired:
                result = None
            if result is not None and result.returncode == 0:
                check_ok(f"SSH connection to {ssh_host}")
            else:
                check_fail(f"SSH connection to {ssh_host}")
                issues.append(f"Check SSH configuration for {ssh_host}")
        else:
            check_fail("TERMINAL_SSH_HOST not set", "(required for TERMINAL_ENV=ssh)")
            issues.append("Set TERMINAL_SSH_HOST in .env")
    
    # Daytona (if using daytona backend)
    if terminal_env == "daytona":
        daytona_key = os.getenv("DAYTONA_API_KEY")
        if daytona_key:
            check_ok("Daytona API key", "(configured)")
        else:
            check_fail("DAYTONA_API_KEY not set", "(required for TERMINAL_ENV=daytona)")
            issues.append("Set DAYTONA_API_KEY environment variable")
        try:
            from daytona import Daytona  # noqa: F401 — SDK presence check
            check_ok("daytona SDK", "(installed)")
        except ImportError:
            check_fail("daytona SDK not installed", "(pip install daytona)")
            issues.append("Install daytona SDK: pip install daytona")

    # Vercel Sandbox (if using vercel_sandbox backend)
    if terminal_env == "vercel_sandbox":
        runtime = os.getenv("TERMINAL_VERCEL_RUNTIME", "node24").strip() or "node24"
        from easybci_lib.tools.terminal_tool import _SUPPORTED_VERCEL_RUNTIMES
        if runtime in _SUPPORTED_VERCEL_RUNTIMES:
            check_ok("Vercel runtime", f"({runtime})")
        else:
            supported = ", ".join(_SUPPORTED_VERCEL_RUNTIMES)
            check_fail("Vercel runtime unsupported", f"({runtime}; use {supported})")
            issues.append(f"Set TERMINAL_VERCEL_RUNTIME to one of: {supported}")

        disk = os.getenv("TERMINAL_CONTAINER_DISK", "51200").strip()
        if disk in {"", "0", "51200"}:
            check_ok("Vercel disk setting", "(uses platform default)")
        else:
            check_fail("Vercel custom disk unsupported", "(reset terminal.container_disk to 51200)")
            issues.append("Vercel Sandbox does not support custom container_disk; use the shared default 51200")

        if importlib.util.find_spec("vercel") is not None:
            check_ok("vercel SDK", "(installed)")
        else:
            check_fail("vercel SDK not installed", "(pip install 'easybci-agent[vercel]')")
            issues.append("Install the Vercel optional dependency: pip install 'easybci-agent[vercel]'")

        auth_status_ok = bool(
            os.getenv("VERCEL_TOKEN") and os.getenv("VERCEL_PROJECT_ID") and os.getenv("VERCEL_TEAM_ID")
        )
        if auth_status_ok:
            check_ok("Vercel auth", "(configured)")
        else:
            check_fail("Vercel auth not configured", "(set VERCEL_TOKEN, VERCEL_PROJECT_ID, VERCEL_TEAM_ID)")
            issues.append(
                "Configure Vercel Sandbox auth with VERCEL_TOKEN, VERCEL_PROJECT_ID, and VERCEL_TEAM_ID"
            )

        persistent = os.getenv("TERMINAL_CONTAINER_PERSISTENT", "true").lower() in {"1", "true", "yes", "on"}
        if persistent:
            check_info("Vercel persistence: snapshot filesystem only; live processes do not survive sandbox recreation")
        else:
            check_info("Vercel persistence: ephemeral filesystem")


    if _is_termux():
        check_info("Termux compatibility fallbacks:")
        for note in _termux_install_all_fallback_notes():
            check_info(note)

    # =========================================================================
    # Check: API connectivity
    # =========================================================================
    print()
    print(color("◆ API Connectivity", Colors.CYAN, Colors.BOLD))

    # Refactor: every connectivity probe below is HTTP-bound and fully
    # independent. Running them in series spent ~5s wall on a typical
    # workstation (2s of that was boto3's IMDS lookup for AWS credentials,
    # which times out unless you're actually on EC2). Threading them with
    # a small executor pool collapses the section to roughly the slowest
    # single probe — about 2s — without changing the output format.
    #
    # Each ``_probe_*`` helper is a pure function: takes its inputs,
    # makes one HTTP/SDK call, returns a ``_ConnectivityResult`` carrying
    # the line(s) to print and any issue strings to append. No globals,
    # no shared mutable state, no printing inside the workers.
    import concurrent.futures as _futures
    from collections import namedtuple as _namedtuple

    _ConnectivityResult = _namedtuple(
        "_ConnectivityResult", ["label", "lines", "issues"]
    )
    _probes: list = []  # list of (label, callable) submitted in display order

    def _probe_openrouter() -> _ConnectivityResult:
        key = os.getenv("OPENROUTER_API_KEY")
        if not key:
            return _ConnectivityResult(
                "OpenRouter API",
                [(color("⚠", Colors.YELLOW), "OpenRouter API",
                  color("(not configured)", Colors.DIM))],
                [],
            )
        try:
            import httpx
            r = httpx.get(
                OPENROUTER_MODELS_URL,
                headers={"Authorization": f"Bearer {key}"},
                timeout=10,
            )
            if r.status_code == 200:
                return _ConnectivityResult(
                    "OpenRouter API",
                    [(color("✓", Colors.GREEN), "OpenRouter API", "")],
                    [],
                )
            if r.status_code == 401:
                return _ConnectivityResult(
                    "OpenRouter API",
                    [(color("✗", Colors.RED), "OpenRouter API",
                      color("(invalid API key)", Colors.DIM))],
                    ["Check OPENROUTER_API_KEY in .env"],
                )
            if r.status_code == 402:
                return _ConnectivityResult(
                    "OpenRouter API",
                    [(color("✗", Colors.RED), "OpenRouter API",
                      color("(out of credits — payment required)", Colors.DIM))],
                    ["OpenRouter account has insufficient credits. "
                     "Fix: run 'easybci config set model.provider <provider>' "
                     "to switch providers, or fund your OpenRouter account "
                     "at https://openrouter.ai/settings/credits"],
                )
            if r.status_code == 429:
                return _ConnectivityResult(
                    "OpenRouter API",
                    [(color("✗", Colors.RED), "OpenRouter API",
                      color("(rate limited)", Colors.DIM))],
                    ["OpenRouter rate limit hit — consider switching to "
                     "a different provider or waiting"],
                )
            return _ConnectivityResult(
                "OpenRouter API",
                [(color("✗", Colors.RED), "OpenRouter API",
                  color(f"(HTTP {r.status_code})", Colors.DIM))],
                [],
            )
        except Exception as e:
            return _ConnectivityResult(
                "OpenRouter API",
                [(color("✗", Colors.RED), "OpenRouter API",
                  color(f"({e})", Colors.DIM))],
                ["Check network connectivity"],
            )

    def _probe_anthropic() -> _ConnectivityResult:
        from easybci_cli.auth import get_anthropic_key
        key = get_anthropic_key()
        if not key:
            return _ConnectivityResult("Anthropic API", [], [])
        try:
            import httpx
            from easybci_agent.anthropic_adapter import (
                _is_oauth_token,
                _COMMON_BETAS,
                _OAUTH_ONLY_BETAS,
                _CONTEXT_1M_BETA,
            )
            headers = {"anthropic-version": "2023-06-01"}
            is_oauth = _is_oauth_token(key)
            if is_oauth:
                headers["Authorization"] = f"Bearer {key}"
                headers["anthropic-beta"] = ",".join(_COMMON_BETAS + _OAUTH_ONLY_BETAS)
            else:
                headers["x-api-key"] = key
            r = httpx.get(
                "https://api.anthropic.com/v1/models",
                headers=headers, timeout=10,
            )
            # Reactive recovery: OAuth subscriptions without 1M context reject the
            # request with 400 "long context beta is not yet available for this
            # subscription". Retry once with that beta stripped so the doctor
            # check doesn't falsely report Anthropic as unreachable.
            if (
                is_oauth
                and r.status_code == 400
                and "long context beta" in r.text.lower()
                and "not yet available" in r.text.lower()
            ):
                headers["anthropic-beta"] = ",".join(
                    [b for b in _COMMON_BETAS if b != _CONTEXT_1M_BETA]
                    + list(_OAUTH_ONLY_BETAS)
                )
                r = httpx.get(
                    "https://api.anthropic.com/v1/models",
                    headers=headers, timeout=10,
                )
            if r.status_code == 200:
                return _ConnectivityResult(
                    "Anthropic API",
                    [(color("✓", Colors.GREEN), "Anthropic API", "")],
                    [],
                )
            if r.status_code == 401:
                return _ConnectivityResult(
                    "Anthropic API",
                    [(color("✗", Colors.RED), "Anthropic API",
                      color("(invalid API key)", Colors.DIM))],
                    [],
                )
            return _ConnectivityResult(
                "Anthropic API",
                [(color("⚠", Colors.YELLOW), "Anthropic API",
                  color("(couldn't verify)", Colors.DIM))],
                [],
            )
        except Exception as e:
            return _ConnectivityResult(
                "Anthropic API",
                [(color("⚠", Colors.YELLOW), "Anthropic API",
                  color(f"({e})", Colors.DIM))],
                [],
            )

    def _probe_apikey_provider(pname, env_vars, default_url, base_env,
                               supports_health_check) -> _ConnectivityResult:
        key = ""
        for ev in env_vars:
            key = os.getenv(ev, "")
            if key:
                break
        if not key:
            return _ConnectivityResult(pname, [], [])
        label = pname.ljust(20)
        if not supports_health_check:
            return _ConnectivityResult(
                pname,
                [(color("✓", Colors.GREEN), label,
                  color("(key configured)", Colors.DIM))],
                [],
            )
        try:
            import httpx
            base = os.getenv(base_env, "") if base_env else ""
            # Auto-detect Kimi Code keys (sk-kimi-) → api.kimi.com/coding/v1
            # (OpenAI-compat surface, which exposes /models for health check).
            if not base and key.startswith("sk-kimi-"):
                base = "https://api.kimi.com/coding/v1"
            # Anthropic-compat endpoints (/anthropic, api.kimi.com/coding
            # with no /v1) don't support /models. Rewrite to OpenAI-compat
            # /v1 surface for health checks.
            if base and base.rstrip("/").endswith("/anthropic"):
                from easybci_agent.auxiliary_client import _to_openai_base_url
                base = _to_openai_base_url(base)
            if base_url_host_matches(base, "api.kimi.com") and base.rstrip("/").endswith("/coding"):
                base = base.rstrip("/") + "/v1"
            url = (base.rstrip("/") + "/models") if base else default_url
            headers = {
                "Authorization": f"Bearer {key}",
                "User-Agent": _EASYBCI_USER_AGENT,
            }
            if base_url_host_matches(base, "api.kimi.com"):
                headers["User-Agent"] = "claude-code/0.1.0"
            r = httpx.get(url, headers=headers, timeout=10)
            if (
                pname == "Alibaba/DashScope"
                and not base
                and r.status_code == 401
            ):
                r = httpx.get(
                    "https://dashscope.aliyuncs.com/compatible-mode/v1/models",
                    headers=headers, timeout=10,
                )
            if r.status_code == 200:
                return _ConnectivityResult(
                    pname,
                    [(color("✓", Colors.GREEN), label, "")],
                    [],
                )
            if r.status_code == 401:
                return _ConnectivityResult(
                    pname,
                    [(color("✗", Colors.RED), label,
                      color("(invalid API key)", Colors.DIM))],
                    [f"Check {env_vars[0]} in .env"],
                )
            return _ConnectivityResult(
                pname,
                [(color("⚠", Colors.YELLOW), label,
                  color(f"(HTTP {r.status_code})", Colors.DIM))],
                [],
            )
        except Exception as e:
            return _ConnectivityResult(
                pname,
                [(color("⚠", Colors.YELLOW), label,
                  color(f"({e})", Colors.DIM))],
                [],
            )

    def _probe_bedrock() -> _ConnectivityResult:
        try:
            from easybci_agent.bedrock_adapter import (
                has_aws_credentials,
                resolve_aws_auth_env_var,
                resolve_bedrock_region,
            )
        except ImportError:
            return _ConnectivityResult("AWS Bedrock", [], [])
        if not has_aws_credentials():
            return _ConnectivityResult("AWS Bedrock", [], [])
        auth_var = resolve_aws_auth_env_var()
        region = resolve_bedrock_region()
        label = "AWS Bedrock".ljust(20)
        try:
            import boto3
            from botocore.config import Config as _BotoConfig
            # Trim retries on the actual Bedrock API call so a transient
            # failure doesn't pad the doctor run by 30+ seconds.
            cfg = _BotoConfig(
                connect_timeout=5,
                read_timeout=10,
                retries={"max_attempts": 1},
            )
            client = boto3.client("bedrock", region_name=region, config=cfg)
            resp = client.list_foundation_models()
            n = len(resp.get("modelSummaries", []))
            return _ConnectivityResult(
                "AWS Bedrock",
                [(color("✓", Colors.GREEN), label,
                  color(f"({auth_var}, {region}, {n} models)", Colors.DIM))],
                [],
            )
        except ImportError:
            return _ConnectivityResult(
                "AWS Bedrock",
                [(color("⚠", Colors.YELLOW), label,
                  color(f"(boto3 not installed — {sys.executable} -m pip install boto3)",
                        Colors.DIM))],
                [f"Install boto3 for Bedrock: {sys.executable} -m pip install boto3"],
            )
        except Exception as e:
            err_name = type(e).__name__
            return _ConnectivityResult(
                "AWS Bedrock",
                [(color("⚠", Colors.YELLOW), label,
                  color(f"({err_name}: {e})", Colors.DIM))],
                [f"AWS Bedrock: {err_name} — check IAM permissions for "
                 f"bedrock:ListFoundationModels"],
            )

    # Build the probe submission list in display order
    _probes.append(("OpenRouter API", _probe_openrouter))
    _probes.append(("Anthropic API", _probe_anthropic))

    global _APIKEY_PROVIDERS_CACHE
    if _APIKEY_PROVIDERS_CACHE is None:
        _APIKEY_PROVIDERS_CACHE = _build_apikey_providers_list()
    for _entry in _APIKEY_PROVIDERS_CACHE:
        _pname, _env_vars, _default_url, _base_env, _supports = _entry
        # Capture loop vars by binding default args — without this, all closures
        # would share the final iteration's values and every probe would hit
        # the last provider's URL.
        _probes.append((_pname, lambda p=_pname, e=_env_vars, u=_default_url,
                                       b=_base_env, s=_supports:
                                _probe_apikey_provider(p, e, u, b, s)))

    _probes.append(("AWS Bedrock", _probe_bedrock))

    # Print a single status line so users see something happening, then
    # fan out. ``\r`` clears it once the first real result line lands.
    print(f"  {color(f'Running {len(_probes)} connectivity checks in parallel…', Colors.DIM)}",
          end="", flush=True)

    # Disable boto3's EC2 instance-metadata-service probe for the duration
    # of the parallel block. boto's default credential chain tries
    # 169.254.169.254 with a multi-second timeout when we're not on EC2,
    # which dominated the section's wall time before this fix
    # (~2s on a developer laptop, even with the rest parallelized).
    # Set on the parent thread before submitting work so the env-var
    # mutation never races with another worker. has_aws_credentials() in
    # the bedrock probe already gates on real env-var creds, so IMDS is
    # never the legitimate source for `easybci doctor`.
    _imds_prev = os.environ.get("AWS_EC2_METADATA_DISABLED")
    os.environ["AWS_EC2_METADATA_DISABLED"] = "true"
    try:
        # 8 workers is plenty — each probe is a single HTTP call plus a TLS
        # handshake. More than that wastes thread-startup cost and risks
        # noisy output if anything ever printed from inside a worker.
        with _futures.ThreadPoolExecutor(max_workers=8,
                                         thread_name_prefix="doctor-probe") as _ex:
            _futures_in_order = [_ex.submit(_fn) for _, _fn in _probes]
            _results = [_f.result() for _f in _futures_in_order]
    finally:
        if _imds_prev is None:
            os.environ.pop("AWS_EC2_METADATA_DISABLED", None)
        else:
            os.environ["AWS_EC2_METADATA_DISABLED"] = _imds_prev

    # Clear the "Running …" line and print all results in submission order.
    print("\r" + " " * 70 + "\r", end="")
    for _r in _results:
        for _glyph, _label, _detail in _r.lines:
            if _detail:
                print(f"  {_glyph} {_label} {_detail}")
            else:
                print(f"  {_glyph} {_label}")
        for _issue in _r.issues:
            issues.append(_issue)

    # =========================================================================
    # Check: Tool Availability
    # =========================================================================
    print()
    print(color("◆ Tool Availability", Colors.CYAN, Colors.BOLD))
    
    try:
        # Add project root to path for imports
        sys.path.insert(0, str(PROJECT_ROOT))
        from easybci_lib.model_tools import check_tool_availability, TOOLSET_REQUIREMENTS
        
        available, unavailable = check_tool_availability()
        available, unavailable = _apply_doctor_tool_availability_overrides(available, unavailable)
        
        for tid in available:
            info = TOOLSET_REQUIREMENTS.get(tid, {})
            check_ok(info.get("name", tid), _doctor_tool_availability_detail(tid))
        
        for item in unavailable:
            env_vars = item.get("missing_vars") or item.get("env_vars") or []
            if env_vars:
                vars_str = ", ".join(env_vars)
                check_warn(item["name"], f"(missing {vars_str})")
            else:
                check_warn(item["name"], "(system dependency not met)")

        # Count disabled tools with API key requirements
        api_disabled = [u for u in unavailable if (u.get("missing_vars") or u.get("env_vars"))]
        if api_disabled:
            issues.append("Run 'easybci setup' to configure missing API keys for full tool access")
    except Exception as e:
        check_warn("Could not check tool availability", f"({e})")
    
    # =========================================================================
    # Check: Skills Hub
    # =========================================================================
    print()
    print(color("◆ Skills Hub", Colors.CYAN, Colors.BOLD))

    hub_dir = EASYBCI_HOME / "skills" / ".hub"
    if hub_dir.exists():
        check_ok("Skills Hub directory exists")
        lock_file = hub_dir / "lock.json"
        if lock_file.exists():
            try:
                import json
                lock_data = json.loads(lock_file.read_text())
                count = len(lock_data.get("installed", {}))
                check_ok(f"Lock file OK ({count} hub-installed skill(s))")
            except Exception:
                check_warn("Lock file", "(corrupted or unreadable)")
        quarantine = hub_dir / "quarantine"
        q_count = sum(1 for d in quarantine.iterdir() if d.is_dir()) if quarantine.exists() else 0
        if q_count > 0:
            check_warn(f"{q_count} skill(s) in quarantine", "(pending review)")
    else:
        check_warn("Skills Hub directory not initialized", "(run: easybci skills list)")

    from easybci_cli.config import get_env_value

    def _gh_authenticated() -> bool:
        """Check if gh CLI is authenticated via token file or device flow."""
        try:
            result = subprocess.run(
                ["gh", "auth", "status", "--json", "authenticated"],
                capture_output=True, timeout=10,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    github_token = get_env_value("GITHUB_TOKEN") or get_env_value("GH_TOKEN")
    if github_token:
        check_ok("GitHub token configured (authenticated API access)")
    elif _gh_authenticated():
        check_ok("GitHub authenticated via gh CLI", "(full API access — no GITHUB_TOKEN needed)")
    else:
        check_warn("No GITHUB_TOKEN", f"(60 req/hr rate limit — set in {_DHH}/.env for better rates)")

    # =========================================================================
    # Memory Provider (only check the active provider, if any)
    # =========================================================================
    print()
    print(color("◆ Memory Provider", Colors.CYAN, Colors.BOLD))

    _active_memory_provider = ""
    try:
        import yaml as _yaml
        _mem_cfg_path = EASYBCI_HOME / "config.yaml"
        if _mem_cfg_path.exists():
            with open(_mem_cfg_path, encoding="utf-8") as _f:
                _raw_cfg = _yaml.safe_load(_f) or {}
            _active_memory_provider = (_raw_cfg.get("memory") or {}).get("provider", "")
    except Exception:
        pass

    if not _active_memory_provider:
        check_ok("Built-in memory active", "(no external provider configured — this is fine)")
    elif _active_memory_provider == "honcho":
        try:
            from services.plugins.memory.honcho.client import HonchoClientConfig, resolve_config_path
            hcfg = HonchoClientConfig.from_global_config()
            _honcho_cfg_path = resolve_config_path()

            if not _honcho_cfg_path.exists():
                check_warn("Honcho config not found", "run: easybci memory setup")
            elif not hcfg.enabled:
                check_info(f"Honcho disabled (set enabled: true in {_honcho_cfg_path} to activate)")
            elif not (hcfg.api_key or hcfg.base_url):
                check_fail("Honcho API key or base URL not set", "run: easybci memory setup")
                issues.append("No Honcho API key — run 'easybci memory setup'")
            else:
                from services.plugins.memory.honcho.client import get_honcho_client, reset_honcho_client
                reset_honcho_client()
                try:
                    get_honcho_client(hcfg)
                    check_ok(
                        "Honcho connected",
                        f"workspace={hcfg.workspace_id} mode={hcfg.recall_mode} freq={hcfg.write_frequency}",
                    )
                except Exception as _e:
                    check_fail("Honcho connection failed", str(_e))
                    issues.append(f"Honcho unreachable: {_e}")
        except ImportError:
            check_fail("honcho-ai not installed", "pip install honcho-ai")
            issues.append("Honcho is set as memory provider but honcho-ai is not installed")
        except Exception as _e:
            check_warn("Honcho check failed", str(_e))
    elif _active_memory_provider == "mem0":
        try:
            from services.plugins.memory.mem0 import _load_config as _load_mem0_config
            mem0_cfg = _load_mem0_config()
            mem0_key = mem0_cfg.get("api_key", "")
            if mem0_key:
                check_ok("Mem0 API key configured")
                check_info(f"user_id={mem0_cfg.get('user_id', '?')}  agent_id={mem0_cfg.get('agent_id', '?')}")
            else:
                check_fail("Mem0 API key not set", "(set MEM0_API_KEY in .env or run easybci memory setup)")
                issues.append("Mem0 is set as memory provider but API key is missing")
        except ImportError:
            check_fail("Mem0 plugin not loadable", "pip install mem0ai")
            issues.append("Mem0 is set as memory provider but mem0ai is not installed")
        except Exception as _e:
            check_warn("Mem0 check failed", str(_e))
    else:
        # Generic check for other memory providers (openviking, hindsight, etc.)
        try:
            from services.plugins.memory import load_memory_provider
            _provider = load_memory_provider(_active_memory_provider)
            if _provider and _provider.is_available():
                check_ok(f"{_active_memory_provider} provider active")
            elif _provider:
                check_warn(f"{_active_memory_provider} configured but not available", "run: easybci memory status")
            else:
                check_warn(f"{_active_memory_provider} plugin not found", "run: easybci memory setup")
        except Exception as _e:
            check_warn(f"{_active_memory_provider} check failed", str(_e))

    # =========================================================================
    # Profiles
    # =========================================================================
    try:
        from easybci_cli.profiles import list_profiles, _get_wrapper_dir, profile_exists
        import re as _re

        named_profiles = [p for p in list_profiles() if not p.is_default]
        if named_profiles:
            print()
            print(color("◆ Profiles", Colors.CYAN, Colors.BOLD))
            check_ok(f"{len(named_profiles)} profile(s) found")
            wrapper_dir = _get_wrapper_dir()
            for p in named_profiles:
                parts = []
                if p.gateway_running:
                    parts.append("gateway running")
                if p.model:
                    parts.append(p.model[:30])
                if not (p.path / "config.yaml").exists():
                    parts.append("⚠ missing config")
                if not (p.path / ".env").exists():
                    parts.append("no .env")
                wrapper = wrapper_dir / p.name
                if not wrapper.exists():
                    parts.append("no alias")
                status = ", ".join(parts) if parts else "configured"
                check_ok(f"  {p.name}: {status}")

            # Check for orphan wrappers
            if wrapper_dir.is_dir():
                for wrapper in wrapper_dir.iterdir():
                    if not wrapper.is_file():
                        continue
                    try:
                        content = wrapper.read_text()
                        if "easybci -p" in content:
                            _m = _re.search(r"easybci -p (\S+)", content)
                            if _m and not profile_exists(_m.group(1)):
                                check_warn(f"Orphan alias: {wrapper.name} → profile '{_m.group(1)}' no longer exists")
                    except Exception:
                        pass
    except ImportError:
        pass
    except Exception:
        pass

    # =========================================================================
    # Parameter-uncertainty registry
    # =========================================================================
    print()
    print(color("◆ Parameter-Uncertainty Registry", Colors.CYAN, Colors.BOLD))
    try:
        from easybci_lib.tools.neural_processing.research.parameter_registry import load_registry
        reg = load_registry(force=True)
        print(color(
            f"  ✓ parameter-uncertainty registry: {len(reg)} operators loaded",
            Colors.GREEN,
        ))
    except Exception as exc:
        msg = f"parameter-uncertainty registry: {exc}"
        print(color(f"  ✗ {msg}", Colors.RED))
        manual_issues.append(msg)

    # =========================================================================
    # Check: Proven-Pipeline Library Hygiene
    # =========================================================================
    print()
    print(color("◆ Proven Pipelines", Colors.CYAN, Colors.BOLD))
    try:
        pp_result = doctor_proven_pipelines()
        scanned = pp_result.get("skills_scanned", 0)
        warnings_list = pp_result.get("warnings", [])
        if not warnings_list:
            check_ok(f"proven-pipelines: {scanned} skill(s) scanned, no issues")
        else:
            check_warn(
                f"proven-pipelines: {len(warnings_list)} skill(s) flagged "
                f"(of {scanned} scanned)"
            )
            for w in warnings_list:
                check_info(f"{w['name']} — {w['issue']} ({w['detail']})")
                check_info(f"  fix: {w['fix']}")
                manual_issues.append(
                    f"proven-pipelines/{w['name']} {w['issue']}: {w['fix']}"
                )
    except Exception as exc:
        msg = f"proven-pipelines scan: {exc}"
        print(color(f"  ✗ {msg}", Colors.RED))
        manual_issues.append(msg)

    # =========================================================================
    # Summary
    # =========================================================================
    print()
    remaining_issues = issues + manual_issues
    if should_fix and fixed_count > 0:
        print(color("─" * 60, Colors.GREEN))
        print(color(f"  Fixed {fixed_count} issue(s).", Colors.GREEN, Colors.BOLD), end="")
        if remaining_issues:
            print(color(f" {len(remaining_issues)} issue(s) require manual intervention.", Colors.YELLOW, Colors.BOLD))
        else:
            print()
        print()
        if remaining_issues:
            for i, issue in enumerate(remaining_issues, 1):
                print(f"  {i}. {issue}")
            print()
    elif remaining_issues:
        print(color("─" * 60, Colors.YELLOW))
        print(color(f"  Found {len(remaining_issues)} issue(s) to address:", Colors.YELLOW, Colors.BOLD))
        print()
        for i, issue in enumerate(remaining_issues, 1):
            print(f"  {i}. {issue}")
        print()
        if not should_fix:
            print(color("  Tip: run 'easybci doctor --fix' to auto-fix what's possible.", Colors.DIM))
    else:
        print(color("─" * 60, Colors.GREEN))
        print(color("  All checks passed! 🎉", Colors.GREEN, Colors.BOLD))

    # === Layout drift probe (Phase 5) ==========================================
    # Peek at recent work_dirs from SessionDB and flag any that fail the layout
    # contract. Best-effort; failures are silent so an unrelated storage issue
    # doesn't break doctor.
    try:
        from easybci_lib.state import SessionDB
        from easybci_lib.tools.neural_processing.export.layout_repair import (
            detect_violations,
        )
        from easybci_agent.i18n import t as _t
        _recent = SessionDB().recent_work_dirs(limit=5)
        _drift = []
        for _wd in _recent:
            _p = Path(_wd)
            if not _p.is_dir():
                continue
            _vs = detect_violations(_p)
            if _vs:
                _drift.append({
                    "work_dir": _wd,
                    "violation_count": len(_vs),
                    "kinds": sorted({v.kind for v in _vs}),
                })
        if _drift:
            print()
            print(color(_t("doctor.layout.drift_hint", count=len(_drift)), Colors.YELLOW))
    except Exception:
        # Best-effort probe; never break doctor on a storage / import issue.
        pass

    print()


# ─── doctor websearch ─────────────────────────────────────────────────────────
#
# Surfaces the same picture as GET /api/web-search/status on the dashboard
# so the WebUI and CLI agree on what each provider's blocker is. Reads the
# registry via the same diagnose_active_provider() helper that
# research_preprocessing's strict gate uses (see Phase 3) — so "running
# `easybci doctor websearch`" tells the user precisely whether the next
# plan_pipeline call will be allowed to consult the web.


def _websearch_status_payload() -> dict:
    """Return the same dict shape /api/web-search/status emits.

    Uses cli_output / colors only at print time; this function is i/o-free
    so --json can dump it verbatim and tests can assert on its keys.
    """
    from easybci_cli.plugins import discover_plugins  # noqa: WPS433
    discover_plugins()

    from easybci_agent.web_search_registry import (
        list_providers,
        get_active_search_provider,
        get_active_search_provider_strict,
        get_active_extract_provider,
        get_active_crawl_provider,
        _read_config_key,
    )

    explicit_search = (
        _read_config_key("web", "search_backend")
        or _read_config_key("web", "backend")
    )
    explicit_extract = (
        _read_config_key("web", "extract_backend")
        or _read_config_key("web", "backend")
    )

    providers = []
    for p in list_providers():
        try:
            available = bool(p.is_available())
        except Exception:
            available = False
        diag = None
        if not available:
            try:
                reason, hint = p.availability_diagnostic()
                diag = {"reason": reason, "fix_hint": hint}
            except Exception:
                diag = {"reason": "not configured", "fix_hint": ""}
        providers.append({
            "name": p.name,
            "display_name": p.display_name,
            "registered": True,
            "supports_search": bool(p.supports_search()),
            "supports_extract": bool(p.supports_extract()),
            "supports_crawl": bool(p.supports_crawl()),
            "is_available": available,
            "configured_as_search": explicit_search == p.name,
            "configured_as_extract": explicit_extract == p.name,
            "diagnostic": diag,
        })

    def _name_or_none(prov) -> str | None:
        return prov.name if prov is not None else None

    return {
        "providers": providers,
        "active_search":        _name_or_none(get_active_search_provider()),
        "active_search_strict": _name_or_none(get_active_search_provider_strict()),
        "active_extract":       _name_or_none(get_active_extract_provider()),
        "active_crawl":         _name_or_none(get_active_crawl_provider()),
    }


def _do_websearch_install(backend: str) -> dict:
    """Run lazy_deps.ensure for exa.

    Returns a dict in the same shape as POST /api/web-search/ensure so
    the CLI and WebUI can render parallel messages.
    """
    if backend == "tavily":
        return {
            "ok": True,
            "backend": "tavily",
            "installed": False,
            "available_after": bool(os.getenv("TAVILY_API_KEY", "").strip()),
            "message": "Tavily uses HTTP only — set TAVILY_API_KEY to activate.",
        }
    if backend not in ("exa",):
        return {
            "ok": False,
            "backend": backend,
            "installed": False,
            "available_after": False,
            "message": f"unknown backend: {backend!r} (expected exa / tavily)",
        }

    from easybci_lib.tools.lazy_deps import (
        ensure as lazy_ensure,
        FeatureUnavailable,
        feature_missing,
    )

    feature = f"search.{backend}"
    try:
        before_missing = bool(feature_missing(feature))
    except Exception:
        before_missing = True

    try:
        lazy_ensure(feature, prompt=False)
    except FeatureUnavailable as exc:
        return {
            "ok": False,
            "backend": backend,
            "installed": False,
            "available_after": False,
            "message": str(exc),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "backend": backend,
            "installed": False,
            "available_after": False,
            "message": f"unexpected error during install: {exc}",
        }

    try:
        from easybci_cli.plugins import discover_plugins  # noqa: WPS433
        discover_plugins()
    except Exception:
        pass

    try:
        from easybci_agent.web_search_registry import get_provider  # noqa: WPS433
        p = get_provider(backend)
        available_after = bool(p and p.is_available())
    except Exception:
        available_after = False

    installed = bool(before_missing) and available_after
    return {
        "ok": True,
        "backend": backend,
        "installed": installed,
        "available_after": available_after,
        "message": (
            f"{backend} installed"
            if installed
            else f"{backend} already installed" if available_after
            else f"{backend} install reported success but package not importable"
        ),
    }


def run_doctor_websearch(args) -> None:
    """Diagnose web-search activation across tavily / exa.

    Mirrors GET /api/web-search/status; --install <backend> calls the same
    lazy-install path POST /api/web-search/ensure uses. --json emits the raw
    dict so scripts can read it; without --json we render a colored table.
    """
    import json as _json
    from easybci_cli.cli_output import (
        print_info, print_success, print_warning, print_error,
    )

    install_target = getattr(args, "install", None)
    want_json = bool(getattr(args, "json", False))

    if install_target:
        result = _do_websearch_install(install_target)
        if want_json:
            print(_json.dumps(result, indent=2))
            sys.exit(0 if result.get("ok") else 1)
        if result.get("ok") and result.get("available_after"):
            if result.get("installed"):
                print_success(f"{install_target} installed")
            else:
                print_info(f"{install_target} already installed")
        elif result.get("ok"):
            print_warning(result.get("message", ""))
        else:
            print_error(result.get("message", ""))
        # Fall through to print the updated status table.

    payload = _websearch_status_payload()

    if want_json:
        print(_json.dumps(payload, indent=2))
        return

    print()
    print(color("EasyBCI Web Search Diagnosis", Colors.CYAN, Colors.BOLD))
    print(color("─" * 60, Colors.CYAN))
    explicit = payload["active_search"] or "—"
    strict = payload["active_search_strict"] or "—"
    extract = payload["active_extract"] or "—"
    crawl = payload["active_crawl"] or "—"
    print_info(f"Active backend (explicit-config wins) : {explicit}")
    print_info(f"Active backend (strict-available)     : {strict}")
    print_info(f"Active extract backend                : {extract}")
    print_info(f"Active crawl backend                  : {crawl}")
    print()
    print(color("Providers", Colors.CYAN, Colors.BOLD))
    print(color("─" * 60, Colors.CYAN))

    for p in payload["providers"]:
        name = p["name"]
        if p["is_available"]:
            print_success(f"{name:<10} {p['display_name']:<28} available")
        else:
            print_error(f"{name:<10} {p['display_name']:<28} not available")
            if p.get("diagnostic"):
                print_info(f"  Reason : {p['diagnostic']['reason']}")
                if p['diagnostic'].get('fix_hint'):
                    print_info(f"  Fix    : {p['diagnostic']['fix_hint']}")

    print()
    if payload["active_search_strict"]:
        print_success(
            "EasyBCI G-2 mandate: research_preprocessing will run on next "
            "plan_pipeline call"
        )
    else:
        print_warning(
            "EasyBCI G-2 mandate: research_preprocessing BLOCKED — set up "
            "at least one provider above"
        )
    print()


def run_doctor_goals(args) -> int:
    """Report analysis_goal name conflicts between builtin REGISTRY and
    third-party ~/.easybci/skills/analysis_goals/*.yaml."""
    # Reload so freshly-edited yaml files in EASYBCI_HOME are picked up.
    importlib.reload(_analysis_goals_module)
    conflicts = getattr(_analysis_goals_module, "_THIRD_PARTY_CONFLICTS", [])

    if getattr(args, "json", False):
        print(json.dumps([
            {"name": c.name, "yaml_path": c.yaml_path, "reason": c.reason}
            for c in conflicts
        ], ensure_ascii=False, indent=2))
        return 1 if conflicts else 0

    if not conflicts:
        print(f"[doctor:goals] no conflicts. Builtin REGISTRY has {len(_analysis_goals_module.REGISTRY)} goals.")
        return 0
    print("[doctor:goals] FOUND conflicts (builtin REGISTRY takes precedence):")
    for c in conflicts:
        print(f"  - {c.name!r} from {c.yaml_path}: {c.reason}")
    return 1


def doctor_proven_pipelines() -> dict:
    """Scan ~/.easybci/skills/proven-pipelines/ for frontmatter problems.

    Reports:
      - analysis_goal ∈ {generic, exploratory} (should not have been crystallized)
      - data_profile.channels / sfreq_hz missing (incomplete provenance)

    Does NOT delete anything — these may be user-edited. Returns a dict with
    ``warnings: list[dict]`` consumable by run_doctor's aggregator.
    """
    from easybci_lib.constants import get_easybci_home

    home = get_easybci_home()
    root = home / "skills" / "proven-pipelines"
    warnings: list[dict] = []
    if not root.is_dir():
        return {"warnings": warnings, "skills_scanned": 0}

    INELIGIBLE_GOALS = {"generic", "exploratory"}
    scanned = 0
    for skill_dir in sorted(root.iterdir()):
        if not skill_dir.is_dir():
            continue
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.exists():
            continue
        scanned += 1
        try:
            text = skill_md.read_text(encoding="utf-8")
        except Exception:
            continue

        fm_match = re.match(r"---\s*\n(.*?)\n---", text, re.DOTALL)
        if not fm_match:
            continue
        fm = fm_match.group(1)

        goal_m = re.search(r"analysis_goal:\s*([\w\-]+)", fm)
        goal = goal_m.group(1) if goal_m else None
        if goal in INELIGIBLE_GOALS:
            warnings.append({
                "name": skill_dir.name, "issue": "ineligible_goal",
                "detail": f"analysis_goal={goal}",
                "fix": f"rm -rf '{skill_dir}'",
            })
            continue

        has_channels = bool(re.search(r"channels:\s*\d+", fm))
        has_sfreq = bool(re.search(r"sfreq_hz:\s*[\d.]+", fm))
        if not (has_channels and has_sfreq):
            warnings.append({
                "name": skill_dir.name, "issue": "incomplete_data_profile",
                "detail": "channels/sfreq_hz missing or non-numeric in frontmatter",
                "fix": f"manually review or rm -rf '{skill_dir}'",
            })

    return {"warnings": warnings, "skills_scanned": scanned}


def doctor_layout(work_dir: str, *, dry_run: bool, json_mode: bool) -> int:
    """Run verify_and_repair on <work_dir> and print a report.

    Returns 0 when remaining_violations == 0, 1 otherwise. Dry-run always
    returns 0 (reporting-only) unless the work_dir doesn't exist (2).
    """
    import json as _json
    from pathlib import Path

    from easybci_lib.tools.neural_processing.export.layout_repair import (
        verify_and_repair,
    )
    from easybci_agent.i18n import t
    from easybci_cli.cli_output import print_error, print_info

    wd = Path(work_dir).expanduser()
    try:
        wd = wd.resolve()
    except OSError as exc:
        msg = t("doctor.layout.resolve_failed", path=work_dir, err=str(exc))
        if json_mode:
            sys.stdout.write(_json.dumps({"success": False, "error": msg}))
            sys.stdout.write("\n")
        else:
            print_error(msg)
        return 2
    if not wd.is_dir():
        msg = t("doctor.layout.not_a_dir", path=str(wd))
        if json_mode:
            sys.stdout.write(_json.dumps({"success": False, "error": msg}))
            sys.stdout.write("\n")
        else:
            print_error(msg)
        return 2

    report = verify_and_repair(
        wd,
        dry_run=dry_run,
        allow_subprocess=(not dry_run),
        write_report=(not dry_run),
    )

    if json_mode:
        sys.stdout.write(_json.dumps(report.to_dict(), indent=2))
        sys.stdout.write("\n")
    else:
        if dry_run:
            print_info(t("doctor.layout.dry_run_header", path=str(wd)))
        else:
            print_info(t("doctor.layout.apply_header", path=str(wd)))
        print_info(t(
            "doctor.layout.summary",
            initial=report.initial_violations,
            remaining=report.remaining_violations,
            rounds=report.rounds,
            elapsed=round(report.wall_clock_s, 2),
        ))
        for fx in report.fixes:
            for note in fx.notes:
                if fx.applied and not fx.residual:
                    marker = "OK "
                elif fx.residual:
                    marker = "!! "
                else:
                    marker = ".  "
                print_info(f"  {marker}[{fx.kind}] {note}")
        if report.unrepairable:
            print_error(t("doctor.layout.unrepairable_header"))
            for u in report.unrepairable:
                print_error(f"  {u}")

    # Dry-run: always exit 0 (reporting-only). Apply-mode: non-zero if
    # residual violations remain.
    if dry_run:
        return 0
    return 0 if report.remaining_violations == 0 else 1

