"""Idempotent sync of main provider creds into auxiliary.* / delegation.*
slots of ``~/.easybci/config.yaml`` plus a ``/v1/models`` probe to confirm
the configured default model is actually served by the provider.

Invoked from ``setup-easybci.sh`` on install (and safe to re-run). Only
fills BLANK slots — never overwrites a user-set base_url / api_key. Round-
trips via ruamel.yaml so comments and key order survive.

The probe catches the failure mode where ``model.default`` (e.g.
``deepseek-v4-flash``) is not actually provisioned on the configured
provider's account, which surfaces at runtime as a misleading
``HTTP 402 Insufficient Balance``.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _easybci_home() -> Path:
    home = os.environ.get("EASYBCI_HOME") or str(Path.home() / ".easybci")
    return Path(home)


def _build_yaml():
    from ruamel.yaml import YAML
    y = YAML(typ="rt")
    y.preserve_quotes = True
    y.allow_unicode = True
    y.default_flow_style = False
    y.indent(mapping=2, sequence=4, offset=2)
    return y


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


# Auxiliary tasks that must NOT inherit the main chat model's credentials
# blindly. The reason string is shown in the setup output so the user
# understands what was skipped and why.
#
# vision      — needs a multimodal model. Most main-chat models (DeepSeek
#               text family, plain Llama, etc.) reject image input; routing
#               vision through them produces 4xx or garbled responses.
# skills_hub  — uses function-calling tool schemas. Some models advertise
#               tool support but mishandle JSON schema edge cases; safer to
#               leave blank and let the user explicitly opt in to a known
#               tool-capable model (or rely on the auto-resolver chain).
_NEEDS_DIFFERENT_CREDS: Dict[str, str] = {
    "vision": "needs a multimodal model — main chat model usually can't process images",
    "skills_hub": "uses function-calling tool schemas; a tool-capable model may differ from main",
}

# Known vision-capable model name patterns, ordered from MOST PREFERRED
# (top of list) to least preferred. Used to scan a provider's
# /v1/models response and suggest a candidate for auxiliary.vision.model
# when the main provider has multimodal SKUs available.
#
# Within a pattern, the suggester prefers the SHORTEST model id (often
# the smallest variant — e.g. ``qwen3-vl-8b-instruct`` over
# ``qwen3-vl-235b-a22b-thinking``), which also aligns with the "weaker
# is better for testing" preference.
#
# Patterns are deliberately specific (e.g. ``qwen-vl-max`` instead of
# bare ``vl`` or ``minimax-vl`` instead of ``minimax-m``) so plain
# text-only chat models don't get suggested as vision backends.
_VISION_MODEL_PATTERNS: Tuple[str, ...] = (
    # Tier 1 — preferred: open-source / mid-tier (latest first within family)
    "qwen3-vl", "qwen2.5-vl", "qwen2-vl", "qwen-vl-max",
    "glm-5v", "glm-4.5v", "glm-4v",
    "mimo-v2.5-pro", "mimo-v2-pro", 
    "internvl3", 
)


# Substrings that indicate an *output-side* image model (text-to-image
# generation, embedding, OCR-only, audio variants) rather than a vision
# model that *accepts* image input. Suggesting any of these for
# auxiliary.vision would break image analysis at runtime.
_VISION_DENY_SUBSTRINGS: Tuple[str, ...] = (
    "image-preview", "image-generate", "image-edit",
    "imagen", "dall-e", "stable-diffusion",
    "embedding", "embed-",
    "-tts", "-stt", "-audio",
    "ocr",  # OCR-specialised models often refuse general vision tasks
    "-instruct-vl-ocr",
)


def _suggest_vision_model(models: List[str]) -> Optional[Tuple[str, List[str]]]:
    """Pick the best vision-capable model from a /v1/models list.

    Returns ``(top_pick, alternatives)`` or None if the provider has no
    obvious multimodal SKU. ``top_pick`` is matched against the earliest
    (most capable) pattern; within a pattern, the shortest model id is
    preferred (typically an alias like ``glm-4.5v`` over a longer
    fork like ``z-ai/glm-4.5v``). ``alternatives`` lists up to five other
    matches diversified ACROSS patterns (one per family) so the user sees
    real choices instead of five variants of the same Claude/GPT.

    Output-side image / embedding / OCR / audio variants are excluded via
    ``_VISION_DENY_SUBSTRINGS`` — substring matching on patterns like
    ``gemini-3-pro`` would otherwise pick up ``gemini-3-pro-image-preview``
    (text-to-image generation, not image-in).
    """
    matches_by_pattern: List[Tuple[int, str]] = []
    for idx, pat in enumerate(_VISION_MODEL_PATTERNS):
        pat_lower = pat.lower()
        for m in models:
            m_lower = m.lower()
            if pat_lower not in m_lower:
                continue
            if any(deny in m_lower for deny in _VISION_DENY_SUBSTRINGS):
                continue
            matches_by_pattern.append((idx, m))
    if not matches_by_pattern:
        return None
    # Earliest pattern wins; within a pattern, shortest id wins.
    matches_by_pattern.sort(key=lambda t: (t[0], len(t[1]), t[1]))
    top_pick = matches_by_pattern[0][1]

    # Diversified alternatives: best (shortest) match from each remaining
    # pattern, in priority order, capped at five.
    seen_patterns = {matches_by_pattern[0][0]}
    seen_ids = {top_pick}
    alternatives: List[str] = []
    for idx, m in matches_by_pattern:
        if idx in seen_patterns or m in seen_ids:
            continue
        seen_patterns.add(idx)
        seen_ids.add(m)
        alternatives.append(m)
        if len(alternatives) >= 5:
            break
    return top_pick, alternatives


def _resolve_template(cfg, main: Dict[str, str]) -> Tuple[Dict[str, str], bool]:
    """Resolve the credential set used as a template for non-special aux tasks.

    Per project policy (user pref): ``auxiliary.compression`` is the
    canonical aux-task config. Other aux tasks (web_extract, session_search,
    approval, mcp, title_generation, curator, triage_specifier) inherit
    from compression so the user can run a cheap text model for all of
    them while keeping ``model.default`` on a heavier reasoning model.

    If compression itself is blank, fall back to ``main`` so first-install
    still works without forcing the user to configure compression first.
    Compression's own blanks are filled from main as a side-effect (so
    the next call sees compression populated and behaves consistently).

    Returns ``(template_creds, used_compression_explicitly)``.
    ``used_compression_explicitly = True`` means compression had at least
    one non-blank credential field — useful for the dual-revert detection
    in ``_sync_aux_and_delegation``.
    """
    aux = cfg.get("auxiliary")
    comp = aux.get("compression") if isinstance(aux, dict) else None
    if not isinstance(comp, dict):
        return main, False

    has_explicit = (
        not _is_blank(comp.get("base_url"))
        or not _is_blank(comp.get("api_key"))
    )

    # Side-effect: backfill compression's own blanks from main so it has
    # a coherent set of creds to template from. This also matters for the
    # case where the user only set ``auxiliary.compression.model`` to
    # something cheap but left base_url/api_key blank — we want the
    # template to use main's URL/key with compression's model.
    #
    # ``provider: auto`` is treated as blank for backfill purposes (it's
    # the auto-resolver placeholder, not a real provider name) — without
    # this, downstream revert detection would mistake ``auto`` for a
    # "snapshotted from compression" signature and trigger spurious
    # revert counts on every re-run.
    if _is_blank(comp.get("base_url")):
        comp["base_url"] = main["base_url"]
    if _is_blank(comp.get("api_key")):
        comp["api_key"] = main["api_key"]
    if _is_blank(comp.get("provider")) or comp.get("provider") == "auto":
        comp["provider"] = main["provider"] or "custom"
    # Compression's model is intentionally NOT auto-filled from main —
    # leaving it blank is a valid signal "use main's model for compression
    # too" (the auxiliary client falls through to main_model when blank).

    template_provider = comp.get("provider") or main["provider"] or "custom"
    if template_provider == "auto":
        # Last-line defence — never let "auto" leak into the template.
        template_provider = main["provider"] or "custom"

    template = {
        "base_url": comp.get("base_url") or main["base_url"],
        "api_key": comp.get("api_key") or main["api_key"],
        "provider": template_provider,
        # Template's model is compression's explicit model if set,
        # otherwise blank so each downstream task keeps its own model
        # value (or stays blank to inherit at runtime).
        "model": comp.get("model") or "",
    }
    return template, has_explicit


# Aux tasks that share the compression template (vs main). Compression
# itself is the source — it does NOT appear in this set.
_COMPRESSION_TEMPLATED_TASKS: frozenset = frozenset({
    "web_extract", "session_search", "approval", "mcp",
    "title_generation", "curator", "triage_specifier",
})


# Web backend auto-fill: maps env-var presence → backend name. Order of
# entries reflects priority: search prefers tavily > exa (tavily has best
# curation), extract prefers tavily > exa. Both backends require an API key;
# no keyless fallback.
_WEB_PROVIDERS_BY_CAPABILITY: Dict[str, Tuple[Tuple[str, str], ...]] = {
    "search":  (("TAVILY_API_KEY", "tavily"), ("EXA_API_KEY", "exa")),
    "extract": (("TAVILY_API_KEY", "tavily"), ("EXA_API_KEY", "exa")),
}


def _read_dotenv_keys(env_path: Path) -> Dict[str, str]:
    """Return KEY=VALUE pairs from a .env file (ignoring comments / blanks)."""
    if not env_path.is_file():
        return {}
    out: Dict[str, str] = {}
    try:
        text = env_path.read_text(encoding="utf-8")
    except OSError:
        return {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k:
            out[k] = v
    return out


def _sync_web_backends(cfg, env_path: Path) -> Tuple[int, List[str]]:
    """Fill blank ``web.search_backend`` / ``web.extract_backend`` based on
    which API keys are present in ``~/.easybci/.env``. Returns
    ``(filled, notes)`` where ``notes`` is a list of human-readable lines
    describing what was set or skipped.
    """
    web = cfg.get("web")
    if not isinstance(web, dict):
        return 0, []

    env_keys = _read_dotenv_keys(env_path)
    notes: List[str] = []
    filled = 0

    for capability, providers in _WEB_PROVIDERS_BY_CAPABILITY.items():
        slot = f"{capability}_backend"
        if not _is_blank(web.get(slot)):
            # User-set or previously-set; preserve.
            continue
        chosen: Optional[str] = None
        for env_var, backend_name in providers:
            if env_var == "" or (env_var in env_keys and env_keys[env_var]):
                chosen = backend_name
                break
        if chosen:
            web[slot] = chosen
            filled += 1
            notes.append(f"  ✓ web.{slot} = {chosen}")
        else:
            notes.append(f"  ⚠ web.{slot} stays blank — no usable provider "
                         f"detected in {env_path}")

    # Also normalize web.backend (the shared fallback used when a
    # capability override is blank). If it's blank but search_backend got
    # set, mirror it so downstream code that only reads `web.backend`
    # still sees a real value.
    if _is_blank(web.get("backend")) and not _is_blank(web.get("search_backend")):
        web["backend"] = web["search_backend"]
        filled += 1
        notes.append(f"  ✓ web.backend = {web['backend']} (mirrored from search_backend)")

    return filled, notes


def _is_interactive() -> bool:
    """True when both stdin and stdout are TTYs and EASYBCI_NON_INTERACTIVE
    isn't set (latter lets CI / scripts force non-interactive even on a TTY).
    """
    if os.environ.get("EASYBCI_NON_INTERACTIVE", "").strip().lower() in {"1", "true", "yes"}:
        return False
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except (AttributeError, OSError):
        return False


def _interactive_vision_picker(
    top_pick: str,
    alternatives: List[str],
    main: Dict[str, str],
) -> Optional[Tuple[str, str]]:
    """Prompt the user to pick a vision model from ``[top_pick] + alternatives``.

    Returns ``(provider, model_id)`` tuple to write into
    ``auxiliary.vision``, or None if the user skipped. Picks use main's
    base_url / api_key (the user's provider hosts these multimodal SKUs
    on the same endpoint).

    Bare RET re-prompts (no default — user must explicitly pick or skip).
    Numbers map to the menu; ``s`` / ``skip`` / ``q`` skip the picker.
    """
    print()
    print("Vision model picker")
    print("───────────────────")
    print("Your main provider lists multimodal SKUs. Pick one to enable")
    print("image input, or skip to leave it blank (auto-resolver fallback).")
    print()

    candidates = [top_pick, *alternatives]
    width = len(str(len(candidates)))
    for i, mid in enumerate(candidates, 1):
        marker = " (recommended)" if i == 1 else ""
        print(f"  [{i:>{width}}] {mid}{marker}")
    print(f"  [{'s':>{width}}] skip — leave auxiliary.vision blank")
    print()

    while True:
        try:
            raw = input("Pick a number or 's' to skip: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()  # newline after ^C / ^D
            return None
        if not raw:
            continue
        if raw in {"s", "skip", "q", "quit", "exit"}:
            return None
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(candidates):
                provider = main["provider"] or "custom"
                return provider, candidates[idx - 1]
        print(f"  ↳ '{raw}' is not a valid choice. Try again or 's' to skip.")



def _sync_aux_and_delegation(cfg, main: Dict[str, str]) -> Tuple[int, int, List[str]]:
    """Fill blank base_url / api_key / provider in every ``auxiliary.<task>.*``
    and ``delegation.*`` slot. Templating rules:

      * ``auxiliary.compression`` itself is filled from main (so a user
        who only set ``compression.model`` gets a coherent compression
        config).
      * Tasks in ``_COMPRESSION_TEMPLATED_TASKS`` (web_extract /
        session_search / approval / mcp / title_generation / curator /
        triage_specifier) inherit from the resolved compression template.
      * Tasks in ``_NEEDS_DIFFERENT_CREDS`` (vision, skills_hub) revert
        any prior auto-snapshot — detected by exact match against EITHER
        compression's OR main's creds (since older syncs filled from main
        and newer ones could fill from compression).
      * ``delegation`` is filled from main (delegation spawns sub-agents
        which need the heavy reasoning model, not the cheap aux model).

    Returns ``(filled, reverted, skipped_notes)``.
    """
    filled = 0
    reverted = 0
    skipped_notes: List[str] = []

    # Resolve template BEFORE iterating so compression-itself's backfill
    # (a side effect of _resolve_template) doesn't get attributed as a
    # snapshotted aux task.
    pre_comp = (cfg.get("auxiliary") or {}).get("compression") or {}
    pre_blanks = sum(1 for k in ("base_url", "api_key", "provider")
                     if _is_blank(pre_comp.get(k)))
    template, _has_explicit_comp = _resolve_template(cfg, main)
    post_comp = (cfg.get("auxiliary") or {}).get("compression") or {}
    post_blanks = sum(1 for k in ("base_url", "api_key", "provider")
                      if _is_blank(post_comp.get(k)))
    filled += max(0, pre_blanks - post_blanks)

    # Both main's and template's creds are valid targets for the revert
    # detection in _NEEDS_DIFFERENT_CREDS — any of these "snapshot
    # signatures" means a previous run filled the slot for us.
    snapshot_sigs = [
        (main["base_url"], main["api_key"], main["provider"] or "custom"),
        (template["base_url"], template["api_key"], template["provider"]),
    ]

    aux = cfg.get("auxiliary")
    if isinstance(aux, dict):
        for task_name, task_cfg in aux.items():
            if not isinstance(task_cfg, dict):
                continue
            if task_name == "compression":
                # Already handled by _resolve_template's side-effect above.
                continue

            if task_name in _NEEDS_DIFFERENT_CREDS:
                reason = _NEEDS_DIFFERENT_CREDS[task_name]
                local_reverted = 0
                cur_url = task_cfg.get("base_url")
                cur_key = task_cfg.get("api_key")
                cur_prov = task_cfg.get("provider")
                # Match against any known snapshot signature (main or
                # compression-template) so older fills get cleared too.
                for sig_url, sig_key, sig_prov in snapshot_sigs:
                    if cur_url == sig_url and not _is_blank(sig_url):
                        task_cfg["base_url"] = ""
                        cur_url = ""
                        local_reverted += 1
                    if cur_key == sig_key and not _is_blank(sig_key):
                        task_cfg["api_key"] = ""
                        cur_key = ""
                        local_reverted += 1
                    if cur_prov == sig_prov and not _is_blank(sig_prov):
                        task_cfg["provider"] = "auto"
                        cur_prov = "auto"
                        local_reverted += 1
                reverted += local_reverted
                if local_reverted:
                    skipped_notes.append(
                        f"  ↺ {task_name}: cleared {local_reverted} auto-snapshotted "
                        f"slot(s) — {reason}. Set "
                        f"`auxiliary.{task_name}.{{provider,model,base_url,api_key}}` "
                        f"explicitly if you want a specific backend."
                    )
                else:
                    skipped_notes.append(
                        f"  • {task_name}: skipped — {reason}. Either leave "
                        f"`provider: auto` (auto-resolver picks a fallback) or "
                        f"set `auxiliary.{task_name}.*` explicitly."
                    )
                continue

            if task_name in _COMPRESSION_TEMPLATED_TASKS:
                if _is_blank(task_cfg.get("base_url")):
                    task_cfg["base_url"] = template["base_url"]; filled += 1
                if _is_blank(task_cfg.get("api_key")):
                    task_cfg["api_key"] = template["api_key"]; filled += 1
                if _is_blank(task_cfg.get("provider")):
                    task_cfg["provider"] = template["provider"]; filled += 1
                continue

            # Unknown task — fall back to main, conservative default.
            if _is_blank(task_cfg.get("base_url")):
                task_cfg["base_url"] = main["base_url"]; filled += 1
            if _is_blank(task_cfg.get("api_key")):
                task_cfg["api_key"] = main["api_key"]; filled += 1
            if _is_blank(task_cfg.get("provider")):
                task_cfg["provider"] = main["provider"] or "custom"; filled += 1

    # delegation uses MAIN, not the compression template — sub-agents
    # spawned via delegation need the heavy reasoning model.
    deleg = cfg.get("delegation")
    if isinstance(deleg, dict):
        if _is_blank(deleg.get("base_url")):
            deleg["base_url"] = main["base_url"]; filled += 1
        if _is_blank(deleg.get("api_key")):
            deleg["api_key"] = main["api_key"]; filled += 1
        if _is_blank(deleg.get("provider")):
            deleg["provider"] = main["provider"] or "custom"; filled += 1
        if _is_blank(deleg.get("model")):
            deleg["model"] = main["model"]; filled += 1

    return filled, reverted, skipped_notes


def _probe_models(
    base_url: str, api_key: str, target_model: str, timeout: float = 10.0,
) -> Tuple[Optional[bool], List[str]]:
    """``GET <base_url>/models`` and check whether ``target_model`` is listed.

    Returns ``(found, available)``:
      - ``found = True``  → target model is in the provider's list
      - ``found = False`` → target model is NOT in the list (warn user)
      - ``found = None``  → could not probe (network / auth / parse failure);
        skip the check rather than surfacing a false negative
    """
    if not base_url or not api_key:
        return None, []
    url = base_url.rstrip("/") + "/models"
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
            json.JSONDecodeError, OSError):
        return None, []

    models: List[str] = []
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        for m in payload["data"]:
            if isinstance(m, dict) and "id" in m:
                models.append(str(m["id"]))
    if not target_model:
        return None, models
    return (target_model in models), models


def _atomic_write(path: Path, yaml_rt, cfg) -> None:
    # Preserve original mode (config.yaml typically lives at 0600 — owner-only —
    # because it stores API keys; tempfile.mkstemp would otherwise leave the
    # replacement at 0600 only on filesystems where rename keeps source mode,
    # but the umask-derived 0664 default leaks the key to other users on most
    # systems). Snapshot the mode before write and restore it after rename.
    try:
        original_mode = path.stat().st_mode & 0o777
    except FileNotFoundError:
        original_mode = 0o600  # safe default for a file containing API keys

    fd, tmp = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.stem}_", suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            yaml_rt.dump(cfg, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        try:
            os.chmod(path, original_mode)
        except OSError:
            pass  # best-effort; mode-restore failure shouldn't fail the install
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def run(probe: bool = True, write: bool = True, interactive: bool = True) -> int:
    """Programmatic entry. Returns POSIX exit status (0 on success)."""
    cfg_path = _easybci_home() / "config.yaml"
    if not cfg_path.exists():
        print(f"sync_config: {cfg_path} not present — nothing to do.")
        return 0

    yaml_rt = _build_yaml()
    try:
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = yaml_rt.load(f)
    except Exception as exc:
        print(f"sync_config: failed to parse {cfg_path}: {exc}", file=sys.stderr)
        return 1

    if not isinstance(cfg, dict):
        print(f"sync_config: {cfg_path} is empty or malformed — skipping.")
        return 0

    model = cfg.get("model") or {}
    main = {
        "base_url": (model.get("base_url") or "").strip(),
        "api_key": (model.get("api_key") or "").strip(),
        "provider": (model.get("provider") or "").strip(),
        "model": (model.get("default") or "").strip(),
    }

    if not main["base_url"] or not main["api_key"]:
        print("sync_config: model.base_url or model.api_key is blank — "
              "skipping snapshot (run `easybci setup` to configure main first).")
        return 0

    if write:
        backup = cfg_path.with_suffix(".yaml.pre_sync.bak")
        if not backup.exists():
            backup.write_text(cfg_path.read_text(encoding="utf-8"), encoding="utf-8")
            try:
                # Backup contains the same API key as config.yaml, so mirror the
                # 0600 owner-only mode (or fall back to 0600 if the source had
                # something more permissive — backups should never widen access).
                src_mode = cfg_path.stat().st_mode & 0o777
                os.chmod(backup, min(src_mode, 0o600) if src_mode else 0o600)
            except OSError:
                pass
        filled, reverted, skipped_notes = _sync_aux_and_delegation(cfg, main)
        # Web search/extract backend auto-fill from .env keys.
        env_path = _easybci_home() / ".env"
        web_filled, web_notes = _sync_web_backends(cfg, env_path)
        filled += web_filled
        if filled or reverted:
            _atomic_write(cfg_path, yaml_rt, cfg)
            parts = []
            if filled:
                parts.append(f"filled {filled} blank slot(s)")
            if reverted:
                parts.append(f"reverted {reverted} mis-snapshotted slot(s)")
            print(f"✓ sync_config: {', '.join(parts)} in auxiliary.* / "
                  f"delegation.* / web.* (main: {main['provider'] or 'custom'} "
                  f"@ {main['base_url']}). Backup at {backup.name}.")
        else:
            print("✓ sync_config: auxiliary.* / delegation.* / web.* already "
                  "in the expected shape — no changes.")
        for note in skipped_notes:
            print(note)
        for note in web_notes:
            print(note)

    if probe:
        found, models = _probe_models(main["base_url"], main["api_key"], main["model"])
        if found is True:
            print(f"✓ sync_config: model '{main['model']}' is listed in "
                  f"{main['base_url']}/models (n={len(models)}). Note that "
                  f"some aggregators publish models in /v1/models that your "
                  f"specific account is not entitled to call — if the agent "
                  f"reports HTTP 402 'Insufficient Balance' on this model "
                  f"despite a positive balance, that's per-account entitlement "
                  f"and you'll need to either top up that model on the provider "
                  f"or pick a different model.default.")
        elif found is False:
            sample = models[:20]
            more = "" if len(models) <= 20 else f"\n     … (+{len(models) - 20} more)"
            print(
                f"⚠️  sync_config: configured model '{main['model']}' is NOT "
                f"in the provider's /v1/models response.\n"
                f"   Provider: {main['base_url']}\n"
                f"   Models the provider does serve:\n"
                + "\n".join(f"     - {m}" for m in sample) + more + "\n"
                f"   Edit ~/.easybci/config.yaml → model.default to one of "
                f"these (or switch providers).\n"
                f"   At runtime an unknown model on this provider surfaces as "
                f"'HTTP 402 Insufficient Balance' — that's the symptom this "
                f"check is meant to prevent."
            )
        else:
            print(f"ℹ️  sync_config: could not probe {main['base_url']}/models "
                  f"(network/auth issue or non-OpenAI-compatible endpoint) — "
                  f"skipping availability check.")

        # Vision-model recommendation: only if vision is currently blank
        # (i.e. relying on auto-resolve) AND we have a usable model list.
        if models:
            aux = cfg.get("auxiliary") or {}
            vis = aux.get("vision") or {}
            if _is_blank(vis.get("base_url")) and _is_blank(vis.get("api_key")):
                pick = _suggest_vision_model(models)
                if pick is not None:
                    top_pick, alternatives = pick
                    if interactive and _is_interactive():
                        chosen = _interactive_vision_picker(
                            top_pick, alternatives, main,
                        )
                        if chosen is not None and write:
                            chosen_provider, chosen_model = chosen
                            # Materialise the pick directly into config.yaml.
                            aux_cfg = cfg.setdefault("auxiliary", {})
                            vis_cfg = aux_cfg.setdefault("vision", {})
                            vis_cfg["provider"] = chosen_provider
                            vis_cfg["model"] = chosen_model
                            vis_cfg["base_url"] = main["base_url"]
                            vis_cfg["api_key"] = main["api_key"]
                            _atomic_write(cfg_path, yaml_rt, cfg)
                            print(f"✓ sync_config: wrote auxiliary.vision = "
                                  f"{chosen_model} (provider={chosen_provider}).")
                        elif chosen is None:
                            print("  ↳ skipped — auxiliary.vision left blank "
                                  "(auto-resolver will pick a fallback at runtime).")
                    else:
                        # Passive suggestion: print the snippet so the user can
                        # paste it into config.yaml manually.
                        alt_line = ""
                        if alternatives:
                            alt_line = ("\n   Other vision-capable models on this "
                                        "provider: " + ", ".join(alternatives))
                        print(
                            f"💡 sync_config: this provider lists vision-capable "
                            f"models. To enable image input via your existing key, "
                            f"set in ~/.easybci/config.yaml:\n"
                            f"     auxiliary:\n"
                            f"       vision:\n"
                            f"         provider: custom\n"
                            f"         model: {top_pick}\n"
                            f"         base_url: {main['base_url']}\n"
                            f"         api_key: <your main key>\n"
                            f"   (Skipping auto-fill because the main chat model "
                            f"'{main['model']}' is not multimodal.)"
                            + alt_line
                        )
                else:
                    print(
                        f"ℹ️  sync_config: no obvious vision-capable model "
                        f"found in this provider's catalog. If you need image "
                        f"input, configure OpenRouter (OPENROUTER_API_KEY) or "
                        f"point auxiliary.vision.* at a multimodal endpoint."
                    )

    return 0


def main() -> int:
    return run(probe=True, write=True)


if __name__ == "__main__":
    sys.exit(main())
