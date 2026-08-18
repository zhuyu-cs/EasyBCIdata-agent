"""Slash command definitions and autocomplete for the EasyBCI CLI.

Central registry for all slash commands. Every consumer -- CLI help, gateway
dispatch, autocomplete -- derives its data from ``COMMAND_REGISTRY``.

To add a command: add a ``CommandDef`` entry to ``COMMAND_REGISTRY``.
To add an alias: set ``aliases=("short",)`` on the existing ``CommandDef``.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from easybci_lib.utils import is_truthy_value

logger = logging.getLogger(__name__)

# prompt_toolkit is an optional CLI dependency — only needed for
# SlashCommandCompleter and SlashCommandAutoSuggest.  Gateway and test
# environments that lack it must still be able to import this module
# for resolve_command, gateway_help_lines, and COMMAND_REGISTRY.
try:
    from prompt_toolkit.auto_suggest import AutoSuggest, Suggestion
    from prompt_toolkit.completion import Completer, Completion
except ImportError:  # pragma: no cover
    AutoSuggest = object  # type: ignore[assignment,misc]
    Completer = object    # type: ignore[assignment,misc]
    Suggestion = None     # type: ignore[assignment]
    Completion = None     # type: ignore[assignment]


# ---------------------------------------------------------------------------
# CommandDef dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CommandDef:
    """Definition of a single slash command."""

    name: str                          # canonical name without slash: "background"
    description: str                   # human-readable description
    category: str                      # "Session", "Configuration", etc.
    aliases: tuple[str, ...] = ()      # alternative names: ("bg",)
    args_hint: str = ""                # argument placeholder: "<prompt>", "[name]"
    subcommands: tuple[str, ...] = ()  # tab-completable subcommands
    cli_only: bool = False             # only available in CLI
    gateway_only: bool = False         # only available in gateway/messaging
    gateway_config_gate: str | None = None  # config dotpath; when truthy, overrides cli_only for gateway


# ---------------------------------------------------------------------------
# Central registry -- single source of truth
# ---------------------------------------------------------------------------

COMMAND_REGISTRY: list[CommandDef] = [
    # Session
    CommandDef("new", "Start a new session (fresh session ID + history)", "Session",
               aliases=("reset",), args_hint="[name]"),
    CommandDef("clear", "Clear screen and start a new session", "Session",
               cli_only=True),
    CommandDef("redraw", "Force a full UI repaint (recovers from terminal drift)", "Session",
               cli_only=True),
    CommandDef("history", "Show conversation history", "Session",
               cli_only=True),
    CommandDef("save", "Save the current conversation", "Session",
               cli_only=True),
    CommandDef("retry", "Retry the last message (resend to agent)", "Session"),
    CommandDef("undo", "Remove the last user/assistant exchange", "Session"),
    CommandDef("title", "Set a title for the current session", "Session",
               args_hint="[name]"),
    CommandDef("handoff", "Hand off this session to a messaging platform", "Session",
               args_hint="<platform>", cli_only=True),
    CommandDef("branch", "Branch the current session (explore a different path)", "Session",
               aliases=("fork",), args_hint="[name]"),
    CommandDef("compress", "Manually compress conversation context", "Session",
               args_hint="[focus topic]"),
    CommandDef("rollback", "List or restore filesystem checkpoints", "Session",
               args_hint="[number]"),
    CommandDef("snapshot", "Create or restore state snapshots of EasyBCI config/state", "Session",
               cli_only=True, aliases=("snap",), args_hint="[create|restore <id>|prune]"),
    CommandDef("stop", "Kill all running background processes", "Session"),
    CommandDef("approve", "Approve a pending dangerous command", "Session",
               gateway_only=True, args_hint="[session|always]"),
    CommandDef("deny", "Deny a pending dangerous command", "Session",
               gateway_only=True),
    CommandDef("background", "Run a prompt in the background", "Session",
               aliases=("bg", "btw"), args_hint="<prompt>"),
    CommandDef("agents", "Show active agents and running tasks", "Session",
               aliases=("tasks",)),
    CommandDef("queue", "Queue a prompt for the next turn (doesn't interrupt)", "Session",
               aliases=("q",), args_hint="<prompt>"),
    CommandDef("steer", "Inject a message after the next tool call without interrupting", "Session",
               args_hint="<prompt>"),
    CommandDef("goal", "Set a standing goal EasyBCI works on across turns until achieved", "Session",
               args_hint="[text | pause | resume | clear | status]"),
    CommandDef("subgoal", "Add or manage extra criteria on the active goal", "Session",
               args_hint="[text | remove N | clear]"),
    CommandDef("status", "Show session info", "Session"),
    CommandDef("whoami", "Show your slash command access (admin / user)", "Info"),
    CommandDef("profile", "Show active profile name and home directory", "Info"),
    CommandDef("sethome", "Set this chat as the home channel", "Session",
               gateway_only=True, aliases=("set-home",)),
    CommandDef("resume", "Resume a previously-named session", "Session",
               args_hint="[name]"),

    # Configuration
    CommandDef("sessions", "Browse and resume previous sessions", "Session"),

    # Configuration
    CommandDef("config", "Show current configuration", "Configuration",
               cli_only=True),
    CommandDef("model", "Switch model for this session", "Configuration",
               aliases=("provider",), args_hint="[model] [--provider name] [--global]"),
    CommandDef("gquota", "Show Google Gemini Code Assist quota usage", "Info",
               cli_only=True),

    CommandDef("personality", "Set a predefined personality", "Configuration",
               args_hint="[name]"),
    CommandDef("statusbar", "Toggle the context/model status bar", "Configuration",
               cli_only=True, aliases=("sb",)),
    CommandDef("verbose", "Cycle tool progress display: off -> new -> all -> verbose",
               "Configuration", cli_only=True,
               gateway_config_gate="display.tool_progress_command"),
    CommandDef("footer", "Toggle gateway runtime-metadata footer on final replies",
               "Configuration", args_hint="[on|off|status]",
               subcommands=("on", "off", "status")),
    CommandDef("yolo", "Toggle YOLO mode (skip all dangerous command approvals)",
               "Configuration"),
    CommandDef("reasoning", "Manage reasoning effort and display", "Configuration",
               args_hint="[level|show|hide]",
               subcommands=("none", "minimal", "low", "medium", "high", "xhigh", "show", "hide", "on", "off")),
    CommandDef("fast", "Toggle fast mode — OpenAI Priority Processing / Anthropic Fast Mode (Normal/Fast)", "Configuration",
               args_hint="[normal|fast|status]",
               subcommands=("normal", "fast", "status", "on", "off")),
    CommandDef("skin", "Show or change the display skin/theme", "Configuration",
               cli_only=True, args_hint="[name]"),
    CommandDef("indicator", "Pick the TUI busy-indicator style", "Configuration",
               cli_only=True, args_hint="[kaomoji|emoji|unicode|ascii]",
               subcommands=("kaomoji", "emoji", "unicode", "ascii")),
    CommandDef("busy", "Control what Enter does while EasyBCI is working", "Configuration",
               cli_only=True, args_hint="[queue|steer|interrupt|status]",
               subcommands=("queue", "steer", "interrupt", "status")),

    # Tools & Skills
    CommandDef("tools", "Manage tools: /tools [list|disable|enable] [name...]", "Tools & Skills",
               args_hint="[list|disable|enable] [name...]", cli_only=True),
    CommandDef("toolsets", "List available toolsets", "Tools & Skills",
               cli_only=True),
    CommandDef("skills", "Search, install, inspect, or manage skills",
               "Tools & Skills", cli_only=True,
               subcommands=("search", "browse", "inspect", "install")),
    CommandDef("curator", "Background skill maintenance (status, run, pin, archive, list-archived)",
               "Tools & Skills", args_hint="[subcommand]",
               subcommands=("status", "run", "pause", "resume", "pin", "unpin", "restore", "list-archived")),
    CommandDef("reload", "Reload .env variables into the running session", "Tools & Skills",
               cli_only=True),
    CommandDef("reload-mcp", "Reload MCP servers from config", "Tools & Skills",
               aliases=("reload_mcp",)),
    CommandDef("reload-skills", "Re-scan ~/.easybci/skills/ for newly installed or removed skills",
               "Tools & Skills", aliases=("reload_skills",)),
    CommandDef("plugins", "List installed plugins and their status",
               "Tools & Skills", cli_only=True),

    # Info
    CommandDef("commands", "Browse all commands and skills (paginated)", "Info",
               gateway_only=True, args_hint="[page]"),
    CommandDef("help", "Show available commands", "Info"),
    CommandDef("restart", "Gracefully restart the gateway after draining active runs", "Session",
               gateway_only=True),
    CommandDef("usage", "Show token usage and rate limits for the current session", "Info"),
    CommandDef("insights", "Show usage insights and analytics", "Info",
               args_hint="[days]"),
    CommandDef("platforms", "Show gateway/messaging platform status", "Info",
               cli_only=True, aliases=("gateway",)),
    CommandDef("copy", "Copy the last assistant response to clipboard", "Info",
               cli_only=True, args_hint="[number]"),
    CommandDef("paste", "Attach clipboard image from your clipboard", "Info",
               cli_only=True),
    CommandDef("image", "Attach a local image file for your next prompt", "Info",
               cli_only=True, args_hint="<path>"),
    CommandDef("update", "Update EasyBCI Agent to the latest version", "Info",
               gateway_only=True),
    CommandDef("debug", "Upload debug report (system info + logs) and get shareable links", "Info"),

    # Exit
    CommandDef("quit", "Exit the CLI", "Exit",
               cli_only=True, aliases=("exit",)),
]


# ---------------------------------------------------------------------------
# Derived lookups -- rebuilt once at import time, refreshed by rebuild_lookups()
# ---------------------------------------------------------------------------

def _build_command_lookup() -> dict[str, CommandDef]:
    """Map every name and alias to its CommandDef."""
    lookup: dict[str, CommandDef] = {}
    for cmd in COMMAND_REGISTRY:
        lookup[cmd.name] = cmd
        for alias in cmd.aliases:
            lookup[alias] = cmd
    return lookup


_COMMAND_LOOKUP: dict[str, CommandDef] = _build_command_lookup()


def resolve_command(name: str) -> CommandDef | None:
    """Resolve a command name or alias to its CommandDef.

    Accepts names with or without the leading slash.
    """
    return _COMMAND_LOOKUP.get(name.lower().lstrip("/"))


def _build_description(cmd: CommandDef) -> str:
    """Build a CLI-facing description string including usage hint."""
    if cmd.args_hint:
        return f"{cmd.description} (usage: /{cmd.name} {cmd.args_hint})"
    return cmd.description


# Backwards-compatible flat dict: "/command" -> description
COMMANDS: dict[str, str] = {}
for _cmd in COMMAND_REGISTRY:
    if not _cmd.gateway_only:
        COMMANDS[f"/{_cmd.name}"] = _build_description(_cmd)
        for _alias in _cmd.aliases:
            COMMANDS[f"/{_alias}"] = f"{_cmd.description} (alias for /{_cmd.name})"

# Backwards-compatible categorized dict
COMMANDS_BY_CATEGORY: dict[str, dict[str, str]] = {}
for _cmd in COMMAND_REGISTRY:
    if not _cmd.gateway_only:
        _cat = COMMANDS_BY_CATEGORY.setdefault(_cmd.category, {})
        _cat[f"/{_cmd.name}"] = COMMANDS[f"/{_cmd.name}"]
        for _alias in _cmd.aliases:
            _cat[f"/{_alias}"] = COMMANDS[f"/{_alias}"]


# Subcommands lookup: "/cmd" -> ["sub1", "sub2", ...]
SUBCOMMANDS: dict[str, list[str]] = {}
for _cmd in COMMAND_REGISTRY:
    if _cmd.subcommands:
        SUBCOMMANDS[f"/{_cmd.name}"] = list(_cmd.subcommands)

# Also extract subcommands hinted in args_hint via pipe-separated patterns
# e.g. args_hint="[on|off|status]" for commands that don't have explicit subcommands.
# NOTE: If a command already has explicit subcommands, this fallback is skipped.
# Use the `subcommands` field on CommandDef for intentional tab-completable args.
_PIPE_SUBS_RE = re.compile(r"[a-z]+(?:\|[a-z]+)+")
for _cmd in COMMAND_REGISTRY:
    key = f"/{_cmd.name}"
    if key in SUBCOMMANDS or not _cmd.args_hint:
        continue
    m = _PIPE_SUBS_RE.search(_cmd.args_hint)
    if m:
        SUBCOMMANDS[key] = m.group(0).split("|")


# ---------------------------------------------------------------------------
# Gateway helpers
# ---------------------------------------------------------------------------

# Set of all command names + aliases recognized by the gateway.
# Includes config-gated commands so the gateway can dispatch them
# (the handler checks the config gate at runtime).
GATEWAY_KNOWN_COMMANDS: frozenset[str] = frozenset(
    name
    for cmd in COMMAND_REGISTRY
    if not cmd.cli_only or cmd.gateway_config_gate
    for name in (cmd.name, *cmd.aliases)
)


def is_gateway_known_command(name: str | None) -> bool:
    """Return True if ``name`` resolves to a gateway-dispatchable slash command.

    This covers both built-in commands (``GATEWAY_KNOWN_COMMANDS`` derived
    from ``COMMAND_REGISTRY``) and plugin-registered commands, which are
    looked up lazily so importing this module never forces plugin
    discovery. Gateway code uses this to decide whether to emit
    ``command:<name>`` hooks — plugin commands get the same lifecycle
    events as built-ins.
    """
    if not name:
        return False
    if name in GATEWAY_KNOWN_COMMANDS:
        return True
    for plugin_name, _description, _args_hint in _iter_plugin_command_entries():
        if plugin_name == name:
            return True
    return False


# Commands with explicit Level-2 running-agent handlers in gateway/run.py.
# Listed here for introspection / tests; semantically a subset of
# "all resolvable commands" — which is the real bypass set (see
# should_bypass_active_session below).
ACTIVE_SESSION_BYPASS_COMMANDS: frozenset[str] = frozenset(
    {
        "agents",
        "approve",
        "background",
        "commands",
        "deny",
        "help",
        "new",
        "profile",
        "queue",
        "restart",
        "status",
        "steer",
        "stop",
        "update",
    }
)


def should_bypass_active_session(command_name: str | None) -> bool:
    """Return True for any resolvable slash command.

    Rationale: every gateway-registered slash command either has a
    specific Level-2 handler in gateway/run.py (/stop, /new, /model,
    /approve, etc.) or reaches the running-agent catch-all that returns
    a "busy — wait or /stop first" response. In both paths the command
    is dispatched, not queued.

    Queueing is always wrong for a recognized slash command because the
    safety net in gateway.run discards any command text that reaches
    the pending queue — which meant a mid-run /model (or /reasoning,
    /insights, /title, /resume, /retry, /undo, /compress,
    /usage, /reload-mcp, /sethome, /reset) would silently
    interrupt the agent AND get discarded, producing a zero-char
    response.

    ACTIVE_SESSION_BYPASS_COMMANDS remains the subset of commands with
    explicit Level-2 handlers; the rest fall through to the catch-all.
    """
    return resolve_command(command_name) is not None if command_name else False


def _resolve_config_gates() -> set[str]:
    """Return canonical names of commands whose ``gateway_config_gate`` is truthy.

    Reads ``config.yaml`` and walks the dot-separated key path for each
    config-gated command.  Returns an empty set on any error so callers
    degrade gracefully.
    """
    gated = [c for c in COMMAND_REGISTRY if c.gateway_config_gate]
    if not gated:
        return set()
    try:
        from easybci_cli.config import read_raw_config
        cfg = read_raw_config()
    except Exception:
        return set()
    result: set[str] = set()
    for cmd in gated:
        val: Any = cfg
        for key in cmd.gateway_config_gate.split("."):
            if isinstance(val, dict):
                val = val.get(key)
            else:
                val = None
                break
        if is_truthy_value(val, default=False):
            result.add(cmd.name)
    return result


def _is_gateway_available(cmd: CommandDef, config_overrides: set[str] | None = None) -> bool:
    """Check if *cmd* should appear in gateway surfaces (help, menus, mappings).

    Unconditionally available when ``cli_only`` is False.  When ``cli_only``
    is True but ``gateway_config_gate`` is set, the command is available only
    when the config value is truthy.  Pass *config_overrides* (from
    ``_resolve_config_gates()``) to avoid re-reading config for every command.
    """
    if not cmd.cli_only:
        return True
    if cmd.gateway_config_gate:
        overrides = config_overrides if config_overrides is not None else _resolve_config_gates()
        return cmd.name in overrides
    return False


def _requires_argument(args_hint: str) -> bool:
    """Return True when selecting a command without text would be incomplete."""
    return args_hint.strip().startswith("<")


def gateway_help_lines() -> list[str]:
    """Generate gateway help text lines from the registry."""
    overrides = _resolve_config_gates()
    lines: list[str] = []
    for cmd in COMMAND_REGISTRY:
        if not _is_gateway_available(cmd, overrides):
            continue
        args = f" {cmd.args_hint}" if cmd.args_hint else ""
        alias_parts: list[str] = []
        for a in cmd.aliases:
            # Skip internal aliases like reload_mcp (underscore variant)
            if a.replace("-", "_") == cmd.name.replace("-", "_") and a != cmd.name:
                continue
            alias_parts.append(f"`/{a}`")
        alias_note = f" (alias: {', '.join(alias_parts)})" if alias_parts else ""
        lines.append(f"`/{cmd.name}{args}` -- {cmd.description}{alias_note}")
    return lines


def _iter_plugin_command_entries() -> list[tuple[str, str, str]]:
    """Yield (name, description, args_hint) tuples for all plugin slash commands.

    Plugin commands are registered via
    :func:`easybci_cli.plugins.PluginContext.register_command`. They behave
    like ``CommandDef`` entries for autocomplete and CLI dispatch.

    Lookup is lazy so importing this module never forces plugin discovery
    (which can trigger filesystem scans and environment-dependent
    behavior).
    """
    try:
        from easybci_cli.plugins import get_plugin_commands
    except Exception:
        return []
    try:
        commands = get_plugin_commands() or {}
    except Exception:
        return []
    entries: list[tuple[str, str, str]] = []
    for name, meta in commands.items():
        if not isinstance(name, str) or not isinstance(meta, dict):
            continue
        description = str(meta.get("description") or f"Run /{name}")
        args_hint = str(meta.get("args_hint") or "").strip()
        entries.append((name, description, args_hint))
    return entries


# ---------------------------------------------------------------------------
# Autocomplete
# ---------------------------------------------------------------------------


# Per-process cache for /model<space> LM Studio autocomplete. Probing on
# every keystroke would block the UI; a short TTL keeps it live without
# hammering the server.
_LMSTUDIO_COMPLETION_CACHE: tuple[float, list[str]] | None = None


def _lmstudio_completion_models() -> list[str]:
    """Locally-loaded LM Studio models for /model autocomplete (cached, gated)."""
    global _LMSTUDIO_COMPLETION_CACHE
    # Gate: don't probe 127.0.0.1 on every keystroke for users who don't use LM Studio.
    if not (os.environ.get("LM_API_KEY") or os.environ.get("LM_BASE_URL")):
        try:
            from easybci_cli.auth import _load_auth_store
            store = _load_auth_store() or {}
            if "lmstudio" not in (store.get("providers") or {}) \
               and "lmstudio" not in (store.get("credential_pool") or {}):
                return []
        except Exception:
            return []
    now = time.time()
    if _LMSTUDIO_COMPLETION_CACHE and (now - _LMSTUDIO_COMPLETION_CACHE[0]) < 30.0:
        return _LMSTUDIO_COMPLETION_CACHE[1]
    try:
        from easybci_cli.models import fetch_lmstudio_models
        models = fetch_lmstudio_models(
            api_key=os.environ.get("LM_API_KEY", ""),
            base_url=os.environ.get("LM_BASE_URL") or "http://127.0.0.1:1234/v1",
            timeout=0.8,
        )
    except Exception:
        models = []
    _LMSTUDIO_COMPLETION_CACHE = (now, models)
    return models


class SlashCommandCompleter(Completer):
    """Autocomplete for built-in slash commands, subcommands, and skill commands."""

    def __init__(
        self,
        skill_commands_provider: Callable[[], Mapping[str, dict[str, Any]]] | None = None,
        command_filter: Callable[[str], bool] | None = None,
    ) -> None:
        self._skill_commands_provider = skill_commands_provider
        self._command_filter = command_filter
        # Cached project file list for fuzzy @ completions
        self._file_cache: list[str] = []
        self._file_cache_time: float = 0.0
        self._file_cache_cwd: str = ""

    def _command_allowed(self, slash_command: str) -> bool:
        if self._command_filter is None:
            return True
        try:
            return bool(self._command_filter(slash_command))
        except Exception:
            return True

    def _iter_skill_commands(self) -> Mapping[str, dict[str, Any]]:
        if self._skill_commands_provider is None:
            return {}
        try:
            return self._skill_commands_provider() or {}
        except Exception:
            return {}

    # Commands that open pickers when run without arguments.
    # These should NOT receive a trailing space in completions because:
    # - The TUI's submit handler applies completions on Enter if input differs
    # - Adding space makes "/model" → "/model " which blocks picker execution
    _PICKER_COMMANDS = frozenset({"model", "skin", "personality"})

    @staticmethod
    def _completion_text(cmd_name: str, word: str) -> str:
        """Return replacement text for a completion.

        When the user has already typed the full command exactly (``/help``),
        returning ``help`` would be a no-op and prompt_toolkit suppresses the
        menu. Appending a trailing space keeps the dropdown visible and makes
        backspacing retrigger it naturally.

        However, commands that open pickers (model, skin, personality) should
        NOT get a trailing space — the TUI would apply the completion on Enter
        and block the picker from opening.
        """
        if cmd_name != word:
            return cmd_name
        # Don't add space for picker commands — allows Enter to execute them
        if cmd_name in SlashCommandCompleter._PICKER_COMMANDS:
            return cmd_name
        return f"{cmd_name} "

    @staticmethod
    def _extract_path_word(text: str) -> str | None:
        """Extract the current word if it looks like a file path.

        Returns the path-like token under the cursor, or None if the
        current word doesn't look like a path.  A word is path-like when
        it starts with ``./``, ``../``, ``~/``, ``/``, or contains a
        ``/`` separator (e.g. ``src/main.py``).
        """
        if not text:
            return None
        # Walk backwards to find the start of the current "word".
        # Words are delimited by spaces, but paths can contain almost anything.
        i = len(text) - 1
        while i >= 0 and text[i] != " ":
            i -= 1
        word = text[i + 1:]
        if not word:
            return None
        # Only trigger path completion for path-like tokens
        if word.startswith(("./", "../", "~/", "/")) or "/" in word:
            return word
        return None

    @staticmethod
    def _path_completions(word: str, limit: int = 30):
        """Yield Completion objects for file paths matching *word*."""
        expanded = os.path.expanduser(word)
        # Split into directory part and prefix to match inside it
        if expanded.endswith("/"):
            search_dir = expanded
            prefix = ""
        else:
            search_dir = os.path.dirname(expanded) or "."
            prefix = os.path.basename(expanded)

        try:
            entries = os.listdir(search_dir)
        except OSError:
            return

        count = 0
        prefix_lower = prefix.lower()
        for entry in sorted(entries):
            if prefix and not entry.lower().startswith(prefix_lower):
                continue
            if count >= limit:
                break

            full_path = os.path.join(search_dir, entry)
            is_dir = os.path.isdir(full_path)

            # Build the completion text (what replaces the typed word)
            if word.startswith("~"):
                display_path = "~/" + os.path.relpath(full_path, os.path.expanduser("~"))
            elif os.path.isabs(word):
                display_path = full_path
            else:
                # Keep relative
                display_path = os.path.relpath(full_path)

            if is_dir:
                display_path += "/"

            suffix = "/" if is_dir else ""
            meta = "dir" if is_dir else _file_size_label(full_path)

            yield Completion(
                display_path,
                start_position=-len(word),
                display=entry + suffix,
                display_meta=meta,
            )
            count += 1

    @staticmethod
    def _extract_context_word(text: str) -> str | None:
        """Extract a bare ``@`` token for context reference completions."""
        if not text:
            return None
        # Walk backwards to find the start of the current word
        i = len(text) - 1
        while i >= 0 and text[i] != " ":
            i -= 1
        word = text[i + 1:]
        if not word.startswith("@"):
            return None
        return word

    def _context_completions(self, word: str, limit: int = 30):
        """Yield Claude Code-style @ context completions.

        Bare ``@`` or ``@partial`` shows static references and matching
        files/folders.  ``@file:path`` and ``@folder:path`` are handled
        by the existing path completion path.
        """
        lowered = word.lower()

        # Static context references
        _STATIC_REFS = (
            ("@diff", "Git working tree diff"),
            ("@staged", "Git staged diff"),
            ("@file:", "Attach a file"),
            ("@folder:", "Attach a folder"),
            ("@git:", "Git log with diffs (e.g. @git:5)"),
            ("@url:", "Fetch web content"),
        )
        for candidate, meta in _STATIC_REFS:
            if candidate.lower().startswith(lowered) and candidate.lower() != lowered:
                yield Completion(
                    candidate,
                    start_position=-len(word),
                    display=candidate,
                    display_meta=meta,
                )

        # If the user typed @file: / @folder: (or just @file / @folder with
        # no colon yet), delegate to path completions.  Accepting the bare
        # form lets the picker surface directories as soon as the user has
        # typed `@folder`, without requiring them to first accept the static
        # `@folder:` hint and re-trigger completion.
        for prefix in ("@file:", "@folder:"):
            bare = prefix[:-1]

            if word == bare or word.startswith(prefix):
                want_dir = prefix == "@folder:"
                path_part = '' if word == bare else word[len(prefix):]
                expanded = os.path.expanduser(path_part)

                if not expanded or expanded == ".":
                    search_dir, match_prefix = ".", ""
                elif expanded.endswith("/"):
                    search_dir, match_prefix = expanded, ""
                else:
                    search_dir = os.path.dirname(expanded) or "."
                    match_prefix = os.path.basename(expanded)

                try:
                    entries = os.listdir(search_dir)
                except OSError:
                    return

                count = 0
                prefix_lower = match_prefix.lower()
                for entry in sorted(entries):
                    if match_prefix and not entry.lower().startswith(prefix_lower):
                        continue
                    full_path = os.path.join(search_dir, entry)
                    is_dir = os.path.isdir(full_path)
                    # `@folder:` must only surface directories; `@file:` only
                    # regular files.  Without this filter `@folder:` listed
                    # every .env / .gitignore in the cwd, defeating the
                    # explicit prefix and confusing users expecting a
                    # directory picker.
                    if want_dir != is_dir:
                        continue
                    if count >= limit:
                        break
                    display_path = os.path.relpath(full_path)
                    suffix = "/" if is_dir else ""
                    meta = "dir" if is_dir else _file_size_label(full_path)
                    completion = f"{prefix}{display_path}{suffix}"
                    yield Completion(
                        completion,
                        start_position=-len(word),
                        display=entry + suffix,
                        display_meta=meta,
                    )
                    count += 1
                return

        # Bare @ or @partial — fuzzy project-wide file search
        query = word[1:]  # strip the @
        yield from self._fuzzy_file_completions(word, query, limit)

    def _get_project_files(self) -> list[str]:
        """Return cached list of project files (refreshed every 5s)."""
        cwd = os.getcwd()
        now = time.monotonic()
        if (
            self._file_cache
            and self._file_cache_cwd == cwd
            and now - self._file_cache_time < 5.0
        ):
            return self._file_cache

        files: list[str] = []
        # Try rg first (fast, respects .gitignore), then fd, then find.
        for cmd in [
            ["rg", "--files", "--sortr=modified", cwd],
            ["rg", "--files", cwd],
            ["fd", "--type", "f", "--base-directory", cwd],
        ]:
            tool = cmd[0]
            if not shutil.which(tool):
                continue
            try:
                proc = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=2,
                    cwd=cwd, encoding="utf-8", errors="replace",
                )
                if proc.returncode == 0 and proc.stdout and proc.stdout.strip():
                    raw = proc.stdout.strip().split("\n")
                    # Store relative paths
                    for p in raw[:5000]:
                        rel = os.path.relpath(p, cwd) if os.path.isabs(p) else p
                        files.append(rel)
                    break
            except (subprocess.TimeoutExpired, OSError):
                continue

        self._file_cache = files
        self._file_cache_time = now
        self._file_cache_cwd = cwd
        return files

    @staticmethod
    def _score_path(filepath: str, query: str) -> int:
        """Score a file path against a fuzzy query. Higher = better match."""
        if not query:
            return 1  # show everything when query is empty

        filename = os.path.basename(filepath)
        lower_file = filename.lower()
        lower_path = filepath.lower()
        lower_q = query.lower()

        # Exact filename match
        if lower_file == lower_q:
            return 100
        # Filename starts with query
        if lower_file.startswith(lower_q):
            return 80
        # Filename contains query as substring
        if lower_q in lower_file:
            return 60
        # Full path contains query
        if lower_q in lower_path:
            return 40
        # Initials / abbreviation match: e.g. "fo" matches "file_operations"
        # Check if query chars appear in order in filename
        qi = 0
        for c in lower_file:
            if qi < len(lower_q) and c == lower_q[qi]:
                qi += 1
        if qi == len(lower_q):
            # Bonus if matches land on word boundaries (after _, -, /, .)
            boundary_hits = 0
            qi = 0
            prev = "_"  # treat start as boundary
            for c in lower_file:
                if qi < len(lower_q) and c == lower_q[qi]:
                    if prev in "_-./":
                        boundary_hits += 1
                    qi += 1
                prev = c
            if boundary_hits >= len(lower_q) * 0.5:
                return 35
            return 25
        return 0

    def _fuzzy_file_completions(self, word: str, query: str, limit: int = 20):
        """Yield fuzzy file completions for bare @query."""
        files = self._get_project_files()

        if not query:
            # No query — show recently modified files (already sorted by mtime)
            for fp in files[:limit]:
                is_dir = fp.endswith("/")
                filename = os.path.basename(fp)
                kind = "folder" if is_dir else "file"
                meta = "dir" if is_dir else _file_size_label(
                    os.path.join(os.getcwd(), fp)
                )
                yield Completion(
                    f"@{kind}:{fp}",
                    start_position=-len(word),
                    display=filename,
                    display_meta=meta,
                )
            return

        # Score and rank
        scored = []
        for fp in files:
            s = self._score_path(fp, query)
            if s > 0:
                scored.append((s, fp))
        scored.sort(key=lambda x: (-x[0], x[1]))

        for _, fp in scored[:limit]:
            is_dir = fp.endswith("/")
            filename = os.path.basename(fp)
            kind = "folder" if is_dir else "file"
            meta = "dir" if is_dir else _file_size_label(
                os.path.join(os.getcwd(), fp)
            )
            yield Completion(
                f"@{kind}:{fp}",
                start_position=-len(word),
                display=filename,
                display_meta=f"{fp}  {meta}" if meta else fp,
            )

    @staticmethod
    def _skin_completions(sub_text: str, sub_lower: str):
        """Yield completions for /skin from available skins."""
        try:
            from easybci_cli.skin_engine import list_skins
            for s in list_skins():
                name = s["name"]
                if name.startswith(sub_lower) and name != sub_lower:
                    yield Completion(
                        name,
                        start_position=-len(sub_text),
                        display=name,
                        display_meta=s.get("description", "") or s.get("source", ""),
                    )
        except Exception:
            pass

    @staticmethod
    def _personality_completions(sub_text: str, sub_lower: str):
        """Yield completions for /personality from configured personalities."""
        try:
            from easybci_cli.config import load_config
            personalities = load_config().get("agent", {}).get("personalities", {})
            if "none".startswith(sub_lower) and "none" != sub_lower:
                yield Completion(
                    "none",
                    start_position=-len(sub_text),
                    display="none",
                    display_meta="clear personality overlay",
                )
            for name, prompt in personalities.items():
                if name.startswith(sub_lower) and name != sub_lower:
                    if isinstance(prompt, dict):
                        meta = prompt.get("description") or prompt.get("system_prompt", "")[:50]
                    else:
                        meta = str(prompt)[:50]
                    yield Completion(
                        name,
                        start_position=-len(sub_text),
                        display=name,
                        display_meta=meta,
                    )
        except Exception:
            pass

    def _model_completions(self, sub_text: str, sub_lower: str):
        """Yield completions for /model from config aliases + built-in aliases."""
        seen = set()
        # Config-based direct aliases (preferred — include provider info)
        try:
            from easybci_cli.model_switch import (
                _ensure_direct_aliases, DIRECT_ALIASES, MODEL_ALIASES,
            )
            _ensure_direct_aliases()
            for name, da in DIRECT_ALIASES.items():
                if name.startswith(sub_lower) and name != sub_lower:
                    seen.add(name)
                    yield Completion(
                        name,
                        start_position=-len(sub_text),
                        display=name,
                        display_meta=f"{da.model} ({da.provider})",
                    )
            # Built-in catalog aliases not already covered
            for name in sorted(MODEL_ALIASES.keys()):
                if name in seen:
                    continue
                if name.startswith(sub_lower) and name != sub_lower:
                    identity = MODEL_ALIASES[name]
                    yield Completion(
                        name,
                        start_position=-len(sub_text),
                        display=name,
                        display_meta=f"{identity.vendor}/{identity.family}",
                    )
        except Exception:
            pass
        # LM Studio: surface locally-loaded models. Gated on the user actually
        # having LM Studio configured (env var or auth-store entry) so we
        # don't probe 127.0.0.1 on every keystroke for users who don't use it.
        for name in _lmstudio_completion_models():
            if name in seen:
                continue
            if name.startswith(sub_lower) and name != sub_lower:
                yield Completion(
                    name,
                    start_position=-len(sub_text),
                    display=name,
                    display_meta="LM Studio",
                )

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if not text.startswith("/"):
            # Try @ context completion (Claude Code-style)
            ctx_word = self._extract_context_word(text)
            if ctx_word is not None:
                yield from self._context_completions(ctx_word)
                return
            # Try file path completion for non-slash input
            path_word = self._extract_path_word(text)
            if path_word is not None:
                yield from self._path_completions(path_word)
            return

        # Check if we're completing a subcommand (base command already typed)
        parts = text.split(maxsplit=1)
        base_cmd = parts[0].lower()
        if len(parts) > 1 or (len(parts) == 1 and text.endswith(" ")):
            sub_text = parts[1] if len(parts) > 1 else ""
            sub_lower = sub_text.lower()

            # Dynamic completions for commands with runtime lists
            if " " not in sub_text:
                if base_cmd == "/model":
                    yield from self._model_completions(sub_text, sub_lower)
                    return
                if base_cmd == "/skin":
                    yield from self._skin_completions(sub_text, sub_lower)
                    return
                if base_cmd == "/personality":
                    yield from self._personality_completions(sub_text, sub_lower)
                    return

            # Static subcommand completions
            if " " not in sub_text and base_cmd in SUBCOMMANDS and self._command_allowed(base_cmd):
                for sub in SUBCOMMANDS[base_cmd]:
                    if sub.startswith(sub_lower) and sub != sub_lower:
                        yield Completion(
                            sub,
                            start_position=-len(sub_text),
                            display=sub,
                        )
            return

        word = text[1:]

        for cmd, desc in COMMANDS.items():
            if not self._command_allowed(cmd):
                continue
            cmd_name = cmd[1:]
            if cmd_name.startswith(word):
                yield Completion(
                    self._completion_text(cmd_name, word),
                    start_position=-len(word),
                    display=cmd,
                    display_meta=desc,
                )

        for cmd, info in self._iter_skill_commands().items():
            cmd_name = cmd[1:]
            if cmd_name.startswith(word):
                description = str(info.get("description", "Skill command"))
                short_desc = description[:50] + ("..." if len(description) > 50 else "")
                yield Completion(
                    self._completion_text(cmd_name, word),
                    start_position=-len(word),
                    display=cmd,
                    display_meta=f"⚡ {short_desc}",
                )

        # Plugin-registered slash commands
        try:
            from easybci_cli.plugins import get_plugin_commands
            for cmd_name, cmd_info in get_plugin_commands().items():
                if cmd_name.startswith(word):
                    desc = str(cmd_info.get("description", "Plugin command"))
                    short_desc = desc[:50] + ("..." if len(desc) > 50 else "")
                    yield Completion(
                        self._completion_text(cmd_name, word),
                        start_position=-len(word),
                        display=f"/{cmd_name}",
                        display_meta=f"🔌 {short_desc}",
                    )
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Inline auto-suggest (ghost text) for slash commands
# ---------------------------------------------------------------------------

class SlashCommandAutoSuggest(AutoSuggest):
    """Inline ghost-text suggestions for slash commands and their subcommands.

    Shows the rest of a command or subcommand in dim text as you type.
    Falls back to history-based suggestions for non-slash input.
    """

    def __init__(
        self,
        history_suggest: AutoSuggest | None = None,
        completer: SlashCommandCompleter | None = None,
    ) -> None:
        self._history = history_suggest
        self._completer = completer  # Reuse its model cache

    def get_suggestion(self, buffer, document):
        text = document.text_before_cursor

        # Only suggest for slash commands
        if not text.startswith("/"):
            # Fall back to history for regular text
            if self._history:
                return self._history.get_suggestion(buffer, document)
            return None

        parts = text.split(maxsplit=1)
        base_cmd = parts[0].lower()

        if len(parts) == 1 and not text.endswith(" "):
            # Still typing the command name: /upd → suggest "ate"
            word = text[1:].lower()
            for cmd in COMMANDS:
                if self._completer is not None and not self._completer._command_allowed(cmd):
                    continue
                cmd_name = cmd[1:]  # strip leading /
                if cmd_name.startswith(word) and cmd_name != word:
                    return Suggestion(cmd_name[len(word):])
            return None

        # Command is complete — suggest subcommands or model names
        sub_text = parts[1] if len(parts) > 1 else ""
        sub_lower = sub_text.lower()

        # Static subcommands
        if self._completer is not None and not self._completer._command_allowed(base_cmd):
            return None
        if base_cmd in SUBCOMMANDS and SUBCOMMANDS[base_cmd]:
            if " " not in sub_text:
                for sub in SUBCOMMANDS[base_cmd]:
                    if sub.startswith(sub_lower) and sub != sub_lower:
                        return Suggestion(sub[len(sub_text):])

        # Fall back to history
        if self._history:
            return self._history.get_suggestion(buffer, document)
        return None


def _file_size_label(path: str) -> str:
    """Return a compact human-readable file size, or '' on error."""
    try:
        size = os.path.getsize(path)
    except OSError:
        return ""
    if size < 1024:
        return f"{size}B"
    if size < 1024 * 1024:
        return f"{size / 1024:.0f}K"
    if size < 1024 * 1024 * 1024:
        return f"{size / (1024 * 1024):.1f}M"
    return f"{size / (1024 * 1024 * 1024):.1f}G"
