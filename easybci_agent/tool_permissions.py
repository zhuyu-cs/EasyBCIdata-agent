"""Per-tool permission tier controller.

Provides layered permission control for tool execution:
- auto: always allowed (read-only tools)
- notify: allowed but logged to UI
- ask-once: prompt on first use, then auto-allow for session
- always-ask: prompt every invocation
- builtin: tool has its own internal approval (e.g. terminal)
"""

import threading
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class PermissionDecision:
    """Result of a permission check."""
    allowed: bool
    needs_prompt: bool
    tier: str
    tool_name: str


VALID_TIERS = frozenset({"auto", "notify", "ask-once", "always-ask", "builtin"})

DEFAULT_TIERS: dict[str, str] = {
    # auto — read-only / informational tools
    "read_file": "auto",
    "search_files": "auto",
    "web_search": "auto",
    "web_extract": "auto",
    "skills_list": "auto",
    "skill_view": "auto",
    "session_search": "auto",
    "clarify": "auto",
    "vision_analyze": "auto",
    "inspect_data": "auto",
    "inspect_neural": "auto",       # alias → inspect_data
    "quality_check": "auto",
    "list_data": "auto",
    "plan_pipeline": "auto",
    "suggest_pipeline": "auto",     # alias → plan_pipeline
    "propose_pipeline": "auto",     # alias → plan_pipeline
    "ha_get_state": "auto",
    "ha_list_entities": "auto",
    "ha_list_services": "auto",
    "pipeline_tracker": "auto",
    # notify — low-risk persistent writes
    "memory": "notify",
    # ask-once — mutating tools, prompt first then auto for session
    "write_file": "ask-once",
    "patch": "ask-once",
    "execute_code": "ask-once",
    "save_processed": "ask-once",
    "confirm_output_format": "ask-once",  # alias → save_processed (confirm=true)
    "segment_data": "ask-once",
    "export_repo": "ask-once",
    "generate_code": "ask-once",    # alias → export_repo (code_only=true)
    "preprocess_neural": "ask-once",
    "bin_spikes": "ask-once",
    "batch_process": "ask-once",
    # always-ask — high-impact delegation
    "delegate_task": "always-ask",
    # builtin — tools with their own internal approval logic
    "terminal": "builtin",
    "process": "builtin",
}


class ToolPermissionController:
    """Per-tool permission tier enforcement.

    Thread-safe. Each session maintains its own set of granted tools
    (for ask-once tier). The controller is instantiated per AIAgent.
    """

    def __init__(self, config_overrides: Optional[dict] = None):
        self._tiers: dict[str, str] = dict(DEFAULT_TIERS)
        if config_overrides:
            for tool_name, tier in config_overrides.items():
                if tier in VALID_TIERS:
                    self._tiers[tool_name] = tier
        self._session_granted: dict[str, set[str]] = {}
        self._lock = threading.Lock()

    def get_tier(self, tool_name: str) -> str:
        """Return the effective tier for a tool. Unknown tools default to ask-once."""
        return self._tiers.get(tool_name, "ask-once")

    def check(self, tool_name: str, session_key: str) -> PermissionDecision:
        """Check if a tool call is permitted under the current tier.

        Returns a PermissionDecision indicating whether to allow, prompt, or block.
        """
        tier = self.get_tier(tool_name)

        if tier == "auto" or tier == "builtin":
            return PermissionDecision(
                allowed=True, needs_prompt=False,
                tier=tier, tool_name=tool_name,
            )

        if tier == "notify":
            return PermissionDecision(
                allowed=True, needs_prompt=False,
                tier=tier, tool_name=tool_name,
            )

        if tier == "ask-once":
            if self.is_session_granted(tool_name, session_key):
                return PermissionDecision(
                    allowed=True, needs_prompt=False,
                    tier=tier, tool_name=tool_name,
                )
            return PermissionDecision(
                allowed=False, needs_prompt=True,
                tier=tier, tool_name=tool_name,
            )

        if tier == "always-ask":
            return PermissionDecision(
                allowed=False, needs_prompt=True,
                tier=tier, tool_name=tool_name,
            )

        return PermissionDecision(
            allowed=False, needs_prompt=True,
            tier="ask-once", tool_name=tool_name,
        )

    def grant_session(self, tool_name: str, session_key: str) -> None:
        """Record that the user approved this tool for the current session."""
        with self._lock:
            self._session_granted.setdefault(session_key, set()).add(tool_name)

    def is_session_granted(self, tool_name: str, session_key: str) -> bool:
        """Check if a tool has been granted for the current session."""
        with self._lock:
            return tool_name in self._session_granted.get(session_key, set())

    def clear_session(self, session_key: str) -> None:
        """Remove all granted permissions for a session."""
        with self._lock:
            self._session_granted.pop(session_key, None)
