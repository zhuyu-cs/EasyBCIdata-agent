"""System prompt builder — assembles the three-tier prompt (stable/context/volatile).

Extracted from run_agent.py AIAgent._build_system_prompt_parts() to support
the composition + delegation decomposition of the monolithic agent class.
"""

import os
from typing import Any, Dict, List, Optional

from easybci_agent.prompt_builder import (
    DEFAULT_AGENT_IDENTITY,
    EASYBCI_AGENT_HELP_GUIDANCE,
    GOOGLE_MODEL_OPERATIONAL_GUIDANCE,
    MEMORY_GUIDANCE,
    NEURAL_TOOL_ROUTING,
    OPENAI_MODEL_EXECUTION_GUIDANCE,
    PLATFORM_HINTS,
    SESSION_SEARCH_GUIDANCE,
    SOURCE_DATA_IMMUTABILITY_CONSTRAINT,
    TOOL_USE_ENFORCEMENT_GUIDANCE,
    TOOL_USE_ENFORCEMENT_MODELS,
    WORK_DIRECTORY_CONSTRAINT,
    WORKFLOW_COMPLIANCE_CONSTRAINT,
    REPRODUCIBILITY_SEED_CONSTRAINT,
    OUTPUT_FORMAT_CONSTRAINT,
    SKILL_IMMUTABILITY_CONSTRAINT,
    EVIDENCE_DRIVEN_PARAMS_CONSTRAINT,
    CODE_STYLE_CONSTRAINT,
    _build_layout_contract_block,
    build_context_files_prompt,
    build_environment_hints,
    build_skills_system_prompt,
    load_soul_md,
)
from easybci_lib.model_tools import get_toolset_for_tool


class SystemPromptBuilder:
    """Assembles the three-tier system prompt for AIAgent.

    Holds no state of its own beyond the agent reference. All config is
    read lazily from the agent instance so the builder always reflects
    the current agent state.
    """

    def __init__(self, agent: Any):
        self._agent = agent

    def build_parts(self, system_message: str = None) -> Dict[str, str]:
        """Assemble the system prompt as three ordered tiers.

        Returns a dict with keys ``stable``, ``context``, ``volatile``.
        """
        agent = self._agent

        # ── Stable tier ────────────────────────────────────────────────
        stable_parts: List[str] = []

        _soul_loaded = False
        if agent.load_soul_identity or not agent.skip_context_files:
            _soul_content = load_soul_md()
            if _soul_content:
                stable_parts.append(_soul_content)
                _soul_loaded = True

        if not _soul_loaded:
            stable_parts.append(DEFAULT_AGENT_IDENTITY)

        # Source data immutability — unconditional, always injected
        stable_parts.append(SOURCE_DATA_IMMUTABILITY_CONSTRAINT)

        # Work directory isolation — unconditional, always injected
        stable_parts.append(WORK_DIRECTORY_CONSTRAINT)

        # Reproducibility seed — unconditional, always injected
        stable_parts.append(REPRODUCIBILITY_SEED_CONSTRAINT)

        # Final output format must be AI-ready — unconditional, always injected
        stable_parts.append(OUTPUT_FORMAT_CONSTRAINT)

        # Skill library is read-only except proven-pipelines — unconditional
        stable_parts.append(SKILL_IMMUTABILITY_CONSTRAINT)

        # Code style for generated Python — unconditional, always injected
        stable_parts.append(CODE_STYLE_CONSTRAINT)

        # Workflow compliance — work_dir sealed, must use tool chain
        if "batch_process_adaptive" in agent.valid_tool_names or "preprocess_neural" in agent.valid_tool_names:
            stable_parts.append(WORKFLOW_COMPLIANCE_CONSTRAINT)

        # Evidence-driven parameter resolution — only when neural tools present
        if "research_parameter" in agent.valid_tool_names or "propose_pipeline" in agent.valid_tool_names:
            stable_parts.append(EVIDENCE_DRIVEN_PARAMS_CONSTRAINT)

        # Neural tool routing guidance — injected when neural tools are available
        if "inspect_neural" in agent.valid_tool_names or "inspect_data" in agent.valid_tool_names:
            stable_parts.append(NEURAL_TOOL_ROUTING)

        # CLI help guidance for self-configuration questions
        if "skill_view" in agent.valid_tool_names:
            stable_parts.append(EASYBCI_AGENT_HELP_GUIDANCE)

        # Tool-aware behavioral guidance
        tool_guidance: List[str] = []
        if "memory" in agent.valid_tool_names:
            tool_guidance.append(MEMORY_GUIDANCE)
        if "session_search" in agent.valid_tool_names:
            tool_guidance.append(SESSION_SEARCH_GUIDANCE)
        if tool_guidance:
            stable_parts.append(" ".join(tool_guidance))

        # Tool-use enforcement
        if agent.valid_tool_names:
            _enforce = agent._tool_use_enforcement
            _inject = False
            if _enforce is True or (isinstance(_enforce, str) and _enforce.lower() in {"true", "always", "yes", "on"}):
                _inject = True
            elif _enforce is False or (isinstance(_enforce, str) and _enforce.lower() in {"false", "never", "no", "off"}):
                _inject = False
            elif isinstance(_enforce, list):
                model_lower = (agent.model or "").lower()
                _inject = any(p.lower() in model_lower for p in _enforce if isinstance(p, str))
            else:
                model_lower = (agent.model or "").lower()
                _inject = any(p in model_lower for p in TOOL_USE_ENFORCEMENT_MODELS)
            if _inject:
                stable_parts.append(TOOL_USE_ENFORCEMENT_GUIDANCE)
                _model_lower = (agent.model or "").lower()
                if "gemini" in _model_lower or "gemma" in _model_lower:
                    stable_parts.append(GOOGLE_MODEL_OPERATIONAL_GUIDANCE)
                if "gpt" in _model_lower:
                    stable_parts.append(OPENAI_MODEL_EXECUTION_GUIDANCE)

        has_skills_tools = any(name in agent.valid_tool_names for name in ['skills_list', 'skill_view', 'skill_manage'])
        if has_skills_tools:
            avail_toolsets = {
                toolset
                for toolset in (
                    get_toolset_for_tool(tool_name) for tool_name in agent.valid_tool_names
                )
                if toolset
            }
            skills_prompt = build_skills_system_prompt(
                available_tools=agent.valid_tool_names,
                available_toolsets=avail_toolsets,
            )
        else:
            skills_prompt = ""
        if skills_prompt:
            stable_parts.append(skills_prompt)

        # Alibaba model identity workaround
        if agent.provider == "alibaba":
            _model_short = agent.model.split("/")[-1] if "/" in agent.model else agent.model
            stable_parts.append(
                f"You are powered by the model named {_model_short}. "
                f"The exact model ID is {agent.model}. "
                f"When asked what model you are, always answer based on this information, "
                f"not on any model name returned by the API."
            )

        _env_hints = build_environment_hints()
        if _env_hints:
            stable_parts.append(_env_hints)

        # Layout Contract block — only present when a preprocess work_dir
        # is active for this thread. Kept in the stable tier so it survives
        # context compression alongside the other environment guidance.
        _layout_block = _build_layout_contract_block()
        if _layout_block:
            stable_parts.append(_layout_block)

        platform_key = (agent.platform or "").lower().strip()
        if platform_key in PLATFORM_HINTS:
            stable_parts.append(PLATFORM_HINTS[platform_key])
        elif platform_key:
            try:
                from services.gateway.platform_registry import platform_registry
                _entry = platform_registry.get(platform_key)
                if _entry and _entry.platform_hint:
                    stable_parts.append(_entry.platform_hint)
            except Exception:
                pass

        # ── Context tier (cwd-dependent, may change between sessions) ─
        context_parts: List[str] = []

        if system_message is not None:
            context_parts.append(system_message)

        if not agent.skip_context_files:
            _context_cwd = os.getenv("TERMINAL_CWD") or None
            context_files_prompt = build_context_files_prompt(
                cwd=_context_cwd, skip_soul=_soul_loaded)
            if context_files_prompt:
                context_parts.append(context_files_prompt)

        # ── Volatile tier (changes per session/turn — never cached) ───
        volatile_parts: List[str] = []

        if agent._memory_store:
            if agent._memory_enabled:
                mem_block = agent._memory_store.format_for_system_prompt("memory")
                if mem_block:
                    volatile_parts.append(mem_block)
            if agent._user_profile_enabled:
                user_block = agent._memory_store.format_for_system_prompt("user")
                if user_block:
                    volatile_parts.append(user_block)

        if agent._memory_manager:
            try:
                _ext_mem_block = agent._memory_manager.build_system_prompt()
                if _ext_mem_block:
                    volatile_parts.append(_ext_mem_block)
            except Exception:
                pass

        from easybci_lib.time_utils import now as _easybci_now
        now = _easybci_now()
        timestamp_line = f"Conversation started: {now.strftime('%A, %B %d, %Y %I:%M %p')}"
        if agent.pass_session_id and agent.session_id:
            timestamp_line += f"\nSession ID: {agent.session_id}"
        if agent.model:
            timestamp_line += f"\nModel: {agent.model}"
        if agent.provider:
            timestamp_line += f"\nProvider: {agent.provider}"
        volatile_parts.append(timestamp_line)

        return {
            "stable":   "\n\n".join(p.strip() for p in stable_parts   if p and p.strip()),
            "context":  "\n\n".join(p.strip() for p in context_parts  if p and p.strip()),
            "volatile": "\n\n".join(p.strip() for p in volatile_parts if p and p.strip()),
        }

    def build(self, system_message: str = None) -> str:
        """Join all tiers into a single system prompt string."""
        parts = self.build_parts(system_message=system_message)
        return "\n\n".join(p for p in (parts["stable"], parts["context"], parts["volatile"]) if p)

    def invalidate(self) -> None:
        """Clear cached prompt and reload memory from disk."""
        self._agent._cached_system_prompt = None
        if self._agent._memory_store:
            self._agent._memory_store.load_from_disk()
