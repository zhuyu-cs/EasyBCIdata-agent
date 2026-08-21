#!/usr/bin/env python3
"""
Toolsets Module

This module provides a flexible system for defining and managing tool aliases/toolsets.
Toolsets allow you to group tools together for specific scenarios and can be composed
from individual tools or other toolsets.

Features:
- Define custom toolsets with specific tools
- Compose toolsets from other toolsets
- Built-in common toolsets for typical use cases
- Easy extension for new toolsets
- Support for dynamic toolset resolution

Usage:
    from easybci_lib.toolsets import get_toolset, resolve_toolset, get_all_toolsets
    
    # Get tools for a specific toolset
    tools = get_toolset("research")
    
    # Resolve a toolset to get all tool names (including from composed toolsets)
    all_tools = resolve_toolset("full_stack")
"""

from typing import List, Dict, Any, Set, Optional


# Neural-preprocessing orchestration helpers required by the
# pipeline orchestrator skill (inspect → propose → generate_code →
# … → export_repo → skill_manage). These were previously reachable ONLY via
# the CLI's unfiltered all-tools default (enabled_toolsets=None), so the WebUI
# (which resolves to the `easybci-webui` toolset) silently lost them and the
# agent fell back to free-writing a script instead of producing the standard
# mini-repo + crystallizing a proven pipeline. Defined once and shared across
# `neural`, `easybci-webui`, and `_EASYBCI_CORE_TOOLS` to prevent drift.
_NEURAL_ORCHESTRATION_TOOLS = [
    "inspect_neural",
    "deep_inspect",
    "inspect_detail",
    # suggest_pipeline / propose_pipeline are hidden aliases of plan_pipeline —
    # they share the same handler and schema. Only plan_pipeline is exposed to
    # the model to avoid ~3.4K tok of duplicate schema overhead. The aliases
    # remain registered (hidden=True) so dispatch still resolves them.
    "mark_proposal_confirmed",
    "generate_code",
    "compare_pipelines",
    "confirm_output_format",
    "research_preprocessing",
    "research_parameter",
    "batch_process_adaptive",
    "clarify",
    # register_io_loader — agent-authored loader plugins for formats the
    # built-in loaders can't read. MUST stay in this list (spread into
    # _EASYBCI_CORE_TOOLS + neural + easybci-webui) or the neural⊆cli subset
    # check drops the ENTIRE neural toolset. See CLAUDE.md sync rule.
    "register_io_loader",
    # register_analysis_goal — agent-authored analysis_goal for methodologies
    # not covered by the built-in REGISTRY. Same sync rule as register_io_loader.
    "register_analysis_goal",
    # `skill_manage` (write access to skills) is intentionally included so the
    # proven-pipeline flywheel can crystallize successful runs into reusable
    # skills on every platform, not just the CLI.
    "skill_manage",
]


# Shared tool list for CLI and API server toolsets.
_EASYBCI_CORE_TOOLS = [
    # Web
    "web_search", "web_extract",
    # Terminal + process management
    "terminal", "process",
    # File manipulation
    "read_file", "write_file", "patch", "search_files",
    # Vision
    "vision_analyze",
    # Skills
    "skills_list", "skill_view",
    # Agent-driven external skill search + install. Belongs to the
    # `skills` toolset alongside skill_manage; listed here so every code-capable
    # session (CLI + WebUI) can reach it. Flows through the same quarantine +
    # security-scan gate as the CLI; degrades gracefully offline (local-dir
    # source + intranet index). See install_skill tool + 02-install-skill-tool.md.
    "install_skill",
    # Planning & memory
    "pipeline_tracker", "memory",
    # Session history search
    "session_search",
    # Code execution + delegation
    "execute_code", "delegate_task",
    # Controlled dependency extension (general capability — request an exact-pinned
    # package instead of raw `pip install`; registered under the `dependency`
    # toolset, MUST be listed here too so every code-capable session can reach it).
    "request_dependency",
    # Neural/BCI data processing
    "inspect_data", "preprocess_neural", "quality_check",
    "segment_data", "save_processed", "plan_pipeline",
    "list_data", "export_repo",
    "bin_spikes", "batch_process",
    "import_reference",
    # Layout repair + resume (registered under the `neural` toolset — MUST be
    # listed here too, otherwise `neural` stops being a subset of the
    # `easybci-cli` / `easybci-webui` composites and `_get_platform_tools`'s
    # subset inference silently drops the ENTIRE neural toolset from sessions.
    # See tools_config._get_platform_tools.)
    "repair_layout", "resume_preprocessing",
    # Neural-preprocessing orchestration helpers (inspect_neural,
    # propose_pipeline, generate_code, skill_manage, clarify,
    # research_preprocessing, …) — shared so CLI and WebUI stay in sync.
    *_NEURAL_ORCHESTRATION_TOOLS,
]


# Core toolset definitions
# These can include individual tools or reference other toolsets
TOOLSETS = {
    # Basic toolsets - individual tool categories
    "web": {
        "description": "Web research and content extraction tools",
        "tools": ["web_search", "web_extract"],
        "includes": []  # No other toolsets included
    },
    
    "search": {
        "description": "Web search only (no content extraction/scraping)",
        "tools": ["web_search"],
        "includes": []
    },
    
    "vision": {
        "description": "Image analysis and vision tools",
        "tools": ["vision_analyze"],
        "includes": []
    },


    "terminal": {
        "description": "Terminal/command execution and process management tools",
        "tools": ["terminal", "process"],
        "includes": []
    },
    
    "moa": {
        "description": "Advanced reasoning and problem-solving tools",
        "tools": ["mixture_of_agents"],
        "includes": []
    },
    
    "skills": {
        "description": "Access and view skill documents with specialized instructions and knowledge",
        "tools": ["skills_list", "skill_view", "install_skill"],
        "includes": []
    },
    
    "file": {
        "description": "File manipulation tools: read, write, patch (with fuzzy matching), and search (content + files)",
        "tools": ["read_file", "write_file", "patch", "search_files"],
        "includes": []
    },

    "pipeline_tracker": {
        "description": "BCI data preprocessing pipeline step tracking",
        "tools": ["pipeline_tracker"],
        "includes": []
    },
    
    "memory": {
        "description": "Persistent memory across sessions (personal notes + user profile)",
        "tools": ["memory"],
        "includes": []
    },
    
    "session_search": {
        "description": "Search and recall past conversations with summarization",
        "tools": ["session_search"],
        "includes": []
    },
    
    "clarify": {
        "description": "Ask the user clarifying questions (multiple-choice or open-ended)",
        "tools": ["clarify"],
        "includes": []
    },
    
    "code_execution": {
        "description": "Run Python scripts that call tools programmatically (reduces LLM round trips)",
        "tools": ["execute_code"],
        "includes": []
    },

    "dependency": {
        "description": "Controlled dependency extension — install an exact-pinned "
                       "package (safer than raw terminal pip)",
        "tools": ["request_dependency"],
        "includes": []
    },
    
    "delegation": {
        "description": "Spawn subagents with isolated context for complex subtasks",
        "tools": ["delegate_task"],
        "includes": []
    },

    # "honcho" toolset removed — Honcho is now a memory provider plugin.
    # Tools are injected via MemoryManager, not the toolset system.

    "neural": {
        "description": "BCI/neural data processing (EEG, sEEG, ECoG, MEG, Spike) — inspect, preprocess, segment, export",
        "tools": [
            "inspect_data", "preprocess_neural", "quality_check",
            "segment_data", "save_processed", "plan_pipeline",
            "list_data", "export_repo",
            "bin_spikes", "batch_process",
            "import_reference",
            *_NEURAL_ORCHESTRATION_TOOLS,
        ],
        "includes": [],
    },

    # Scenario-specific toolsets
    
    "debugging": {
        "description": "Debugging and troubleshooting toolkit",
        "tools": ["terminal", "process"],
        "includes": ["web", "file"]  # For searching error messages and solutions, and file operations
    },
    
    "safe": {
        "description": "Safe toolkit without terminal access",
        "tools": [],
        "includes": ["web", "vision"]
    },
    
    # ==========================================================================
    # EasyBCI platform toolsets (CLI, WebUI API server)
    # ==========================================================================

    "easybci-webui": {
        "description": "WebUI backend — full agent tools accessible via HTTP/WebSocket",
        # Single source of truth: identical to easybci-cli. Avoids the
        # issubset() drift where a new neural tool registered via
        # registry.register(toolset="neural", ...) would silently kick the
        # entire neural toolset out of api_server's enabled_toolsets, because
        # the reverse-inference in tools_config._get_platform_tools requires
        # neural.tools.issubset(easybci-webui.tools).
        "tools": _EASYBCI_CORE_TOOLS,
        "includes": []
    },

    "easybci-cli": {
        "description": "Full interactive CLI toolset",
        "tools": _EASYBCI_CORE_TOOLS,
        "includes": []
    },
}



def get_toolset(name: str) -> Optional[Dict[str, Any]]:
    """
    Get a toolset definition by name.
    
    Args:
        name (str): Name of the toolset
        
    Returns:
        Dict: Toolset definition with description, tools, and includes
        None: If toolset not found
    """
    toolset = TOOLSETS.get(name)

    try:
        from easybci_lib.tools.registry import registry
    except Exception:
        return toolset if toolset else None

    if toolset:
        merged_tools = sorted(
            set(toolset.get("tools", []))
            | set(registry.get_tool_names_for_toolset(name))
        )
        return {**toolset, "tools": merged_tools}

    registry_toolset = name
    description = f"Plugin toolset: {name}"
    alias_target = registry.get_toolset_alias_target(name)

    if name not in _get_plugin_toolset_names():
        registry_toolset = alias_target
        if not registry_toolset:
            return None
        description = f"MCP server '{name}' tools"
    else:
        reverse_aliases = {
            canonical: alias
            for alias, canonical in _get_registry_toolset_aliases().items()
            if alias not in TOOLSETS
        }
        alias = reverse_aliases.get(name)
        if alias:
            description = f"MCP server '{alias}' tools"

    return {
        "description": description,
        "tools": registry.get_tool_names_for_toolset(registry_toolset),
        "includes": [],
    }


def resolve_toolset(name: str, visited: Set[str] = None) -> List[str]:
    """
    Recursively resolve a toolset to get all tool names.
    
    This function handles toolset composition by recursively resolving
    included toolsets and combining all tools.
    
    Args:
        name (str): Name of the toolset to resolve
        visited (Set[str]): Set of already visited toolsets (for cycle detection)
        
    Returns:
        List[str]: List of all tool names in the toolset
    """
    if visited is None:
        visited = set()
    
    # Special aliases that represent all tools across every toolset
    # This ensures future toolsets are automatically included without changes.
    if name in {"all", "*"}:
        all_tools: Set[str] = set()
        for toolset_name in get_toolset_names():
            # Use a fresh visited set per branch to avoid cross-branch contamination
            resolved = resolve_toolset(toolset_name, visited.copy())
            all_tools.update(resolved)
        return sorted(all_tools)

    # Check for cycles / already-resolved (diamond deps).
    # Silently return [] — either this is a diamond (not a bug, tools already
    # collected via another path) or a genuine cycle (safe to skip).
    if name in visited:
        return []

    visited.add(name)

    # Get toolset definition
    toolset = get_toolset(name)
    if not toolset:
        # Auto-generate a toolset for plugin platforms (easybci-<name>).
        # Gives them _EASYBCI_CORE_TOOLS plus any tools the plugin registered
        # into a toolset matching the platform name.
        if name.startswith("easybci-"):
            platform_name = name[len("easybci-"):]
            try:
                from services.gateway.platform_registry import platform_registry
                if platform_registry.is_registered(platform_name):
                    plugin_tools = set(_EASYBCI_CORE_TOOLS)
                    try:
                        from easybci_lib.tools.registry import registry
                        plugin_tools.update(
                            e.name for e in registry._tools.values()
                            if e.toolset == platform_name
                        )
                    except Exception:
                        pass
                    return list(plugin_tools)
            except Exception:
                pass

        return []

    # Collect direct tools
    tools = set(toolset.get("tools", []))

    # Recursively resolve included toolsets, sharing the visited set across
    # sibling includes so diamond dependencies are only resolved once and
    # cycle warnings don't fire multiple times for the same cycle.
    for included_name in toolset.get("includes", []):
        included_tools = resolve_toolset(included_name, visited)
        tools.update(included_tools)
    
    return sorted(tools)


def resolve_multiple_toolsets(toolset_names: List[str]) -> List[str]:
    """
    Resolve multiple toolsets and combine their tools.
    
    Args:
        toolset_names (List[str]): List of toolset names to resolve
        
    Returns:
        List[str]: Combined list of all tool names (deduplicated)
    """
    all_tools = set()
    
    for name in toolset_names:
        tools = resolve_toolset(name)
        all_tools.update(tools)
    
    return sorted(all_tools)


def _get_plugin_toolset_names() -> Set[str]:
    """Return toolset names registered by plugins (from the tool registry).

    These are toolsets that exist in the registry but not in the static
    ``TOOLSETS`` dict — i.e. they were added by plugins at load time.
    """
    try:
        from easybci_lib.tools.registry import registry
        return {
            toolset_name
            for toolset_name in registry.get_registered_toolset_names()
            if toolset_name not in TOOLSETS
        }
    except Exception:
        return set()


def _get_registry_toolset_aliases() -> Dict[str, str]:
    """Return explicit toolset aliases registered in the live registry."""
    try:
        from easybci_lib.tools.registry import registry
        return registry.get_registered_toolset_aliases()
    except Exception:
        return {}


def get_all_toolsets() -> Dict[str, Dict[str, Any]]:
    """
    Get all available toolsets with their definitions.

    Includes both statically-defined toolsets and plugin-registered ones.
    
    Returns:
        Dict: All toolset definitions
    """
    result = dict(TOOLSETS)
    aliases = _get_registry_toolset_aliases()
    for ts_name in _get_plugin_toolset_names():
        display_name = ts_name
        for alias, canonical in aliases.items():
            if canonical == ts_name and alias not in TOOLSETS:
                display_name = alias
                break
        if display_name in result:
            continue
        toolset = get_toolset(display_name)
        if toolset:
            result[display_name] = toolset
    return result


def get_toolset_names() -> List[str]:
    """
    Get names of all available toolsets (excluding aliases).

    Includes plugin-registered toolset names.
    
    Returns:
        List[str]: List of toolset names
    """
    names = set(TOOLSETS.keys())
    aliases = _get_registry_toolset_aliases()
    for ts_name in _get_plugin_toolset_names():
        for alias, canonical in aliases.items():
            if canonical == ts_name and alias not in TOOLSETS:
                names.add(alias)
                break
        else:
            names.add(ts_name)
    return sorted(names)




def validate_toolset(name: str) -> bool:
    """
    Check if a toolset name is valid.
    
    Args:
        name (str): Toolset name to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    # Accept special alias names for convenience
    if name in {"all", "*"}:
        return True
    if name in TOOLSETS:
        return True
    if name in _get_plugin_toolset_names():
        return True
    return name in _get_registry_toolset_aliases()


def create_custom_toolset(
    name: str,
    description: str,
    tools: List[str] = None,
    includes: List[str] = None
) -> None:
    """
    Create a custom toolset at runtime.
    
    Args:
        name (str): Name for the new toolset
        description (str): Description of the toolset
        tools (List[str]): Direct tools to include
        includes (List[str]): Other toolsets to include
    """
    TOOLSETS[name] = {
        "description": description,
        "tools": tools or [],
        "includes": includes or []
    }




def get_toolset_info(name: str) -> Dict[str, Any]:
    """
    Get detailed information about a toolset including resolved tools.
    
    Args:
        name (str): Toolset name
        
    Returns:
        Dict: Detailed toolset information
    """
    toolset = get_toolset(name)
    if not toolset:
        return None
    
    resolved_tools = resolve_toolset(name)
    
    return {
        "name": name,
        "description": toolset["description"],
        "direct_tools": toolset["tools"],
        "includes": toolset["includes"],
        "resolved_tools": resolved_tools,
        "tool_count": len(resolved_tools),
        "is_composite": bool(toolset["includes"])
    }




if __name__ == "__main__":
    print("Toolsets System Demo")
    print("=" * 60)
    
    print("\nAvailable Toolsets:")
    print("-" * 40)
    for name, toolset in get_all_toolsets().items():
        info = get_toolset_info(name)
        composite = "[composite]" if info["is_composite"] else "[leaf]"
        print(f"  {composite} {name:20} - {toolset['description']}")
        print(f"     Tools: {len(info['resolved_tools'])} total")
    
    print("\nToolset Resolution Examples:")
    print("-" * 40)
    for name in ["web", "terminal", "safe", "debugging"]:
        tools = resolve_toolset(name)
        print(f"\n  {name}:")
        print(f"    Resolved to {len(tools)} tools: {', '.join(sorted(tools))}")
    
    print("\nMultiple Toolset Resolution:")
    print("-" * 40)
    combined = resolve_multiple_toolsets(["web", "vision", "terminal"])
    print("  Combining ['web', 'vision', 'terminal']:")
    print(f"    Result: {', '.join(sorted(combined))}")
    
    print("\nCustom Toolset Creation:")
    print("-" * 40)
    create_custom_toolset(
        name="my_custom",
        description="My custom toolset for specific tasks",
        tools=["web_search"],
        includes=["terminal", "vision"]
    )
    custom_info = get_toolset_info("my_custom")
    print("  Created 'my_custom' toolset:")
    print(f"    Description: {custom_info['description']}")
    print(f"    Resolved tools: {', '.join(custom_info['resolved_tools'])}")
