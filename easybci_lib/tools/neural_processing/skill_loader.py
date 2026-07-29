"""Skill Loader — auto-discovery and registration of custom skills on startup.

Scans the processing/custom/ directory for .py files with SKILL_META dicts
and registers them in the tool registry. Called during Orchestrator initialization
to ensure skills persist across restarts.

Also handles versioned skills (func_name_v{N}.py) — latest version is registered
by default.
"""

import importlib.util
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

_CUSTOM_DIR = Path(__file__).parent / "custom"
_INDEX_FILE = _CUSTOM_DIR / "SKILLS_INDEX.yaml"
_loaded = False


def ensure_custom_skills_loaded():
    """Load and register all custom skills (idempotent — safe to call multiple times)."""
    global _loaded
    if _loaded:
        return
    _loaded = True
    discover_and_register()


def discover_and_register() -> List[str]:
    """Scan custom skills directory and register each in the tool registry.

    Returns list of registered tool names.
    """
    if not _CUSTOM_DIR.exists():
        _CUSTOM_DIR.mkdir(parents=True, exist_ok=True)
        return []

    registered = []
    skills = _discover_skill_files()

    for skill_info in skills:
        try:
            _register_custom_skill(skill_info)
            registered.append(skill_info["tool_name"])
            logger.info("Loaded custom skill: %s", skill_info["tool_name"])
        except Exception as e:
            logger.warning("Failed to load custom skill '%s': %s", skill_info["file"], e)

    return registered


def _discover_skill_files() -> List[Dict[str, Any]]:
    """Find all custom skill .py files and extract metadata.

    Handles versioning: if func_name_v1.py and func_name_v2.py exist,
    only the latest version is registered as the primary tool.
    """
    skills: Dict[str, Dict[str, Any]] = {}  # base_name -> best version info

    for f in sorted(_CUSTOM_DIR.glob("*.py")):
        if f.name.startswith("_"):
            continue

        meta = _extract_skill_meta(f)
        if meta is None:
            continue

        func_name = meta.get("func_name", f.stem)
        # Parse version from filename: {name}_v{N}.py
        base_name, version = _parse_version(f.stem, func_name)

        existing = skills.get(base_name)
        if existing is None or version > existing.get("version", 1):
            skills[base_name] = {
                "file": str(f),
                "func_name": func_name,
                "base_name": base_name,
                "version": version,
                "meta": meta,
                "tool_name": f"custom_{base_name}",
            }

    return list(skills.values())


def _extract_skill_meta(filepath: Path) -> Optional[Dict[str, Any]]:
    """Load a .py file and extract its SKILL_META dict."""
    try:
        spec = importlib.util.spec_from_file_location(filepath.stem, str(filepath))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        meta = getattr(module, "SKILL_META", None)
        if isinstance(meta, dict) and "name" in meta:
            return meta
    except Exception as e:
        logger.debug("Cannot extract SKILL_META from %s: %s", filepath, e)
    return None


def _register_custom_skill(skill_info: Dict[str, Any]):
    """Register a custom skill in the tool registry."""
    from core.registry import registry

    meta = skill_info["meta"]
    func_name = skill_info["func_name"]
    filepath = skill_info["file"]
    tool_name = skill_info["tool_name"]
    description = meta.get("description", f"Custom skill: {func_name}")
    params = meta.get("params", {})

    def _handler(data_path: str, modality: str = "auto", **kwargs) -> dict:
        from easybci_lib.tools.neural_processing.io import load_neural

        spec = importlib.util.spec_from_file_location(func_name, filepath)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        skill_func = getattr(mod, func_name)

        data_dict = load_neural(data_path, modality=modality)
        result = skill_func(data_dict, **kwargs)
        return {
            "success": True,
            "data_shape": list(result["data"].shape),
            "frequency": result["frequency"],
            "channels": result.get("channels", []),
        }

    schema = {
        "type": "object",
        "properties": {
            "data_path": {"type": "string", "description": "Path to neural data file"},
            "modality": {"type": "string", "enum": ["auto", "eeg", "seeg", "ecog", "meg"]},
        },
        "required": ["data_path"],
    }
    for k, v in params.items():
        schema["properties"][k] = {"type": "string", "description": f"Parameter: {k}"}

    registry.register(
        name=tool_name,
        category="custom",
        description=f"[Custom] {description}",
        schema=schema,
        handler=_handler,
    )


def _parse_version(stem: str, func_name: str) -> tuple:
    """Parse version from filename stem. Returns (base_name, version_int)."""
    import re
    match = re.match(r"^(.+)_v(\d+)$", stem)
    if match:
        return match.group(1), int(match.group(2))
    return stem, 1


def save_to_index(skill_meta: Dict[str, Any], filepath: str):
    """Add or update a skill entry in SKILLS_INDEX.yaml."""
    _CUSTOM_DIR.mkdir(parents=True, exist_ok=True)

    index = _load_index()
    entries = index.get("custom_skills", [])

    func_name = skill_meta.get("func_name", "")
    # Update existing or append
    found = False
    for entry in entries:
        if entry.get("func_name") == func_name:
            entry["file"] = filepath
            entry["description"] = skill_meta.get("description", "")
            entry["version"] = entry.get("version", 0) + 1
            entry["updated_at"] = skill_meta.get("created_at", "")
            found = True
            break

    if not found:
        entries.append({
            "func_name": func_name,
            "name": skill_meta.get("name", func_name),
            "description": skill_meta.get("description", ""),
            "file": filepath,
            "version": 1,
            "created_at": skill_meta.get("created_at", ""),
            "params": skill_meta.get("params", {}),
        })

    index["custom_skills"] = entries
    _save_index(index)


def _load_index() -> Dict[str, Any]:
    """Load SKILLS_INDEX.yaml or return empty structure."""
    if _INDEX_FILE.exists():
        try:
            with open(_INDEX_FILE, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                return data if isinstance(data, dict) else {}
        except Exception:
            return {}
    return {}


def _save_index(data: Dict[str, Any]):
    """Write SKILLS_INDEX.yaml."""
    with open(_INDEX_FILE, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


def list_custom_skills() -> List[Dict[str, str]]:
    """List all registered custom skills from the index."""
    index = _load_index()
    return index.get("custom_skills", [])
