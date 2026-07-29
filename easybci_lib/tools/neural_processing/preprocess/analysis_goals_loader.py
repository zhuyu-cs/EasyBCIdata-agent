"""Load third-party analysis_goal yaml files into REGISTRY.

Path: ~/.easybci/skills/analysis_goals/*.yaml — one file per custom goal.

Conflict policy: builtin REGISTRY entries WIN. Conflicts are returned to the
caller so `easybci doctor goals` can surface them as warnings.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml

from easybci_lib.constants import get_easybci_home

logger = logging.getLogger(__name__)


@dataclass
class GoalConflict:
    name: str
    yaml_path: str
    reason: str


def _third_party_dir() -> Optional[Path]:
    try:
        d = get_easybci_home() / "skills" / "analysis_goals"
        return d if d.is_dir() else None
    except Exception:  # noqa: BLE001
        return None


def load_and_merge_third_party(
    registry: Dict[str, Any],
    spec_factory: Optional[Callable[..., Any]] = None,
) -> List[GoalConflict]:
    """Load every yaml under ~/.easybci/skills/analysis_goals/ and merge into
    ``registry``. Returns list of conflicts (third-party tried to claim a name
    that's already registered).

    ``spec_factory`` is an injectable factory — typically ``AnalysisGoalSpec``
    from ``analysis_goals``. Passing it in keeps this module free of a
    circular import on ``analysis_goals``; the default lazily resolves it
    from the sibling module at call time (executed only after the parent
    module has fully initialised, so the cycle is safe).
    """
    if spec_factory is None:
        from .analysis_goals import AnalysisGoalSpec
        spec_factory = AnalysisGoalSpec

    conflicts: List[GoalConflict] = []
    d = _third_party_dir()
    if d is None:
        return conflicts

    for yfile in sorted(d.glob("*.yaml")):
        try:
            data = yaml.safe_load(yfile.read_text(encoding="utf-8")) or {}
        except Exception as exc:  # noqa: BLE001
            logger.warning("failed to parse %s: %s", yfile, exc)
            continue
        name = str(data.get("name", "")).strip()
        if not name:
            conflicts.append(GoalConflict(name="", yaml_path=str(yfile), reason="missing 'name' field"))
            continue
        if name in registry:
            conflicts.append(GoalConflict(
                name=name, yaml_path=str(yfile),
                reason="builtin REGISTRY entry takes precedence",
            ))
            continue
        try:
            spec = spec_factory(
                name=name,
                display_name=dict(data.get("display_name") or {"en": name, "zh": name}),
                description=str(data.get("description", "")),
                inject_drop_bads=bool(data.get("inject_drop_bads", True)),
                inject_drop_nondata=bool(data.get("inject_drop_nondata", True)),
                allow_aggressive_notch=bool(data.get("allow_aggressive_notch", True)),
                allow_ica=bool(data.get("allow_ica", True)),
                produces_figures=bool(data.get("produces_figures", True)),
                crystallize_eligible=bool(data.get("crystallize_eligible", True)),
                notes=str(data.get("notes", "")),
            )
        except Exception as exc:  # noqa: BLE001
            conflicts.append(GoalConflict(
                name=name, yaml_path=str(yfile),
                reason=f"failed to build spec: {exc}",
            ))
            continue
        registry[name] = spec
        logger.info("registered third-party goal %s from %s", name, yfile)
    return conflicts
