"""CLI handlers + dashboard REST wrappers for internalized-knowledge management.

Both the CLI subcommands and dashboard REST endpoints share these wrappers, so
behaviour stays identical and there's only one place to add features.
"""
from __future__ import annotations

import datetime as _dt
import json
from typing import Any, Dict, List, Optional

from easybci_agent.i18n import t
from easybci_cli import cli_output
from easybci_lib.constants import get_easybci_home, get_skills_dir
from easybci_lib.tools.neural_processing.research.knowledge_internalizer import (
    KnowledgeInternalizer,
)


def _load_internalizer() -> KnowledgeInternalizer:
    """Construct a ``KnowledgeInternalizer`` rooted at the active EASYBCI_HOME.

    Routing through ``get_easybci_home`` (CLAUDE.md: not ``Path.home()``) is
    what makes CLI subcommands honour ``EASYBCI_HOME`` overrides in tests +
    multi-profile setups.
    """
    try:
        log_path = get_easybci_home() / "internalized_knowledge.json"
        skills_dir = get_skills_dir() / "bci" / "paradigms"
        return KnowledgeInternalizer(log_path=log_path, skills_dir=skills_dir)
    except Exception:  # noqa: BLE001
        return KnowledgeInternalizer()


# ---------------------------------------------------------------- CLI handlers


def cmd_internalizations_list(args: Any, json_mode: bool = False) -> int:
    ki = _load_internalizer()
    entries = ki.get_all()
    if json_mode:
        out = [
            {
                "internalization_id": e.internalization_id,
                "skill_name": e.skill_name,
                "confidence": e.confidence,
                "internalized_at": e.internalized_at,
                "verified": e.verified,
                "legacy": not e.internalization_id,
            }
            for e in entries
        ]
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0
    if not entries:
        cli_output.print_info(t("internalization.list_empty"))
        return 0
    cli_output.print_info(t("internalization.list_header"))
    for e in entries:
        when = ""
        if e.internalized_at:
            when = _dt.datetime.utcfromtimestamp(e.internalized_at).strftime("%Y-%m-%d")
        flag = t("internalization.legacy_flag") if not e.internalization_id else ""
        id_col = e.internalization_id or "------"
        cli_output.print_info(
            f"{id_col:<10} {e.skill_name:<20} {e.confidence:<10.2f} {when:<12} {flag}"
        )
    return 0


def cmd_internalization_show(args: Any) -> int:
    ki = _load_internalizer()
    target_id = getattr(args, "id", None)
    if not target_id:
        cli_output.print_warning(t("internalization.missing_id_arg"))
        return 1
    for e in ki.get_all():
        if e.internalization_id == target_id:
            if getattr(args, "json", False):
                print(json.dumps(e.to_dict(), ensure_ascii=False, indent=2))
            else:
                cli_output.print_info(f"id:          {e.internalization_id}")
                cli_output.print_info(f"skill:       {e.skill_name}")
                cli_output.print_info(f"confidence:  {e.confidence}")
                cli_output.print_info(f"source:      {', '.join(e.source_urls)}")
                cli_output.print_info("--- content ---")
                print(e.content)
            return 0
    cli_output.print_warning(t("internalization.no_such_id", id=target_id))
    return 1


def cmd_revoke_internalization(args: Any) -> int:
    ki = _load_internalizer()
    target_id = getattr(args, "id", None)
    reason = getattr(args, "reason", "") or "(no reason given)"
    allow_unsafe = bool(getattr(args, "allow_unsafe", False))
    if not target_id:
        cli_output.print_warning(t("internalization.missing_id_arg"))
        return 1
    try:
        result = ki.revoke(internalization_id=target_id, reason=reason, allow_unsafe=allow_unsafe)
    except KeyError:
        cli_output.print_warning(t("internalization.no_such_id", id=target_id))
        return 1
    except RuntimeError as exc:
        cli_output.print_warning(str(exc))
        cli_output.print_info(t("internalization.hand_edited_hint"))
        return 2
    cli_output.print_info(t(
        "internalization.revoke_success",
        id=target_id,
        path=result.skill_path,
        lines=result.removed_lines,
    ))
    return 0


# ----------------------------------------------------- Dashboard REST wrappers


def list_internalizations_dict() -> List[Dict[str, Any]]:
    ki = _load_internalizer()
    return [
        {
            "internalization_id": e.internalization_id,
            "skill_name": e.skill_name,
            "confidence": e.confidence,
            "internalized_at": e.internalized_at,
            "verified": e.verified,
            "legacy": not e.internalization_id,
        }
        for e in ki.get_all()
    ]


def get_internalization_dict(internalization_id: str) -> Optional[Dict[str, Any]]:
    ki = _load_internalizer()
    for e in ki.get_all():
        if e.internalization_id == internalization_id:
            return e.to_dict()
    return None


def revoke_internalization_dict(
    internalization_id: str,
    reason: str,
    allow_unsafe: bool,
) -> Dict[str, Any]:
    ki = _load_internalizer()
    result = ki.revoke(
        internalization_id=internalization_id,
        reason=reason,
        allow_unsafe=allow_unsafe,
    )
    return {
        "skill_path": result.skill_path,
        "internalization_id": result.internalization_id,
        "removed_lines": result.removed_lines,
    }
