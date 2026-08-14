#!/usr/bin/env python3
"""install_skill — agent-driven external skill search + install (Phase B).

The sanctioned alternative to waiting for a human to run ``easybci skills
install`` on the CLI. When the agent discovers it is missing a methodology, it
can ``search`` candidate skills across all configured sources and ``install`` a
chosen one — flowing through the *exact same* quarantine + security-scan gate as
the CLI (:func:`easybci_cli.skills_hub.do_install`). No new security bypass.

Air-gap degradation ladder (see ``improved_docs/plans/agent-self-extension/
02-install-skill-tool.md`` §0): search aggregates candidates across
public sources → the configurable local-dir source → intranet HTTP index, each
independently fail-open. When *no* source is reachable, the return carries
``degraded: true`` + a ``fix_hint`` naming three offline paths — never a silent
empty result (principle §二·五 #2: observable, never silent).

Belongs to the ``skills`` toolset (alongside ``skill_manage`` / ``skills_list`` /
``skill_view``), which is covered by ``_EASYBCI_CORE_TOOLS``.
"""
from __future__ import annotations

import json
import logging
import shutil
from typing import Any, Dict, List, Optional, Tuple

from easybci_lib.tools.registry import registry

logger = logging.getLogger(__name__)


INSTALL_SKILL_SCHEMA = {
    "name": "install_skill",
    "description": (
        "Search for and install external agent skills when you find you're "
        "missing a methodology. Two actions: action='search' (query the "
        "configured skill sources and return candidates, each with a "
        "trust_level for you to judge) and action='install' (fetch a chosen "
        "skill by name/identifier from a prior search, run it through the "
        "security-scan gate, and persist it). After installing, use skill_view "
        "to read it. Works offline too: if the public sources are unreachable "
        "(intranet / air-gapped), it falls back to a configurable local skill "
        "directory (EASYBCI_LOCAL_SKILLS_DIR / skills.local_source_dir) and an "
        "intranet HTTP index. If nothing is reachable, the result says so and "
        "tells you how to add a source or hand-author one with skill_manage."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["search", "install"],
                "description": "'search' to find candidates, 'install' to install one.",
            },
            "query": {
                "type": "string",
                "description": "Search keywords (action='search'), e.g. 'csp motor imagery'.",
            },
            "name": {
                "type": "string",
                "description": "Skill name/identifier to install (action='install'), "
                               "taken from a prior search candidate's 'identifier'.",
            },
            "category": {
                "type": "string",
                "description": "OPTIONAL install category bucket under ~/.easybci/skills/. "
                               "Defaults to the source's own category.",
            },
            "force": {
                "type": "boolean",
                "description": "OPTIONAL: reinstall if already installed. Does NOT "
                               "bypass the security scan for community skills.",
            },
        },
        "required": ["action"],
    },
}


# Categorization for the degradation surface. A source that returns candidates
# is "ok"; one that times out is "timeout"; the local-dir source with no
# configured directory is "unconfigured"; anything else that came back empty is
# "empty" (public sources unreachable on an air-gapped box land here too).
def _build_sources_status(
    sources: List[Any],
    source_counts: Dict[str, int],
    timed_out_ids: List[str],
) -> Dict[str, str]:
    status: Dict[str, str] = {}
    timed_out = set(timed_out_ids or [])
    for src in sources:
        sid = src.source_id()
        if sid in timed_out:
            status[sid] = "timeout"
        elif source_counts.get(sid, 0) > 0:
            status[sid] = "ok"
        elif sid == "local-dir" and getattr(src, "_dir", None) is None:
            status[sid] = "unconfigured"
        else:
            status[sid] = "empty"
    return status


def _search(query: str, limit: int = 10) -> Tuple[list, Dict[str, str]]:
    """Search all configured sources; return (candidates, sources_status).

    Uses parallel_search_sources (not unified_search) because it exposes
    per-source counts + timeouts, which is exactly what the degradation surface
    needs. Fail-open per source is already handled inside it.
    """
    from easybci_lib.tools.skills_hub import (
        GitHubAuth, create_source_router, parallel_search_sources,
    )

    sources = create_source_router(GitHubAuth())
    all_results, source_counts, timed_out_ids = parallel_search_sources(
        sources, query=query, per_source_limits={}, source_filter="all",
    )
    status = _build_sources_status(sources, source_counts, timed_out_ids)

    # Merge + de-dupe by identifier, cap to limit (mirrors unified_search intent).
    seen: set = set()
    merged = []
    for meta in all_results:
        key = getattr(meta, "identifier", None) or getattr(meta, "name", None)
        if key in seen:
            continue
        seen.add(key)
        merged.append(meta)
        if len(merged) >= limit:
            break
    return merged, status


def _meta_to_dict(meta: Any) -> Dict[str, Any]:
    return {
        "name": getattr(meta, "name", ""),
        "description": getattr(meta, "description", ""),
        "source": getattr(meta, "source", ""),
        "identifier": getattr(meta, "identifier", ""),
        "trust_level": getattr(meta, "trust_level", "community"),
        "tags": list(getattr(meta, "tags", []) or []),
    }


_DEGRADED_FIX_HINT = (
    "No skill source returned candidates (likely offline / intranet / air-gapped). "
    "Three offline options: (a) drop skill bundles into the local skill directory "
    "($EASYBCI_LOCAL_SKILLS_DIR or config skills.local_source_dir, default "
    "~/.easybci/local-skills/) and search again; (b) point at an intranet HTTP "
    "index and set security.allow_private_urls=true (or EASYBCI_ALLOW_PRIVATE_URLS=1); "
    "(c) hand-author the skill locally with skill_manage(action='create')."
)


def _resolve_meta_and_bundle(identifier: str, sources: List[Any]):
    """Inlined equivalent of the CLI's _resolve_source_meta_and_bundle.

    (Inlined rather than imported: the CLI helper lives in the easybci_cli layer,
    and importing up from easybci_lib would be a layering inversion.)
    Walks sources, returns (meta, bundle, matched_source); each source's
    inspect/fetch is guarded so one failing source doesn't abort resolution.
    """
    meta = None
    bundle = None
    matched = None
    for src in sources:
        if meta is None:
            try:
                meta = src.inspect(identifier)
                if meta:
                    matched = src
            except Exception:
                meta = None
        try:
            bundle = src.fetch(identifier)
        except Exception:
            bundle = None
        if bundle:
            matched = src
            if meta is None:
                try:
                    meta = src.inspect(identifier)
                except Exception:
                    meta = None
            break
    return meta, bundle, matched


def _handle_install_skill(args, **kw):
    """search | install external skills. Mirrors CLI do_install security gate.

    Never raises into the agent loop — all failures are returned as JSON.
    """
    if not isinstance(args, dict):
        return json.dumps({"success": False, "error": "invalid args"})
    action = (args.get("action") or "install").strip()

    try:
        from easybci_lib.tools.skills_hub import (
            GitHubAuth, create_source_router, ensure_hub_dirs,
            quarantine_bundle, install_from_quarantine, HubLockFile,
            append_audit_log,
        )
        from easybci_lib.tools.skills_guard import (
            scan_skill, should_allow_install, format_scan_report,
        )
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"success": False, "error": f"skills hub unavailable: {exc}"})

    # ----- search --------------------------------------------------------
    if action == "search":
        query = (args.get("query") or "").strip()
        if not query:
            return json.dumps({
                "success": False, "error": "query required for action='search'",
                "fix_hint": "pass query='<keywords>' describing the methodology you need",
            })
        try:
            metas, sources_status = _search(query, limit=int(args.get("limit", 10)))
        except Exception as exc:  # noqa: BLE001
            logger.exception("install_skill search failed")
            return json.dumps({"success": False, "error": f"{type(exc).__name__}: {exc}"})

        candidates = [_meta_to_dict(m) for m in metas]
        degraded = len(candidates) == 0
        payload: Dict[str, Any] = {
            "success": True,
            "candidates": candidates,
            "sources_status": sources_status,
            "degraded": degraded,
        }
        if degraded:
            payload["fix_hint"] = _DEGRADED_FIX_HINT
        return json.dumps(payload, default=str)

    # ----- install -------------------------------------------------------
    identifier = (args.get("name") or args.get("identifier") or "").strip()
    category = (args.get("category") or "").strip()
    force = bool(args.get("force"))
    if not identifier:
        return json.dumps({
            "success": False, "error": "name required for action='install'",
            "fix_hint": "pass name='<identifier>' from a prior install_skill search candidate",
        })

    try:
        ensure_hub_dirs()
        sources = create_source_router(GitHubAuth())
        meta, bundle, _matched = _resolve_meta_and_bundle(identifier, sources)
        if bundle is None:
            return json.dumps({
                "success": False, "stage": "fetch",
                "error": f"skill {identifier!r} not found from any reachable source",
                "fix_hint": "run install_skill action='search' first, then install by "
                            "the candidate's exact 'identifier'; if offline, see the "
                            "search result's fix_hint for local/intranet options",
            })

        # Auto-detect category for official / local-dir skills (identifier like
        # "official/cat/skill" or "local/cat/skill").
        if not category and bundle.source in ("official", "local-dir"):
            id_parts = bundle.identifier.split("/")
            if len(id_parts) >= 3:
                category = id_parts[1]

        # Already-installed check
        lock = HubLockFile()
        if lock.get_installed(bundle.name) and not force:
            return json.dumps({
                "success": False, "stage": "already_installed",
                "error": f"{bundle.name!r} is already installed",
                "fix_hint": "pass force=true to reinstall, or just skill_view it",
            })

        # Security gate — identical to CLI do_install.
        try:
            q_path = quarantine_bundle(bundle)
        except ValueError as exc:
            append_audit_log("BLOCKED", bundle.name, bundle.source,
                             bundle.trust_level, "invalid_path", str(exc))
            return json.dumps({
                "success": False, "stage": "quarantine", "error": str(exc),
                "fix_hint": "the bundle contained an unsafe file path; do not force",
            })

        scan_source = getattr(bundle, "identifier", "") or getattr(meta, "identifier", "") or identifier
        result = scan_skill(q_path, source=scan_source)
        allowed, reason = should_allow_install(result, force=force)
        # should_allow_install returns (None, reason) for a "needs confirm"
        # decision; `not None` is True, so that correctly lands here too.
        if not allowed:
            shutil.rmtree(q_path, ignore_errors=True)
            append_audit_log("BLOCKED", bundle.name, bundle.source,
                             bundle.trust_level, result.verdict,
                             f"{len(result.findings)}_findings")
            return json.dumps({
                "success": False, "stage": "scan", "error": reason,
                "scan_report": format_scan_report(result),
                "fix_hint": "skill flagged by the security scan; do not force unless "
                            "you have reviewed the files and trust the source",
            }, default=str)

        install_dir = install_from_quarantine(q_path, bundle.name, category, bundle, result)
    except Exception as exc:  # noqa: BLE001
        logger.exception("install_skill install failed")
        return json.dumps({"success": False, "error": f"{type(exc).__name__}: {exc}"})

    return json.dumps({
        "success": True,
        "name": bundle.name,
        "source": bundle.source,
        "trust_level": bundle.trust_level,
        "installed_path": str(install_dir),
        "next_action": {
            "next_tool": "skill_view",
            "must_present": True,
            "reason": "skill_installed",
            "hint": f"Skill '{bundle.name}' installed. Use skill_view('{bundle.name}') "
                    "to read it, then apply the methodology.",
        },
    }, default=str)


registry.register(
    name="install_skill",
    toolset="skills",
    schema=INSTALL_SKILL_SCHEMA,
    handler=_handle_install_skill,
    emoji="\U0001f4e6",  # 📦
)
