"""Helper to build the 'Retracted citations' banner that gets prepended to
plan/reasoning.md when a pipeline uses default values whose origin is marked
retracted/withdrawn in the most recent citation_audit log.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Set, Tuple

from .citation_audit import CitationAuditLog
from .citation_checker import CitationCheckResult


_FLAGGED_STATUSES = {"retracted", "withdrawn", "revised"}


def latest_flagged_citation_ids(audit_log_path: Path) -> Set[str]:
    """Return citation_ids whose most-recent audit event is retracted/withdrawn/revised."""
    log = CitationAuditLog(path=audit_log_path)
    events = log.read_all()
    latest: dict[str, CitationCheckResult] = {}
    for e in events:
        latest[e.citation_id] = e
    return {cid for cid, ev in latest.items() if ev.status in _FLAGGED_STATUSES}


def build_banner(flagged_in_use: List[Tuple[str, str]]) -> str:
    """flagged_in_use = [(operator_name, citation_label), ...]
    Returns a markdown banner block. Empty string if nothing to warn about.
    """
    if not flagged_in_use:
        return ""
    lines = ["> ⚠ **Retracted / withdrawn citations in active defaults**", ">"]
    for op, label in flagged_in_use:
        lines.append(f"> - `{op}` default cites *{label}* — flagged by `easybci registry check`")
    lines.append(">")
    lines.append("> Review with `easybci registry check` and update via the parameter_uncertainty yaml.")
    return "\n".join(lines) + "\n\n"
