"""Read-side self-heal: backfill SQLite from the agent's full JSON session log.

WebUI persistence has only two real sources — SQLite (agent flush via
run_agent._flush_messages_to_session_db) and the agent's full JSON log at
``<home>/sessions/session_<id>.json`` (run_agent._save_session_log, which has a
never-shrink guard). The WebUI ``/v1/runs`` SSE path never writes JSONL, and the
Dashboard read path (web_server.get_session_messages) reads SQLite only — so a
dropped flush is invisible on refresh even though the full JSON log still holds
every message.

This module reconciles on read: if the JSON log holds MORE rows than SQLite
AND is a positional superset (same rows in the same order, plus a longer tail),
append the missing tail into SQLite (idempotent by position) so the session
becomes whole. Since 2026-08 the JSON log MAY be a post-compression working
state that is SHORTER than SQLite (summary + recent turns) — in that case this
is a no-op and SQLite, which holds the full pre-compression history, stays
authoritative. Fail-open — any error leaves SQLite untouched and the caller
still returns whatever SQLite has.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from easybci_lib.constants import get_easybci_home

logger = logging.getLogger(__name__)


def _agent_log_path(session_id: str) -> Path:
    # Mirrors run_agent.py:1406 — logs_dir = get_easybci_home() / "sessions",
    # session_log_file = logs_dir / f"session_{session_id}.json".
    return get_easybci_home() / "sessions" / f"session_{session_id}.json"


def _load_agent_log_messages(session_id: str) -> List[Dict[str, Any]]:
    path = _agent_log_path(session_id)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    msgs = data.get("messages") if isinstance(data, dict) else None
    return msgs if isinstance(msgs, list) else []


def reconcile_session_from_agent_log(db, session_id: str) -> int:
    """Backfill SQLite from the agent JSON log when the log is longer.

    Returns the number of rows appended (0 = no-op / nothing to heal).

    Idempotent by position: only log rows *beyond* the count already present in
    SQLite are appended, so repeated reads never duplicate. This is a plain
    dropped-flush self-heal and fires ONLY when the log is strictly LONGER than
    SQLite; a shorter post-compression log is a no-op (SQLite is authoritative).
    Fail-open: any error returns the count appended so far and leaves the rest
    of SQLite untouched.
    """
    try:
        sql_rows = db.get_messages(session_id)
    except Exception:
        return 0
    log_rows = _load_agent_log_messages(session_id)
    if not log_rows or len(log_rows) <= len(sql_rows):
        return 0

    tail = log_rows[len(sql_rows):]
    appended = 0
    for msg in tail:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "unknown")
        try:
            db.append_message(
                session_id=session_id,
                role=role,
                content=msg.get("content"),
                tool_name=msg.get("tool_name"),
                tool_calls=msg.get("tool_calls"),
                tool_call_id=msg.get("tool_call_id"),
                finish_reason=msg.get("finish_reason"),
                reasoning=msg.get("reasoning") if role == "assistant" else None,
                reasoning_content=msg.get("reasoning_content") if role == "assistant" else None,
                reasoning_details=msg.get("reasoning_details") if role == "assistant" else None,
                tool_status=msg.get("tool_status") if role == "tool" else None,
                tool_duration=msg.get("tool_duration") if role == "tool" else None,
            )
            appended += 1
        except Exception as e:
            # Stop at the first failure — don't punch holes in ordering by
            # skipping a row and writing later ones.
            logger.debug("reconcile append halted for %s at row +%d: %s", session_id, appended, e)
            break
    return appended
