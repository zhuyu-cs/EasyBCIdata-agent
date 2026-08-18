#!/usr/bin/env python3
"""Diagnose WebUI history loss: reconcile SQLite `messages` vs JSONL transcript.

Usage:
    venv/bin/python scripts/diagnose_history_persistence.py <session_id>
    venv/bin/python scripts/diagnose_history_persistence.py --all
    venv/bin/python scripts/diagnose_history_persistence.py --all --verbose

For each session it prints per-role counts from SQLite and from the JSONL
transcript, and flags sessions where JSONL has MORE user/assistant turns than
SQLite — the signature of a dropped flush (modes M1/M2/M3).

Why this reconciliation is authoritative: the gateway writes every new turn to
JSONL unconditionally (services/gateway/session.py:1063 `append_to_transcript`),
but only writes SQLite when `skip_db=False`. For WebUI sessions `skip_db` is
True (the agent's `_flush_messages_to_session_db` is trusted to have written
SQLite already, run.py:6374). So JSONL is the near-complete ground truth and
SQLite is what the WebUI actually reads back (web_server.py `GET
/api/sessions/{id}/messages`). JSONL > SQLite for user/assistant rows == the
turns a user sees vanish on refresh.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

# EASYBCI_HOME-aware imports — never hardcode ~/.easybci (CLAUDE.md).
from easybci_lib.constants import get_easybci_home
from easybci_lib.state import SessionDB

# Roles that a user actually sees in the WebUI transcript. tool/system rows are
# infrastructure and are intentionally filtered by the client
# (useConversation.ts transformApiMessage), so a JSONL>SQLite gap on THOSE is
# not user-visible loss — we only alarm on user/assistant.
_VISIBLE_ROLES = ("user", "assistant")


def _sessions_dir() -> Path:
    """Resolve the gateway transcript directory.

    Mirrors services/gateway/config.py:255 — sessions_dir defaults to
    ``get_easybci_home() / "sessions"``. A non-default config-relocated
    sessions_dir is out of scope for this offline diagnostic.
    """
    return get_easybci_home() / "sessions"


def _sqlite_role_counts(db: SessionDB, session_id: str) -> Counter:
    rows = db.get_messages(session_id)  # SELECT * ... ORDER BY id (state.py:1841)
    return Counter(r.get("role", "unknown") for r in rows)


def _jsonl_role_counts(session_id: str) -> Counter | None:
    """Read the JSONL transcript for a session, if present.

    Transcript path is exactly ``<sessions_dir>/<session_id>.jsonl``
    (services/gateway/session.py:1029 get_transcript_path). Returns None when
    no transcript file exists.
    """
    path = _sessions_dir() / f"{session_id}.jsonl"
    if not path.exists():
        return None
    counts: Counter = Counter()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        # session_meta entries carry no conversational role — skip them so they
        # don't inflate the "unknown" bucket.
        if obj.get("role") == "session_meta":
            continue
        nested = obj.get("message")
        role = obj.get("role") or (nested.get("role") if isinstance(nested, dict) else None) or "unknown"
        counts[role] += 1
    return counts


def _report(session_id: str, db: SessionDB, verbose: bool = False) -> bool:
    sq = _sqlite_role_counts(db, session_id)
    js = _jsonl_role_counts(session_id)
    suspicious = False
    lines: list[str] = []
    if js is not None:
        for role in _VISIBLE_ROLES:
            if js.get(role, 0) > sq.get(role, 0):
                lines.append(
                    f"  ⚠️  MISSING: JSONL has {js.get(role, 0)} '{role}' rows but "
                    f"SQLite only {sq.get(role, 0)} — {js.get(role, 0) - sq.get(role, 0)} dropped"
                )
                suspicious = True
    # Only print healthy sessions in verbose mode to keep --all output focused
    # on the sessions that actually lost rows.
    if suspicious or verbose:
        print(f"\n=== session {session_id} ===")
        print(f"  SQLite : {dict(sq)}")
        print(f"  JSONL  : {dict(js) if js is not None else '(no transcript file)'}")
        for ln in lines:
            print(ln)
    return suspicious


def _all_session_ids(db: SessionDB) -> list[str]:
    """Enumerate every session id, including child (compression/subagent) ones.

    list_sessions_rich returns dicts keyed by 'id' (state.py:1398 docstring).
    include_children=True so compression continuations (where flushed rows
    actually land) aren't hidden.
    """
    rows = db.list_sessions_rich(limit=100000, include_children=True)
    ids: list[str] = []
    for row in rows:
        sid = row.get("id") if isinstance(row, dict) else None
        if sid:
            ids.append(sid)
    return ids


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("session_id", nargs="?", help="session id to inspect")
    ap.add_argument("--all", action="store_true", help="scan every session")
    ap.add_argument("--verbose", action="store_true", help="also print healthy sessions")
    args = ap.parse_args()

    db = SessionDB()
    if args.all:
        any_bad = False
        n = 0
        for sid in _all_session_ids(db):
            n += 1
            if _report(sid, db, verbose=args.verbose):
                any_bad = True
        print(f"\nscanned {n} session(s)")
        print("⚠️ found sessions with dropped rows" if any_bad else "✅ no drops detected")
        return 1 if any_bad else 0
    if not args.session_id:
        ap.error("provide a session_id or --all")
    bad = _report(args.session_id, db, verbose=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
