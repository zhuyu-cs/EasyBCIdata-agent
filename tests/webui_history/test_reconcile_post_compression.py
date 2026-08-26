"""Reconcile safety once the JSON log may be a post-compression working state.

After the shrink change, session_<id>.json can hold FEWER rows than SQLite
(summary + recent turns), and its front row may be a synthetic compression
summary. reconcile_session_from_agent_log must:
  * no-op when the log is shorter than SQLite (never delete/rewrite SQLite);
  * still backfill the genuine dropped-flush case (log strictly longer AND a
    positional superset);
  * never append a misaligned tail from a shorter/summary-fronted log.
"""
from __future__ import annotations

import json
from pathlib import Path

from easybci_lib.session_reconcile import reconcile_session_from_agent_log


def _write_log(home: Path, sid: str, messages):
    d = home / "sessions"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"session_{sid}.json").write_text(
        json.dumps({"session_id": sid, "message_count": len(messages), "messages": messages}),
        encoding="utf-8",
    )


def test_shorter_log_is_noop(isolated_db, isolated_easybci_home):
    """Post-compression log shorter than SQLite → reconcile must not touch SQLite."""
    db = isolated_db
    sid = "s_short"
    db.create_session(session_id=sid, source="test")
    for i in range(6):  # SQLite holds the full pre-compression history
        db.append_message(session_id=sid, role="user", content=f"full{i}")
    # JSON log is the compressed working state: summary + 2 recent turns.
    _write_log(isolated_easybci_home, sid, [
        {"role": "system", "content": "[summary]"},
        {"role": "user", "content": "full4"},
        {"role": "user", "content": "full5"},
    ])
    appended = reconcile_session_from_agent_log(db, sid)
    assert appended == 0
    assert len(db.get_messages(sid)) == 6  # unchanged


def test_genuine_dropped_flush_still_backfills(isolated_db, isolated_easybci_home):
    """Real self-heal case: log is a strict positional superset → tail appended."""
    db = isolated_db
    sid = "s_heal"
    db.create_session(session_id=sid, source="test")
    for i in range(3):
        db.append_message(session_id=sid, role="user", content=f"m{i}")
    _write_log(isolated_easybci_home, sid, [
        {"role": "user", "content": "m0"},
        {"role": "user", "content": "m1"},
        {"role": "user", "content": "m2"},
        {"role": "user", "content": "m3"},  # dropped flush
        {"role": "user", "content": "m4"},
    ])
    appended = reconcile_session_from_agent_log(db, sid)
    assert appended == 2
    assert len(db.get_messages(sid)) == 5
