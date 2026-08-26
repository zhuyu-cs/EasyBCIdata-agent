"""Read-side title backfill (plan Task 3C / sidebar "Untitled" self-heal).

A heuristic/LLM title is written only on the run path. Sessions that never hit
it (legacy rows, reconcile-backfilled rows) keep title=NULL and show "Untitled"
in the sidebar even though the conversation is intact. ``backfill_missing_title``
derives one from the first user message on read: idempotent, never clobbers an
existing title, safe when there is no user message yet.
"""
from __future__ import annotations

from easybci_lib.state import SessionDB


def test_backfill_derives_from_first_user_message(isolated_db):
    db = isolated_db
    try:
        db.create_session(session_id="20260101_100000_notitle", source="test")
        db.append_message(
            session_id="20260101_100000_notitle",
            role="user",
            content="把我的 EEG 数据预处理一下做分类",
        )
        assert db.get_session_title("20260101_100000_notitle") is None
        title = db.backfill_missing_title("20260101_100000_notitle")
        assert title and "把我的 EEG" in title
        # Persisted.
        assert db.get_session_title("20260101_100000_notitle") == title
    finally:
        db.close()


def test_backfill_is_idempotent(isolated_db):
    db = isolated_db
    try:
        db.create_session(session_id="20260101_100000_x", source="test")
        db.append_message(session_id="20260101_100000_x", role="user", content="first message")
        first = db.backfill_missing_title("20260101_100000_x")
        second = db.backfill_missing_title("20260101_100000_x")
        assert first == second
    finally:
        db.close()


def test_backfill_never_clobbers_existing_title(isolated_db):
    db = isolated_db
    try:
        db.create_session(session_id="20260101_100000_named", source="test")
        db.set_session_title("20260101_100000_named", "My Manual Title", source="manual")
        db.append_message(
            session_id="20260101_100000_named", role="user", content="different first message",
        )
        assert db.backfill_missing_title("20260101_100000_named") == "My Manual Title"
    finally:
        db.close()


def test_backfill_safe_when_no_user_message(isolated_db):
    db = isolated_db
    try:
        db.create_session(session_id="20260101_100000_empty", source="test")
        assert db.backfill_missing_title("20260101_100000_empty") is None
    finally:
        db.close()


def test_get_first_user_message_ignores_assistant_rows(isolated_db):
    db = isolated_db
    try:
        db.create_session(session_id="20260101_100000_mix", source="test")
        db.append_message(session_id="20260101_100000_mix", role="assistant", content="system-ish")
        db.append_message(session_id="20260101_100000_mix", role="user", content="the real first")
        assert db.get_first_user_message("20260101_100000_mix") == "the real first"
    finally:
        db.close()
