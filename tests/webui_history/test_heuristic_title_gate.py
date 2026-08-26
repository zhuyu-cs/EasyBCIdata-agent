"""Heuristic-title gate fix (WebUI 'Untitled Session' primary root cause).

The instant heuristic title is written by APIServerAdapter._maybe_set_heuristic_title
on the FIRST user turn. The old gate counted ANY user message in
conversation_history and bailed if >0 — but the WebUI includes the CURRENT
user message in that history, so prior_user was always ≥1 and the heuristic
title NEVER fired on the WebUI path, leaving every session NULL → all showing
"Untitled Session". The fix discounts the current turn before counting.

These tests drive the real method with a minimal adapter shell whose
_ensure_session_db returns an isolated DB, so we assert the DB end-state.
"""
from __future__ import annotations

import pytest

from services.gateway.platforms.api_server import APIServerAdapter


def _make_first_message(role="user", content="预处理我的 EEG 数据做分类"):
    return {"role": role, "content": content}


@pytest.fixture
def adapter_with_db(isolated_db):
    """A bare adapter shell wired to the isolated DB for title writes."""
    adapter = APIServerAdapter.__new__(APIServerAdapter)
    adapter._ensure_session_db = lambda: isolated_db  # type: ignore[attr-defined]
    return adapter, isolated_db


def test_webui_first_turn_writes_heuristic_title(adapter_with_db):
    """WebUI includes the current user msg in history → title must still fire."""
    adapter, db = adapter_with_db
    msg = "预处理我的 EEG 数据做分类"
    adapter._maybe_set_heuristic_title(
        user_message=msg,
        conversation_history=[_make_first_message(content=msg)],  # incl current
        session_id="20260813_100000_webui1",
    )
    title = db.get_session_title("20260813_100000_webui1")
    assert title is not None and "EEG" in title


def test_native_first_turn_writes_heuristic_title(adapter_with_db):
    """Native path passes empty history on turn 1 → title fires."""
    adapter, db = adapter_with_db
    adapter._maybe_set_heuristic_title(
        user_message="hello world",
        conversation_history=[],
        session_id="20260813_100000_native1",
    )
    assert db.get_session_title("20260813_100000_native1") == "hello world"


def test_second_turn_does_not_write(adapter_with_db):
    """A genuine 2nd user turn (prior turn present) must NOT (re)fire."""
    adapter, db = adapter_with_db
    # WebUI turn 2: history holds prior user + assistant + the current user msg.
    adapter._maybe_set_heuristic_title(
        user_message="second question",
        conversation_history=[
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": "second question"},
        ],
        session_id="20260813_100000_turn2",
    )
    # No title written because this is not the first turn.
    assert db.get_session_title("20260813_100000_turn2") is None


def test_distinct_first_messages_give_distinct_titles(adapter_with_db):
    """Two different chats must not collapse to the same placeholder."""
    adapter, db = adapter_with_db
    adapter._maybe_set_heuristic_title(
        user_message="classify motor imagery EEG",
        conversation_history=[{"role": "user", "content": "classify motor imagery EEG"}],
        session_id="20260813_100000_a",
    )
    adapter._maybe_set_heuristic_title(
        user_message="source localization for MEG",
        conversation_history=[{"role": "user", "content": "source localization for MEG"}],
        session_id="20260813_100000_b",
    )
    ta = db.get_session_title("20260813_100000_a")
    tb = db.get_session_title("20260813_100000_b")
    assert ta and tb and ta != tb
