"""LLM auto-title collision retry ('all same name' secondary contributor).

The sessions.title UNIQUE index makes set_session_title raise when another
session already holds the exact title. The LLM-title path used to catch that
and give up (leaving the 2nd session NULL → "Untitled"). It now retries with a
" (N)" suffix, mirroring the heuristic path, so similar conversations get
distinct titles instead of collapsing to the placeholder.

We drive the real ``auto_title_session`` with a monkeypatched ``generate_title``
so no auxiliary LLM call is made.
"""
from __future__ import annotations

import easybci_agent.title_generator as tg


def test_llm_title_collision_gets_suffix(isolated_db, monkeypatch):
    db = isolated_db
    db.create_session(session_id="s1", source="test")
    db.create_session(session_id="s2", source="test")

    # Force both conversations to produce the SAME generated title.
    monkeypatch.setattr(tg, "generate_title", lambda *a, **k: "EEG Preprocessing")

    tg.auto_title_session(
        db, "s1", user_message="u1", assistant_response="a1",
    )
    tg.auto_title_session(
        db, "s2", user_message="u2", assistant_response="a2",
    )

    t1 = db.get_session_title("s1")
    t2 = db.get_session_title("s2")
    assert t1 == "EEG Preprocessing"
    assert t2 == "EEG Preprocessing (2)"  # suffixed, NOT NULL
    assert t1 != t2


def test_llm_title_none_leaves_prior_heuristic(isolated_db, monkeypatch):
    """When the aux LLM fails (generate_title → None), a prior heuristic title
    survives instead of being wiped to NULL."""
    db = isolated_db
    db.create_session(session_id="s3", source="test")
    db.set_session_title("s3", "heuristic first msg", source="heuristic")

    monkeypatch.setattr(tg, "generate_title", lambda *a, **k: None)
    tg.auto_title_session(db, "s3", user_message="u", assistant_response="a")

    assert db.get_session_title("s3") == "heuristic first msg"
