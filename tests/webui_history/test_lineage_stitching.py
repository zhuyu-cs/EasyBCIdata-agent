"""Lineage-aware read stitching of pre-fix split sessions (plan Task 3 read-side).

Before the in-place-compression fix, mid-task compression SPLIT one logical
task into a parent→child chain (parent_session_id). Clicking the root in the
WebUI sidebar (roots-only listing) used to show only the pre-compression half.
``get_messages_with_lineage`` stitches the whole chain root→tip, de-duping a
replayed user seed at the boundary, and preserving every column the frontend
needs. Unsplit (post-fix) sessions pass through unchanged.
"""
from __future__ import annotations

from easybci_lib.state import SessionDB


def _add(db, sid, pairs, parent=None):
    db.create_session(session_id=sid, source="test", parent_session_id=parent)
    for role, content in pairs:
        db.append_message(session_id=sid, role=role, content=content)


def test_normal_split_stitches_full_history(isolated_db):
    db = isolated_db
    try:
        _add(db, "20260101_100000_root",
             [("user", "q1"), ("assistant", "a1"), ("user", "q2"), ("assistant", "a2")])
        _add(db, "20260101_110000_child",
             [("assistant", "[summary]"), ("user", "q3"), ("assistant", "a3")],
             parent="20260101_100000_root")

        assert db._full_lineage_chain("20260101_100000_root") == [
            "20260101_100000_root", "20260101_110000_child",
        ]
        # Resolves the same whole chain when clicked from the child (mid node).
        assert db._full_lineage_chain("20260101_110000_child") == [
            "20260101_100000_root", "20260101_110000_child",
        ]

        contents = [m["content"] for m in db.get_messages_with_lineage("20260101_100000_root")]
        assert contents == ["q1", "a1", "q2", "a2", "[summary]", "q3", "a3"]
    finally:
        db.close()


def test_empty_parent_recovers_from_child(isolated_db):
    """The message_count=0 bug: parent never flushed, child holds everything."""
    db = isolated_db
    try:
        db.create_session(session_id="20260201_100000_root2", source="test")
        _add(db, "20260201_110000_child2",
             [("user", "x1"), ("assistant", "y1"), ("user", "x2"), ("assistant", "y2")],
             parent="20260201_100000_root2")
        contents = [m["content"] for m in db.get_messages_with_lineage("20260201_100000_root2")]
        assert contents == ["x1", "y1", "x2", "y2"]
    finally:
        db.close()


def test_unsplit_session_passthrough(isolated_db):
    db = isolated_db
    try:
        _add(db, "20260301_120000_solo", [("user", "hi"), ("assistant", "hello")])
        assert db._full_lineage_chain("20260301_120000_solo") == ["20260301_120000_solo"]
        contents = [m["content"] for m in db.get_messages_with_lineage("20260301_120000_solo")]
        assert contents == ["hi", "hello"]
    finally:
        db.close()


def test_dedup_is_not_overeager_across_boundary(isolated_db):
    """Dedup only drops a replayed user turn with no intervening assistant.

    A same-content user message that legitimately follows an assistant turn
    must be KEPT — guards against over-eager dedup swallowing real turns.
    """
    db = isolated_db
    try:
        _add(db, "20260401_100000_r", [("user", "only-q"), ("assistant", "only-a")])
        _add(db, "20260401_110000_c",
             [("user", "only-a"), ("assistant", "next")],
             parent="20260401_100000_r")
        # 'only-a' as a user message follows assistant 'only-a' → not a dup; kept.
        contents = [m["content"] for m in db.get_messages_with_lineage("20260401_100000_r")]
        assert contents == ["only-q", "only-a", "only-a", "next"]
    finally:
        db.close()


def test_dedup_drops_replayed_user_seed(isolated_db):
    """A child whose leading user turn repeats the parent's last user turn
    (no assistant in between across the boundary) is de-duped."""
    db = isolated_db
    try:
        # Parent ends on a user turn (assistant answer lived only in the child's
        # summary) — the child replays that same user turn as its seed.
        _add(db, "20260501_100000_r", [("user", "shared-q")])
        _add(db, "20260501_110000_c",
             [("user", "shared-q"), ("assistant", "answer")],
             parent="20260501_100000_r")
        contents = [m["content"] for m in db.get_messages_with_lineage("20260501_100000_r")]
        # The replayed 'shared-q' collapses to one.
        assert contents == ["shared-q", "answer"]
    finally:
        db.close()
