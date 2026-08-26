"""session_id no-rotation invariant (plan Task 3A + this-round hardening).

User's hard rule: "一个任务，只要没有退出，就不该有新的 session 文件". The JSON log
filename is frozen from session_id at construction, so the only way a live task
spawns a SECOND file is a mid-run session_id mutation. AIAgent.session_id is a
property that is LOCKED for the duration of a run (run_conversation) and rejects
any divergent write while locked — reject-and-log, never raise, so a stray
rotation can't crash a live task.

These tests exercise the property directly on a bare instance (no full __init__)
so they stay fast and independent of provider/config wiring.
"""
from __future__ import annotations

import logging

import pytest

import run_agent


@pytest.fixture
def bare_agent():
    """An AIAgent shell with only the session_id machinery initialized."""
    agent = run_agent.AIAgent.__new__(run_agent.AIAgent)
    return agent


def test_construction_sets_id(bare_agent):
    # Unlocked, _session_id unset → first assignment always takes.
    bare_agent.session_id = "20260813_120000_abc123"
    assert bare_agent.session_id == "20260813_120000_abc123"


def test_idle_rotation_allowed(bare_agent):
    # Legitimate /new, /branch, /resume happen BETWEEN runs (unlocked).
    bare_agent.session_id = "20260813_120000_abc123"
    bare_agent.session_id = "20260813_130000_def456"
    assert bare_agent.session_id == "20260813_130000_def456"


def test_midrun_rotation_refused(bare_agent, caplog):
    bare_agent.session_id = "20260813_130000_def456"
    bare_agent._session_id_locked = True
    with caplog.at_level(logging.ERROR):
        bare_agent.session_id = "20260813_140000_ghi789"  # must be ignored
    # Original id preserved — no second file will ever be minted.
    assert bare_agent.session_id == "20260813_130000_def456"
    assert any("session-invariant" in r.message or "REFUSED" in r.message
               for r in caplog.records)


def test_midrun_refusal_does_not_raise(bare_agent):
    bare_agent.session_id = "20260813_130000_def456"
    bare_agent._session_id_locked = True
    # Reject-and-log: must NOT raise (a stray rotation can't crash a live task).
    bare_agent.session_id = "20260813_999999_zzz999"
    assert bare_agent.session_id == "20260813_130000_def456"


def test_same_id_write_while_locked_is_noop(bare_agent):
    bare_agent.session_id = "20260813_130000_def456"
    bare_agent._session_id_locked = True
    bare_agent.session_id = "20260813_130000_def456"  # identical → fine
    assert bare_agent.session_id == "20260813_130000_def456"


def test_unlock_then_rotate_allowed(bare_agent):
    bare_agent.session_id = "20260813_130000_def456"
    bare_agent._session_id_locked = True
    bare_agent.session_id = "20260813_140000_ghi789"  # refused
    bare_agent._session_id_locked = False
    bare_agent.session_id = "20260813_150000_jkl000"  # now allowed
    assert bare_agent.session_id == "20260813_150000_jkl000"


def test_run_conversation_wrapper_locks_and_releases():
    """The run wrapper must lock for the run and release in finally, even on error."""
    agent = run_agent.AIAgent.__new__(run_agent.AIAgent)
    agent.session_id = "20260813_120000_abc123"
    observed = {}

    def fake_impl(*args, **kwargs):
        observed["locked_during_run"] = agent._session_id_locked
        # A mid-run rotation attempt (e.g. compression) must be refused here.
        agent.session_id = "20260813_140000_rotated"
        observed["id_during_run"] = agent.session_id
        raise RuntimeError("boom")  # ensure finally still releases

    agent._run_conversation_impl = fake_impl
    with pytest.raises(RuntimeError):
        agent.run_conversation("hi")

    assert observed["locked_during_run"] is True
    assert observed["id_during_run"] == "20260813_120000_abc123"  # rotation refused
    assert agent._session_id_locked is False  # released in finally
    # Between runs (unlocked) rotation works again.
    agent.session_id = "20260813_160000_newturn"
    assert agent.session_id == "20260813_160000_newturn"
