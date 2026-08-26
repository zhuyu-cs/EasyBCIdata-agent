"""Post-compression JSON-log shrink: the never-shrink guard must exempt a run
where compression actually happened, but keep protecting ordinary resumed runs
from clobbering a fuller log with a partial one.

Drives the real AIAgent._save_session_log on a bare agent shell (no LLM, no DB
row creation) with session_log_file pointed at a tmp path.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from run_agent import AIAgent


@pytest.fixture
def log_agent(tmp_path):
    """A bare AIAgent shell wired only enough to exercise _save_session_log."""
    agent = AIAgent.__new__(AIAgent)
    agent._session_id = "20260813_shrink"
    agent.session_log_file = tmp_path / "session_20260813_shrink.json"
    agent.logs_dir = tmp_path
    agent.model = "test-model"
    agent.base_url = ""
    agent.platform = "test"
    agent.tools = []
    agent._cached_system_prompt = "SYS"
    agent.verbose_logging = False
    agent._session_messages = []
    agent._compression_occurred = False
    # session_start / _clean_session_content are needed by _save_session_log
    from datetime import datetime
    agent.session_start = datetime(2026, 8, 13, 10, 0, 0)
    agent._clean_session_content = lambda c: c  # type: ignore[attr-defined]
    return agent


def _write(agent, n):
    agent._save_session_log([{"role": "user", "content": f"m{i}"} for i in range(n)])


def _count(agent):
    data = json.loads(Path(agent.session_log_file).read_text(encoding="utf-8"))
    return data["message_count"]


def test_guard_blocks_shrink_without_compression(log_agent):
    """Ordinary resumed run: a shorter write must NOT clobber a fuller log."""
    _write(log_agent, 10)
    log_agent._compression_occurred = False
    _write(log_agent, 4)          # partial resume tries to shrink
    assert _count(log_agent) == 10  # protected


def test_guard_allows_shrink_after_compression(log_agent):
    """Compression happened this run: the shorter working state IS written."""
    _write(log_agent, 10)
    log_agent._compression_occurred = True
    _write(log_agent, 4)          # post-compression working state
    assert _count(log_agent) == 4   # exemption fired


def test_post_compression_can_grow_again(log_agent):
    """After a compression shrink, later turns keep appending on top."""
    _write(log_agent, 10)
    log_agent._compression_occurred = True
    _write(log_agent, 4)
    _write(log_agent, 6)          # 2 more turns after compression
    assert _count(log_agent) == 6   # no longer stuck at the old high-water mark
