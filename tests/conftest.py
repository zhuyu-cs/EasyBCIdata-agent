"""Shared test fixtures.

Autouse ``isolated_easybci_home`` points ``EASYBCI_HOME`` at a per-test tmpdir
so nothing touches the developer's real ``~/.easybci`` (profiles, sessions,
skills). Convention documented in CLAUDE.md.
"""
from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def isolated_easybci_home(tmp_path, monkeypatch):
    home = tmp_path / "easybci_home"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("EASYBCI_HOME", str(home))
    yield home


@pytest.fixture
def isolated_db(isolated_easybci_home):
    """A SessionDB bound to the per-test isolated home.

    NOTE: ``state.DEFAULT_DB_PATH`` is frozen at import time (before the
    ``EASYBCI_HOME`` monkeypatch runs), so a bare ``SessionDB()`` would hit the
    developer's REAL ~/.easybci/state.db. Always construct the DB via this
    fixture (explicit ``db_path``) to stay isolated.
    """
    from easybci_lib.state import SessionDB

    db = SessionDB(db_path=isolated_easybci_home / "state.db")
    try:
        yield db
    finally:
        db.close()

