"""Task 0 (Phase B air-gap) — LocalDirSkillSource + get_local_skills_dir.

Verifies the offline fallback lane: a configurable local directory serves skills
with zero network, and stays fully transparent (empty results) when unconfigured.
"""
from __future__ import annotations


def _write_skill(root, name, description="d"):
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\nbody\n",
        encoding="utf-8",
    )
    return d


def test_get_local_skills_dir_resolution(tmp_path, monkeypatch):
    from easybci_lib.constants import get_local_skills_dir, get_easybci_home

    # 1. env var wins
    monkeypatch.setenv("EASYBCI_LOCAL_SKILLS_DIR", str(tmp_path / "env-dir"))
    assert get_local_skills_dir() == tmp_path / "env-dir"

    # 2. explicit default (config value) used when env unset
    monkeypatch.delenv("EASYBCI_LOCAL_SKILLS_DIR", raising=False)
    assert get_local_skills_dir(default=tmp_path / "cfg-dir") == tmp_path / "cfg-dir"

    # 3. falls back to home/local-skills
    assert get_local_skills_dir() == get_easybci_home() / "local-skills"


def test_local_source_lists_and_fetches(tmp_path, monkeypatch):
    _write_skill(tmp_path, "pca-denoise", "PCA-based denoising")
    monkeypatch.setenv("EASYBCI_LOCAL_SKILLS_DIR", str(tmp_path))

    from easybci_lib.tools.skills_hub import LocalDirSkillSource

    src = LocalDirSkillSource()
    assert src.source_id() == "local-dir"

    metas = src.search("pca")
    assert any(m.name == "pca-denoise" for m in metas)
    meta = next(m for m in metas if m.name == "pca-denoise")
    assert meta.trust_level == "community"  # user-supplied, not builtin
    assert meta.source == "local-dir"

    bundle = src.fetch(meta.identifier)
    assert bundle is not None
    assert "SKILL.md" in bundle.files
    assert bundle.trust_level == "community"

    # inspect resolves by name too
    assert src.inspect("pca-denoise") is not None


def test_local_source_unconfigured_is_silent(tmp_path, monkeypatch):
    # No env var, and the default home/local-skills does not exist.
    monkeypatch.delenv("EASYBCI_LOCAL_SKILLS_DIR", raising=False)
    from easybci_lib.tools.skills_hub import LocalDirSkillSource

    src = LocalDirSkillSource()
    assert src.search("anything") == []
    assert src.fetch("x") is None
    assert src.inspect("x") is None


def test_local_source_path_traversal_blocked(tmp_path, monkeypatch):
    _write_skill(tmp_path, "safe")
    monkeypatch.setenv("EASYBCI_LOCAL_SKILLS_DIR", str(tmp_path))
    from easybci_lib.tools.skills_hub import LocalDirSkillSource

    src = LocalDirSkillSource()
    # A traversal identifier must not escape the configured dir.
    assert src.fetch("local/../../etc") is None


def test_router_includes_local_dir_source(monkeypatch):
    from easybci_lib.tools.skills_hub import create_source_router

    sources = create_source_router()
    ids = [s.source_id() for s in sources]
    assert "local-dir" in ids
    # Not part of the API-source short-circuit set — always participates.
    from easybci_lib.tools import skills_hub as sh
    # Sanity: the frozenset used by parallel_search_sources excludes local-dir.
    # (Guards Task 0 Step 4's claim.)
    import inspect
    srctext = inspect.getsource(sh.parallel_search_sources)
    assert "local-dir" not in srctext.split("_api_source_ids = frozenset(")[1].split(")")[0]
