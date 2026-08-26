"""Phase C: request_dependency — controlled agent dependency extension.

Covers the lazy_deps runtime-allowlist union + the request_dependency tool.
The install path is always mocked (_venv_pip_install + feature_missing) so tests
never hit PyPI / the network.
"""
import json

import pytest


# ---------------------------------------------------------------------------
# Task 1: lazy_deps runtime allowlist union + request()
# ---------------------------------------------------------------------------

def test_spec_must_be_exact_pinned():
    from easybci_lib.tools.lazy_deps import _spec_is_exact_pinned
    assert _spec_is_exact_pinned("numpy==1.26.4")
    assert not _spec_is_exact_pinned("numpy>=1.26")   # range rejected (§三 hard constraint)
    assert not _spec_is_exact_pinned("numpy")          # no version rejected
    assert not _spec_is_exact_pinned("numpy==1.0; rm -rf /")  # shell injection
    assert not _spec_is_exact_pinned("git+https://x/y.git")   # url/git+


def test_request_rejects_unsafe_spec():
    from easybci_lib.tools.lazy_deps import request, FeatureUnavailable
    with pytest.raises(FeatureUnavailable):
        request("adhoc.evil", ("pkg==1.0; rm -rf /",))
    with pytest.raises(FeatureUnavailable):
        request("adhoc.gitdep", ("git+https://x/y.git",))
    with pytest.raises(FeatureUnavailable):
        request("adhoc.range", ("pkg>=1.0",))


def test_request_registers_into_runtime_allowlist(monkeypatch):
    from easybci_lib.tools import lazy_deps as ld
    monkeypatch.setattr(
        ld, "_venv_pip_install",
        lambda specs, **kw: ld._InstallResult(True, "", ""))
    monkeypatch.setattr(ld, "feature_missing", lambda f: ())
    ld.request("adhoc.somepkg", ("somepkg==1.2.3",))
    # After registering into the runtime union, is_available recognises it.
    assert ld.is_available("adhoc.somepkg")
    assert "adhoc.somepkg" in ld._RUNTIME_DEPS


def test_request_persists_to_disk(monkeypatch):
    from easybci_lib.tools import lazy_deps as ld
    from easybci_lib.constants import get_easybci_home
    monkeypatch.setattr(
        ld, "_venv_pip_install",
        lambda specs, **kw: ld._InstallResult(True, "", ""))
    monkeypatch.setattr(ld, "feature_missing", lambda f: ())
    ld.request("adhoc.persistme", ("persistme==0.1.0",))
    p = get_easybci_home() / "runtime_lazy_deps.json"
    assert p.exists()
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data.get("adhoc.persistme") == ["persistme==0.1.0"]


# ---------------------------------------------------------------------------
# Task 2: request_dependency agent tool
# ---------------------------------------------------------------------------

def test_tool_installs_safe_pkg(monkeypatch):
    from easybci_lib.tools import lazy_deps as ld
    monkeypatch.setattr(
        ld, "_venv_pip_install",
        lambda s, **kw: ld._InstallResult(True, "", ""))
    monkeypatch.setattr(ld, "feature_missing", lambda f: ())
    from easybci_lib.tools.request_dependency_tool import _handle_request_dependency
    res = json.loads(_handle_request_dependency({
        "package": "somepkg", "version": "1.2.3", "purpose": "need X for Y"}))
    assert res["success"] is True, res
    assert res["installed"] == "somepkg==1.2.3"


def test_tool_rejects_range_with_fix_hint():
    from easybci_lib.tools.request_dependency_tool import _handle_request_dependency
    res = json.loads(_handle_request_dependency({"package": "numpy", "version": ">=1.0"}))
    assert res["success"] is False
    assert "==" in res.get("fix_hint", "")


def test_tool_respects_global_kill_switch(monkeypatch):
    from easybci_lib.tools import lazy_deps as ld
    monkeypatch.setattr(ld, "_allow_lazy_installs", lambda: False)
    from easybci_lib.tools.request_dependency_tool import _handle_request_dependency
    res = json.loads(_handle_request_dependency({"package": "somepkg", "version": "1.2.3"}))
    assert res["success"] is False
    assert "disabled" in (res.get("error", "")).lower()


def test_tool_missing_fields_rejected():
    from easybci_lib.tools.request_dependency_tool import _handle_request_dependency
    res = json.loads(_handle_request_dependency({"package": "somepkg"}))
    assert res["success"] is False


def test_tool_registered_and_discoverable():
    from easybci_lib.tools import registry as _reg
    _reg.discover_builtin_tools()
    names = {d["function"]["name"] for d in _reg.registry.get_definitions(
        {"request_dependency"}, quiet=True)}
    assert "request_dependency" in names


def test_request_dependency_in_core_tools():
    from easybci_lib.toolsets import resolve_toolset
    assert "request_dependency" in resolve_toolset("easybci-cli")
    assert "request_dependency" in resolve_toolset("easybci-webui")


def test_neural_subset_invariant_holds():
    """request_dependency must not break the neural ⊆ easybci-cli invariant."""
    from easybci_lib.toolsets import resolve_toolset
    assert set(resolve_toolset("neural")).issubset(set(resolve_toolset("easybci-cli")))
