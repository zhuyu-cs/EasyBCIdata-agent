"""Phase D: SkillWriter dead-code removal regression guard.

The interactive ``SkillWriter`` (custom single-function operator skills) was
superseded by skill_manage / register_io_loader / crystallize and had no agent
caller. It must stay gone, and the real skill-writing / crystallization paths
must remain intact.
"""
import importlib

import pytest


def test_skillwriter_gone():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("easybci_lib.tools.neural_processing.skill_writer")


def test_registry_still_imports_clean():
    from easybci_lib.tools import registry
    registry.discover_builtin_tools()
    names = {d["function"]["name"] for d in registry.registry.get_definitions(
        {"skill_manage"}, quiet=True)}
    assert "skill_manage" in names   # the real skill-writing path is intact


def test_crystallize_path_intact():
    import easybci_lib.tools.neural_processing.export.contract_check as c
    assert hasattr(c, "maybe_crystallize_proven")
    assert hasattr(c, "_render_skill_md")
