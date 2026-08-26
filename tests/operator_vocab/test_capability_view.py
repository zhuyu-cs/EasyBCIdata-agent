"""Lock the operator single-source-of-truth so the two executors never drift.

The silent-skip bug was caused by two hand-maintained operator lists (the
runtime engine's ``AVAILABLE_STEPS`` and the codegen bundle's ``_OPS``) that
diverged. These tests assert both are DERIVED from
``operator_vocab.OPERATOR_EXECUTORS`` and stay a subset of the canonical set.
"""
from __future__ import annotations

import re
from pathlib import Path

from easybci_lib.tools.neural_processing.preprocess.operator_vocab import (
    CANONICAL_OPERATORS,
    OPERATOR_EXECUTORS,
    codegen_operators,
    engine_operators,
)


def test_capability_table_matches_canonical_set():
    assert set(OPERATOR_EXECUTORS) == set(CANONICAL_OPERATORS)
    # every canonical op runnable by at least one executor
    assert all(v for v in OPERATOR_EXECUTORS.values())


def test_engine_available_steps_derives_from_vocab():
    from easybci_lib.tools.neural_processing.preprocess.pipeline import AVAILABLE_STEPS

    assert AVAILABLE_STEPS == engine_operators()
    assert set(AVAILABLE_STEPS).issubset(CANONICAL_OPERATORS)


def _extract_codegen_ops_keys() -> set[str]:
    """Parse the _OPS dict keys out of the generator template source.

    _OPS lives inside a template string (it is source that gets embedded into
    the generated standalone script), so we read it textually rather than
    importing it.
    """
    src = Path(
        "easybci_lib/tools/neural_processing/codegen/generator.py"
    ).read_text(encoding="utf-8")
    # grab the _OPS = {{ ... }} block
    m = re.search(r"_OPS = \{\{(.*?)\}\}", src, re.DOTALL)
    assert m, "could not locate _OPS block in generator.py"
    return set(re.findall(r'"([a-z_]+)":\s*op_', m.group(1)))


def test_codegen_ops_subset_of_canonical():
    keys = _extract_codegen_ops_keys()
    assert keys, "no _OPS keys parsed"
    # codegen must not invent operators outside the canonical vocabulary
    assert keys.issubset(CANONICAL_OPERATORS), keys - CANONICAL_OPERATORS
    # and it must match exactly what the capability view says codegen runs
    assert keys == set(codegen_operators()), (
        keys ^ set(codegen_operators())
    )


def test_no_synonyms_leak_into_codegen_ops():
    # highpass/lowpass are synonyms, not canonical — must NOT appear as _OPS keys
    keys = _extract_codegen_ops_keys()
    assert "highpass" not in keys
    assert "lowpass" not in keys
