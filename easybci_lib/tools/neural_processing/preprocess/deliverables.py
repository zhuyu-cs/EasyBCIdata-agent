"""Single-source-of-truth for the `deliverables` concept.

`deliverables` is the delivery-output axis: which artefact families a run
produces. It replaces the old "goal.produces_ai_ready auto-inference" as the
gate for beyond-NWB outputs. `preprocessed` (NWB) is ALWAYS implied. `ai_ready`
(epochs.pkl) is opt-in — added only when the user explicitly asks at confirm.

Consumers: neural_tools.py (propose/confirm/codegen), contract_check.py,
repo_builder.py.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

VALID_DELIVERABLES = {"preprocessed", "ai_ready"}
DEFAULT_DELIVERABLES: List[str] = ["preprocessed"]

# Canonical ordering for stable, deduplicated output.
_ORDER = ["preprocessed", "ai_ready"]


def normalize_deliverables(deliverables: Optional[List[str]]) -> List[str]:
    """Validate + canonicalise a deliverables list.

    - None → a fresh copy of DEFAULT_DELIVERABLES.
    - `preprocessed` is always included (NWB is the universal baseline).
    - Deduplicated and ordered per `_ORDER`.
    - Unknown values raise ValueError (only preprocessed|ai_ready supported;
      future formats extend VALID_DELIVERABLES).
    """
    if deliverables is None:
        return list(DEFAULT_DELIVERABLES)
    requested = set(deliverables)
    unknown = requested - VALID_DELIVERABLES
    if unknown:
        raise ValueError(
            f"unknown deliverable(s): {sorted(unknown)}; "
            f"supported: {sorted(VALID_DELIVERABLES)}"
        )
    requested.add("preprocessed")
    return [d for d in _ORDER if d in requested]


def resolve_deliverables(
    record: Optional[dict],
    work_dir: Optional[Path] = None,
) -> List[str]:
    """Read deliverables from a record (proposal.json / pipeline_record.json /
    confirm marker) with backward-compat fallback for legacy records that
    predate the field.

    - Explicit `deliverables` field present → normalize + return it.
    - Absent → default to ["preprocessed"]; additionally infer `ai_ready` when
      AI_ready artefacts already exist on disk under work_dir (so contract_check
      does not false-report a legacy completed run as missing ai_ready).
    """
    if record and isinstance(record.get("deliverables"), list):
        return normalize_deliverables(record["deliverables"])
    result = list(DEFAULT_DELIVERABLES)
    if work_dir is not None:
        ai_root = Path(work_dir) / "preprocessed_output" / "AI_ready"
        try:
            if ai_root.is_dir() and any(ai_root.rglob("*_epochs.pkl")):
                result.append("ai_ready")
        except OSError:
            pass
    return result
