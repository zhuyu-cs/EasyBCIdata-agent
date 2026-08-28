"""Resume-phase hint: inject a factual phase snapshot into the resumed
conversation so the LLM does not have to reconstruct pipeline state from
a truncated tool-call replay after an API disconnect.

Kept factual on purpose — the hint lists what's on disk plus a one-line
next-step recommendation. The LLM decides what to do; guardrails against
regressions live in ``_reject_if_already_confirmed`` on the tool side.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional


def _phase_facts(wd: Path) -> dict:
    confirmed = (wd / "middle_process" / "proposal.confirmed").is_file()
    staged = (wd / "middle_process" / "proposal.staged.json").is_file()
    code = (wd / "code" / "pipeline.py").is_file()
    nwb_dir = wd / "preprocessed_output" / "preprocessed"
    has_nwb = nwb_dir.is_dir() and any(nwb_dir.rglob("*.nwb"))
    record = wd / "plan" / "pipeline_record.json"
    finalized_ok = False
    if record.is_file():
        try:
            finalized_ok = json.loads(
                record.read_text(encoding="utf-8")
            ).get("status") == "ok"
        except (OSError, json.JSONDecodeError):
            finalized_ok = False
    return {
        "confirmed": confirmed,
        "staged": staged,
        "code": code,
        "has_nwb": has_nwb,
        "finalized_ok": finalized_ok,
    }


def _next_step(f: dict) -> str:
    if f["finalized_ok"]:
        return "Pipeline finalized. Ask the user what to do next; do not re-run."
    if f["has_nwb"]:
        return (
            "NWB output exists but not finalized. Run qc.py / vis.py, "
            "then export_repo — do NOT re-run preprocess_neural."
        )
    if f["code"]:
        return (
            "code/pipeline.py exists. Run preprocess_neural to execute it — "
            "do NOT regenerate code unless the user asks to modify steps."
        )
    if f["confirmed"]:
        return (
            "Phase 1 closed. Continue with generate_code(work_dir=...). "
            "Do NOT call plan_pipeline / propose_pipeline / suggest_pipeline "
            "— those will be rejected."
        )
    if f["staged"]:
        return (
            "A proposal is staged awaiting user decision. Present the "
            "staged proposal and call mark_proposal_confirmed."
        )
    return (
        "No confirmed proposal yet. Start from inspect_data / deep_inspect "
        "/ plan_pipeline as usual."
    )


def build_resume_phase_hint(session_db, session_id: str) -> Optional[str]:
    """Return a factual phase snapshot for the resumed session's work_dir,
    or None if hinting is disabled / unresolvable.
    """
    if os.environ.get("EASYBCI_TUI_RESUME_PHASE_HINT_DISABLE"):
        return None
    if session_db is None or not session_id:
        return None
    try:
        wd_str = session_db.get_session_work_dir(session_id)
    except Exception:
        return None
    if not wd_str:
        return None
    wd = Path(wd_str)
    if not wd.is_dir():
        return None
    f = _phase_facts(wd)
    lines = [
        f"[Resume phase snapshot for work_dir={wd}]",
        f"• proposal.confirmed: {'yes' if f['confirmed'] else 'no'}",
        f"• proposal.staged.json: {'yes' if f['staged'] else 'no'}",
        f"• code/pipeline.py: {'yes' if f['code'] else 'no'}",
        f"• preprocessed/*.nwb: {'yes' if f['has_nwb'] else 'no'}",
        f"• finalized: {'yes' if f['finalized_ok'] else 'no'}",
        f"Guidance: {_next_step(f)}",
    ]
    return "\n".join(lines)


def build_resume_phase_hint_short(session_db, session_id: str) -> Optional[str]:
    """One-line human summary suitable for CLI stdout ('Resume phase: ...').
    Returns None under the same conditions as ``build_resume_phase_hint``.
    """
    if os.environ.get("EASYBCI_TUI_RESUME_PHASE_HINT_DISABLE"):
        return None
    if session_db is None or not session_id:
        return None
    try:
        wd_str = session_db.get_session_work_dir(session_id)
    except Exception:
        return None
    if not wd_str:
        return None
    wd = Path(wd_str)
    if not wd.is_dir():
        return None
    f = _phase_facts(wd)
    return (
        f"proposal.confirmed={'yes' if f['confirmed'] else 'no'}, "
        f"code={'yes' if f['code'] else 'no'}, "
        f"NWB={'yes' if f['has_nwb'] else 'no'}, "
        f"finalized={'yes' if f['finalized_ok'] else 'no'}"
    )


__all__ = ["build_resume_phase_hint", "build_resume_phase_hint_short"]
