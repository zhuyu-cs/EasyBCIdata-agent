"""Hard-constraint errors raised by finalize / contract_check.

These two errors are the ONLY way the export layer signals "this run did
not produce a valid mini-repo." Callers (gateway, CLI) catch them and
surface them to the user instead of writing out husk artifacts.
"""
from __future__ import annotations

from typing import List


class IncompleteRunError(Exception):
    """The run did not produce enough signal to write a real plan/.

    Raised by finalize._recover_from_middle_process when both modality and
    paradigm could only be filled with the string ``"unknown"`` — the
    previous behavior was to write a husk plan/proposal.json with these
    placeholders, which downstream tools (proven_match, baseline compare,
    skill_writer) silently treated as a real result.
    """

    def __init__(self, reason: str, *, work_dir: str = ""):
        super().__init__(reason)
        self.reason = reason
        self.work_dir = work_dir


class LayoutContractError(Exception):
    """The work_dir's layout violates the mini-repo contract.

    Raised by verify_layout_strict when either:
      - one of the required top-level dirs is missing
      - plan/proposal.json or plan/pipeline_record.json contains "unknown"
        in a critical field (modality / paradigm / analysis_goal)
    """

    def __init__(
        self,
        message: str,
        *,
        missing: List[str] | None = None,
        husk_fields: List[str] | None = None,
    ):
        super().__init__(message)
        self.missing = list(missing or [])
        self.husk_fields = list(husk_fields or [])
