"""Unified operator exception layer.

Every operator step that fails must raise :class:`EasyBCIOperatorError`
rather than letting the underlying numpy / mne / scipy exception escape.
The executor's 3-tier error_recovery chain then inspects ``recoverable``
to decide between immediate surface-to-user and a rule → memory → web
recovery attempt.
"""
from __future__ import annotations

from typing import Optional


class EasyBCIOperatorError(Exception):
    """Raised when an operator step cannot complete.

    Attributes
    ----------
    operator : str
        Step name (e.g. "bandpass_filter").
    reason : str
        Human-readable cause; surfaced to the user verbatim.
    recoverable : bool
        ``True`` means the error_recovery layer should attempt a fallback
        step before surfacing to the user.  ``False`` means a hard
        contract violation (modality mismatch, NaN-only input, etc.) —
        surface immediately.
    fallback_step : Optional[str]
        Suggested fallback step string (e.g. "bandpass:1,40").  Used by
        the rule-based recovery tier when ``recoverable=True``.
    modality : Optional[str]
        Modality that triggered the error, when relevant; used by tests
        and by the memory-match tier of error_recovery.
    """

    def __init__(
        self,
        operator: str,
        reason: str,
        *,
        recoverable: bool = True,
        fallback_step: Optional[str] = None,
        modality: Optional[str] = None,
    ) -> None:
        super().__init__(f"{operator}: {reason}")
        self.operator = operator
        self.reason = reason
        self.recoverable = recoverable
        self.fallback_step = fallback_step
        self.modality = modality

    def to_dict(self) -> dict:
        """Structured form for serialisation into agent-visible errors."""
        return {
            "operator": self.operator,
            "reason": self.reason,
            "recoverable": self.recoverable,
            "fallback_step": self.fallback_step,
            "modality": self.modality,
        }
