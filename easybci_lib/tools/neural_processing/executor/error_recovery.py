"""Adaptive error recovery — learning from past fixes.

Extends the existing auto_fixer and feedback modules with:
1. Error pattern memory: persists (error_pattern, fix_action, context) triples
2. Progressive retry: rule-based → memory-based → escalation
3. Dynamic remedy table: learns from proven pipeline fixes

Storage: ~/.easybci/error_remedies.json
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_REMEDIES_PATH_DEFAULT = Path.home() / ".easybci" / "error_remedies.json"


@dataclass
class ErrorRemedy:
    """A learned error→fix mapping."""
    error_pattern: str
    fix_action: str
    context: Dict[str, Any] = field(default_factory=dict)
    success_count: int = 0
    failure_count: int = 0
    last_used: float = 0.0
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_pattern": self.error_pattern,
            "fix_action": self.fix_action,
            "context": self.context,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "last_used": self.last_used,
            "confidence": round(self.confidence, 3),
        }


@dataclass
class RecoveryPlan:
    """A recovery plan with escalation levels."""
    level: int  # 1=rule, 2=memory, 3=escalation
    strategy: str
    fix_actions: List[str] = field(default_factory=list)
    explanation: str = ""
    from_memory: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level,
            "strategy": self.strategy,
            "fix_actions": self.fix_actions,
            "explanation": self.explanation,
            "from_memory": self.from_memory,
        }


class ErrorRemedyStore:
    """Persistent store for learned error→fix mappings."""

    def __init__(self, store_path: Optional[Path] = None):
        self.store_path = store_path or _REMEDIES_PATH_DEFAULT
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self._remedies: List[Dict[str, Any]] = self._load()

    def _load(self) -> List[Dict[str, Any]]:
        if self.store_path.exists():
            try:
                with open(self.store_path, encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        return []

    def _save(self) -> None:
        tmp = self.store_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._remedies, f, indent=2, ensure_ascii=False)
        tmp.rename(self.store_path)

    def record_fix(
        self,
        error_pattern: str,
        fix_action: str,
        success: bool,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a fix attempt (success or failure).

        If the same error_pattern + fix_action already exists, update counts.
        Otherwise create a new entry.
        """
        normalized = _normalize_error(error_pattern)

        for entry in self._remedies:
            if entry["error_pattern"] == normalized and entry["fix_action"] == fix_action:
                if success:
                    entry["success_count"] += 1
                else:
                    entry["failure_count"] += 1
                entry["last_used"] = time.time()
                total = entry["success_count"] + entry["failure_count"]
                entry["confidence"] = entry["success_count"] / total if total > 0 else 0
                self._save()
                return

        # New entry
        self._remedies.append({
            "error_pattern": normalized,
            "fix_action": fix_action,
            "context": context or {},
            "success_count": 1 if success else 0,
            "failure_count": 0 if success else 1,
            "last_used": time.time(),
            "confidence": 1.0 if success else 0.0,
        })
        self._save()

    def find_remedy(
        self,
        error_message: str,
        min_confidence: float = 0.5,
    ) -> Optional[ErrorRemedy]:
        """Find a previously successful fix for a similar error.

        Matches against stored patterns using substring and fuzzy matching.
        Returns the highest-confidence match.
        """
        normalized = _normalize_error(error_message)
        candidates = []

        for entry in self._remedies:
            if entry["confidence"] < min_confidence:
                continue
            pattern = entry["error_pattern"]
            # Exact substring match
            if pattern in normalized or normalized in pattern:
                candidates.append(entry)
                continue
            # Key terms overlap
            pattern_terms = set(pattern.lower().split())
            error_terms = set(normalized.lower().split())
            overlap = len(pattern_terms & error_terms) / max(len(pattern_terms), 1)
            if overlap > 0.6:
                candidates.append(entry)

        if not candidates:
            return None

        # Return highest confidence match
        best = max(candidates, key=lambda e: e["confidence"])
        return ErrorRemedy(**{k: v for k, v in best.items()})

    def get_all(self) -> List[ErrorRemedy]:
        """Get all stored remedies."""
        return [ErrorRemedy(**e) for e in self._remedies]

    def get_stats(self) -> Dict[str, Any]:
        """Get summary statistics."""
        total = len(self._remedies)
        high_conf = sum(1 for e in self._remedies if e.get("confidence", 0) >= 0.8)
        total_uses = sum(e.get("success_count", 0) + e.get("failure_count", 0) for e in self._remedies)
        return {
            "total_remedies": total,
            "high_confidence": high_conf,
            "total_uses": total_uses,
        }


def plan_recovery(
    error_message: str,
    retry_count: int,
    current_steps: List[str],
    modality: str = "",
    paradigm: str = "",
    store: Optional[ErrorRemedyStore] = None,
    *,
    operator_error: Optional[Any] = None,
) -> RecoveryPlan:
    """Create a progressive recovery plan based on retry count.

    Escalation levels:
      1 (retry 0-1): Rule-based fix from static table + memory
      2 (retry 2): Memory-based fix from past successful remedies
      3 (retry 3+): Suggest manual intervention or web research

    When ``operator_error`` is supplied as a
    :class:`easybci_lib.tools.neural_processing.operator_errors.EasyBCIOperatorError`:
      * ``recoverable=False`` → return an empty plan with action="surface"
        so the executor escalates straight to the user without burning
        retries.
      * ``recoverable=True`` + ``fallback_step`` → seed Level-1 with the
        operator-supplied fallback (the operator knows its own failure
        mode better than the generic rule table).

    Parameters
    ----------
    error_message : str
        The error that occurred.
    retry_count : int
        Number of retries already attempted.
    current_steps : list of str
        Current pipeline steps.
    modality, paradigm : str
        Data context for better matching.
    store : ErrorRemedyStore, optional
        Override store (for testing).
    operator_error : EasyBCIOperatorError, optional
        Structured operator error.  Takes precedence over heuristic
        recovery when present.

    Returns
    -------
    RecoveryPlan with fix actions and explanation.
    """
    # Handle structured operator errors before generic recovery.
    try:
        from easybci_lib.tools.neural_processing.operator_errors import (
            EasyBCIOperatorError,
        )
    except Exception:  # noqa: BLE001
        EasyBCIOperatorError = ()  # type: ignore[assignment]
    if operator_error is not None and isinstance(operator_error, EasyBCIOperatorError):
        if not operator_error.recoverable:
            return RecoveryPlan(
                level=3,
                strategy="surface",
                fix_actions=[],
                explanation=(
                    f"{operator_error.operator}: {operator_error.reason} "
                    "(non-recoverable — surfacing to user)."
                ),
                from_memory=False,
            )
        if operator_error.fallback_step:
            return RecoveryPlan(
                level=1,
                strategy="replace_step",
                fix_actions=[
                    f"replace_step:{operator_error.operator}={operator_error.fallback_step}",
                ],
                explanation=(
                    f"{operator_error.operator}: {operator_error.reason} → "
                    f"falling back to {operator_error.fallback_step!r}."
                ),
                from_memory=False,
            )

    if store is None:
        store = ErrorRemedyStore()

    # Level 1: Rule-based + memory lookup
    if retry_count <= 1:
        # Try memory first (faster, learned)
        remedy = store.find_remedy(error_message)
        if remedy and remedy.confidence >= 0.7:
            return RecoveryPlan(
                level=1,
                strategy="memory_match",
                fix_actions=[remedy.fix_action],
                explanation=(
                    f"Found previously successful fix (confidence: {remedy.confidence:.0%}): "
                    f"{remedy.fix_action}"
                ),
                from_memory=True,
            )

        # Fall back to rule-based analysis
        rule_fixes = _rule_based_fix(error_message, current_steps)
        if rule_fixes:
            return RecoveryPlan(
                level=1,
                strategy="rule_based",
                fix_actions=rule_fixes,
                explanation=f"Applying rule-based fix: {', '.join(rule_fixes)}",
            )

    # Level 2: Memory with lower confidence threshold
    if retry_count == 2:
        remedy = store.find_remedy(error_message, min_confidence=0.3)
        if remedy:
            return RecoveryPlan(
                level=2,
                strategy="memory_low_confidence",
                fix_actions=[remedy.fix_action],
                explanation=(
                    f"Trying lower-confidence remedy (confidence: {remedy.confidence:.0%}): "
                    f"{remedy.fix_action}"
                ),
                from_memory=True,
            )

        # Try dynamic remedies from proven pipelines
        dynamic_fixes = _dynamic_remedy_from_experience(error_message, modality, paradigm)
        if dynamic_fixes:
            return RecoveryPlan(
                level=2,
                strategy="experience_based",
                fix_actions=dynamic_fixes,
                explanation="Applying fix learned from similar processing records.",
                from_memory=True,
            )

    # Level 3: Escalation
    return RecoveryPlan(
        level=3,
        strategy="escalation",
        fix_actions=[],
        explanation=(
            f"After {retry_count} retries, automatic recovery exhausted. "
            f"Recommend: (1) web research for '{_extract_error_key(error_message)}', "
            f"(2) manual parameter adjustment, or (3) try a different pipeline approach."
        ),
    )


def _rule_based_fix(error_message: str, current_steps: List[str]) -> List[str]:
    """Apply static rules to generate fix actions."""
    fixes = []
    msg_lower = error_message.lower()

    if "nan" in msg_lower or "non-finite" in msg_lower:
        if not any("fill_nan" in s for s in current_steps):
            fixes.append("prepend:fill_nan")

    if "singular" in msg_lower or "covariance" in msg_lower:
        if any("ica" in s for s in current_steps):
            fixes.append("remove:ica")

    if "memory" in msg_lower or "memoryerror" in msg_lower or "oom" in msg_lower:
        fixes.append("chunked_processing")
        if any("resample" in s for s in current_steps):
            fixes.append("modify:resample:128")
        else:
            fixes.append("prepend:resample:128")

    if "flat" in msg_lower or "zero variance" in msg_lower:
        if not any("drop_bads" in s for s in current_steps):
            fixes.append("prepend:drop_bads")

    if "filter" in msg_lower and "too short" in msg_lower:
        if any("bandpass" in s for s in current_steps):
            fixes.append("remove:bandpass")

    if "shape" in msg_lower and "mismatch" in msg_lower:
        if not any("pick_channels" in s for s in current_steps):
            fixes.append("prepend:pick_channels:eeg")

    return fixes


def _dynamic_remedy_from_experience(
    error_message: str,
    modality: str,
    paradigm: str,
) -> List[str]:
    """Look up remedies from the experience store (processing records)."""
    try:
        from easybci_lib.tools.neural_processing.experience import ExperienceStore
        store = ExperienceStore()
        records = store.load_records(modality=modality, paradigm=paradigm, limit=50)

        # Find records that had retries and succeeded
        for rec in reversed(records):
            retries = rec.get("retries", {}).get("events", [])
            if not retries:
                continue
            outcome = rec.get("outcome", {})
            if not outcome.get("success"):
                continue

            # Check if any retry event matches our error
            for retry_evt in retries:
                if retry_evt.get("trigger") == "error":
                    # Similar error?
                    past_error = retry_evt.get("error_message", "")
                    if _errors_similar(error_message, past_error):
                        steps_after = retry_evt.get("steps_after", [])
                        steps_before = retry_evt.get("steps_before", [])
                        # Return the diff (what was added)
                        added = [s for s in steps_after if s not in steps_before]
                        if added:
                            return [f"add:{s}" for s in added]
    except Exception:
        pass
    return []


def _errors_similar(a: str, b: str) -> bool:
    """Check if two error messages are semantically similar."""
    a_terms = set(_normalize_error(a).lower().split())
    b_terms = set(_normalize_error(b).lower().split())
    if not a_terms or not b_terms:
        return False
    overlap = len(a_terms & b_terms) / min(len(a_terms), len(b_terms))
    return overlap > 0.5


def _normalize_error(msg: str) -> str:
    """Normalize an error message for matching (remove paths, numbers, etc.)."""
    # Remove file paths
    msg = re.sub(r'(/[\w/.-]+)', '<PATH>', msg)
    # Remove specific numbers (line numbers, array sizes)
    msg = re.sub(r'\b\d{3,}\b', '<NUM>', msg)
    # Remove hex addresses
    msg = re.sub(r'0x[0-9a-f]+', '<ADDR>', msg)
    return msg.strip()


def _extract_error_key(msg: str) -> str:
    """Extract a short key phrase from an error for web search."""
    # Get the last line (usually the most specific)
    lines = msg.strip().split('\n')
    last = lines[-1] if lines else msg
    # Truncate to reasonable search length
    return last[:100].strip()
