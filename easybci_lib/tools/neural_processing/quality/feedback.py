"""QC-driven feedback — automatic parameter adjustment when quality checks fail.

When preprocessing produces output that doesn't pass QC, this module suggests
remedies (pipeline modifications) and applies them. The orchestrator can then
re-run the pipeline with adjusted steps.

Flow:
  1. QC returns issues (flat, amplitude, nan, variance)
  2. suggest_remedy() maps issues → concrete pipeline modifications
  3. apply_remedy() inserts/modifies steps in the pipeline
  4. Orchestrator re-runs with new steps (max 3 retries)
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MAX_RETRIES = 10


@dataclass
class Remedy:
    """A suggested fix for a QC issue."""
    issue_check: str
    action: str
    reason: str
    insert_position: str = "before_scale"  # before_scale, prepend, append
    priority: int = 0  # higher = apply first


@dataclass
class FeedbackResult:
    """Result of the feedback analysis."""
    should_retry: bool = False
    remedies: List[Remedy] = field(default_factory=list)
    new_steps: List[str] = field(default_factory=list)
    explanation: str = ""


# Issue → remedy mapping (ordered by priority)
_REMEDY_TABLE: List[Dict[str, Any]] = [
    {
        "check": "nan",
        "severity_min": "warning",
        "action": "fill_nan",
        "reason": "NaN values detected — inserting fill_nan to replace with interpolated values",
        "position": "prepend",
        "priority": 10,
    },
    {
        "check": "flat",
        "severity_min": "warning",
        "action": "drop_bads",
        "reason": "Flat (zero-variance) channels detected — adding drop_bads to remove them",
        "position": "prepend",
        "priority": 8,
    },
    {
        "check": "amplitude",
        "severity_min": "warning",
        "action": "clip:500",
        "reason": "Extreme amplitude values — adding clip:500 to limit outliers",
        "position": "before_scale",
        "priority": 5,
    },
    {
        "check": "variance",
        "severity_min": "warning",
        "action": "drop_bads",
        "reason": "Abnormal variance channels — adding drop_bads to remove outlier channels",
        "position": "prepend",
        "priority": 7,
    },
    {
        "check": "inf",
        "severity_min": "warning",
        "action": "fill_nan",
        "reason": "Inf values detected — inserting fill_nan to replace non-finite values",
        "position": "prepend",
        "priority": 9,
    },
]


def suggest_remedy(
    qc_result: Dict[str, Any],
    current_steps: List[str],
    retry_count: int = 0,
) -> FeedbackResult:
    """Analyze QC issues and suggest pipeline adjustments.

    Parameters
    ----------
    qc_result : dict
        Output from validate_signal() with 'passed', 'issues', 'stats'.
    current_steps : list of str
        Current pipeline steps.
    retry_count : int
        How many retries have already been attempted.

    Returns
    -------
    FeedbackResult with suggested changes.
    """
    if retry_count >= _MAX_RETRIES:
        return FeedbackResult(
            should_retry=False,
            explanation=f"Maximum retries ({_MAX_RETRIES}) reached. Manual review recommended.",
        )

    issues = qc_result.get("issues", [])
    if not issues:
        return FeedbackResult(should_retry=False, explanation="No issues found.")

    # Only retry for warnings, not hard errors (like format/load failures)
    actionable_issues = [
        i for i in issues
        if i.get("check") in ("nan", "inf", "flat", "amplitude", "variance")
    ]

    if not actionable_issues:
        return FeedbackResult(
            should_retry=False,
            explanation="Issues found but none are auto-remediable.",
        )

    remedies = []
    for issue in actionable_issues:
        check = issue["check"]
        for entry in _REMEDY_TABLE:
            if entry["check"] == check:
                action = entry["action"]
                # Don't add a remedy if the step already exists
                if _step_already_present(action, current_steps):
                    continue
                remedies.append(Remedy(
                    issue_check=check,
                    action=action,
                    reason=entry["reason"],
                    insert_position=entry["position"],
                    priority=entry["priority"],
                ))
                break

    if not remedies:
        # Static table exhausted — try learned remedies from error_recovery store
        learned = _try_learned_remedies(actionable_issues, current_steps, retry_count)
        if learned:
            return learned

        # Then try web search for advanced solutions
        web_remedies = _search_web_for_remedies(actionable_issues, current_steps, qc_result)
        if web_remedies:
            return web_remedies

        return FeedbackResult(
            should_retry=False,
            explanation="All suggested remedies are already in the pipeline. Manual review needed.",
        )

    # Deduplicate (same action)
    seen_actions = set()
    unique_remedies = []
    for r in sorted(remedies, key=lambda x: -x.priority):
        if r.action not in seen_actions:
            seen_actions.add(r.action)
            unique_remedies.append(r)

    new_steps = apply_remedies(current_steps, unique_remedies)
    explanations = [r.reason for r in unique_remedies]

    return FeedbackResult(
        should_retry=True,
        remedies=unique_remedies,
        new_steps=new_steps,
        explanation=" | ".join(explanations),
    )


def apply_remedies(steps: List[str], remedies: List[Remedy]) -> List[str]:
    """Apply remedy actions to the pipeline steps.

    Inserts new steps at the appropriate position without duplicating.
    """
    result = list(steps)

    for remedy in sorted(remedies, key=lambda r: -r.priority):
        action = remedy.action
        if _step_already_present(action, result):
            continue

        if remedy.insert_position == "prepend":
            result.insert(0, action)
        elif remedy.insert_position == "append":
            result.append(action)
        elif remedy.insert_position == "before_scale":
            idx = _find_step_index(result, "scale")
            if idx is not None:
                result.insert(idx, action)
            else:
                result.append(action)

    return result


def format_qc_suggestion(qc_result: Dict[str, Any], current_steps: List[str]) -> str:
    """Generate a human-readable suggestion string for tool result injection.

    Used in Agent mode to hint the LLM about possible fixes.
    """
    feedback = suggest_remedy(qc_result, current_steps)
    if not feedback.should_retry:
        return ""

    parts = ["[Auto-suggestion] QC issues detected. Possible fixes:"]
    for remedy in feedback.remedies:
        parts.append(f"  - Add '{remedy.action}' ({remedy.reason})")
    parts.append(f"  Suggested pipeline: {feedback.new_steps}")
    return "\n".join(parts)


def _step_already_present(action: str, steps: List[str]) -> bool:
    """Check if an action (or equivalent) is already in the pipeline."""
    base = action.split(":")[0]
    for s in steps:
        if s.split(":")[0] == base:
            return True
    return False


def _find_step_index(steps: List[str], step_prefix: str) -> Optional[int]:
    """Find index of a step by its prefix."""
    for i, s in enumerate(steps):
        if s.startswith(step_prefix):
            return i
    return None


def _search_web_for_remedies(
    issues: List[Dict[str, Any]],
    current_steps: List[str],
    qc_result: Dict[str, Any],
) -> Optional[FeedbackResult]:
    """Attempt to find remedies via web search when local table is exhausted.

    Only called when _REMEDY_TABLE has no applicable solutions (all known
    remedies are already in the pipeline). Searches for specialized fixes.

    Returns FeedbackResult if web search yields actionable advice, else None.
    """
    try:
        from easybci_lib.tools.neural_processing.research.complexity_classifier import classify_complexity
        from easybci_lib.tools.neural_processing.research.query_builder import build_queries
        from easybci_lib.tools.neural_processing.research.search_cache import SearchCache
        from easybci_lib.tools.neural_processing.research.evidence_synthesizer import synthesize_evidence
        from easybci_agent.web_search_registry import get_active_search_provider
    except ImportError:
        return None

    provider = get_active_search_provider()
    if provider is None:
        return None

    issue_names = [i.get("check", "") for i in issues]
    question = (
        f"EEG preprocessing QC failed with issues: {', '.join(issue_names)}. "
        f"Current pipeline: {' → '.join(current_steps)}. "
        f"How to fix persistent quality issues after standard remedies exhausted"
    )

    cache = SearchCache()
    cache_key_paradigm = "qc_" + "_".join(sorted(issue_names))
    cached = cache.get("eeg", cache_key_paradigm, question)
    if cached and cached.get("recommendations"):
        new_steps = _apply_web_remedies(current_steps, cached["recommendations"])
        if new_steps != current_steps:
            citations = cached.get("citations") or []
            first_url = (citations[0].get("url") if citations else None) or "web"
            return FeedbackResult(
                should_retry=True,
                new_steps=new_steps,
                explanation=f"Web research suggests: {', '.join(cached['recommendations'])}. "
                            f"Sources: {first_url}",
            )
        return None

    queries = build_queries(
        level=2,
        modality="eeg",
        paradigm="",
        question=question,
        context={"qc_issues": issue_names},
    )

    search_results = []
    for sq in queries[:3]:
        try:
            response = provider.search(sq.query, limit=3)
            if response and response.get("success"):
                search_results.append({
                    "query": sq.query,
                    "purpose": sq.purpose,
                    "results": response.get("data", {}).get("web", []),
                })
        except Exception:
            continue

    if not search_results:
        return None

    report = synthesize_evidence(
        search_results=search_results,
        modality="eeg",
        question=question,
    )

    if report.confidence < 0.3 or not report.recommendations:
        return None

    cache.put("eeg", cache_key_paradigm, question, report.to_dict())

    # Internalize high-confidence findings into domain skills
    if report.confidence >= 0.8:
        try:
            from easybci_lib.tools.neural_processing.research.knowledge_internalizer import try_internalize_from_evidence
            try_internalize_from_evidence(
                report=report.to_dict(),
                modality="eeg",
                paradigm="",
                question=question,
                user_confirmed=False,
            )
        except Exception:
            pass

    new_steps = _apply_web_remedies(current_steps, report.recommendations)
    if new_steps == current_steps:
        return None

    return FeedbackResult(
        should_retry=True,
        new_steps=new_steps,
        explanation=(
            f"Web research (confidence {report.confidence:.0%}) suggests adding: "
            f"{', '.join(report.recommendations)}. {report.rationale}"
        ),
    )


def _apply_web_remedies(current_steps: List[str], suggested_steps: List[str]) -> List[str]:
    """Merge web-suggested steps into current pipeline without duplicates."""
    result = list(current_steps)
    for step in suggested_steps:
        if not _step_already_present(step, result):
            # Insert before scale if it exists, otherwise append
            idx = _find_step_index(result, "scale")
            if idx is not None:
                result.insert(idx, step)
            else:
                result.append(step)
    return result


def _try_learned_remedies(
    issues: List[Dict[str, Any]],
    current_steps: List[str],
    retry_count: int,
) -> Optional[FeedbackResult]:
    """Consult the ErrorRemedyStore for previously successful fixes."""
    try:
        from easybci_lib.tools.neural_processing.executor.error_recovery import ErrorRemedyStore
    except ImportError:
        return None

    store = ErrorRemedyStore()
    issue_descriptions = [
        f"{i.get('check', 'unknown')}: {i.get('message', '')}" for i in issues
    ]
    combined_error = " | ".join(issue_descriptions)

    remedy = store.find_remedy(combined_error, min_confidence=0.5)
    if not remedy:
        return None

    action = remedy.fix_action
    if _step_already_present(action, current_steps):
        return None

    new_steps = list(current_steps)
    if action.startswith("prepend:"):
        new_steps.insert(0, action.split(":", 1)[1])
    elif action.startswith("remove:"):
        target = action.split(":", 1)[1]
        new_steps = [s for s in new_steps if not s.startswith(target)]
    elif action.startswith("add:"):
        step_to_add = action.split(":", 1)[1]
        idx = _find_step_index(new_steps, "scale")
        if idx is not None:
            new_steps.insert(idx, step_to_add)
        else:
            new_steps.append(step_to_add)
    else:
        idx = _find_step_index(new_steps, "scale")
        if idx is not None:
            new_steps.insert(idx, action)
        else:
            new_steps.append(action)

    if new_steps == current_steps:
        return None

    return FeedbackResult(
        should_retry=True,
        new_steps=new_steps,
        explanation=(
            f"Learned remedy (confidence {remedy.confidence:.0%}): {remedy.fix_action}. "
            f"Based on {remedy.success_count} prior successes."
        ),
    )


def record_remedy_outcome(
    error_message: str,
    fix_action: str,
    success: bool,
    context: Optional[Dict[str, Any]] = None,
) -> None:
    """Record whether a remedy fix succeeded, updating the learning store."""
    try:
        from easybci_lib.tools.neural_processing.executor.error_recovery import ErrorRemedyStore
        store = ErrorRemedyStore()
        store.record_fix(error_message, fix_action, success=success, context=context)
    except Exception:
        logger.debug("Failed to record remedy outcome", exc_info=True)
