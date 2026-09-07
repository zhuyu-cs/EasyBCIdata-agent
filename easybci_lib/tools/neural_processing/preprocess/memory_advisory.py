"""Memory footprint advisory -- pure function.

Takes the already-computed peak_native / budget / goal / current sfreq and
returns a structured advisory dict when memory pressure looks concerning.
Consumed by plan_pipeline / propose_pipeline / suggest_pipeline tool
returns to surface memory constraints to the LLM at proposal time (before
the runtime _gate_peak_mb fallback silently decimates).

No I/O, no filesystem access. Pure function of the inputs.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

from .analysis_goals import REGISTRY

_DEFAULT_WARN_RATIO = 0.5


def _env_warn_ratio() -> float:
    """Read EASYBCI_MEMORY_FOOTPRINT_WARN_RATIO from env, clamp to [0.1, 0.95]."""
    raw = os.environ.get("EASYBCI_MEMORY_FOOTPRINT_WARN_RATIO", "").strip()
    if not raw:
        return _DEFAULT_WARN_RATIO
    try:
        v = float(raw)
    except ValueError:
        return _DEFAULT_WARN_RATIO
    return max(0.1, min(0.95, v))


def build_memory_footprint(
    peak_native_mb: Optional[float],
    memory_budget_mb: Optional[float],
    analysis_goal: Optional[str],
    current_sfreq_hz: Optional[float],
) -> Optional[Dict[str, Any]]:
    """Build a structured memory-footprint advisory for the tool return.

    Args:
        peak_native_mb: Estimated peak working-set of this file at native
            sampling rate (from batch_memory_plan.json:peak_native_max_mb).
        memory_budget_mb: Available memory budget in MB (from
            batch_memory_plan.json:memory_budget_mb).
        analysis_goal: One of the built-in goals or a third-party goal
            name. Consulted for min_sfreq_hz recommendation.
        current_sfreq_hz: The file's native sampling rate (Hz).

    Returns:
        None when memory usage looks safe (ratio < warn threshold, or
        inputs missing/invalid). Otherwise a dict with:
            {
                "peak_native_mb": float,
                "budget_mb": float,
                "ratio": float,        # peak/budget
                "warn_threshold": float,
                "severity": "warn" | "critical",  # critical: ratio > 1.0
                "goal_min_sfreq_hz": float | None,
                "current_sfreq_hz": float | None,
                "recommendation": str,   # human-readable one-liner
            }
    """
    try:
        peak = float(peak_native_mb) if peak_native_mb is not None else None
        budget = float(memory_budget_mb) if memory_budget_mb is not None else None
    except (TypeError, ValueError):
        return None
    if not (peak and budget and peak > 0 and budget > 0):
        return None

    ratio = peak / budget
    warn_thresh = _env_warn_ratio()
    if ratio < warn_thresh:
        return None

    severity = "critical" if ratio > 1.0 else "warn"

    goal_min_sfreq: Optional[float] = None
    if analysis_goal:
        spec = REGISTRY.get(analysis_goal)
        if spec is not None:
            goal_min_sfreq = spec.min_sfreq_hz

    if severity == "critical":
        head = (
            f"Your file's peak native memory footprint (~{peak:.0f} MB) "
            f"EXCEEDS the current budget (~{budget:.0f} MB, ratio={ratio:.1f}x). "
            f"Pipeline.py will hit runtime decimation fallback or OOM."
        )
    else:
        head = (
            f"Your file's peak native memory footprint (~{peak:.0f} MB) "
            f"is {ratio*100:.0f}% of the current budget (~{budget:.0f} MB)."
        )

    if goal_min_sfreq and current_sfreq_hz and current_sfreq_hz > goal_min_sfreq:
        reduction = (1.0 - (goal_min_sfreq / current_sfreq_hz)) * 100
        tail = (
            f" Consider adding `resample:{int(goal_min_sfreq)}` to your steps: "
            f"analysis_goal='{analysis_goal}' only requires >= {goal_min_sfreq:.0f} Hz "
            f"(you're at {current_sfreq_hz:.0f} Hz -- a {reduction:.0f}% data reduction)."
        )
    elif goal_min_sfreq and current_sfreq_hz and current_sfreq_hz <= goal_min_sfreq:
        tail = (
            f" Your current sampling rate ({current_sfreq_hz:.0f} Hz) is already at or "
            f"below the analysis_goal's Nyquist floor ({goal_min_sfreq:.0f} Hz) -- "
            f"downsampling is not an option. Consider excluding this file, running "
            f"batch_process_adaptive with a smaller EASYBCI_BATCH_CHUNK, or using a "
            f"machine with more RAM."
        )
    else:
        tail = (
            " Consider adding a `resample:<sfreq>` step to reduce data size -- "
            "a value that satisfies your analysis's Nyquist requirement is safest. "
            "Or exclude this file and run on a larger machine."
        )

    return {
        "peak_native_mb": round(peak, 1),
        "budget_mb": round(budget, 1),
        "ratio": round(ratio, 3),
        "warn_threshold": warn_thresh,
        "severity": severity,
        "goal_min_sfreq_hz": goal_min_sfreq,
        "current_sfreq_hz": (float(current_sfreq_hz) if current_sfreq_hz else None),
        "recommendation": head + tail,
    }


def format_footprint_prompt_block(advisory: Optional[Dict[str, Any]]) -> str:
    """Render the advisory dict as a human-readable prompt block for
    reasoning.md / proposal envelope injection. Empty string when advisory
    is None or falsy."""
    if not advisory:
        return ""
    lines = [
        "## Memory footprint advisory",
        "",
        f"- Peak native working-set: {advisory['peak_native_mb']:.0f} MB",
        f"- Available budget:        {advisory['budget_mb']:.0f} MB",
        f"- Ratio:                   {advisory['ratio']:.2f}x  (severity: {advisory['severity']})",
    ]
    if advisory.get("goal_min_sfreq_hz"):
        lines.append(f"- Goal's Nyquist floor:    {advisory['goal_min_sfreq_hz']:.0f} Hz")
    if advisory.get("current_sfreq_hz"):
        lines.append(f"- File's native sfreq:     {advisory['current_sfreq_hz']:.0f} Hz")
    lines.append("")
    lines.append(f"**Recommendation:** {advisory['recommendation']}")
    return "\n".join(lines)
