"""Conflict resolver — domain-skill defaults vs. web-research recommendations.

Reconciles ``research_preprocessing`` output against the per-paradigm
PIPELINE_RECOMMENDATIONS table:

| Situation                           | Action                                              |
|-------------------------------------|-----------------------------------------------------|
| recommendation == default           | direct adoption, conflicts=[]                       |
| recommendation != default, conf < 0.5 | keep skill default, decision=ignore_web           |
| recommendation != default, conf ≥ 0.5 | re-research the contested param, use re-research |
| re-research inconclusive            | keep skill default, decision=ignore_web_inconclusive |

The re-research step calls ``research_parameter`` (the existing single-
parameter tool). All inputs/outputs are plain dicts so the resolver is easy
to unit-test in isolation; no I/O, no tool-registry access here — the
``_research_parameter`` callback is injected by the caller (defaults to a
real registry lookup).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


_HIGH_CONFIDENCE_THRESHOLD = 0.5
_RERESEARCH_MIN_CONFIDENCE = 0.5


def _default_research_parameter(*, operator: str, parameter: str, **context: Any) -> Optional[Dict[str, Any]]:
    """Default re-research callback — invokes the registered ``research_parameter`` tool.

    Returns the parsed dict response, or ``None`` if the tool is unavailable
    or returned a non-success envelope.
    """
    try:
        from easybci_lib.tools.registry import registry
    except ImportError:
        return None
    entry = registry.get_entry("research_parameter")
    if entry is None or entry.handler is None:
        return None
    try:
        raw = entry.handler({
            "operator": operator,
            "parameter": parameter,
            **context,
        })
    except Exception as exc:
        logger.debug("research_parameter call raised: %s", exc)
        return None
    if isinstance(raw, dict):
        result = raw
    elif isinstance(raw, str):
        try:
            result = json.loads(raw)
        except (TypeError, ValueError):
            return None
    else:
        return None
    if not result.get("success"):
        return None
    return result


def resolve_conflicts(
    skill_defaults: Dict[str, Any],
    web_recos: Dict[str, Any],
    confidence: float,
    *,
    operator: str = "",
    modality: str = "",
    paradigm: str = "",
    research_parameter: Optional[Callable[..., Optional[Dict[str, Any]]]] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Reconcile skill_defaults vs. web_recos at parameter level.

    Parameters
    ----------
    skill_defaults : dict
        ``{param_name: value}`` from the domain-skill registry.
    web_recos : dict
        ``{param_name: value}`` extracted from the web evidence.
    confidence : float
        Overall confidence of the web research (0–1). Per-parameter
        confidences override this if available; this is the global default.
    operator, modality, paradigm : str
        Forwarded to ``research_parameter`` when re-research fires.
    research_parameter : callable, optional
        Override the re-research callback. Default is the real registry tool.

    Returns
    -------
    final_params : dict
        Merged parameter map (every key from skill_defaults present;
        web-only keys appended when they don't collide).
    conflicts : list of dict
        Per-disagreement record with keys ``param``, ``skill_default``,
        ``web_recommended``, ``decision``, ``reason``. Decisions:
          - ``follow_web`` (re-research confirmed)
          - ``ignore_web`` (web confidence < 0.5)
          - ``ignore_web_inconclusive`` (re-research returned no value)
    """
    research_fn = research_parameter or _default_research_parameter
    final: Dict[str, Any] = dict(skill_defaults or {})
    conflicts: List[Dict[str, Any]] = []

    web_recos = web_recos or {}

    for param, web_value in web_recos.items():
        if param not in final:
            # web-only key: adopt without conflict.
            final[param] = web_value
            continue
        skill_value = final[param]
        if skill_value == web_value:
            continue  # exact match, no conflict to record

        # Disagreement.
        if confidence < _HIGH_CONFIDENCE_THRESHOLD:
            conflicts.append({
                "param": param,
                "skill_default": skill_value,
                "web_recommended": web_value,
                "decision": "ignore_web",
                "reason": (
                    f"web confidence {confidence:.2f} < {_HIGH_CONFIDENCE_THRESHOLD}; "
                    "keeping domain-skill default"
                ),
            })
            continue

        # confidence ≥ 0.5 → re-research this single parameter.
        rr = research_fn(
            operator=operator,
            parameter=param,
            modality=modality,
            paradigm=paradigm,
        )
        rr_value = (rr or {}).get("value")
        rr_conf = float((rr or {}).get("confidence") or 0.0)

        if rr is None or rr_value is None or rr_conf < _RERESEARCH_MIN_CONFIDENCE:
            conflicts.append({
                "param": param,
                "skill_default": skill_value,
                "web_recommended": web_value,
                "decision": "ignore_web_inconclusive",
                "reason": (
                    f"re-research returned no usable value "
                    f"(rr_value={rr_value!r}, rr_conf={rr_conf:.2f}); "
                    "keeping domain-skill default"
                ),
            })
            continue

        final[param] = rr_value
        conflicts.append({
            "param": param,
            "skill_default": skill_value,
            "web_recommended": web_value,
            "decision": "follow_web",
            "reason": (
                f"re-research via research_parameter returned "
                f"{rr_value!r} with confidence {rr_conf:.2f}"
            ),
            "rerun": {
                "value": rr_value,
                "confidence": rr_conf,
            },
        })

    return final, conflicts


__all__ = ["resolve_conflicts"]
