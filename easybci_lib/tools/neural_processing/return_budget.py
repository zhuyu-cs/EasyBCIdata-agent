"""Global tool-return size gate for neural tools.

Progressively strips low-priority fields when a payload exceeds the
token budget before it enters the LLM context window.

Design principle: information is never LOST — it is either kept inline
(compact summary), offloaded to disk (with a path reference), or made
available via inspect_detail. The gate only changes the DENSITY of
information in the LLM context, not its availability.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

MAX_RETURN_CHARS = 4000

_LOW_PRIORITY_KEYS = frozenset({
    "note", "hint", "scenario_bias", "negatives_hint",
})

_TRUNCATABLE_LIST_KEYS = frozenset({
    "warnings", "stdout_tail", "stderr_tail", "sample_rows",
    "all_signal_files", "sidecar_files",
})

# Fields that must NEVER be truncated — they are pass-through content
# the LLM shows to the user verbatim, or contain disk paths needed for
# downstream tools.
_PROTECTED_KEYS = frozenset({
    "presentation_block", "work_dir", "report_path", "staged_path",
    "success", "error", "fix_hint", "next_action",
    "presented_steps_expected", "awaiting_confirmation",
})

_MAX_LIST_ITEMS = 8


def cap_return(payload: dict[str, Any], budget: int = MAX_RETURN_CHARS) -> str:
    """Serialize payload to JSON; if over budget, progressively strip fields.

    Degradation order:
      1. Remove low-priority instructional keys (note/hint/scenario_bias)
      2. Truncate truncatable list fields to _MAX_LIST_ITEMS
      3. Adaptive string truncation on NON-PROTECTED fields
      4. If still over: keep only protected keys + fingerprint + summary

    Returns the JSON string. Protected fields are never altered.
    """
    out = json.dumps(payload, ensure_ascii=False, default=str)
    if len(out) <= budget:
        return out

    for key in _LOW_PRIORITY_KEYS:
        payload.pop(key, None)
    out = json.dumps(payload, ensure_ascii=False, default=str)
    if len(out) <= budget:
        return out

    _truncate_lists(payload)
    out = json.dumps(payload, ensure_ascii=False, default=str)
    if len(out) <= budget:
        return out

    _adaptive_truncate_strings(payload, budget)
    out = json.dumps(payload, ensure_ascii=False, default=str)
    if len(out) <= budget:
        return out

    # Last resort: shed non-protected, non-essential keys entirely.
    # Information is NOT lost — it exists on disk (report_path, staged_path).
    essential = {}
    for key, val in payload.items():
        if key in _PROTECTED_KEYS:
            essential[key] = val
        elif key == "report" and isinstance(val, dict):
            essential["report"] = {
                k: v for k, v in val.items()
                if k in ("fingerprint", "channel_quality_summary", "degraded")
            }
        elif key in ("fingerprint", "channel_quality_summary",
                     "analysis_goal", "modality", "paradigm", "degraded",
                     "elapsed_s", "proven_recommendation"):
            essential[key] = val
    essential["_compacted"] = True
    essential["_retrieve_detail"] = "Use inspect_detail tool for full data."
    return json.dumps(essential, ensure_ascii=False, default=str)


def _truncate_lists(d: dict[str, Any]) -> None:
    for key, val in list(d.items()):
        if isinstance(val, list) and key in _TRUNCATABLE_LIST_KEYS and len(val) > _MAX_LIST_ITEMS:
            d[key] = val[:_MAX_LIST_ITEMS]
            d[f"_{key}_total"] = len(val)
        elif isinstance(val, dict):
            _truncate_lists(val)


def _adaptive_truncate_strings(payload: dict[str, Any], budget: int) -> None:
    """Adaptively truncate NON-PROTECTED string fields based on available budget.

    Strategy: measure total eligible string content vs available headroom,
    then distribute the cut proportionally. Protected keys (presentation_block,
    paths, errors) are never touched.
    """
    out = json.dumps(payload, ensure_ascii=False, default=str)
    overshoot = len(out) - budget
    if overshoot <= 0:
        return

    long_strings = _collect_long_strings(payload, min_len=80)
    if not long_strings:
        return

    total_string_chars = sum(info["len"] for info in long_strings)
    chars_to_cut = min(overshoot + 200, total_string_chars)

    for info in long_strings:
        proportion = info["len"] / total_string_chars
        cut = int(chars_to_cut * proportion)
        new_len = max(info["len"] - cut, 60)
        if new_len < info["len"]:
            original = info["ref"][info["key"]]
            info["ref"][info["key"]] = original[:new_len] + f"...[{info['len']}]"


def _collect_long_strings(
    d: dict[str, Any] | list, min_len: int = 80, _out: list | None = None,
    _parent_key: str = "",
) -> list[dict[str, Any]]:
    """Walk payload tree and collect references to truncatable string fields.

    Skips fields in _PROTECTED_KEYS.
    """
    if _out is None:
        _out = []
    if isinstance(d, dict):
        for key, val in d.items():
            if key in _PROTECTED_KEYS:
                continue
            if isinstance(val, str) and len(val) >= min_len:
                _out.append({"ref": d, "key": key, "len": len(val)})
            elif isinstance(val, (dict, list)):
                _collect_long_strings(val, min_len, _out, _parent_key=key)
    elif isinstance(d, list):
        for i, item in enumerate(d):
            if isinstance(item, str) and len(item) >= min_len:
                _out.append({"ref": d, "key": i, "len": len(item)})
            elif isinstance(item, (dict, list)):
                _collect_long_strings(item, min_len, _out)
    return _out
