"""Redact privacy-sensitive frontmatter before pushing to federation.

Default policy: cohort_tag is REDACTED (clinical details). User must
explicitly opt-in with allow_cohort=True.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


_SENSITIVE_KEYS_DEFAULT: List[str] = ["cohort_tag"]


def redact_frontmatter(
    fm: Dict[str, Any],
    *,
    allow_cohort: bool = False,
    extra_redact: Optional[List[str]] = None,
) -> Dict[str, Any]:
    to_remove = list(_SENSITIVE_KEYS_DEFAULT) if not allow_cohort else []
    if extra_redact:
        to_remove.extend(extra_redact)
    return {k: v for k, v in fm.items() if k not in to_remove}
