"""cohort_taxonomy.yaml loader + cohort similarity scoring.

Used by the pluggable SimilarityDimension when cohort matching is enabled.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml

logger = logging.getLogger(__name__)

_TAXONOMY_PATH = Path(__file__).parent / "cohort_taxonomy.yaml"


@lru_cache(maxsize=1)
def load_taxonomy() -> Dict[str, Dict[str, Any]]:
    """Return {cohort_name: {"parent": str | None, ...}}. Empty dict if file missing."""
    if not _TAXONOMY_PATH.exists():
        logger.warning("cohort_taxonomy.yaml missing at %s", _TAXONOMY_PATH)
        return {}
    try:
        return yaml.safe_load(_TAXONOMY_PATH.read_text(encoding="utf-8")) or {}
    except Exception as exc:  # noqa: BLE001
        logger.warning("failed to load cohort_taxonomy: %s", exc)
        return {}


def cohort_similarity(a: str, b: str) -> float:
    """Return 1.0 for exact match, 0.5 for sibling (same parent), 0.0 else.

    Unknown cohorts or empty strings return 0.0.
    """
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    tax = load_taxonomy()
    pa = tax.get(a, {}).get("parent")
    pb = tax.get(b, {}).get("parent")
    if pa is None or pb is None:
        return 0.0
    if pa == pb:
        return 0.5
    return 0.0
