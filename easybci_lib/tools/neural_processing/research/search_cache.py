"""File-based search result cache for BCI preprocessing research.

Avoids redundant web searches for the same questions. Cached by
(modality, paradigm, question_hash) with a 7-day TTL.

Storage: ~/.easybci/cache/research/<hash>.json
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_TTL_SECONDS = 7 * 24 * 3600  # 7 days

# Bumped when the cached payload schema changes. Old keys hash to a
# different digest and stale files become orphans (cleared by TTL or
# clear_expired). Bump alongside any breaking change to the dict that
# `_handle_research_preprocessing` / `_research_for_suggestion` cache.
_CACHE_SCHEMA_VERSION = "v2"


class SearchCache:
    """File-based search result cache."""

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        ttl_seconds: int = _DEFAULT_TTL_SECONDS,
    ):
        if cache_dir is None:
            try:
                from easybci_lib.constants import get_easybci_home
                cache_dir = get_easybci_home() / "cache" / "research"
            except ImportError:
                cache_dir = Path.home() / ".easybci" / "cache" / "research"
        self._cache_dir = cache_dir
        self._ttl = ttl_seconds

    def _ensure_dir(self) -> None:
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _make_key(self, modality: str, paradigm: str, question: str) -> str:
        """Create a stable cache key from search parameters."""
        raw = (
            f"v={_CACHE_SCHEMA_VERSION}|"
            f"{modality.lower()}|{paradigm.lower()}|{question.lower().strip()}"
        )
        return hashlib.sha256(raw.encode(encoding="utf-8")).hexdigest()[:24]

    def _path_for_key(self, key: str) -> Path:
        return self._cache_dir / f"{key}.json"

    def get(
        self,
        modality: str,
        paradigm: str,
        question: str,
    ) -> Optional[Dict[str, Any]]:
        """Retrieve cached result if it exists and hasn't expired.

        Returns None on miss (not found or expired).
        """
        key = self._make_key(modality, paradigm, question)
        path = self._path_for_key(key)

        if not path.exists():
            return None

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.debug("Cache read error for key %s: %s", key, exc)
            return None

        cached_at = data.get("cached_at", 0)
        if time.time() - cached_at > self._ttl:
            logger.debug("Cache expired for key %s", key)
            try:
                path.unlink()
            except OSError:
                pass
            return None

        logger.debug("Cache hit for key %s", key)
        return data.get("payload")

    def put(
        self,
        modality: str,
        paradigm: str,
        question: str,
        payload: Dict[str, Any],
    ) -> None:
        """Store a search result in the cache."""
        self._ensure_dir()
        key = self._make_key(modality, paradigm, question)
        path = self._path_for_key(key)

        envelope = {
            "cached_at": time.time(),
            "modality": modality,
            "paradigm": paradigm,
            "question": question,
            "payload": payload,
        }

        try:
            path.write_text(
                json.dumps(envelope, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.warning("Cache write failed for key %s: %s", key, exc)

    def invalidate(self, modality: str, paradigm: str, question: str) -> bool:
        """Remove a specific cache entry. Returns True if entry existed."""
        key = self._make_key(modality, paradigm, question)
        path = self._path_for_key(key)
        if path.exists():
            try:
                path.unlink()
                return True
            except OSError:
                pass
        return False

    def clear_expired(self) -> int:
        """Remove all expired entries. Returns count removed."""
        if not self._cache_dir.exists():
            return 0

        removed = 0
        now = time.time()

        for path in self._cache_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if now - data.get("cached_at", 0) > self._ttl:
                    path.unlink()
                    removed += 1
            except (json.JSONDecodeError, OSError):
                try:
                    path.unlink()
                    removed += 1
                except OSError:
                    pass

        return removed

    def stats(self) -> Dict[str, int]:
        """Return cache statistics."""
        if not self._cache_dir.exists():
            return {"total": 0, "valid": 0, "expired": 0, "size_bytes": 0}

        total = 0
        valid = 0
        size_bytes = 0
        now = time.time()

        for path in self._cache_dir.glob("*.json"):
            total += 1
            size_bytes += path.stat().st_size
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if now - data.get("cached_at", 0) <= self._ttl:
                    valid += 1
            except (json.JSONDecodeError, OSError):
                pass

        return {
            "total": total,
            "valid": valid,
            "expired": total - valid,
            "size_bytes": size_bytes,
        }

    # ----- Per-parameter keying (evidence-driven-params) -----

    def _make_param_key(
        self, modality: str, paradigm: str,
        parameter: str, registry_version: str,
    ) -> str:
        raw = (
            f"v={registry_version}|"
            f"m={modality.lower()}|"
            f"p={paradigm.lower()}|"
            f"param={parameter.lower()}"
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]

    def get_parameter(
        self, modality: str, paradigm: str,
        parameter: str, registry_version: str,
    ) -> Optional[Dict[str, Any]]:
        key = self._make_param_key(modality, paradigm, parameter, registry_version)
        path = self._path_for_key(key)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        if time.time() - data.get("cached_at", 0) > self._ttl:
            try:
                path.unlink()
            except OSError:
                pass
            return None
        return data.get("payload")

    def put_parameter(
        self, modality: str, paradigm: str,
        parameter: str, registry_version: str,
        result: Dict[str, Any],
    ) -> None:
        self._ensure_dir()
        key = self._make_param_key(modality, paradigm, parameter, registry_version)
        path = self._path_for_key(key)
        payload = {"cached_at": time.time(), "payload": result}
        path.write_text(json.dumps(payload), encoding="utf-8")
