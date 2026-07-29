"""Research → Skill internalization — web knowledge lifecycle management.

Extends the research module with:
1. EvidenceReport caching: stores synthesized reports (not just raw search results)
2. High-confidence internalization: writes proven recommendations into domain skills
3. Knowledge expiration: 6-month TTL with re-verification before use

Storage:
  - Evidence cache: ~/.easybci/cache/evidence/<hash>.json
  - Internalized log: ~/.easybci/internalized_knowledge.json
"""

import datetime as _dt
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from easybci_lib.constants import get_easybci_home, get_skills_dir

from .internalization_audit import AuditEvent, InternalizationAuditLog
from .internalization_marker import (
    compute_internalization_id,
    find_marker_block,
    strip_marker_block,
    wrap_with_marker,
)

logger = logging.getLogger(__name__)

_EVIDENCE_TTL_SECONDS = 30 * 24 * 3600  # 30 days for evidence cache
_KNOWLEDGE_TTL_SECONDS = 180 * 24 * 3600  # 6 months for internalized knowledge
_INTERNALIZED_PATH_DEFAULT = Path.home() / ".easybci" / "internalized_knowledge.json"


@dataclass
class InternalizedKnowledge:
    """A piece of knowledge internalized from web research into a domain skill."""
    skill_name: str
    section: str
    content: str
    source_query: str
    source_urls: List[str] = field(default_factory=list)
    confidence: float = 0.0
    internalized_at: float = 0.0
    expires_at: float = 0.0
    verified: bool = False
    verification_count: int = 0
    internalization_id: str = ""  # 8-hex id for marker block; empty for legacy entries

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill_name": self.skill_name,
            "section": self.section,
            "content": self.content,
            "source_query": self.source_query,
            "source_urls": self.source_urls,
            "confidence": round(self.confidence, 3),
            "internalized_at": self.internalized_at,
            "expires_at": self.expires_at,
            "verified": self.verified,
            "verification_count": self.verification_count,
            "internalization_id": self.internalization_id,
        }

    def is_expired(self) -> bool:
        return time.time() > self.expires_at


@dataclass
class RevokeResult:
    """Returned by KnowledgeInternalizer.revoke()."""
    skill_path: str
    internalization_id: str
    removed_lines: int
    reason: str


class EvidenceCache:
    """Extended cache that stores full synthesized EvidenceReports.

    Unlike SearchCache (raw results, 7-day TTL), this stores processed
    evidence with longer TTL (30 days) and tracks confidence + usage.
    """

    def __init__(self, cache_dir: Optional[Path] = None, ttl_seconds: int = _EVIDENCE_TTL_SECONDS):
        if cache_dir is None:
            try:
                cache_dir = get_easybci_home() / "cache" / "evidence"
            except Exception:  # noqa: BLE001
                cache_dir = Path.home() / ".easybci" / "cache" / "evidence"
        self._cache_dir = cache_dir
        self._ttl = ttl_seconds

    def _ensure_dir(self) -> None:
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _make_key(self, modality: str, paradigm: str, question: str) -> str:
        raw = f"{modality.lower()}|{paradigm.lower()}|{question.lower().strip()}"
        return hashlib.sha256(raw.encode(encoding="utf-8")).hexdigest()[:24]

    def get(self, modality: str, paradigm: str, question: str) -> Optional[Dict[str, Any]]:
        """Retrieve a cached EvidenceReport if valid."""
        key = self._make_key(modality, paradigm, question)
        path = self._cache_dir / f"{key}.json"

        if not path.exists():
            return None

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

        cached_at = data.get("cached_at", 0)
        if time.time() - cached_at > self._ttl:
            try:
                path.unlink()
            except OSError:
                pass
            return None

        # Track usage
        data["access_count"] = data.get("access_count", 0) + 1
        data["last_accessed"] = time.time()
        try:
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError:
            pass

        return data.get("report")

    def put(
        self,
        modality: str,
        paradigm: str,
        question: str,
        report: Dict[str, Any],
    ) -> None:
        """Store a synthesized EvidenceReport."""
        self._ensure_dir()
        key = self._make_key(modality, paradigm, question)
        path = self._cache_dir / f"{key}.json"

        envelope = {
            "cached_at": time.time(),
            "modality": modality,
            "paradigm": paradigm,
            "question": question,
            "report": report,
            "access_count": 0,
            "last_accessed": time.time(),
        }

        try:
            path.write_text(
                json.dumps(envelope, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.warning("Evidence cache write failed: %s", exc)

    def get_high_confidence_reports(self, min_confidence: float = 0.8) -> List[Dict[str, Any]]:
        """Get all cached reports above a confidence threshold."""
        if not self._cache_dir.exists():
            return []

        results = []
        now = time.time()

        for path in self._cache_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if now - data.get("cached_at", 0) > self._ttl:
                    continue
                report = data.get("report", {})
                if report.get("confidence", 0) >= min_confidence:
                    results.append({
                        "modality": data.get("modality", ""),
                        "paradigm": data.get("paradigm", ""),
                        "question": data.get("question", ""),
                        "report": report,
                        "access_count": data.get("access_count", 0),
                    })
            except (json.JSONDecodeError, OSError):
                continue

        return results


class KnowledgeInternalizer:
    """Manages the lifecycle of internalized web knowledge in domain skills.

    When a high-confidence EvidenceReport is confirmed by the user, this class:
    1. Appends the recommendation to the relevant domain skill file
    2. Records it in the internalized_knowledge.json log with a 6-month TTL
    3. On expiration, marks the knowledge as needing re-verification
    """

    def __init__(self, log_path: Optional[Path] = None, skills_dir: Optional[Path] = None):
        self._log_path = log_path or _INTERNALIZED_PATH_DEFAULT
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: List[Dict[str, Any]] = self._load_log()

        if skills_dir is None:
            try:
                skills_dir = get_skills_dir() / "bci" / "paradigms"
            except Exception:  # noqa: BLE001
                skills_dir = Path(__file__).resolve().parent.parent.parent.parent / "skills" / "bci" / "paradigms"
        self._skills_dir = skills_dir

    def _load_log(self) -> List[Dict[str, Any]]:
        if self._log_path.exists():
            try:
                return json.loads(self._log_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
        return []

    def _save_log(self) -> None:
        tmp = self._log_path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(self._entries, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp.rename(self._log_path)

    def internalize(
        self,
        skill_name: str,
        report: Dict[str, Any],
        question: str,
        user_confirmed: bool = True,
    ) -> Optional[InternalizedKnowledge]:
        """Internalize a high-confidence EvidenceReport into a domain skill.

        Parameters
        ----------
        skill_name : str
            Target skill file name (without .md extension).
        report : dict
            The EvidenceReport dict (with recommendations, parameters, rationale).
        question : str
            The original research question.
        user_confirmed : bool
            Whether the user explicitly confirmed this recommendation.

        Returns
        -------
        InternalizedKnowledge if successfully written, None otherwise.
        """
        confidence = report.get("confidence", 0)
        if confidence < 0.8 and not user_confirmed:
            logger.info("Skipping internalization: confidence %.2f < 0.8 and not user-confirmed", confidence)
            return None

        recommendations = report.get("recommendations", [])
        rationale = report.get("rationale", "")
        citations = report.get("citations", [])
        caveats = report.get("caveats", [])

        if not recommendations and not rationale:
            return None

        # Build the content block to append
        content_lines = [
            "",
            f"### Research-Based Suggestion (confidence: {confidence:.0%})",
            "",
            f"**Question**: {question}",
            "",
        ]
        if recommendations:
            content_lines.append(f"**Recommended steps**: `{'` → `'.join(recommendations)}`")
            content_lines.append("")
        if rationale:
            content_lines.append(f"**Rationale**: {rationale}")
            content_lines.append("")
        if caveats:
            content_lines.append("**Caveats**:")
            for caveat in caveats:
                content_lines.append(f"- {caveat}")
            content_lines.append("")
        source_urls: List[str] = []
        if citations:
            source_urls = [c.get("url", "") for c in citations[:5] if c.get("url")]
            if source_urls:
                content_lines.append(f"*Sources*: {', '.join(source_urls[:3])}")
                content_lines.append("")

        now = time.time()
        expires = now + _KNOWLEDGE_TTL_SECONDS
        content = "\n".join(content_lines)

        # Write to the skill file
        skill_path = self._skills_dir / f"{skill_name}.md"
        if not skill_path.exists():
            logger.warning("Skill file not found: %s", skill_path)
            return None

        try:
            existing = skill_path.read_text(encoding="utf-8")
            source_url_for_id = source_urls[0] if source_urls else question
            internalization_id = compute_internalization_id(
                skill_path=str(skill_path),
                source_url=source_url_for_id,
                content=content.strip(),
            )
            if find_marker_block(existing, internalization_id=internalization_id) is not None:
                logger.debug("Marker block %s already present in %s", internalization_id, skill_name)
                return None

            date_iso = _dt.datetime.utcfromtimestamp(now).strftime("%Y-%m-%d")
            wrapped = wrap_with_marker(
                body=content,
                internalization_id=internalization_id,
                date_iso=date_iso,
                source=source_url_for_id,
                confidence=("high" if confidence >= 0.8 else "provisional"),
            )

            section_header = "## Web Research Insights"
            if section_header not in existing:
                updated = existing.rstrip() + "\n\n" + section_header + "\n\n" + wrapped + "\n"
            else:
                updated = existing.rstrip() + "\n\n" + wrapped + "\n"

            skill_path.write_text(updated, encoding="utf-8")
        except OSError as exc:
            logger.warning("Failed to write to skill %s: %s", skill_name, exc)
            return None

        # Log the internalization
        knowledge = InternalizedKnowledge(
            skill_name=skill_name,
            section="Web Research Insights",
            content=content.strip(),
            source_query=question,
            source_urls=source_urls,
            confidence=confidence,
            internalized_at=now,
            expires_at=expires,
            verified=user_confirmed,
            verification_count=1 if user_confirmed else 0,
            internalization_id=internalization_id,
        )

        self._entries.append(knowledge.to_dict())
        self._save_log()

        # Also record this internalisation as an event in the jsonl audit log.
        try:
            audit = InternalizationAuditLog(
                path=get_easybci_home() / "internalization_audit.jsonl"
            )
            audit.append(AuditEvent(
                event="internalize",
                internalization_id=internalization_id,
                skill_path=str(skill_path),
                source_url=source_url_for_id,
                content_excerpt=content.strip()[:200],
                timestamp_iso=_dt.datetime.utcfromtimestamp(now).isoformat() + "Z",
                confidence=("high" if confidence >= 0.8 else "provisional"),
            ))
        except Exception:  # noqa: BLE001
            logger.debug("audit log write failed", exc_info=True)

        logger.info("Internalized knowledge into skill '%s' (confidence: %.0f%%)", skill_name, confidence * 100)
        return knowledge

    def get_expired(self) -> List[InternalizedKnowledge]:
        """Get all internalized knowledge entries that have expired (need re-verification)."""
        now = time.time()
        expired = []
        for entry in self._entries:
            if entry.get("expires_at", 0) < now:
                expired.append(InternalizedKnowledge(**entry))
        return expired

    def mark_verified(self, skill_name: str, source_query: str) -> bool:
        """Mark a knowledge entry as re-verified, extending its TTL."""
        now = time.time()
        for entry in self._entries:
            if entry["skill_name"] == skill_name and entry["source_query"] == source_query:
                entry["verified"] = True
                entry["verification_count"] = entry.get("verification_count", 0) + 1
                entry["expires_at"] = now + _KNOWLEDGE_TTL_SECONDS
                self._save_log()
                return True
        return False

    def revoke(
        self,
        *,
        internalization_id: str,
        reason: str,
        allow_unsafe: bool = False,
    ) -> "RevokeResult":
        """Revoke a previously-internalized knowledge block.

        Steps:
        1. Find entry in self._entries by internalization_id
        2. Locate marker block in skill file
        3. Verify body still matches stored content (unless allow_unsafe)
        4. Strip marker block; remove entry; persist; emit audit `revoke` event.
        """
        entry = None
        for e in self._entries:
            if e.get("internalization_id") == internalization_id and internalization_id:
                entry = e
                break
        if entry is None:
            raise KeyError(f"no such internalization: {internalization_id}")

        skill_name = entry["skill_name"]
        skill_path = self._skills_dir / f"{skill_name}.md"
        if not skill_path.exists():
            raise FileNotFoundError(f"skill file missing: {skill_path}")

        existing = skill_path.read_text(encoding="utf-8")
        found_body = find_marker_block(existing, internalization_id=internalization_id)
        if found_body is None:
            raise RuntimeError(
                f"marker block {internalization_id} not in {skill_path}; was the file edited?"
            )

        # hand-edit detection: marker body must equal what we stored
        if not allow_unsafe and found_body.strip() != entry.get("content", "").strip():
            raise RuntimeError(
                "marker body was hand-edited (body differs from stored content); "
                "pass allow_unsafe=True to revoke anyway."
            )

        new_doc, removed_lines = strip_marker_block(existing, internalization_id=internalization_id)
        skill_path.write_text(new_doc, encoding="utf-8")

        self._entries.remove(entry)
        self._save_log()

        try:
            audit = InternalizationAuditLog(
                path=get_easybci_home() / "internalization_audit.jsonl"
            )
            audit.append(AuditEvent(
                event="revoke",
                internalization_id=internalization_id,
                skill_path=str(skill_path),
                source_url=(entry.get("source_urls") or [""])[0],
                content_excerpt=entry.get("content", "")[:200],
                timestamp_iso=_dt.datetime.utcnow().isoformat() + "Z",
                confidence=str(entry.get("confidence", "")),
                revoke_reason=reason,
            ))
        except Exception:  # noqa: BLE001
            logger.debug("audit write on revoke failed", exc_info=True)

        return RevokeResult(
            skill_path=str(skill_path),
            internalization_id=internalization_id,
            removed_lines=removed_lines,
            reason=reason,
        )

    def remove_expired_from_skills(self) -> List[str]:
        """Remove expired knowledge blocks from skill files.

        Returns list of skill names that were modified.
        """
        now = time.time()
        modified_skills = []

        for entry in list(self._entries):
            if entry.get("expires_at", 0) >= now:
                continue

            skill_name = entry["skill_name"]
            content = entry.get("content", "")
            skill_path = self._skills_dir / f"{skill_name}.md"

            if not skill_path.exists() or not content:
                continue

            try:
                existing = skill_path.read_text(encoding="utf-8")
                if content in existing:
                    updated = existing.replace(content, "")
                    # Clean up double blank lines
                    while "\n\n\n" in updated:
                        updated = updated.replace("\n\n\n", "\n\n")
                    skill_path.write_text(updated, encoding="utf-8")
                    modified_skills.append(skill_name)
                    logger.info("Removed expired knowledge from skill '%s'", skill_name)
            except OSError:
                continue

            # Remove from log
            self._entries.remove(entry)

        if modified_skills:
            self._save_log()

        return modified_skills

    def get_all(self) -> List[InternalizedKnowledge]:
        """Get all internalized knowledge entries."""
        return [InternalizedKnowledge(**e) for e in self._entries]

    def get_stats(self) -> Dict[str, Any]:
        """Get summary statistics."""
        now = time.time()
        total = len(self._entries)
        expired = sum(1 for e in self._entries if e.get("expires_at", 0) < now)
        verified = sum(1 for e in self._entries if e.get("verified", False))
        skills = list(set(e.get("skill_name", "") for e in self._entries))

        return {
            "total_entries": total,
            "expired": expired,
            "active": total - expired,
            "verified": verified,
            "skills_touched": skills,
        }


def try_internalize_from_evidence(
    report: Dict[str, Any],
    modality: str,
    paradigm: str,
    question: str,
    user_confirmed: bool = True,
) -> Optional[Dict[str, Any]]:
    """Convenience function: attempt to internalize an EvidenceReport.

    Determines the target skill from modality/paradigm and delegates to
    KnowledgeInternalizer.

    Returns the internalized knowledge dict, or None if not applicable.
    """
    confidence = report.get("confidence", 0)
    if confidence < 0.8 and not user_confirmed:
        return None

    skill_name = _resolve_skill_name(modality, paradigm)
    if not skill_name:
        return None

    internalizer = KnowledgeInternalizer()
    knowledge = internalizer.internalize(
        skill_name=skill_name,
        report=report,
        question=question,
        user_confirmed=user_confirmed,
    )

    return knowledge.to_dict() if knowledge else None


def _resolve_skill_name(modality: str, paradigm: str) -> str:
    """Map modality + paradigm to the target skill file name."""
    paradigm_map = {
        "motor_imagery": "motor_imagery",
        "mi": "motor_imagery",
        "p300": "p300_erp",
        "erp": "p300_erp",
        "ssvep": "ssvep",
        "sleep": "sleep_staging",
        "emotion": "emotion_recognition",
        "neurofeedback": "eeg_general",
        "tms": "eeg_general",
    }

    modality_map = {
        "eeg": "eeg_general",
        "ecog": "ecog",
        "seeg": "ieeg_depth",
        "meg": "meg",
        "fnirs": "fnirs",
        "spike": "eeg_general",
    }

    # Paradigm takes priority (more specific)
    paradigm_lower = paradigm.lower().replace("-", "_").replace(" ", "_")
    for key, skill in paradigm_map.items():
        if key in paradigm_lower:
            return skill

    # Fall back to modality
    modality_lower = modality.lower()
    return modality_map.get(modality_lower, "eeg_general")
