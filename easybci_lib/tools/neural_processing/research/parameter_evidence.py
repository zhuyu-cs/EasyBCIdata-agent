"""ParameterEvidence and Citation dataclasses — single canonical serialization
for per-parameter evidence in proposed pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional, Tuple

SourceLiteral = Literal[
    "web", "empirical_default", "registry_miss", "user_provided"
]


@dataclass(frozen=True)
class Citation:
    url: str
    title: str = ""
    snippet: str = ""

    def to_dict(self) -> dict:
        return {"url": self.url, "title": self.title, "snippet": self.snippet}

    @classmethod
    def from_dict(cls, d: dict) -> "Citation":
        return cls(
            url=d.get("url", ""),
            title=d.get("title", ""),
            snippet=d.get("snippet", ""),
        )


@dataclass(frozen=True)
class ParameterEvidence:
    operator: str
    parameter: str
    value: Any
    source: SourceLiteral
    confidence: float
    citations: Tuple[Citation, ...] = ()
    summary: str = ""
    rationale: str = ""
    default_origin: str = ""
    fallback_reason: str = ""
    cache_key: str = ""
    registry_version: str = ""
    needs_user: bool = False
    attempted_evidence: Optional[dict] = None
    previous_evidence: Optional["ParameterEvidence"] = None

    def to_dict(self) -> dict:
        d = {
            "operator": self.operator,
            "parameter": self.parameter,
            "value": self.value,
            "source": self.source,
            "confidence": self.confidence,
            "citations": [c.to_dict() for c in self.citations],
            "summary": self.summary,
            "rationale": self.rationale,
            "default_origin": self.default_origin,
            "fallback_reason": self.fallback_reason,
            "cache_key": self.cache_key,
            "registry_version": self.registry_version,
            "needs_user": self.needs_user,
            "attempted_evidence": self.attempted_evidence,
        }
        if self.previous_evidence is not None:
            d["previous_evidence"] = self.previous_evidence.to_dict()
        return d

    def to_dict_json(self) -> str:
        import json
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def from_dict(cls, d: dict) -> "ParameterEvidence":
        prev = d.get("previous_evidence")
        return cls(
            operator=d.get("operator", ""),
            parameter=d.get("parameter", ""),
            value=d.get("value"),
            source=d.get("source", "empirical_default"),
            confidence=float(d.get("confidence", 0.0)),
            citations=tuple(Citation.from_dict(c) for c in d.get("citations", []) or []),
            summary=d.get("summary", ""),
            rationale=d.get("rationale", "") or "",
            default_origin=d.get("default_origin", ""),
            fallback_reason=d.get("fallback_reason", ""),
            cache_key=d.get("cache_key", ""),
            registry_version=d.get("registry_version", ""),
            needs_user=bool(d.get("needs_user", False)),
            attempted_evidence=d.get("attempted_evidence"),
            previous_evidence=cls.from_dict(prev) if prev else None,
        )
