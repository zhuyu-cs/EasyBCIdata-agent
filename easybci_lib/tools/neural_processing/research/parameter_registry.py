"""Parameter-uncertainty registry: 14 YAMLs (one per operator) declaring
which preprocessing parameters benefit from web research and supplying
deterministic empirical-default fallbacks.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml

logger = logging.getLogger(__name__)

ResearchTrigger = Literal[
    "paradigm_dependent", "region_dependent", "data_dependent",
    "never", "user_only",
]

_REGISTRY_DIR = Path(__file__).parent / "parameter_uncertainty"
_DEFAULT_KEY = "_default"


@dataclass(frozen=True)
class DefaultEntry:
    value: Any
    unit: str = ""
    # Structured origin (was a single string).
    # Back-compat: from_dict() accepts both legacy string-form and new object-form.
    origin_citation: str = ""
    origin_url: Optional[str] = None
    origin_doi: Optional[str] = None
    origin_arxiv_id: Optional[str] = None
    confidence_tier: str = "established"  # established | provisional | experimental
    last_verified: Optional[str] = None    # ISO date "YYYY-MM-DD"
    notes: str = ""

    @property
    def origin(self) -> str:
        """Back-compat alias: legacy callers reading .origin get the citation string."""
        return self.origin_citation

    @classmethod
    def from_dict(cls, d: Any) -> "DefaultEntry":
        """Accept both new object-form origin and legacy string-form origin."""
        if not isinstance(d, dict):
            return cls(value=d)

        value = d.get("value")
        unit = d.get("unit", "")
        origin = d.get("origin")

        if isinstance(origin, str):
            # legacy string form → wrap with default tier
            return cls(
                value=value,
                unit=unit,
                origin_citation=origin,
                confidence_tier="established",
            )
        if isinstance(origin, dict):
            return cls(
                value=value,
                unit=unit,
                origin_citation=str(origin.get("citation", "")),
                origin_url=origin.get("url"),
                origin_doi=origin.get("doi"),
                origin_arxiv_id=origin.get("arxiv_id"),
                confidence_tier=str(origin.get("confidence_tier", "established")),
                last_verified=origin.get("last_verified"),
                notes=str(origin.get("notes", "")),
            )
        return cls(value=value, unit=unit)


@dataclass(frozen=True)
class EvidenceBundle:
    """Returned by lookup_with_evidence — value + full origin metadata."""
    value: Any
    origin_citation: str
    origin_url: Optional[str]
    origin_doi: Optional[str]
    origin_arxiv_id: Optional[str]
    confidence_tier: str
    last_verified: Optional[str]
    notes: str


@dataclass(frozen=True)
class ParamEntry:
    operator: str
    parameter: str
    research_trigger: ResearchTrigger
    canonical_question: str
    empirical_defaults: Dict[str, DefaultEntry]
    sanity_range: Optional[tuple] = None

    def get_default(self, modality: str, paradigm: str) -> DefaultEntry:
        for key in (paradigm, modality, _DEFAULT_KEY):
            if key and key in self.empirical_defaults:
                return self.empirical_defaults[key]
        return DefaultEntry(value=None)


@dataclass(frozen=True)
class OperatorSchema:
    operator: str
    version: int
    parameters: Dict[str, ParamEntry]


_REGISTRY_CACHE: Optional[Dict[str, OperatorSchema]] = None
_REGISTRY_HASH: Optional[str] = None


def _yaml_files() -> List[Path]:
    if not _REGISTRY_DIR.exists():
        return []
    return sorted(_REGISTRY_DIR.glob("*.yaml"))


def load_registry(force: bool = False) -> Dict[str, OperatorSchema]:
    """Read every YAML, validate, return {operator_name: OperatorSchema}.

    Cached for the process lifetime. ``force=True`` re-reads from disk.
    """
    global _REGISTRY_CACHE, _REGISTRY_HASH
    if _REGISTRY_CACHE is not None and not force:
        return _REGISTRY_CACHE

    out: Dict[str, OperatorSchema] = {}
    hasher = hashlib.sha256()
    for path in _yaml_files():
        raw_bytes = path.read_bytes()
        hasher.update(path.name.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(raw_bytes)
        hasher.update(b"\0")
        data = yaml.safe_load(raw_bytes.decode("utf-8")) or {}
        op = str(data.get("operator") or path.stem)
        version = int(data.get("version", 1))
        params: Dict[str, ParamEntry] = {}
        for pname, pdata in (data.get("parameters") or {}).items():
            if not isinstance(pdata, dict):
                raise ValueError(
                    f"{path.name}:parameters.{pname} must be a mapping; "
                    f"got {type(pdata).__name__}"
                )
            trigger = pdata.get("research_trigger", "never")
            if trigger not in (
                "paradigm_dependent", "region_dependent",
                "data_dependent", "never", "user_only",
            ):
                raise ValueError(
                    f"{path.name}:parameters.{pname}: invalid research_trigger {trigger!r}"
                )
            cq = str(pdata.get("canonical_question", "")).strip()
            defaults_block = pdata.get("empirical_defaults") or {}
            if not isinstance(defaults_block, dict) or _DEFAULT_KEY not in defaults_block:
                raise ValueError(
                    f"{path.name}:parameters.{pname}.empirical_defaults missing required '_default'"
                )
            defaults = {k: DefaultEntry.from_dict(v) for k, v in defaults_block.items()}
            sanity = pdata.get("sanity_range")
            sanity_tuple = tuple(sanity) if isinstance(sanity, list) and len(sanity) == 2 else None
            params[pname] = ParamEntry(
                operator=op,
                parameter=pname,
                research_trigger=trigger,
                canonical_question=cq,
                empirical_defaults=defaults,
                sanity_range=sanity_tuple,
            )
        out[op] = OperatorSchema(operator=op, version=version, parameters=params)

    _REGISTRY_CACHE = out
    _REGISTRY_HASH = hasher.hexdigest()[:16]
    return out


def lookup(operator: str, parameter: str) -> Optional[ParamEntry]:
    reg = load_registry()
    schema = reg.get(operator)
    if schema is None:
        return None
    return schema.parameters.get(parameter)


def get_default(
    operator: str, parameter: str, modality: str, paradigm: str,
) -> Optional[DefaultEntry]:
    entry = lookup(operator, parameter)
    if entry is None:
        return None
    return entry.get_default(modality=modality, paradigm=paradigm)


def needs_research(
    operator: str, parameter: str,
    fingerprint: Optional[Dict[str, Any]] = None,
) -> bool:
    entry = lookup(operator, parameter)
    if entry is None:
        return False
    if entry.research_trigger in ("never", "user_only"):
        return False
    if entry.research_trigger == "data_dependent":
        return _fingerprint_is_anomalous(fingerprint or {})
    return True


def _fingerprint_is_anomalous(fp: Dict[str, Any]) -> bool:
    """Trigger data_dependent research only when fingerprint signals oddness."""
    try:
        nch = int(fp.get("n_channels", 0) or 0)
        sf = float(fp.get("sfreq", 0) or 0.0)
    except (TypeError, ValueError):
        return False
    if nch >= 256:
        return True
    if sf >= 5000:
        return True
    if fp.get("artifact_density_high"):
        return True
    return False


def render_canonical_question(
    operator: str, parameter: str,
    modality: str = "", paradigm: str = "",
) -> str:
    entry = lookup(operator, parameter)
    if entry is None:
        return ""
    return entry.canonical_question.format(
        modality=modality, paradigm=paradigm,
    ).strip()


def registry_version_hash() -> str:
    """Stable hex hash of every YAML's content. Participates in the
    research_parameter cache key.
    """
    if _REGISTRY_CACHE is None:
        load_registry()
    return _REGISTRY_HASH or ""


def lookup_with_evidence(
    *,
    operator: str,
    parameter: str,
    paradigm: str,
    modality: str = "",
) -> Optional[EvidenceBundle]:
    """Return EvidenceBundle for (operator, parameter, paradigm).

    Falls back to ``_default`` if paradigm-specific entry not present. Returns
    None if operator or parameter is unknown.
    """
    entry = lookup(operator, parameter)
    if entry is None:
        return None
    default = entry.get_default(modality=modality, paradigm=paradigm)
    if default is None or default.value is None:
        return None
    return EvidenceBundle(
        value=default.value,
        origin_citation=default.origin_citation,
        origin_url=default.origin_url,
        origin_doi=default.origin_doi,
        origin_arxiv_id=default.origin_arxiv_id,
        confidence_tier=default.confidence_tier,
        last_verified=default.last_verified,
        notes=default.notes,
    )
