"""LLM-driven query planner for BCI preprocessing research.

The template query builder (:mod:`query_builder`) mechanically concatenates
modality + paradigm + method keywords into search strings. That "keyword soup"
often fails to surface the right papers / toolbox docs / manuals — the LLM
never *understands* the need before searching.

This module adds that missing step: a single auxiliary-LLM call reads the
structured need (modality, paradigm, analysis goal, data fingerprint, user
intent, known problems) and emits a handful of PRECISE, source-targeted
queries using correct academic terminology.

Design contract:
  - Provider-agnostic: emits plain query strings + a source-type tag only;
    nothing here is coupled to Exa / Tavily / any backend.
  - Fail-safe: any failure (aux unavailable, bad JSON, empty output) returns
    ``None`` so the caller falls back to the template builder. This module
    NEVER raises to its caller.
  - Cached on disk (7-day TTL) keyed by the need, so repeat runs are free.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .query_builder import (
    SearchQuery,
    _DOMAINS_ACADEMIC,
    _DOMAINS_COMMUNITY,
    _DOMAINS_DOCS,
)

logger = logging.getLogger(__name__)

_CACHE_TTL_SECONDS = 7 * 24 * 60 * 60  # 7 days
# Bump when the prompt / output shape changes so stale cached plans expire.
_PLANNER_VERSION = "planner_v2"

_MAX_QUERIES = 8

# Map the LLM's declared source_type onto the authoritative-domain lists that
# query_builder already curates. Metadata only for now (not hard-filtered into
# the search API), so the ranking/whitelist stages can still favor them.
_SOURCE_TYPE_DOMAINS: Dict[str, List[str]] = {
    "paper": _DOMAINS_ACADEMIC,
    "toolbox_docs": _DOMAINS_DOCS,
    "manual": _DOMAINS_DOCS,
    "dataset": ["openneuro.org", "physionet.org"] + _DOMAINS_ACADEMIC,
    "community": _DOMAINS_COMMUNITY,
}


_PLANNER_PROMPT = """You are a neuroscience research librarian. Given a \
neural-data preprocessing need, produce PRECISE web-search queries that will \
surface the most authoritative sources. Understand the intent first, then \
target each query at ONE source type using correct academic terminology \
(spelled-out modality names, standard method names, canonical toolbox names).

Need:
- Modality: {modality}
- Paradigm: {paradigm}
- Analysis goal: {analysis_goal}
- Data: {data_profile}
- User intent / question: {question}
- Known problems (if any): {problems}

Return ONLY one JSON object (no markdown fencing):
{{
  "queries": [
    {{"query": "<precise natural-language search string>",
      "source_type": "paper|toolbox_docs|manual|dataset|community",
      "rationale": "<why this query, one clause>"}}
  ]
}}
Rules:
- 6-8 queries. Cover the methodological axes the need implies (filtering,
  artifact handling, referencing, epoching/segmentation) plus anything the
  user intent or known problems specifically call for.
- Balance source types for breadth: include at least ONE `paper` query and at
  least ONE `toolbox_docs` query. When a canonical toolbox exists for the
  modality (MNE-Python/EEGLAB/FieldTrip for M/EEG, MNE-NIRS for fNIRS,
  SpikeInterface/Kilosort for spikes), add a `manual` query targeting its
  official preprocessing documentation.
- Use spelled-out AND acronym forms of the modality; use real toolbox names
  (MNE-Python, EEGLAB, FieldTrip, MNE-NIRS, Kilosort, SpikeInterface...) where
  a docs/manual query fits.
- Prefer terms that appear in peer-reviewed papers and official docs, not blogs.
- Do NOT invent dataset names; only add a dataset query if one is named in the
  intent."""


def _cache_dir() -> Path:
    from easybci_lib.constants import get_easybci_home

    d = get_easybci_home() / "cache" / "query_plan"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cache_key(need: Dict[str, Any], model_id: str) -> str:
    raw = "|".join([
        str(need.get("modality", "")),
        str(need.get("paradigm", "")),
        str(need.get("analysis_goal", "")),
        str(need.get("question", "")),
        str(need.get("level", "")),
        model_id,
        _PLANNER_VERSION,
    ]).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:24]


def _cache_get(key: str) -> Optional[list]:
    path = _cache_dir() / f"{key}.json"
    if not path.exists():
        return None
    try:
        age = time.time() - path.stat().st_mtime
        if age > _CACHE_TTL_SECONDS:
            try:
                path.unlink()
            except OSError:
                pass
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.debug("query_plan cache read failed for %s: %s", key, exc)
        return None


def _cache_put(key: str, value: list) -> None:
    try:
        path = _cache_dir() / f"{key}.json"
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    except OSError as exc:
        logger.debug("query_plan cache write failed for %s: %s", key, exc)


def _resolve_aux_model_id() -> str:
    """Read ``auxiliary.web_extract.model`` (same aux slot the extractor uses)
    so a model change invalidates cached plans."""
    try:
        from easybci_cli.config import load_config
    except Exception:
        return "unknown"
    try:
        cfg = load_config() or {}
        aux = cfg.get("auxiliary") or {}
        web_extract = aux.get("web_extract") or {}
        return str(web_extract.get("model") or "").strip() or "unknown"
    except Exception:  # noqa: BLE001 — config read shouldn't be fatal
        return "unknown"


def _planner_enabled() -> bool:
    """``web.query_planner`` gate — default True. Escape hatch to force the
    template path."""
    try:
        from easybci_cli.config import load_config

        cfg = load_config() or {}
        web = cfg.get("web") or {}
        val = web.get("query_planner", True)
        return bool(val)
    except Exception:  # noqa: BLE001
        return True


def _format_data_profile(fingerprint: Optional[Dict[str, Any]]) -> str:
    if not isinstance(fingerprint, dict) or not fingerprint:
        return "unknown"
    n_ch = fingerprint.get("n_channels")
    fs = fingerprint.get("frequency_hz") or fingerprint.get("frequency")
    dur = fingerprint.get("duration_s") or fingerprint.get("duration")
    parts = []
    if n_ch:
        parts.append(f"{n_ch} channels")
    if fs:
        parts.append(f"{fs} Hz")
    if dur:
        parts.append(f"{dur}s")
    return ", ".join(parts) if parts else "unknown"


def _format_problems(context: Optional[Dict[str, Any]]) -> str:
    if not isinstance(context, dict):
        return "none"
    bits = []
    failed = context.get("failed_steps") or context.get("failed_remedies")
    if failed:
        bits.append(f"failed steps: {failed}")
    qc = context.get("qc_issues")
    if qc:
        bits.append(f"QC issues: {qc}")
    return "; ".join(str(b) for b in bits) if bits else "none"


def _parse_plan(text: str) -> Optional[dict]:
    """Reuse the robust JSON parser from evidence_synthesizer."""
    try:
        from .evidence_synthesizer import _parse_json_robust
    except ImportError:
        try:
            return json.loads(text)
        except (ValueError, TypeError):
            return None
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = "\n".join(
            ln for ln in cleaned.split("\n") if not ln.strip().startswith("```")
        )
    return _parse_json_robust(cleaned, tag="query_plan")


def _plan_to_queries(plan: dict) -> List[SearchQuery]:
    """Convert the parsed LLM plan into deduped SearchQuery objects."""
    raw_queries = plan.get("queries") if isinstance(plan, dict) else None
    if not isinstance(raw_queries, list):
        return []

    out: List[SearchQuery] = []
    seen: set = set()
    for item in raw_queries:
        if not isinstance(item, dict):
            continue
        q = str(item.get("query", "")).strip()
        if not q:
            continue
        norm = " ".join(q.lower().split())
        if norm in seen:
            continue
        seen.add(norm)
        source_type = str(item.get("source_type", "")).strip().lower()
        domains = _SOURCE_TYPE_DOMAINS.get(source_type, _DOMAINS_DOCS + _DOMAINS_ACADEMIC)
        out.append(SearchQuery(
            query=q,
            purpose=f"LLM:{source_type or 'general'}",
            priority=10,
            preferred_domains=list(domains),
        ))
        if len(out) >= _MAX_QUERIES:
            break
    return out


def plan_queries(
    *,
    modality: str = "",
    paradigm: str = "",
    analysis_goal: str = "",
    question: str = "",
    level: int = 1,
    fingerprint: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None,
) -> Optional[List[SearchQuery]]:
    """Plan precise search queries via the auxiliary LLM.

    Returns a list of :class:`SearchQuery` on success, or ``None`` on ANY
    failure / disabled gate / empty output so the caller can fall back to the
    template builder. Never raises.
    """
    if not _planner_enabled():
        return None

    need = {
        "modality": modality,
        "paradigm": paradigm,
        "analysis_goal": analysis_goal,
        "question": question,
        "level": level,
    }
    model_id = _resolve_aux_model_id()
    key = _cache_key(need, model_id)

    cached = _cache_get(key)
    if cached is not None:
        queries = _plan_to_queries({"queries": cached})
        return queries or None

    try:
        from easybci_agent.auxiliary_client import call_llm, extract_content_or_reasoning
    except ImportError:
        return None

    from easybci_lib.tools._llm_overflow import call_llm_with_overflow_retry

    prompt = _PLANNER_PROMPT.format(
        modality=modality or "unknown",
        paradigm=paradigm or "general",
        analysis_goal=analysis_goal or "generic",
        data_profile=_format_data_profile(fingerprint),
        question=question or "(none)",
        problems=_format_problems(context),
    )

    try:
        response = call_llm_with_overflow_retry(
            call_llm=call_llm,
            task="web_extract",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1024,
            fallback_input_chars=64_000,
        )
        text = extract_content_or_reasoning(response) or ""
    except Exception as exc:  # noqa: BLE001 — fail safe, caller uses templates
        logger.debug("query planner LLM failed: %s", exc)
        return None

    if not text.strip():
        return None

    plan = _parse_plan(text)
    if not isinstance(plan, dict):
        return None

    queries = _plan_to_queries(plan)
    if not queries:
        return None

    # Cache the raw query dicts (post-dedup) so a re-run reproduces the plan.
    _cache_put(key, [
        {"query": q.query, "source_type": q.purpose.split(":", 1)[-1]}
        for q in queries
    ])
    return queries
