"""Construct domain-aware search queries for BCI preprocessing research.

Builds queries that target authoritative neuroscience sources (MNE docs,
PubMed, NeuroStars, EEGLAB wiki) with correct academic terminology.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SearchQuery:
    """A single search query with metadata."""
    query: str
    purpose: str
    priority: int = 0
    preferred_domains: List[str] = field(default_factory=list)


# Authoritative domains by category
_DOMAINS_DOCS = ["mne.tools", "eeglab.org", "fieldtriptoolbox.org"]
_DOMAINS_ACADEMIC = ["pubmed.ncbi.nlm.nih.gov", "scholar.google.com", "arxiv.org"]
_DOMAINS_COMMUNITY = ["neurostars.org", "github.com"]

# Modality-to-terminology mapping for search precision
_MODALITY_TERMS: Dict[str, Dict[str, str]] = {
    "eeg": {
        "full_name": "electroencephalography EEG",
        "artifact_terms": "eye blink EOG muscle EMG",
        "tool_names": "MNE-Python EEGLAB",
    },
    "meg": {
        "full_name": "magnetoencephalography MEG",
        "artifact_terms": "environmental noise SSS Maxwell filter",
        "tool_names": "MNE-Python FieldTrip",
    },
    "seeg": {
        "full_name": "stereoelectroencephalography sEEG depth electrodes",
        "artifact_terms": "interictal spikes HFO",
        "tool_names": "MNE-Python AnyWave",
    },
    "ecog": {
        "full_name": "electrocorticography ECoG",
        "artifact_terms": "line noise electrode drift",
        "tool_names": "MNE-Python FieldTrip",
    },
    "spike": {
        "full_name": "neural spike sorting extracellular recording",
        "artifact_terms": "noise cluster artifact",
        "tool_names": "Kilosort SpikeInterface MountainSort",
    },
    "fnirs": {
        "full_name": "functional near-infrared spectroscopy fNIRS",
        "artifact_terms": "motion artifact short channel scalp coupling",
        "tool_names": "MNE-NIRS Homer3 SNIRF",
    },
    "ieeg": {
        "full_name": "intracranial EEG iEEG",
        "artifact_terms": "interictal epileptiform discharge IED",
        "tool_names": "MNE-Python iELVis",
    },
}


def build_queries(
    level: int,
    modality: str = "",
    paradigm: str = "",
    question: str = "",
    context: Optional[Dict[str, Any]] = None,
    *,
    analysis_goal: str = "",
    fingerprint: Optional[Dict[str, Any]] = None,
) -> List[SearchQuery]:
    """Build search queries appropriate to the complexity level.

    An LLM query planner (:func:`query_planner.plan_queries`) runs first: it
    reads the structured need and emits precise, source-targeted queries. On
    any failure / disabled gate it returns ``None`` and we fall back to the
    deterministic template builders below — so this function never regresses.

    Parameters
    ----------
    level : int
        Complexity level (1-3). Level 0 should not reach here.
    modality : str
        Neural data modality (eeg, meg, seeg, etc.)
    paradigm : str
        Processing paradigm (motor_imagery, p300, etc.)
    question : str
        Specific question or user intent
    context : dict or None
        Additional context (fingerprint, failed_steps, qc_issues, etc.)
    analysis_goal : str
        Analysis goal (classification, source_localization, ...) — fed to the
        planner so queries reflect the downstream task.
    fingerprint : dict or None
        Data fingerprint (n_channels / sfreq / duration) for the planner.

    Returns
    -------
    list of SearchQuery
        Ordered list of queries to execute (highest priority first)
    """
    if context is None:
        context = {}

    # Fall back to context-embedded fields when not passed explicitly, so the
    # existing callers that only populate `context` still feed the planner.
    if not fingerprint:
        fingerprint = context.get("fingerprint") if isinstance(context, dict) else None
    if not analysis_goal and isinstance(context, dict):
        analysis_goal = context.get("analysis_goal", "") or ""

    # LLM query planner first (provider-agnostic; fails safe to templates).
    try:
        from .query_planner import plan_queries

        planned = plan_queries(
            modality=modality,
            paradigm=paradigm,
            analysis_goal=analysis_goal,
            question=question,
            level=level,
            fingerprint=fingerprint,
            context=context,
        )
        if planned:
            planned.sort(key=lambda q: q.priority, reverse=True)
            return planned
    except Exception as exc:  # noqa: BLE001 — planner must never break search
        logger.debug("query planner unavailable, using templates: %s", exc)

    mod_lower = modality.lower() if modality else ""
    mod_info = _MODALITY_TERMS.get(mod_lower, _MODALITY_TERMS.get("eeg", {}))

    queries: List[SearchQuery] = []

    if level == 1:
        queries = _build_level1_queries(mod_lower, paradigm, question)
    elif level == 2:
        queries = _build_level2_queries(mod_lower, mod_info, paradigm, question, context)
    elif level >= 3:
        queries = _build_level3_queries(mod_lower, mod_info, paradigm, question, context)

    queries.sort(key=lambda q: q.priority, reverse=True)
    return queries


def _build_level1_queries(
    modality: str,
    paradigm: str,
    question: str,
    *,
    preferred_domains=None,
) -> List[SearchQuery]:
    """Level-1 queries: natural-language, method-oriented probes.

    Uses the spelled-out modality name (e.g. "electroencephalography EEG")
    and academic phrasing so search backends return methods papers / toolbox
    docs rather than overview blogs. Fewer, richer queries than the old
    keyword-soup form — faster and less noisy.
    """
    mod = (modality or "").lower().strip() or "eeg"
    para = (paradigm or "general").lower().strip()
    para_text = para.replace("_", " ")
    mod_full = _MODALITY_TERMS.get(mod, {}).get("full_name", mod)

    # (topic, extra synonyms/context) — kept to ~4 axes to bound latency.
    method_axes = (
        ("bandpass and notch filtering", "recommended filter cutoff frequencies"),
        ("ICA artifact removal", "independent component analysis eye blink muscle"),
        ("re-referencing and montage", "average reference common average"),
        ("epoching baseline correction preprocessing pipeline", ""),
    )

    domains = preferred_domains or (_DOMAINS_DOCS + _DOMAINS_ACADEMIC)
    out: List[SearchQuery] = []
    for topic, extra in method_axes:
        q = f"{mod_full} {para_text} {topic} {extra} best practices parameters".strip()
        # collapse any double spaces left by an empty `extra`
        q = " ".join(q.split())
        out.append(SearchQuery(
            query=q,
            purpose=f"L1:{topic}",
            priority=10,
            preferred_domains=domains,
        ))

    return out


def _build_level2_queries(
    modality: str,
    mod_info: Dict[str, str],
    paradigm: str,
    question: str,
    context: Dict[str, Any],
) -> List[SearchQuery]:
    """Level 2: 3-5 method research queries."""
    queries = []
    full_name = mod_info.get("full_name", modality)
    tool_names = mod_info.get("tool_names", "MNE-Python")

    # Query 1: Method overview
    if question:
        queries.append(SearchQuery(
            query=f"{question} {full_name} preprocessing pipeline tutorial",
            purpose="method_overview",
            priority=10,
            preferred_domains=_DOMAINS_DOCS + _DOMAINS_ACADEMIC,
        ))

    # Query 2: Paradigm-specific best practices
    if paradigm:
        queries.append(SearchQuery(
            query=f"{paradigm} {modality} preprocessing best practices recommended steps",
            purpose="paradigm_practices",
            priority=9,
            preferred_domains=_DOMAINS_ACADEMIC,
        ))

    # Query 3: Parameter recommendations
    queries.append(SearchQuery(
        query=f"{full_name} {paradigm or 'analysis'} filter parameters frequency band selection",
        purpose="parameter_selection",
        priority=8,
        preferred_domains=_DOMAINS_DOCS,
    ))

    # Query 4: If QC failed, search for specific issue remedies
    qc_issues = context.get("qc_issues", [])
    if qc_issues:
        issue_terms = " ".join(qc_issues[:3])
        queries.append(SearchQuery(
            query=f"{modality} preprocessing {issue_terms} solution fix",
            purpose="qc_remedy_search",
            priority=9,
            preferred_domains=_DOMAINS_COMMUNITY + _DOMAINS_DOCS,
        ))

    # Query 5: Code examples
    queries.append(SearchQuery(
        query=f"{tool_names} {paradigm or modality} preprocessing example code pipeline",
        purpose="code_examples",
        priority=6,
        preferred_domains=_DOMAINS_DOCS + _DOMAINS_COMMUNITY,
    ))

    return queries[:5]  # Level 2: max 5 queries


def _build_level3_queries(
    modality: str,
    mod_info: Dict[str, str],
    paradigm: str,
    question: str,
    context: Dict[str, Any],
) -> List[SearchQuery]:
    """Level 3: 5-10 deep investigation queries across multiple sources."""
    queries = []
    full_name = mod_info.get("full_name", modality)
    tool_names = mod_info.get("tool_names", "MNE-Python")

    # Start with Level 2 queries as base
    queries.extend(_build_level2_queries(modality, mod_info, paradigm, question, context))

    # Additional academic searches
    if question:
        queries.append(SearchQuery(
            query=f"{question} {modality} signal processing methodology review",
            purpose="academic_review",
            priority=7,
            preferred_domains=_DOMAINS_ACADEMIC,
        ))

    # Search for recent publications (method advances)
    queries.append(SearchQuery(
        query=f"{full_name} {paradigm or 'preprocessing'} pipeline 2024 2025 comparison",
        purpose="recent_advances",
        priority=6,
        preferred_domains=_DOMAINS_ACADEMIC,
    ))

    # Community discussions for practical insights
    queries.append(SearchQuery(
        query=f"{modality} {paradigm or question} preprocessing issues discussion",
        purpose="community_insights",
        priority=5,
        preferred_domains=_DOMAINS_COMMUNITY,
    ))

    # Alternative tools/approaches
    queries.append(SearchQuery(
        query=f"{paradigm or modality} preprocessing alternative methods comparison {tool_names}",
        purpose="alternative_approaches",
        priority=4,
        preferred_domains=_DOMAINS_DOCS + _DOMAINS_ACADEMIC,
    ))

    # Dataset-specific known issues (if dataset name mentioned)
    dataset_name = context.get("dataset_name", "")
    if dataset_name:
        queries.append(SearchQuery(
            query=f"{dataset_name} dataset preprocessing known issues channel naming",
            purpose="dataset_specific",
            priority=8,
            preferred_domains=_DOMAINS_COMMUNITY,
        ))

    return queries[:10]  # Level 3: max 10 queries
