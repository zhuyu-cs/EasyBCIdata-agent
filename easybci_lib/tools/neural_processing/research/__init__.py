"""Web-search-backed research for BCI preprocessing decisions.

Provides evidence-based pipeline recommendations when static domain skills
are insufficient. Integrates with the existing WebSearchProvider registry.

Modules:
    complexity_classifier — determines search level (0-3) from data context
    query_planner         — LLM-driven precise query planning (provider-agnostic)
    query_builder         — constructs domain-aware search queries (template fallback)
    search_cache          — file-based result caching (7-day TTL)
    evidence_synthesizer  — summarizes search results into actionable advice
    knowledge_internalizer — caches evidence reports + internalizes into skills
"""

from easybci_lib.tools.neural_processing.research.complexity_classifier import classify_complexity
from easybci_lib.tools.neural_processing.research.query_builder import build_queries
from easybci_lib.tools.neural_processing.research.query_planner import plan_queries
from easybci_lib.tools.neural_processing.research.search_cache import SearchCache
from easybci_lib.tools.neural_processing.research.evidence_synthesizer import synthesize_evidence
from easybci_lib.tools.neural_processing.research.knowledge_internalizer import (
    EvidenceCache,
    KnowledgeInternalizer,
    try_internalize_from_evidence,
)

__all__ = [
    "classify_complexity",
    "build_queries",
    "plan_queries",
    "SearchCache",
    "synthesize_evidence",
    "EvidenceCache",
    "KnowledgeInternalizer",
    "try_internalize_from_evidence",
]
