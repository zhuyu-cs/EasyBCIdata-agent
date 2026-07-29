"""Synthesize web search results into actionable pipeline recommendations.

Takes raw search results and uses the auxiliary LLM to extract structured
preprocessing advice — recommended steps, parameters, and rationale with
source attribution.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Imported at module level (not in the function body) so tests can
# monkeypatch it, and per the repo's "imports at top" rule. This is
# cycle-safe: citation_extractor only back-imports evidence_synthesizer
# inside a function body, never at module load.
from .citation_extractor import extract_citation
from .confidence import compute_evidence_confidence

# Breadth defaults. Kept config-driven because widening costs latency/tokens
# on the weak aux model. max_sources = kept citations (was hard 5); the weak
# endpoint is protected by the low _MAX_EXTRACT_WORKERS below, not by a small K.
_MAX_SOURCES_DEFAULT = 10


def _load_research_cfg() -> dict:
    """Read the ``web.research`` config sub-block. Returns {} on any failure.

    Lazy import of load_config (module stays import-cheap and headless-safe),
    mirroring citation_extractor._resolve_aux_model_id.
    """
    try:
        from easybci_cli.config import load_config
        cfg = load_config() or {}
        web = cfg.get("web") or {}
        research = web.get("research") or {}
        return research if isinstance(research, dict) else {}
    except Exception:  # noqa: BLE001 — config read must never be fatal
        return {}


def _resolve_max_sources() -> int:
    try:
        val = int(_load_research_cfg().get("max_sources", _MAX_SOURCES_DEFAULT))
    except (TypeError, ValueError):
        val = _MAX_SOURCES_DEFAULT
    return max(1, val)
# Per-citation extractions run concurrently (blocking LLM calls). Kept low
# (2) so the weak custom aux endpoint (deepseek-v4-pro) isn't overloaded —
# 5 concurrent heavy requests measurably raised its truncation / "not
# relevant" rate. The on-disk cache makes repeats free.
_MAX_EXTRACT_WORKERS = 2


@dataclass
class EvidenceReport:
    """Structured output from evidence synthesis.

    Field names match the canonical ``plan/web_evidence.json`` schema
    (``recommendations`` / ``citations``) so ``to_dict()`` round-trips
    cleanly into ``_shape_web_evidence_payload`` without key remapping.
    """
    confidence: float = 0.0
    recommendations: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    parameters_extracted: List[str] = field(default_factory=list)
    rationale: str = ""
    citations: List[Dict[str, str]] = field(default_factory=list)
    caveats: List[str] = field(default_factory=list)
    raw_excerpts: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "confidence": self.confidence,
            "recommendations": self.recommendations,
            "parameters": self.parameters,
            "parameters_extracted": self.parameters_extracted,
            "rationale": self.rationale,
            "citations": self.citations,
            "caveats": self.caveats,
        }


_SYNTHESIS_PROMPT = """\
You are a neuroscience signal processing expert. You are given the TOP most \
relevant sources about BCI / neural-data preprocessing (already ranked and \
extracted). Extract actionable preprocessing pipeline recommendations.

Context:
- Modality: {modality}
- Paradigm: {paradigm}
- Question: {question}

Sources (ranked most-relevant first):
{search_results}

Respond with a JSON object (no markdown fencing):
{{
  "confidence": <0.0-1.0, how confident are you in this recommendation>,
  "recommended_steps": [<pipeline step strings like "notch:50", "bandpass:1,40", "resample:256">],
  "parameters": {{<key-value pairs for specific parameter recommendations>}},
  "rationale": "<2-3 sentences explaining the choice, citing sources by title>",
  "caveats": [<caveats or uncertainties>]
}}

Rules:
- Base recommendations ONLY on the provided sources — do not invent values.
- Use the step format: "operator:param1,param2" (e.g. "bandpass:0.5,40").
- If sources disagree, pick the majority / most-conservative value and record
  the disagreement in caveats.
- confidence < 0.3 if only one source supports a step, 0.3-0.6 for 2 sources,
  > 0.6 for 3+ agreeing sources.
- If the sources are insufficient, return confidence=0 with empty steps.
"""


def synthesize_evidence(
    search_results: List[Dict[str, Any]],
    modality: str = "",
    paradigm: str = "",
    question: str = "",
    use_llm: bool = True,
) -> EvidenceReport:
    """Synthesize search results into an EvidenceReport.

    Parameters
    ----------
    search_results : list of dict
        Raw search results. Each dict has keys: "query", "results" (list of
        {"title", "url", "description"} dicts), and optionally "extracted_content".
    modality : str
        Neural data modality
    paradigm : str
        Processing paradigm
    question : str
        The specific question being researched
    use_llm : bool
        If True, use auxiliary LLM for synthesis. If False, use rule-based fallback.

    Returns
    -------
    EvidenceReport
    """
    if not search_results:
        return EvidenceReport(confidence=0.0, rationale="No search results available.")

    # Collect citations first (full-fidelity; no char truncation), rank them
    # and keep only the max_sources most relevant BEFORE the (expensive)
    # extraction loop, then run the extractions in parallel, then build
    # all_snippets so the aggregate LLM input prefers extracted text.
    # Extraction focuses each citation on the specific parameter retrieval
    # question.
    all_citations: List[Dict[str, Any]] = []
    extracted_by_url: Dict[str, str] = {}
    seen_urls: dict = {}  # url -> index into all_citations

    for result_set in search_results:
        results = result_set.get("results", [])
        for r in results:
            title = r.get("title", "") or ""
            desc = r.get("description", "") or r.get("snippet", "") or ""
            url = r.get("url", "") or ""
            if not url:
                continue
            relevance = float(r.get("_relevance", 0.0) or 0.0)
            whitelisted = bool(r.get("_whitelisted", False))
            if url not in seen_urls:
                seen_urls[url] = len(all_citations)
                all_citations.append({
                    "url": url,
                    "title": title,
                    "snippet": desc,
                    "_raw_desc": desc,  # internal — used to build all_snippets
                    "_relevance": relevance,
                    "_whitelisted": whitelisted,
                })
            else:
                # Same URL surfaced by another query — keep the strongest signal.
                existing = all_citations[seen_urls[url]]
                if relevance > existing.get("_relevance", 0.0):
                    existing["_relevance"] = relevance
                existing["_whitelisted"] = existing.get("_whitelisted", False) or whitelisted

        # Track per-result-set extracted content keyed by any URL in the
        # set, so the extraction helper can see full-page text. Multiple
        # citations from the same result_set share one extracted_content.
        extracted = result_set.get("extracted_content", "") or ""
        if extracted:
            for r in results:
                u = r.get("url", "") or ""
                if u and u not in extracted_by_url:
                    extracted_by_url[u] = extracted

    # Rank and keep only the top-K citations (whitelist > relevance > snippet
    # richness). This is both the primary speed win (extract K instead of ~30)
    # and the product requirement: web_evidence.json carries only the top-K
    # most relevant sources' steps/params.
    all_citations.sort(
        key=lambda c: (
            c.get("_whitelisted", False),
            c.get("_relevance", 0.0),
            len(c.get("_raw_desc") or c.get("snippet") or ""),
        ),
        reverse=True,
    )
    all_citations = all_citations[:_resolve_max_sources()]

    # Per-citation extraction (cached on disk; question-focused). Run in
    # parallel since extract_citation is a blocking LLM call and never raises
    # (it returns a structured extract_error), so one bad source can't abort
    # the batch. The on-disk cache is thread-safe: each write targets a
    # distinct sha256(url|question|model_id) file.
    from concurrent.futures import ThreadPoolExecutor

    def _extract_one(citation: Dict[str, Any]) -> tuple:
        url_for_extract = citation.get("url", "") or ""
        full_text = extracted_by_url.get(url_for_extract, "")
        result = extract_citation(
            url=url_for_extract,
            question=question or "",
            raw_snippet=citation.get("snippet", "") or "",
            full_text=full_text,
        )
        return citation, result

    if all_citations:
        with ThreadPoolExecutor(max_workers=min(_MAX_EXTRACT_WORKERS, len(all_citations))) as ex:
            for citation, result in ex.map(_extract_one, all_citations):
                citation["key_information"] = result["key_information"]
                citation["key_params"] = result["key_params"]
                citation["extract_error"] = result["extract_error"]

    all_snippets: List[str] = []
    for citation in all_citations:
        title = citation.get("title", "") or ""
        raw = citation.get("_raw_desc") or citation.get("snippet") or ""
        info = citation.get("key_information") or ""
        params = citation.get("key_params") or []
        # The raw snippet is ALWAYS included — a weak-model extraction must
        # never shadow real source text (root cause of the empty-recommendations
        # regression). The key_information summary is appended only as an
        # explicit hint, and only when the extraction demonstrably engaged with
        # the source (it pulled structured key_params). A model refusal like
        # "Not relevant to the question." has no key_params, so its unreliable
        # summary is not forwarded — but the raw snippet still is.
        parts: List[str] = []
        if raw:
            parts.append(raw)
        if params and info.strip() and info.strip() != raw.strip():
            parts.append(f"(key info: {info.strip()})")
        if parts:
            all_snippets.append(f"[{title}] " + " ".join(parts))

    # Strip internal fields so they don't leak to web_evidence.json
    for citation in all_citations:
        citation.pop("_raw_desc", None)
        citation.pop("_relevance", None)
        citation.pop("_whitelisted", None)

    # Append any extracted content blobs (full-fidelity; the LLM call is
    # wrapped by call_llm_with_overflow_retry in _synthesize_with_llm,
    # which handles context overflow by retrying with bounded input).
    for extracted in extracted_by_url.values():
        if extracted:
            all_snippets.append(extracted)

    if not all_snippets:
        return EvidenceReport(confidence=0.0, rationale="Search returned no useful content.")

    if use_llm:
        report = _synthesize_with_llm(all_snippets, all_citations, modality, paradigm, question)
    else:
        report = _synthesize_rule_based(all_snippets, all_citations, modality, paradigm, question)

    # Salvage: fold successfully-extracted per-citation key_params into a
    # DEDICATED parameters_extracted field (deduped, order-preserving). These
    # are `param=value` strings — they must NOT pollute `recommendations`,
    # which downstream (conflict_resolver, codegen) treats as pipeline steps.
    salvaged: List[str] = []
    for citation in all_citations:
        for kp in citation.get("key_params") or []:
            if kp and kp not in salvaged:
                salvaged.append(kp)
    if salvaged:
        report.parameters_extracted = salvaged

    # Evidence-driven confidence: reflects how much corroborated evidence was
    # gathered, independent of whether the aggregate LLM step parsed. Take the
    # max so a genuine LLM score is never lowered, but a pinned-0.2 fallback is
    # lifted when the salvaged evidence supports it.
    evidence_conf = compute_evidence_confidence(all_citations)
    report.confidence = max(float(report.confidence or 0.0), evidence_conf)

    # Repair rationale: if the aggregate LLM parse failed (blank rationale) but
    # we DO have evidence, synthesize a short factual rationale from the kept
    # citation titles so the output is never a blank-rationale/full-recs mix.
    if not (report.rationale or "").strip() and (report.recommendations or report.parameters_extracted):
        titles = [c.get("title", "").strip() for c in all_citations if c.get("title", "").strip()]
        if titles:
            shown = "; ".join(titles[:3])
            more = f" (+{len(titles) - 3} more)" if len(titles) > 3 else ""
            noun = "source" if len(titles) == 1 else "sources"
            # Only claim per-source extraction when the evidence we actually
            # have came from that path (salvaged params, no aggregate recs).
            if report.parameters_extracted and not report.recommendations:
                suffix = ("Aggregate LLM synthesis was unavailable; parameters "
                          "were taken directly from per-source extraction.")
            else:
                suffix = "Aggregate LLM synthesis did not return a rationale."
            report.rationale = (
                f"Synthesized from {len(titles)} {noun} including: {shown}{more}. {suffix}"
            )
    return report


def _synthesize_with_llm(
    snippets: List[str],
    citations: List[Dict[str, str]],
    modality: str,
    paradigm: str,
    question: str,
) -> EvidenceReport:
    """Use auxiliary LLM to synthesize evidence.

    Sends full-fidelity snippets (no static char cap). On a classified
    context-overflow error the overflow-retry helper slices the user
    message to 64k chars and retries once.
    """
    try:
        from easybci_agent.auxiliary_client import call_llm, extract_content_or_reasoning
    except ImportError:
        logger.warning("auxiliary_client not available, falling back to rule-based synthesis")
        return _synthesize_rule_based(snippets, citations, modality, paradigm, question)

    from easybci_lib.tools._llm_overflow import call_llm_with_overflow_retry

    combined = "\n\n".join(snippets[:_resolve_max_sources()])
    prompt = _SYNTHESIS_PROMPT.format(
        modality=modality or "unknown",
        paradigm=paradigm or "unknown",
        question=question or "general preprocessing",
        search_results=combined,
    )

    try:
        response = call_llm_with_overflow_retry(
            call_llm=call_llm,
            task="web_extract",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=2048,
            fallback_input_chars=64_000,
        )
        text = extract_content_or_reasoning(response)
    except Exception as exc:
        logger.warning("LLM synthesis failed: %s", exc)
        return _synthesize_rule_based(snippets, citations, modality, paradigm, question)

    return _parse_llm_response(text, citations)


def _parse_llm_response(text: str, citations: List[Dict[str, str]]) -> EvidenceReport:
    """Parse the LLM JSON response into an EvidenceReport."""
    # Strip markdown fencing if present
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines)

    data = _parse_json_robust(cleaned, tag="evidence_synthesis_aggregate")
    if data is None:
        logger.warning("Failed to parse LLM synthesis response")
        # Do NOT put the raw (often reasoning-prose) text into the user-facing
        # rationale — leave it blank so synthesize_evidence's repair path can
        # synthesize a clean rationale from citation titles. Keep a truncated
        # copy in caveats for debugging traceability only.
        return EvidenceReport(
            confidence=0.2,
            rationale="",
            citations=citations[:_resolve_max_sources()],
            caveats=[
                "LLM response could not be parsed as structured data",
                f"raw_unparsed_response: {cleaned[:300]}",
            ],
        )

    return EvidenceReport(
        confidence=min(1.0, max(0.0, float(data.get("confidence", 0.0)))),
        recommendations=data.get("recommended_steps", []),
        parameters=data.get("parameters", {}),
        rationale=data.get("rationale", ""),
        citations=citations[:_resolve_max_sources()],
        caveats=data.get("caveats", []),
    )


def _parse_json_robust(text: str, tag: str = "evidence_synthesis") -> Optional[Dict[str, Any]]:
    """Best-effort parse of a JSON object that may be malformed or truncated.

    Tries direct parse, then first ``{``-to-last-``}`` slice (handles
    wrapping prose), then tool-call truncation repair (handles tail
    truncation when ``max_tokens`` cuts the response mid-string).
    Returns parsed dict on success, ``None`` on failure.

    ``tag`` distinguishes log lines from different callsites
    (``evidence_synthesis_aggregate`` vs ``citation_extract``) so the
    operator can see which path is producing repairs.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    if start < 0:
        return None

    end = text.rfind("}") + 1
    if end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass

    from easybci_agent.sanitization import _repair_tool_call_arguments
    repaired = _repair_tool_call_arguments(text[start:], tag)
    if repaired and repaired != "{}":
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass
    return None


def _synthesize_rule_based(
    snippets: List[str],
    citations: List[Dict[str, str]],
    modality: str,
    paradigm: str,
    question: str,
) -> EvidenceReport:
    """Fallback rule-based synthesis when LLM is unavailable.

    Extracts frequency values and step names from snippets using pattern matching.
    """
    import re

    steps: List[str] = []
    params: Dict[str, Any] = {}
    caveats = ["Rule-based synthesis (LLM unavailable) — verify recommendations manually"]

    combined = " ".join(snippets).lower()

    # Extract frequency band mentions
    bandpass_match = re.search(r"bandpass[:\s]+(\d+\.?\d*)\s*[-–to]+\s*(\d+\.?\d*)\s*hz", combined)
    if bandpass_match:
        lo, hi = bandpass_match.group(1), bandpass_match.group(2)
        steps.append(f"bandpass:{lo},{hi}")
        params["bandpass_low"] = float(lo)
        params["bandpass_high"] = float(hi)

    # Extract notch filter mentions
    notch_match = re.search(r"notch[:\s]+(\d+)\s*hz", combined)
    if notch_match:
        freq = notch_match.group(1)
        steps.append(f"notch:{freq}")
        params["notch_freq"] = int(freq)

    # Extract resampling mentions
    resample_match = re.search(r"resamp(?:le|ling)[:\s]+(\d+)\s*hz", combined)
    if resample_match:
        rate = resample_match.group(1)
        steps.append(f"resample:{rate}")
        params["resample_rate"] = int(rate)

    confidence = min(0.4, len(steps) * 0.15)

    return EvidenceReport(
        confidence=confidence,
        recommendations=steps,
        parameters=params,
        rationale=f"Extracted from {len(snippets)} search result snippets for {modality} {paradigm}.",
        citations=citations[:_resolve_max_sources()],
        caveats=caveats,
    )


import re as _re_param

_NUMERIC_VALUE_PATTERNS = [
    _re_param.compile(r"(?:=|\s)\s*([0-9]+(?:\.[0-9]+)?)\s*(?:hz|μv|uv|s|ms|%|x|×)?", _re_param.IGNORECASE),
]


def _extract_numeric_candidates(text: str):
    out = []
    for pat in _NUMERIC_VALUE_PATTERNS:
        for m in pat.finditer(text or ""):
            try:
                out.append(float(m.group(1)))
            except (TypeError, ValueError):
                continue
    return out


def synthesize_parameter(
    *,
    operator: str,
    parameter: str,
    search_results: list,
    sanity_range=None,
):
    """Synthesize a single-parameter recommendation from search snippets."""
    candidates = []
    citations = []
    snippets = []
    for sq in search_results or []:
        for r in sq.get("results") or []:
            text = " ".join(filter(None, [r.get("title"), r.get("snippet")]))
            for v in _extract_numeric_candidates(text):
                candidates.append(v)
            citations.append({
                "url": r.get("url", ""),
                "title": r.get("title", ""),
                "snippet": r.get("snippet") or "",
            })
            if r.get("snippet"):
                snippets.append(r["snippet"])

    if not candidates:
        return None

    if sanity_range is not None:
        lo, hi = float(sanity_range[0]), float(sanity_range[1])
        in_range = [v for v in candidates if lo <= v <= hi]
        if not in_range:
            return {
                "value": None,
                "confidence": 0.0,
                "summary": "All candidates out of sanity range.",
                "citations": citations[:_resolve_max_sources()],
                "rejected_reason": "out_of_range",
                "raw_candidates": candidates[:10],
            }
        candidates = in_range

    candidates.sort()
    median = candidates[len(candidates) // 2]
    band = max(0.1, abs(median) * 0.25)
    agreement = sum(1 for v in candidates if abs(v - median) <= band)
    confidence = min(0.95, agreement / max(3, len(candidates)))

    summary = " ".join(snippets[:2])

    return {
        "value": median,
        "confidence": round(confidence, 2),
        "summary": summary,
        "citations": citations[:_resolve_max_sources()],
    }

