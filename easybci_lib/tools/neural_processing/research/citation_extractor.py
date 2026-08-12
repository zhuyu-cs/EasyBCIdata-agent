"""Per-citation LLM key-information extraction focused on the search question.

Each citation collected by ``synthesize_evidence`` carries a raw ``snippet``
(typically Exa's ``highlights`` field, full-fidelity). This module produces a
``key_information`` summary that focuses the snippet on the **specific**
parameter-retrieval question (e.g. "bandpass / notch / ICA"), plus a structured
``key_params`` list of ``param=value`` strings the LLM extracted.

Extraction calls go through the auxiliary task ``web_extract`` and results are
cached on disk under ``~/.easybci/cache/citation_extract/`` keyed by
sha256(url|question|model_id) so repeat runs incur zero LLM cost.

NOTE: the model-facing wording is deliberately plain "extract key information"
— never "distill" — because the weak auxiliary model (deepseek-v4-pro) was
observed to under-extract / return "not relevant" when asked to "distill".
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_CACHE_TTL_SECONDS = 7 * 24 * 60 * 60  # 7 days

_EXTRACT_PROMPT = """You are extracting the key preprocessing information from a search result.

Search question:
{question}

Citation snippet (raw, from search backend):
{snippet}

{full_text_block}

Return ONLY a single JSON object with this exact shape — no preamble,
no markdown fencing. IMPORTANT: emit "key_params" FIRST so the structured
field is preserved even if the response is truncated:
{{
  "key_params": ["<param>=<value>", ...],   // e.g. "bandpass_low=1", "ica_method=runica"; empty list if none
  "key_information": "<2-4 sentences focused on the search question, summarizing what THIS citation contributes. Use specific numeric parameters/step names from the citation when present. If the citation truly contains nothing about the question, return an empty string here and an empty key_params list.>"
}}"""


def _cache_dir() -> Path:
    from easybci_lib.constants import get_easybci_home

    d = get_easybci_home() / "cache" / "citation_extract"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cache_key(url: str, question: str, model_id: str) -> str:
    raw = f"{url}|{question}|{model_id}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:24]


def _cache_get(key: str) -> dict | None:
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
        logger.debug("citation_extract cache read failed for %s: %s", key, exc)
        return None


def _cache_put(key: str, value: dict) -> None:
    try:
        path = _cache_dir() / f"{key}.json"
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    except OSError as exc:
        logger.debug("citation_extract cache write failed for %s: %s", key, exc)


def _resolve_aux_model_id() -> str:
    """Read ``auxiliary.web_extract.model`` from config; fall back to 'unknown'.

    Used as a cache-key dimension so a run with a different aux model
    won't reuse the prior model's extracted outputs.
    """
    try:
        from easybci_cli.config import load_config
    except Exception:
        return "unknown"
    try:
        cfg = load_config() or {}
        aux = cfg.get("auxiliary") or {}
        web_extract = aux.get("web_extract") or {}
        model = str(web_extract.get("model") or "").strip()
        return model or "unknown"
    except Exception:  # noqa: BLE001 — config read shouldn't be fatal
        return "unknown"


def _resolve_llm_timeout(default: float = 25.0) -> float:
    """SHORT per-call timeout for the per-citation extraction LLM call,
    overriding the 360s ``auxiliary.web_extract.timeout`` it would inherit."""
    try:
        from easybci_cli.config import load_config

        research = ((load_config() or {}).get("web") or {}).get("research") or {}
        return max(0.0, float(research.get("llm_timeout_seconds", default)))
    except Exception:  # noqa: BLE001 — config read shouldn't be fatal
        return default


def _resolve_extract_max_tokens(default: int = 16384) -> int:
    """Output-token ceiling for the per-citation extraction LLM call.

    The extracted JSON (a handful of parameter strings + a short quote per
    citation) is small, but the aux model may be a REASONING model that spends
    the budget thinking BEFORE emitting the JSON. The old 1024 ceiling let the
    thinking phase consume it and cut the JSON mid-structure — the repair path
    then salvaged valid JSON but dropped the tail, so citations contributed no
    parameters to the evidence. A generous default leaves room for the answer
    to complete after reasoning; override via ``web.research.extract_max_tokens``.
    """
    try:
        from easybci_cli.config import load_config

        research = ((load_config() or {}).get("web") or {}).get("research") or {}
        return max(1024, int(research.get("extract_max_tokens", default)))
    except Exception:  # noqa: BLE001 — config read shouldn't be fatal
        return default


def _parse_extract_response(text: str) -> dict | None:
    """Best-effort parse of the LLM's JSON response, reusing the robust
    parser from evidence_synthesizer."""
    try:
        from easybci_lib.tools.neural_processing.research.evidence_synthesizer import (
            _parse_json_robust,
        )
    except ImportError:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = [ln for ln in cleaned.split("\n") if not ln.strip().startswith("```")]
        cleaned = "\n".join(lines)
    return _parse_json_robust(cleaned, tag="citation_extract")


def extract_citation(
    *,
    url: str = "",
    question: str,
    raw_snippet: str,
    full_text: str = "",
) -> dict:
    """Run a question-focused LLM extraction on one citation.

    Returns a dict with fixed shape::

        {
          "key_information": str,      # LLM summary; "" on failure
          "key_params": list[str],     # e.g. ["bandpass_low=1"]
          "extract_error": str | False,  # False on success, full error string on failure
        }

    Behavior:
      - cache hit (under 7-day TTL): return cached dict immediately
      - cache miss: call_llm(task="web_extract") with overflow-retry
      - on any failure (aux unavailable / context exceeded after retry /
        unparseable response / empty response / other exception): return
        empty key_information with a non-null extract_error string

    Cache writes happen ONLY when a usable (non-empty) extraction is produced,
    so failures and empty junk don't poison the cache.
    """
    model_id = _resolve_aux_model_id()
    key = _cache_key(url or "", question or "", model_id)

    cached = _cache_get(key)
    if cached is not None:
        # Old cache entries stored success as None; surface as False per the
        # current contract (False = no error, string = the error).
        if cached.get("extract_error") is None:
            cached["extract_error"] = False
        return cached

    try:
        from easybci_agent.auxiliary_client import call_llm, extract_content_or_reasoning
    except ImportError:
        return {
            "key_information": "",
            "key_params": [],
            "extract_error": "aux model unavailable",
        }

    from easybci_lib.tools._llm_overflow import call_llm_with_overflow_retry

    full_text_block = (
        f"Full extracted page text (when available):\n{full_text}\n"
        if full_text
        else ""
    )
    prompt = _EXTRACT_PROMPT.format(
        question=question or "",
        snippet=raw_snippet or "",
        full_text_block=full_text_block,
    )
    messages = [{"role": "user", "content": prompt}]

    try:
        response = call_llm_with_overflow_retry(
            call_llm=call_llm,
            task="web_extract",
            timeout=_resolve_llm_timeout(),
            messages=messages,
            temperature=0.1,
            max_tokens=_resolve_extract_max_tokens(),
            fallback_input_chars=64_000,
        )
    except Exception as exc:  # noqa: BLE001 — propagate as structured error
        return {
            "key_information": "",
            "key_params": [],
            "extract_error": f"{type(exc).__name__}: {str(exc)[:200]}",
        }

    text = extract_content_or_reasoning(response) or ""
    if not text.strip():
        return {
            "key_information": "",
            "key_params": [],
            "extract_error": "empty response",
        }

    parsed = _parse_extract_response(text)
    if parsed is None:
        return {
            "key_information": "",
            "key_params": [],
            "extract_error": "unparseable LLM response",
        }

    key_information = str(parsed.get("key_information", "")).strip()
    key_params_raw = parsed.get("key_params", [])
    if isinstance(key_params_raw, list):
        key_params = [str(x).strip() for x in key_params_raw if str(x).strip()]
    else:
        key_params = []

    result = {
        "key_information": key_information,
        "key_params": key_params,
        "extract_error": False,
    }
    # Only cache extractions that produced usable content — an empty
    # key_information + empty key_params is almost always weak-model junk and
    # must not be pinned for the 7-day TTL (cache poisoning). Failures already
    # return early above without caching.
    if key_information or key_params:
        _cache_put(key, result)
    return result
