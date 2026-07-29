"""Evidence-driven confidence scoring for web research synthesis.

Pure functions (no I/O, no LLM). The score reflects how much *corroborated*
preprocessing evidence was actually gathered, not merely whether the aggregate
synthesis LLM call parsed. Two signals, combined additively:

  coverage      — saturating in the number of sources that extracted params;
                  this is the dominant signal ("how much did we find")
  corroboration — fraction of param-keys backed by >=2 independent sources;
                  applied as a small BONUS (0-20%), not a gate

confidence = min(1.0, coverage * (1 + CORROBORATION_BONUS * corroboration))

Coverage leads: lots of extracted evidence => high score. Corroboration only
nudges upward when independent sources agree on the same parameter key — it is
NOT a multiplicative gate, because the weak aux model emits inconsistent key
names run-to-run, so cross-source key agreement is a weak (bonus-only) signal.
There is deliberately NO cleanliness factor: coverage already uses the count of
*effective* sources, so multiplying again by effective/total would penalize the
same fact twice. If NO source extracted any params the score is exactly 0.0.
"""

from __future__ import annotations

from typing import Any, Dict, List

# Base of the saturating coverage curve. Smaller -> saturates faster.
_COVERAGE_BASE = 0.6
# Corroboration contributes a bonus of up to this fraction (not a gate).
_CORROBORATION_BONUS = 0.20
# Noise suffixes stripped during key normalization so obvious synonyms collapse
# (highpass_filter / high_pass_cutoff share a stem; ica_method / ica_algorithm -> ica).
_NOISE_SUFFIXES = ("_method", "_algorithm", "_filter", "_cutoff", "_threshold")


def normalize_param_key(key: str) -> str:
    """Lowercase, unify separators, strip a small set of noise suffixes.

    Conservative: never returns "" for a non-empty input by stripping a suffix
    (if stripping would empty the key, the original stem is kept).
    """
    k = (key or "").strip().lower()
    for sep in (" ", "-", "/", "."):
        k = k.replace(sep, "_")
    while "__" in k:
        k = k.replace("__", "_")
    k = k.strip("_")
    for suf in _NOISE_SUFFIXES:
        if k.endswith(suf) and len(k) > len(suf):
            k = k[: -len(suf)]
            break
    return k


def _param_key(param: str) -> str:
    """Extract the normalized key from a 'key=value' string (or the whole token)."""
    head = param.split("=", 1)[0] if "=" in param else param
    return normalize_param_key(head)


def compute_evidence_confidence(citations: List[Dict[str, Any]]) -> float:
    """Compute confidence in [0, 1] from per-citation extraction results.

    Each citation is a dict with at least ``key_params`` (list[str]). An
    ``extract_error`` truthy value marks a failed extraction (does not count
    as an effective source). Returns 0.0 when no citation extracted any params.
    """
    if not citations:
        return 0.0

    key_to_sources: Dict[str, set] = {}
    n_effective = 0

    for idx, c in enumerate(citations):
        params = c.get("key_params") or []
        if not params:
            continue
        n_effective += 1
        for p in params:
            nk = _param_key(str(p))
            if nk:
                key_to_sources.setdefault(nk, set()).add(idx)

    if n_effective == 0 or not key_to_sources:
        return 0.0

    coverage = 1.0 - (_COVERAGE_BASE ** n_effective)

    n_keys = len(key_to_sources)
    n_multi = sum(1 for srcs in key_to_sources.values() if len(srcs) >= 2)
    corroboration = n_multi / n_keys if n_keys else 0.0

    score = coverage * (1.0 + _CORROBORATION_BONUS * corroboration)
    return max(0.0, min(1.0, round(score, 3)))
