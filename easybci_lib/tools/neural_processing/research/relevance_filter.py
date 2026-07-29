"""Relevance gate for web search results returned by research_preprocessing.

Three-tier judgment, applied IN ORDER:
1. blacklist domain → drop, reason="blacklisted_domain"
2. whitelist domain → keep, reason="whitelisted_domain"
3. keyword score    → keep if >= threshold else drop, reason="low_score"
"""
from __future__ import annotations

import re
from typing import List, Tuple
from urllib.parse import urlparse


# Base relevance score assigned to whitelisted academic domains. Higher than
# the keep threshold (0.35) so academic sources rank above keyword-scored
# blog/doc hits during the downstream top-K citation selection.
WHITELIST_BASE = 0.8


_WHITELIST_SUFFIXES = (
    "pubmed.ncbi.nlm.nih.gov",
    "ncbi.nlm.nih.gov",
    "arxiv.org",
    "biorxiv.org",
    "medrxiv.org",
    "mne.tools",
    "nature.com",
    "sciencedirect.com",
    "ieeexplore.ieee.org",
    "frontiersin.org",
    "plos.org",
    "wiley.com",
    "springer.com",
    "link.springer.com",
    "tandfonline.com",
    "cell.com",
    "jneurosci.org",
    "academic.oup.com",
    "iopscience.iop.org",
    "openneuro.org",
    "bids-specification.readthedocs.io",
    "neuroimage.usc.edu",
    "fieldtriptoolbox.org",
    "brainstorm-tools.org",
    "eeglab.org",
    "nilearn.github.io",
)

# Tool/library documentation & official toolbox sites. Whitelisted so the
# evidence is not limited to journal papers — MNE / EEGLAB / FieldTrip /
# SpikeInterface / Kilosort / nilearn / sklearn / pytorch docs are
# authoritative preprocessing references in their own right.
_TOOLBOX_DOC_SUFFIXES = (
    "readthedocs.io",
    "github.io",
    "fieldtriptoolbox.org",
    "sccn.ucsd.edu",
    "scikit-learn.org",
    "pytorch.org",
    "numpy.org",
    "scipy.org",
    "neuraldatascience.io",
    "nipy.org",
)


def _host(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower()
    except Exception:
        return ""


_BLACKLIST_SUFFIXES = (
    "medium.com",
    "reddit.com",
    "quora.com",
    "csdn.net",
    "blog.csdn.net",
    "stackoverflow.com",
    "stackexchange.com",
    "zhihu.com",
    "jianshu.com",
    "cnblogs.com",
    "tumblr.com",
    "wordpress.com",
    "blogspot.com",
    "linkedin.com",
    "twitter.com",
    "x.com",
    "facebook.com",
    "youtube.com",
    "tiktok.com",
)


def is_whitelisted_domain(url: str) -> bool:
    """Return True if URL's host ends with a whitelisted academic or tool-doc suffix."""
    host = _host(url)
    if not host:
        return False
    suffixes = _WHITELIST_SUFFIXES + _TOOLBOX_DOC_SUFFIXES
    return any(host == s or host.endswith("." + s) for s in suffixes)


def is_blacklisted_domain(url: str) -> bool:
    """Return True if URL's host ends with a blacklisted suffix.

    Blacklist takes precedence over whitelist when both match (which should
    not normally happen — these are disjoint sets — but be defensive)."""
    host = _host(url)
    if not host:
        return False
    return any(host == s or host.endswith("." + s) for s in _BLACKLIST_SUFFIXES)


# Methodology keywords that lift relevance — neuroscience preprocessing vocabulary
_METHOD_KEYWORDS = (
    "filter", "bandpass", "lowpass", "highpass", "notch", "bandstop",
    "ica", "artifact", "preprocessing", "preprocess",
    "epoch", "epoching", "segmentation", "baseline", "reference", "montage", "channel",
    "csp", "psd", "spectral", "rereference", "resample", "car",
    "downsample", "interpolate", "drop", "bad", "clean", "asr",
    "pipeline", "classification", "decoding", "feature",
    "detrend", "demean", "wavelet", "common", "average",
    "rejection", "component", "independent",
)

# Modality synonym expansion — a snippet using the spelled-out term
# ("electroencephalography") should still hit the modality token ("eeg").
# Keys and values are matched against the tokenized text, so multi-word
# phrases are stored as individual tokens where needed.
_MODALITY_SYNONYMS = {
    "eeg": {"electroencephalography", "electroencephalogram", "electroencephalographic"},
    "meg": {"magnetoencephalography", "magnetoencephalogram"},
    "seeg": {"intracranial", "depth", "stereoeeg", "stereo"},
    "ecog": {"intracranial", "electrocorticography", "electrocorticogram"},
    "ieeg": {"intracranial", "depth"},
    "fnirs": {"nirs", "infrared", "hemodynamic"},
    "spike": {"extracellular", "unit", "units", "neuron", "neuronal", "firing"},
}


_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_]+")


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text or "")}


def _modality_match_tokens(modality: str) -> set[str]:
    """Return the set of tokens that should count as a modality hit.

    Combines the raw modality tokens with their spelled-out synonyms so
    academic phrasing ("electroencephalography") is not under-scored.
    """
    base = _tokens(modality)
    expanded = set(base)
    for tok in base:
        expanded |= _MODALITY_SYNONYMS.get(tok, set())
    return expanded


def score_result(
    item: dict,
    *,
    modality: str,
    paradigm: str,
    question: str,
) -> float:
    """Score a single search result in [0, 1] for relevance.

    The score is the sum of three weighted bonuses (each capped before sum):

    - modality_hit  (0.35) — modality token (or spelled-out synonym) appears
      in title or snippet
    - paradigm_hit  (0.32) — paradigm token appears in title or snippet
    - method_hits   (0.05 each, capped at 0.35) — methodology keywords

    Final value is clamped to [0, 1].
    """
    title = (item.get("title") or "")
    snippet = (item.get("snippet") or "")
    text_tokens = _tokens(title) | _tokens(snippet)

    score = 0.0

    modality_tokens = _modality_match_tokens(modality)
    if modality_tokens & text_tokens:
        score += 0.35

    paradigm_tokens = _tokens(paradigm.replace("_", " ")) if paradigm else set()
    if paradigm_tokens & text_tokens:
        score += 0.32

    method_hits = sum(1 for kw in _METHOD_KEYWORDS if kw in text_tokens)
    score += min(0.35, 0.05 * method_hits)

    return max(0.0, min(1.0, score))


def filter_results(
    items: list[dict],
    *,
    modality: str,
    paradigm: str,
    question: str,
    threshold: float = 0.35,
) -> Tuple[List[dict], List[dict]]:
    """Partition ``items`` into (kept, dropped).

    Each ``kept`` entry is a shallow copy carrying two extra fields used by
    the downstream top-K ranking in ``evidence_synthesizer``:

    - ``_relevance`` : float — the numeric relevance score (whitelisted
      domains get ``WHITELIST_BASE`` so academic sources rank above
      keyword-scored blog hits).
    - ``_whitelisted`` : bool — True for whitelisted academic domains.

    Each ``dropped`` entry is enriched with ``"reason"`` (one of
    ``"blacklisted_domain"`` / ``"low_score"``) and, for low-score drops,
    the computed ``"score"``.

    Ordering rule (applied per-item, in order):
      1. blacklisted domain → drop
      2. whitelisted domain → keep (``_relevance = WHITELIST_BASE``)
      3. otherwise → score; keep iff ``score >= threshold``
    """
    kept: list[dict] = []
    dropped: list[dict] = []

    for item in items:
        url = item.get("url") or ""

        if is_blacklisted_domain(url):
            dropped.append({**item, "reason": "blacklisted_domain"})
            continue

        if is_whitelisted_domain(url):
            kept.append({**item, "_relevance": WHITELIST_BASE, "_whitelisted": True})
            continue

        s = score_result(item, modality=modality, paradigm=paradigm, question=question)
        if s >= threshold:
            kept.append({**item, "_relevance": round(s, 3), "_whitelisted": False})
        else:
            dropped.append({**item, "reason": "low_score", "score": round(s, 3)})

    return kept, dropped
