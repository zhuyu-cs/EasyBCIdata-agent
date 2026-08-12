"""Label-driven time-segment rejection.

The gold sEEG workflow discards an entire recording when any event label
matches a reject keyword (seizure / stimulation / interictal discharge). This
module instead excises only the labelled time windows (± a pad) and keeps the
rest of the recording — recovering the clean majority of otherwise-dropped data.

Two pieces:
- ``keyword_matches_label`` — word-START boundary, case-insensitive matching.
  Fires when a keyword begins a word in the label, so ``IID`` matches ``IIDa``
  and ``Stim`` matches ``Stim Start D1-D2`` (real clinical markers), but a
  keyword embedded mid-word (``preStimulus``) does NOT match. This replaces the
  gold ``re.compile("|".join(kw))`` bare-substring match, which had no
  boundary and could over-match future labels.
- ``label_reject_mask`` — turn matching annotations into a boolean keep-mask
  over samples, padding each window and clamping to recording bounds.

Pure/stdlib+numpy only.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence

import numpy as np


# Multilingual floor of reject keywords. This is a STARTING POINT, never
# authoritative — it is always union'd with gold + agent-supplied keywords, and
# labels it fails to match are surfaced via collect_unmatched_labels so a human
# or agent can top it up. Covers common seizure / stimulation / epileptiform
# terms across English, Chinese, and a few frequent abbreviations.
DEFAULT_REJECT_KEYWORDS: List[str] = [
    # English — seizure / ictal
    "Seiz", "Seizure", "Ictal", "SZ\\b", "pre-ictal", "post-ictal",
    "epilep", "convuls",
    # English — epileptiform discharges
    "IID", "spike", "sharp wave", "polyspike", "discharge",
    # English — stimulation
    "Stim", "stimulat", "electrical stim",
    # Chinese
    "发作", "癫", "痫", "刺激", "电刺激", "痉挛",
]

# Semantic word-roots that mark a label as clinically suspicious even when it is
# NOT in any keyword list. Used only to *flag for review* (never to auto-reject),
# so it is deliberately broad and multilingual.
_SUSPICIOUS_ROOTS = [
    "seiz", "ictal", "epilep", "convuls", "spike", "discharge", "polyspike",
    "stim", "stimulat", "anfall", "crise", "krampf",  # de/fr seizure words
    "发作", "癫", "痫", "刺激", "痉挛", "放电",
]
_SUSPICIOUS_RX = re.compile("|".join(re.escape(r) for r in _SUSPICIOUS_ROOTS),
                            re.IGNORECASE)


def merge_keywords(*keyword_lists: Optional[Sequence[str]]) -> List[str]:
    """Union multiple keyword lists, de-duplicating case-insensitively.

    First occurrence (by original casing) wins. Empty/whitespace entries and
    None lists are ignored. Order: entries appear in the order first seen.
    """
    seen: set = set()
    merged: List[str] = []
    for lst in keyword_lists:
        for kw in lst or []:
            kw = str(kw).strip()
            if not kw:
                continue
            key = kw.lower()
            if key in seen:
                continue
            seen.add(key)
            merged.append(kw)
    return merged


def collect_unmatched_labels(
    labels: Sequence[str], keywords: Sequence[str]
) -> List[str]:
    """Distinct labels present in the recording that NO keyword matched.

    Sorted, de-duplicated. This is the observability hook: an agent inspects
    these to decide whether a new environment uses labels the keyword list
    doesn't cover yet.
    """
    rx = _compile(keywords)
    out: set = set()
    for lab in labels or []:
        lab_s = str(lab)
        if rx is None or rx.search(lab_s) is None:
            out.add(lab_s)
    return sorted(out)


def flag_suspicious_labels(labels: Sequence[str]) -> List[str]:
    """From (typically unmatched) labels, return those that look clinically
    suspicious by multilingual seizure/stim word-roots — candidates the agent
    should review and possibly add as reject keywords. Never auto-rejects."""
    flagged: List[str] = []
    for lab in labels or []:
        lab_s = str(lab)
        if _SUSPICIOUS_RX.search(lab_s) is not None:
            flagged.append(lab_s)
    return flagged


def _compile(keywords: Sequence[str]) -> Optional[re.Pattern]:
    """Compile keywords into a single word-START-boundary, case-insensitive regex.

    ``(?<![0-9A-Za-z])`` asserts the char before the keyword is not
    alphanumeric — i.e. the keyword starts a word. We deliberately do NOT
    require a trailing boundary, so ``IID`` still matches ``IIDa``/``IIDs``.
    Keyword strings that already contain a regex boundary (the gold config
    ships ``"SZ\\b"``) are honoured as-is.
    """
    parts: List[str] = []
    for kw in keywords:
        kw = str(kw).strip()
        if not kw:
            continue
        # If the author baked in \b (e.g. "SZ\b"), trust their pattern verbatim.
        if "\\b" in kw:
            parts.append(kw)
        else:
            parts.append(r"(?<![0-9A-Za-z])" + re.escape(kw))
    if not parts:
        return None
    return re.compile("|".join(parts), re.IGNORECASE)


def keyword_matches_label(label: str, keywords: Sequence[str]) -> bool:
    """True if any keyword matches ``label`` at a word start (case-insensitive)."""
    rx = _compile(keywords)
    if rx is None:
        return False
    return rx.search(str(label)) is not None


def label_reject_mask(
    annotations: Optional[Dict[str, Sequence]],
    keywords: Sequence[str],
    sfreq: float,
    n_samples: int,
    pad_s: float = 1.0,
) -> np.ndarray:
    """Boolean keep-mask (shape ``(n_samples,)``): True = keep, False = reject.

    Any annotation whose description matches a keyword marks
    ``[onset - pad_s, onset + duration + pad_s]`` for rejection. Windows are
    clamped to ``[0, n_samples)``. Instantaneous events (duration 0) still
    excise a ``±pad_s`` window.
    """
    keep = np.ones(int(n_samples), dtype=bool)
    if not annotations or sfreq <= 0 or n_samples <= 0:
        return keep
    rx = _compile(keywords)
    if rx is None:
        return keep

    onsets = annotations.get("onset") or []
    durations = annotations.get("duration") or []
    descriptions = annotations.get("description") or []
    for i, desc in enumerate(descriptions):
        if rx.search(str(desc)) is None:
            continue
        onset = float(onsets[i]) if i < len(onsets) else 0.0
        dur = float(durations[i]) if i < len(durations) else 0.0
        start_s = onset - pad_s
        stop_s = onset + dur + pad_s
        start = max(0, int(np.floor(start_s * sfreq)))
        stop = min(int(n_samples), int(np.ceil(stop_s * sfreq)))
        if stop > start:
            keep[start:stop] = False
    return keep
