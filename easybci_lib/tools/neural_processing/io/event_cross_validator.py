"""Multi-event source cross-validation.

When multiple event sources exist (embedded annotations in the data file,
external event files, BIDS sidecars), this module loads all sources,
compares timestamps for consistency, and reports discrepancies.

The researcher can then choose the authoritative source.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def cross_validate_events(
    embedded_events: Optional[List[Dict[str, Any]]] = None,
    external_paths: Optional[List[str]] = None,
    frequency: Optional[float] = None,
    tolerance_s: float = 0.05,
) -> Dict[str, Any]:
    """Compare event timestamps across multiple sources for consistency.

    Parameters
    ----------
    embedded_events : list of dict, optional
        Events extracted from the data file itself (MNE annotations,
        EEGLAB EEG.event, FieldTrip trl).
    external_paths : list of str, optional
        Paths to external event files (CSV, JSON, etc.).
    frequency : float, optional
        Sampling rate (needed for sample→second conversion).
    tolerance_s : float
        Maximum time difference (seconds) to consider two events as matching.

    Returns
    -------
    Dict with:
        sources: list of {name, n_events, path_or_origin}
        agreement: float (0-1) — fraction of events that match across sources
        discrepancies: list of {time, source_a, source_b, delta_s, description}
        recommended_source: str — which source appears most reliable
        summary: str — human-readable summary
    """
    from easybci_lib.tools.neural_processing.io.event_loader import load_events

    sources: List[Dict[str, Any]] = []

    # Collect embedded events as first source
    if embedded_events:
        sources.append({
            "name": "embedded",
            "origin": "data_file_annotations",
            "events": _normalize_events(embedded_events),
        })

    # Load external event files
    for ext_path in (external_paths or []):
        if not Path(ext_path).exists():
            continue
        try:
            ext_events = load_events(ext_path, frequency=frequency)
            if ext_events:
                sources.append({
                    "name": Path(ext_path).stem,
                    "origin": ext_path,
                    "events": _normalize_events(ext_events),
                })
        except Exception as exc:
            logger.debug("Failed to load events from %s: %s", ext_path, exc)

    if len(sources) < 2:
        return {
            "sources": [{"name": s["name"], "n_events": len(s["events"]), "origin": s["origin"]} for s in sources],
            "agreement": 1.0 if sources else 0.0,
            "discrepancies": [],
            "recommended_source": sources[0]["name"] if sources else "",
            "summary": (
                f"Only {len(sources)} event source(s) available — cross-validation not applicable."
                if sources else "No event sources found."
            ),
        }

    # Pairwise comparison between all source pairs
    all_discrepancies: List[Dict[str, Any]] = []
    pair_agreements: List[float] = []

    for i in range(len(sources)):
        for j in range(i + 1, len(sources)):
            agreement, discrepancies = _compare_sources(
                sources[i], sources[j], tolerance_s
            )
            pair_agreements.append(agreement)
            all_discrepancies.extend(discrepancies)

    overall_agreement = float(np.mean(pair_agreements)) if pair_agreements else 0.0

    # Determine recommended source (most events + best consistency)
    recommended = _select_recommended_source(sources, all_discrepancies)

    # Source summaries (without event arrays)
    source_summaries = [
        {"name": s["name"], "n_events": len(s["events"]), "origin": s["origin"]}
        for s in sources
    ]

    # Build summary text
    summary_parts = [
        f"{len(sources)} event sources compared.",
        f"Overall agreement: {overall_agreement:.1%}.",
    ]
    if all_discrepancies:
        summary_parts.append(
            f"{len(all_discrepancies)} discrepancies found "
            f"(max delta: {max(d['delta_s'] for d in all_discrepancies):.3f}s)."
        )
    else:
        summary_parts.append("All sources are consistent within tolerance.")
    summary_parts.append(f"Recommended source: '{recommended}'.")

    return {
        "sources": source_summaries,
        "agreement": round(overall_agreement, 3),
        "discrepancies": all_discrepancies[:50],  # cap at 50 to avoid huge output
        "n_discrepancies_total": len(all_discrepancies),
        "recommended_source": recommended,
        "summary": " ".join(summary_parts),
    }


def _normalize_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize event list to standard format with onset/type."""
    normalized = []
    for ev in events:
        onset = ev.get("onset", ev.get("start", 0.0))
        duration = ev.get("duration", 0.0)
        ev_type = ev.get("type", ev.get("description", ev.get("label", "unknown")))
        try:
            onset = float(onset)
            duration = float(duration)
        except (TypeError, ValueError):
            continue
        normalized.append({
            "onset": onset,
            "duration": duration,
            "type": str(ev_type),
        })
    normalized.sort(key=lambda x: x["onset"])
    return normalized


def _compare_sources(
    source_a: Dict[str, Any],
    source_b: Dict[str, Any],
    tolerance_s: float,
) -> Tuple[float, List[Dict[str, Any]]]:
    """Compare two event sources for timestamp agreement.

    Returns (agreement_ratio, list_of_discrepancies).
    """
    events_a = source_a["events"]
    events_b = source_b["events"]

    if not events_a or not events_b:
        return 0.0, []

    # Match events by closest onset time
    onsets_a = np.array([e["onset"] for e in events_a])
    onsets_b = np.array([e["onset"] for e in events_b])

    matched = 0
    discrepancies: List[Dict[str, Any]] = []
    used_b = set()

    for i, onset_a in enumerate(onsets_a):
        # Find closest event in B
        diffs = np.abs(onsets_b - onset_a)
        best_j = int(np.argmin(diffs))
        delta = float(diffs[best_j])

        if best_j in used_b:
            # Already matched — find next best
            sorted_indices = np.argsort(diffs)
            found = False
            for candidate_j in sorted_indices:
                if candidate_j not in used_b:
                    best_j = int(candidate_j)
                    delta = float(diffs[best_j])
                    found = True
                    break
            if not found:
                continue

        if delta <= tolerance_s:
            matched += 1
            used_b.add(best_j)

            # Check type agreement
            type_a = events_a[i].get("type", "")
            type_b = events_b[best_j].get("type", "")
            if type_a != type_b and type_a and type_b:
                discrepancies.append({
                    "time": round(onset_a, 4),
                    "source_a": source_a["name"],
                    "source_b": source_b["name"],
                    "delta_s": round(delta, 4),
                    "description": (
                        f"Type mismatch at t={onset_a:.3f}s: "
                        f"'{type_a}' ({source_a['name']}) vs "
                        f"'{type_b}' ({source_b['name']})"
                    ),
                    "category": "type_mismatch",
                })
        else:
            discrepancies.append({
                "time": round(onset_a, 4),
                "source_a": source_a["name"],
                "source_b": source_b["name"],
                "delta_s": round(delta, 4),
                "description": (
                    f"Timing discrepancy at t={onset_a:.3f}s: "
                    f"nearest in '{source_b['name']}' is {delta:.3f}s away"
                ),
                "category": "timing_mismatch",
            })

    # Events in B with no match in A
    unmatched_b = set(range(len(events_b))) - used_b
    for j in unmatched_b:
        discrepancies.append({
            "time": round(onsets_b[j], 4),
            "source_a": source_a["name"],
            "source_b": source_b["name"],
            "delta_s": 0.0,
            "description": (
                f"Extra event in '{source_b['name']}' at t={onsets_b[j]:.3f}s "
                f"with no match in '{source_a['name']}'"
            ),
            "category": "extra_event",
        })

    total = max(len(events_a), len(events_b))
    agreement = matched / total if total > 0 else 0.0

    return round(agreement, 3), discrepancies


def _select_recommended_source(
    sources: List[Dict[str, Any]],
    discrepancies: List[Dict[str, Any]],
) -> str:
    """Select the most reliable event source based on heuristics.

    Priority:
    1. Embedded annotations (closest to recording)
    2. BIDS-style external files (community standard)
    3. Source with most events
    4. Source with fewest discrepancies attributed to it
    """
    if not sources:
        return ""

    # Score each source
    scores: Dict[str, float] = {}
    for s in sources:
        name = s["name"]
        score = 0.0

        # Bonus for embedded (closest to data)
        if name == "embedded" or "annotation" in s.get("origin", "").lower():
            score += 2.0

        # Bonus for event count (more events = more information)
        score += len(s["events"]) * 0.01

        # Penalty for being cited in discrepancies
        n_issues = sum(
            1 for d in discrepancies
            if d.get("source_b") == name and d.get("category") == "extra_event"
        )
        score -= n_issues * 0.1

        scores[name] = score

    # Return highest scoring
    best = max(scores, key=lambda k: scores[k])
    return best
