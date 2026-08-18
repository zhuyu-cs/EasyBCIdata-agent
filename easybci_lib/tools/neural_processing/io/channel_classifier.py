"""Channel classifier — categorize each channel as
data / marker / physio / misc / bad, to drive non-data-channel filtering.

Pure functions only. No file IO, no mutation of inputs, no ``import mne``
(we consume the ``ch_types`` list that the loader already obtained from MNE).

Categories
----------
- data    : target neural signal (eeg/mag/grad/seeg/ecog/dbs/ref_meg) — keep
- marker  : pure trigger/marker (stim, or name like STI/Trigger/Status/...) — must-drop
- physio  : physiological reference (eog/emg/ecg/resp) — suggest-drop, default keep
- misc    : misc / unknown non-data — suggest-drop, default keep
- bad     : already-flagged bad channel — left to existing drop_bads
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

# MNE ch_type -> our category
_TYPE_TO_CATEGORY: Dict[str, str] = {
    "eeg": "data", "mag": "data", "grad": "data",
    "seeg": "data", "ecog": "data", "dbs": "data",
    "ref_meg": "data",                       # MEG reference — keep
    "stim": "marker",
    "eog": "physio", "emg": "physio", "ecg": "physio",
    "resp": "physio", "bio": "physio",
    "misc": "misc",
}

_RE_MARKER = re.compile(
    r"(STI[\s_]?\d*|Trigger|Status|Event|Marker|Sync|Photodiode|TTL|^DC\d*$)",
    re.IGNORECASE,
)
_RE_PHYSIO = re.compile(
    r"(VEOG|HEOG|EOG|EMG|Chin|Leg|ECG|EKG|Resp|Pleth|SpO2)",
    re.IGNORECASE,
)

# PSG auxiliary channels — physiological but ESSENTIAL to a sleep study.
# In PSG context these are kept (not suggest-dropped). Names follow the
# Compumedics/AASM montage seen in .SLP StudyCfg.
_RE_PSG_AUX = re.compile(
    r"(SpO2|SaO2|Pleth|Pulse|Ox[\s_]?Status|NPress|CPress|Airflow|Flow|"
    r"Thor|Abdo|Effort|Snore|Sound|Therm|Thermistor|Position|Pos|Leg|Limb)",
    re.IGNORECASE,
)

# Spike data has no channel-type concept.
_NON_APPLICABLE_MODALITIES = {"spike", "spikes", "unit"}


def _category_from_name(name: str) -> str:
    if _RE_MARKER.search(name):
        return "marker"
    if _RE_PHYSIO.search(name):
        return "physio"
    return "data"   # optimistic: unknown names default to data, never auto-dropped


def classify_channels(
    channels: List[str],
    *,
    ch_types: Optional[List[str]] = None,
    modality: str = "",
    bad_channels: Optional[List[str]] = None,
    psg_context: bool = False,
) -> Dict:
    """Classify each channel name into a category.

    When ``ch_types`` is provided AND its length matches ``channels``, the
    types win; the name regex only fills in unknown types. This is
    deliberately conservative: a channel correctly typed ``misc`` (e.g. DC*)
    stays ``misc`` (suggest-drop) rather than being upgraded to ``marker``
    (must-drop).
    """
    bad_set = set(bad_channels or [])

    if (modality or "").lower() in _NON_APPLICABLE_MODALITIES:
        return {
            "applicable": False, "used_fallback": False,
            "categories": {}, "summary": {},
            "must_drop": [], "suggest_drop": [], "psg_aux": [],
        }

    use_types = bool(ch_types) and len(ch_types) == len(channels)
    categories: Dict[str, str] = {}

    for i, name in enumerate(channels):
        if name in bad_set:
            categories[name] = "bad"
            continue
        if use_types:
            cat = _TYPE_TO_CATEGORY.get((ch_types[i] or "").lower())
            if cat is None:
                cat = _category_from_name(name)   # unknown type → name heuristic
        else:
            cat = _category_from_name(name)
        categories[name] = cat

    summary: Dict[str, int] = {}
    for cat in categories.values():
        summary[cat] = summary.get(cat, 0) + 1

    must_drop = [n for n, c in categories.items() if c == "marker"]
    suggest_drop = [n for n, c in categories.items() if c in ("physio", "misc")]

    psg_aux = [n for n in channels if _RE_PSG_AUX.search(n)] if psg_context else []
    if psg_context:
        aux_set = set(psg_aux)
        suggest_drop = [n for n in suggest_drop if n not in aux_set]

    return {
        "applicable": True,
        "used_fallback": not use_types,
        "categories": categories,
        "summary": summary,
        "must_drop": must_drop,
        "suggest_drop": suggest_drop,
        "psg_aux": psg_aux,
    }
