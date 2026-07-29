"""Classify preprocessing complexity to determine web search level.

Levels:
    0 — Direct: standard paradigm, domain skill covers it fully
    1 — Parameter Lookup: known paradigm but uncertain parameters
    2 — Method Research: non-standard paradigm or failed QC remedies
    3 — Deep Investigation: unknown territory, multi-source research needed
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Paradigms that have explicit domain skill coverage
_COVERED_PARADIGMS: Dict[str, set] = {
    "eeg": {
        "motor_imagery", "mi", "erp", "p300", "ssvep",
        "sleep", "sleep_staging", "emotion", "emotion_recognition",
        "general", "default",
    },
    "seeg": {"epilepsy", "default"},
    "ecog": {"default", "motor", "speech"},
    "meg": {"default", "auditory", "visual", "motor"},
    "spike": {"sorting", "default"},
    "fnirs": {"default", "motor", "cognitive"},
    "ieeg": {"epilepsy", "default"},
}

# Standard modalities with full pipeline recommendations
_STANDARD_MODALITIES = {"eeg", "seeg", "ecog", "meg", "spike", "fnirs", "ieeg"}

# Keywords indicating non-standard methods that need research
_ADVANCED_METHOD_KEYWORDS = {
    "riemannian", "asr", "artifact subspace reconstruction",
    "deep learning", "autoencoder", "ica label", "iclabel",
    "tms-eeg", "tms", "concurrent fmri", "fmri-eeg",
    "microstate", "source localization", "beamforming",
    "connectivity", "granger", "phase amplitude coupling",
    "pac", "cross-frequency", "hilbert-huang", "emd",
    "wavelet packet", "adaptive filter", "kalman",
    "real-time", "online", "neurofeedback",
    "high-density", "hd-eeg", "256 channel", "128 channel",
    "infant", "neonatal", "pediatric",
    "animal", "rodent", "monkey", "neuropixels",
    "dbs", "deep brain stimulation",
}

# Keywords indicating simple parameter questions (Level 1)
_PARAMETER_KEYWORDS = {
    "what frequency", "which band", "cutoff",
    "sampling rate", "resample to", "how many components",
    "threshold", "reference", "montage", "channel names",
    "notch frequency", "power line", "50 hz", "60 hz",
}


def classify_complexity(
    fingerprint: Optional[Dict[str, Any]] = None,
    user_intent: str = "",
    modality: str = "",
    paradigm: str = "",
    matched_skill: Optional[str] = None,
    proven_match: bool = False,
    qc_failures: int = 0,
    failed_remedies: Optional[List[str]] = None,
) -> int:
    """Determine search complexity level (0-3).

    Parameters
    ----------
    fingerprint : dict or None
        Output from inspect_data tool (modality, n_channels, frequency_hz, etc.)
    user_intent : str
        Natural language description of what the user wants
    modality : str
        Detected or stated modality (eeg, meg, seeg, etc.)
    paradigm : str
        Detected or stated paradigm (motor_imagery, p300, etc.)
    matched_skill : str or None
        Name of the domain skill that matched, if any
    proven_match : bool
        Whether a proven pipeline was found for this scenario
    qc_failures : int
        Number of QC failures in the current session
    failed_remedies : list or None
        Remedies that were tried and failed

    Returns
    -------
    int
        Complexity level 0-3
    """
    if failed_remedies is None:
        failed_remedies = []

    intent_lower = user_intent.lower()

    # Level 3: Deep investigation triggers
    if qc_failures >= 3:
        logger.info("Level 3: multiple QC failures (%d) with exhausted remedies", qc_failures)
        return 3

    if len(failed_remedies) >= 3:
        logger.info("Level 3: %d remedies failed", len(failed_remedies))
        return 3

    if modality and modality.lower() not in _STANDARD_MODALITIES:
        logger.info("Level 3: unknown modality '%s'", modality)
        return 3

    # Level 0: Proven pipeline match
    if proven_match:
        logger.debug("Level 0: proven pipeline match")
        return 0

    # Level 0: Standard paradigm fully covered
    mod_lower = (modality or "").lower()
    par_lower = (paradigm or "").lower()

    if matched_skill and mod_lower in _COVERED_PARADIGMS:
        covered = _COVERED_PARADIGMS[mod_lower]
        if par_lower in covered or par_lower == "default":
            if not _has_advanced_keywords(intent_lower):
                logger.debug("Level 0: standard paradigm '%s/%s' with skill '%s'",
                             mod_lower, par_lower, matched_skill)
                return 0

    # Level 2: Advanced method keywords in user intent
    if _has_advanced_keywords(intent_lower):
        logger.info("Level 2: advanced method keywords detected in intent")
        return 2

    # Level 2: Non-standard paradigm (modality known but paradigm not covered)
    if mod_lower in _COVERED_PARADIGMS and par_lower:
        covered = _COVERED_PARADIGMS[mod_lower]
        if par_lower not in covered and par_lower != "default":
            logger.info("Level 2: paradigm '%s' not in covered set for '%s'",
                        par_lower, mod_lower)
            return 2

    # Level 2: QC failure with failed remedies
    if qc_failures >= 1 and len(failed_remedies) >= 1:
        logger.info("Level 2: QC failure with %d failed remedies", len(failed_remedies))
        return 2

    # Level 1: Parameter uncertainty indicators
    if _has_parameter_keywords(intent_lower):
        logger.info("Level 1: parameter-related question detected")
        return 1

    # Level 1: No matched skill but modality is known
    if not matched_skill and mod_lower in _STANDARD_MODALITIES:
        logger.info("Level 1: known modality '%s' but no matched skill", mod_lower)
        return 1

    # Level 1: Fingerprint shows unusual characteristics
    if fingerprint and _has_unusual_characteristics(fingerprint):
        logger.info("Level 1: unusual data characteristics detected")
        return 1

    # Default: Level 0 for anything that doesn't trigger above
    logger.debug("Level 0: default (standard scenario)")
    return 0


def _has_advanced_keywords(text: str) -> bool:
    """Check if text contains advanced method keywords requiring research."""
    return any(kw in text for kw in _ADVANCED_METHOD_KEYWORDS)


def _has_parameter_keywords(text: str) -> bool:
    """Check if text contains parameter-related question keywords."""
    return any(kw in text for kw in _PARAMETER_KEYWORDS)


def _has_unusual_characteristics(fingerprint: Dict[str, Any]) -> bool:
    """Detect unusual data characteristics from fingerprint.

    "Unusual" means values outside typical ranges that suggest
    non-standard equipment or acquisition settings.
    """
    freq = fingerprint.get("frequency_hz", 0)
    n_channels = fingerprint.get("n_channels", 0)

    # Extremely high sampling rate (>5kHz) suggests intracranial or special setup
    if freq > 5000:
        return True

    # Very high channel count (>256) suggests HD-EEG or special system
    if n_channels > 256:
        return True

    # Very low sampling rate (<64 Hz) suggests unusual acquisition
    if 0 < freq < 64:
        return True

    # Very few channels (<3) for EEG might indicate unusual setup
    modality = fingerprint.get("modality", "")
    if modality == "eeg" and 0 < n_channels < 3:
        return True

    return False
