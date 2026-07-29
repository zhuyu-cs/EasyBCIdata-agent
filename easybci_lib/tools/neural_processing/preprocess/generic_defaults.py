"""Generic safe-default pipeline parameters.

When ``analysis_goal == "generic"`` (the user did not signal a specific
downstream — classification / source_localization / etc.), the proposer falls
back to broad-coverage, low-risk defaults. This module is the single source of
truth for the Generic-column defaults.

Note: this module returns the step list *before* ``_enforce_clean_output``
injects ``drop_bads:auto`` and ``drop_nondata_channels:data_only``. The
codegen layer adds those at script-generation time. The fingerprint dict
exposes ``drop_bads_will_be_injected`` so reasoning.md can disclose the
upcoming auto-injection honestly without the proposer needing to mutate the
step list.

Public surface
--------------
``generic_pipeline_defaults(fingerprint) -> dict`` — returns a dict with:
  - ``steps``: ordered list of step strings (``"notch:50"``, ``"bandpass:1,40"``, …)
  - ``segment_method``: ``"sliding"``
  - ``segment_duration``: 2.0
  - ``stride``: 1.0
  - ``rationale``: parallel list of short reasons keyed to ``steps``

Note: ``output_format`` is no longer carried — see
:mod:`easybci_lib.tools.neural_processing.output.format_policy` for the
single source of truth (NWB-only for the ``preprocessed/`` layer).
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


_DEFAULT_LINE_FREQ = 50  # Hz, used when fingerprint hints are absent

_GENERIC_BANDPASS_LOW = 1  # Hz — strips drift, mu/beta/low-gamma intact
_GENERIC_BANDPASS_HIGH = 40  # Hz

_RESAMPLE_THRESHOLD_HZ = 1000.0  # only downsample if fs > 1000 Hz
_RESAMPLE_TARGET_HZ = 500.0
_ICA_MIN_CHANNELS = 16  # ICA defaults off below this — too few components


def _detect_line_freq(fingerprint: Dict[str, Any]) -> int:
    """Best-effort line-frequency pick (50 vs 60 Hz).

    The Phase-1 contract says "auto 50/60 by region; default 50 when
    uncertain". We don't have geolocation here — but the fingerprint may
    carry a ``line_freq`` hint or the channel-name set may include "line50"/
    "line60" markers from notch-aware loaders. Default to 50.
    """
    if not isinstance(fingerprint, dict):
        return _DEFAULT_LINE_FREQ
    hint = fingerprint.get("line_freq")
    if isinstance(hint, (int, float)) and 40 <= hint <= 70:
        return 60 if abs(hint - 60) < abs(hint - 50) else 50
    if isinstance(hint, str):
        try:
            val = float(hint)
        except ValueError:
            return _DEFAULT_LINE_FREQ
        if 40 <= val <= 70:
            return 60 if abs(val - 60) < abs(val - 50) else 50
    return _DEFAULT_LINE_FREQ


def generic_pipeline_defaults(fingerprint: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Build the Generic Safe Defaults pipeline for the given data fingerprint.

    Parameters
    ----------
    fingerprint
        Optional dict with any of: ``n_channels`` (int), ``frequency_hz`` /
        ``frequency`` (Hz), ``line_freq`` (Hz hint), ``modality`` (str).
        Missing fields fall back to conservative assumptions.

    Returns
    -------
    dict with keys ``steps`` (list[str]), ``rationale`` (list[str], same
    length), ``segment_method``, ``segment_duration``, ``stride``,
    ``analysis_goal`` (always ``"generic"`` for clarity).
    """
    fp = dict(fingerprint or {})
    n_ch = fp.get("n_channels")
    fs = fp.get("frequency_hz", fp.get("frequency"))

    line_freq = _detect_line_freq(fp)

    steps: List[str] = []
    rationale: List[str] = []

    # 1. Markers-only cleanup — keep physio for ICA-EOG-aware downstream.
    steps.append("drop_nondata_channels:markers_only")
    rationale.append(
        "Generic Safe Default: drop pure marker/trigger channels only at this "
        "stage. Physiological references (EOG/ECG) are retained because we do "
        "not know whether the downstream analysis will benefit from "
        "EOG-aware ICA. Final output cleanup (post-ICA data_only) is "
        "applied separately by codegen when analysis_goal==generic."
    )

    # 2. Notch — line-frequency suppression.
    steps.append(f"notch:{line_freq}")
    rationale.append(
        f"Generic Safe Default: notch filter at {line_freq} Hz to suppress "
        "power-line interference. Region-aware autodetection picks 60 Hz when "
        "the fingerprint signals it; 50 Hz is the conservative fallback."
    )

    # 3. Bandpass 1–40 Hz — broad band, no paradigm assumption.
    steps.append(f"bandpass:{_GENERIC_BANDPASS_LOW},{_GENERIC_BANDPASS_HIGH}")
    rationale.append(
        f"Generic Safe Default: {_GENERIC_BANDPASS_LOW}–{_GENERIC_BANDPASS_HIGH} Hz "
        "bandpass strips slow drift below 1 Hz and high-frequency noise above "
        "40 Hz while preserving mu, beta, and low gamma rhythms. We do not "
        "assume a specific oscillation band because the user did not name a "
        "downstream task."
    )

    # 4. ICA — only when channel count permits stable decomposition.
    if isinstance(n_ch, int) and n_ch >= _ICA_MIN_CHANNELS:
        steps.append(f"ica:{min(n_ch, 30)}")
        rationale.append(
            f"Generic Safe Default: ICA enabled because the recording has "
            f"{n_ch} ≥ {_ICA_MIN_CHANNELS} channels. Below this threshold the "
            "decomposition is unstable; here it is safe to extract up to "
            "min(n_channels, 30) components."
        )

    # 5. Resample — only when fs is excessive.
    if isinstance(fs, (int, float)) and fs > _RESAMPLE_THRESHOLD_HZ:
        steps.append(f"resample:{int(_RESAMPLE_TARGET_HZ)}")
        rationale.append(
            f"Generic Safe Default: source rate {fs} Hz exceeds "
            f"{_RESAMPLE_THRESHOLD_HZ} Hz; downsample to "
            f"{int(_RESAMPLE_TARGET_HZ)} Hz to halve data volume while "
            "remaining well above the Nyquist criterion for the 40 Hz "
            "bandpass."
        )

    # 6. Robust scaling — generic safety against per-channel amplitude drift.
    steps.append("scale:robust")
    rationale.append(
        "Generic Safe Default: robust per-channel scaling (median + IQR) "
        "normalises amplitudes without being skewed by outlier spikes. This "
        "produces well-behaved features whether the downstream is "
        "classification, feature extraction, or pure exploration."
    )

    return {
        "steps": steps,
        "rationale": rationale,
        "segment_method": "sliding",
        "segment_duration": 2.0,
        "stride": 1.0,
        "analysis_goal": "generic",
        "fingerprint": {
            "n_channels": n_ch,
            "frequency_hz": fs,
            "line_freq": line_freq,
            "ica_enabled": isinstance(n_ch, int) and n_ch >= _ICA_MIN_CHANNELS,
            "resampled": isinstance(fs, (int, float)) and fs > _RESAMPLE_THRESHOLD_HZ,
            # codegen._enforce_clean_output injects drop_bads:auto for goal=generic
            "drop_bads_will_be_injected": True,
        },
    }


__all__ = ["generic_pipeline_defaults"]
