"""Canonical step-key parsing.

Pipeline steps are written as ``operator[:params]`` strings (e.g. ``"ica:eog"``,
``"bandpass:0.5,40"``, ``"drop_bads:auto"``). The proven-pipeline body parser
keeps the full string (``"ica:eog"``); the experience store's
``NegativeExample.failed_step`` ought to refer to the *operator* only so set
intersections at the penalty layer line up. This module is the single seam.

The whitelist is sourced from ``preprocess.pipeline.AVAILABLE_STEPS`` so it
never drifts from the actual step dispatcher.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Tuple


@lru_cache(maxsize=1)
def known_operators() -> frozenset[str]:
    """Lazy-load AVAILABLE_STEPS to avoid eager-importing MNE-heavy pipeline."""
    try:
        from easybci_lib.tools.neural_processing.preprocess.pipeline import AVAILABLE_STEPS
        return frozenset(AVAILABLE_STEPS)
    except Exception:
        # Fallback to the historical short list from proven_match — keeps
        # canonical_step usable even when MNE is unavailable.
        return frozenset({
            "pick_channels", "drop_bads", "drop_nondata_channels",
            "interpolate_bads", "car", "bipolar_ref", "notch", "bandpass",
            "hilbert", "ica", "resample", "scale", "clip", "fill_nan",
            "extract_psd_bands", "extract_csp", "extract_tfr", "extract_connectivity",
        })


def canonical_step(s: str) -> Tuple[str, str]:
    """Split a step string into ``(operator, params)``.

    Examples
    --------
    >>> canonical_step("ica:eog")
    ('ica', 'eog')
    >>> canonical_step("bandpass:0.5,40")
    ('bandpass', '0.5,40')
    >>> canonical_step("ica")
    ('ica', '')
    >>> canonical_step("notarealstep")
    ('', 'notarealstep')

    Whitespace around the operator and the params half is stripped. Unknown
    operators (not in ``AVAILABLE_STEPS``) return ``("", original)`` so the
    caller can choose to reject or store under ``failure_mode="legacy_unparseable"``.
    """
    if not isinstance(s, str):
        return "", str(s) if s is not None else ""
    raw = s.strip()
    if not raw:
        return "", ""
    if ":" in raw:
        op, _, params = raw.partition(":")
    else:
        op, params = raw, ""
    op = op.strip().lower()
    params = params.strip()
    if op not in known_operators():
        return "", raw
    return op, params


def canonical_op(s: str) -> str:
    """Return only the operator name (empty string when unknown)."""
    return canonical_step(s)[0]
