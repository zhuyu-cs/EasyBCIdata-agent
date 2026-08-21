"""Single source of truth for preprocessing operator names.

Two execution paths historically kept their own operator vocabularies, which
drifted apart and let LLM-authored synonyms (``highpass`` / ``lowpass`` /
``bad_channels``) slip through to the bottom layer and get *silently skipped*:

- the runtime engine dispatcher: ``preprocess.pipeline.preprocess`` /
  ``AVAILABLE_STEPS``
- the codegen standalone bundle: ``codegen.generator._OPS``

This module unifies both vocabularies and provides pre-execution normalization
so a non-canonical-but-recognisable name is *corrected* (not skipped) before it
ever reaches an executor, and a truly unknown name fails loud with a suggestion.

Design notes
------------
- ``CANONICAL_OPERATORS`` is the union of every operator either executor can
  run. ``AVAILABLE_STEPS`` (engine) and ``_OPS`` (codegen) should derive their
  membership from here so they never drift again.
- ``OPERATOR_SYNONYMS`` maps a non-canonical operator token to a callable that
  rewrites the whole ``operator[:param]`` step into its canonical form. This is
  a function (not a static string) because ``highpass:X`` / ``lowpass:X`` must
  move the cutoff into the correct side of ``bandpass``.
- Normalization is ALWAYS applied for known synonyms (per product decision:
  "必须全部纠正"). Only genuinely unmappable names raise.
"""
from __future__ import annotations

import difflib
from typing import Callable, Dict, List, Tuple

# The canonical operator vocabulary — union of engine + codegen + skill executors.
CANONICAL_OPERATORS: frozenset[str] = frozenset({
    # channel ops
    "pick_channels", "drop_bads", "drop_nondata_channels", "interpolate_bads",
    # reference
    "car", "bipolar_ref",
    # filter / transform
    "notch", "bandpass", "hilbert",
    # artifact
    "ica",
    # resample / scale / clean
    "resample", "scale", "clip", "fill_nan",
    # segment / label ops
    "reject_by_labels",
    # spike ops (codegen standalone)
    "threshold_spike", "mua_binning",
    # feature extraction (epoched)
    "extract_psd_bands", "extract_csp", "extract_tfr", "extract_connectivity",
    # ── Extended (skill-documented) ──
    "epoch", "segment", "baseline_correct", "reject_epochs",
    "define_events", "import_events", "repair_events", "select_events",
    "attach_metadata",
    "detect_bads", "mark_bads", "set_channel_types", "set_montage",
    "derive_bipolar_channel", "minmax_scale",
    "ic_classify", "manual_ic_selection", "detect_artifact_spans",
    "interpolate_artifact", "reject_bad_segments", "wavelet_ica",
    "overlap_regression",
    "maxwell_filter", "ctf_grad_comp", "estimate_head_position",
    "align_head_position",
    "detrend", "filter_bank", "smooth",
    "reref_channels",
    "dss",
    "aggregate_bands", "amplitude_modulation",
    "graph_metrics",
    "concatenate", "crop", "split_runs", "equalize_channels",
    "exclude_subjects", "sort_epochs",
    "no_op",
    "sleep_stager",
})


# ── Capability view ─────────────────────────────────────────────────────────
# Which executor(s) can actually run each canonical operator. This is the
# single source of truth both vocabularies derive from — the runtime engine's
# ``AVAILABLE_STEPS`` and the codegen standalone bundle's ``_OPS`` MUST NOT
# hardcode their own lists; they read ``engine_operators()`` / ``codegen_operators()``
# here instead, so the two can never drift apart again (the root cause of the
# silent-skip bug). Insertion order is preserved to keep a stable display order.
ENGINE = "engine"
CODEGEN = "codegen"
SKILL = "skill"

OPERATOR_EXECUTORS: Dict[str, frozenset[str]] = {
    # channel ops
    "pick_channels":         frozenset({ENGINE}),
    "drop_bads":             frozenset({ENGINE, CODEGEN}),
    "drop_nondata_channels": frozenset({ENGINE, CODEGEN}),
    "interpolate_bads":      frozenset({ENGINE}),
    # reference
    "car":                   frozenset({ENGINE, CODEGEN}),
    "bipolar_ref":           frozenset({ENGINE}),
    # filter / transform
    "notch":                 frozenset({ENGINE, CODEGEN}),
    "bandpass":              frozenset({ENGINE, CODEGEN}),
    "hilbert":               frozenset({ENGINE}),
    # artifact
    "ica":                   frozenset({ENGINE, CODEGEN}),
    # resample / scale / clean
    "resample":              frozenset({ENGINE, CODEGEN}),
    "scale":                 frozenset({ENGINE, CODEGEN}),
    "clip":                  frozenset({ENGINE, CODEGEN}),
    "fill_nan":              frozenset({ENGINE, CODEGEN}),
    # segment / label ops
    "reject_by_labels":      frozenset({ENGINE, CODEGEN}),
    # spike ops (codegen standalone only)
    "threshold_spike":       frozenset({CODEGEN}),
    "mua_binning":           frozenset({CODEGEN}),
    # feature extraction (epoched — engine only)
    "extract_psd_bands":     frozenset({ENGINE}),
    "extract_csp":           frozenset({ENGINE}),
    "extract_tfr":           frozenset({ENGINE}),
    "extract_connectivity":  frozenset({ENGINE}),
    # ── Extended operators (skill-documented, codegen emits inline code) ──
    # epoch / segmentation
    "epoch":                 frozenset({SKILL}),
    "segment":               frozenset({SKILL}),
    "baseline_correct":      frozenset({SKILL}),
    "reject_epochs":         frozenset({SKILL}),
    # event / trial management
    "define_events":         frozenset({SKILL}),
    "import_events":         frozenset({SKILL}),
    "repair_events":         frozenset({SKILL}),
    "select_events":         frozenset({SKILL}),
    "attach_metadata":       frozenset({SKILL}),
    # channel (extended)
    "detect_bads":           frozenset({SKILL}),
    "mark_bads":             frozenset({SKILL}),
    "set_channel_types":     frozenset({SKILL}),
    "set_montage":           frozenset({SKILL}),
    "derive_bipolar_channel": frozenset({SKILL}),
    "minmax_scale":          frozenset({SKILL}),
    # adaptive cleaning (extended)
    "ic_classify":           frozenset({SKILL}),
    "manual_ic_selection":   frozenset({SKILL}),
    "detect_artifact_spans": frozenset({SKILL}),
    "interpolate_artifact":  frozenset({SKILL}),
    "reject_bad_segments":   frozenset({SKILL}),
    "wavelet_ica":           frozenset({SKILL}),
    "overlap_regression":    frozenset({SKILL}),
    # MEG hardware
    "maxwell_filter":        frozenset({SKILL}),
    "ctf_grad_comp":         frozenset({SKILL}),
    "estimate_head_position": frozenset({SKILL}),
    "align_head_position":   frozenset({SKILL}),
    # filter (extended)
    "detrend":               frozenset({SKILL}),
    "filter_bank":           frozenset({SKILL}),
    "smooth":                frozenset({SKILL}),
    # reference (extended)
    "reref_channels":        frozenset({SKILL}),
    # spatial (extended)
    "dss":                   frozenset({SKILL}),
    # spectral (extended)
    "aggregate_bands":       frozenset({SKILL}),
    "amplitude_modulation":  frozenset({SKILL}),
    # connectivity (extended)
    "graph_metrics":         frozenset({SKILL}),
    # dataset-level
    "concatenate":           frozenset({SKILL}),
    "crop":                  frozenset({SKILL}),
    "split_runs":            frozenset({SKILL}),
    "equalize_channels":     frozenset({SKILL}),
    "exclude_subjects":      frozenset({SKILL}),
    "sort_epochs":           frozenset({SKILL}),
    # misc
    "no_op":                 frozenset({SKILL}),
    # qc
    "sleep_stager":          frozenset({SKILL}),
}

# Invariant: the capability table and CANONICAL_OPERATORS describe the same set,
# and every canonical operator is runnable by at least one executor. Enforced at
# import so a future edit that adds an op to one place but not the other fails
# loud immediately (unit-tested too).
assert set(OPERATOR_EXECUTORS) == set(CANONICAL_OPERATORS), (
    "OPERATOR_EXECUTORS and CANONICAL_OPERATORS out of sync: "
    f"{set(OPERATOR_EXECUTORS) ^ set(CANONICAL_OPERATORS)}"
)
assert all(v for v in OPERATOR_EXECUTORS.values()), (
    "every canonical operator must be runnable by at least one executor"
)


def engine_operators() -> List[str]:
    """Canonical operators the runtime engine implements (declaration order)."""
    return [op for op, ex in OPERATOR_EXECUTORS.items() if ENGINE in ex]


def codegen_operators() -> List[str]:
    """Canonical operators the codegen standalone bundle implements."""
    return [op for op, ex in OPERATOR_EXECUTORS.items() if CODEGEN in ex]


class UnknownOperatorError(ValueError):
    """Raised when a step's operator is neither canonical nor a known synonym."""


def _split(step_str: str) -> Tuple[str, str]:
    raw = (step_str or "").strip()
    if ":" in raw:
        op, _, param = raw.partition(":")
        return op.strip().lower(), param.strip()
    return raw.lower(), ""


def _syn_highpass(param: str) -> str:
    # highpass:X → keep frequencies above X → bandpass with l_freq=X, h_freq=None
    lo = param.strip() if param.strip() else "1.0"
    return f"bandpass:{lo},"


def _syn_lowpass(param: str) -> str:
    # lowpass:X → keep frequencies below X → bandpass with l_freq=None, h_freq=X
    hi = param.strip() if param.strip() else "40.0"
    return f"bandpass:,{hi}"


def _syn_drop_bads(param: str) -> str:
    return f"drop_bads:{param}" if param.strip() else "drop_bads"


# Synonym operator-token → rewrite(param) -> canonical step string.
OPERATOR_SYNONYMS: Dict[str, Callable[[str], str]] = {
    "highpass": _syn_highpass,
    "high_pass": _syn_highpass,
    "hpf": _syn_highpass,
    "lowpass": _syn_lowpass,
    "low_pass": _syn_lowpass,
    "lpf": _syn_lowpass,
    "bad_channels": _syn_drop_bads,
    "drop_bad_channels": _syn_drop_bads,
    "reject_channels": _syn_drop_bads,
    # Skill doc long-names → canonical short-names
    "notch_filter": lambda p: f"notch:{p}" if p else "notch",
    "bandpass_filter": lambda p: f"bandpass:{p}" if p else "bandpass",
}


def normalize_step(step_str: str) -> Tuple[str, bool]:
    """Normalize a single ``operator[:param]`` step to canonical form.

    Returns ``(canonical_step, was_normalized)``.

    - Canonical operator → returned unchanged (``was_normalized=False``), but
      lower-cased/trimmed on the operator token.
    - Known synonym → rewritten to canonical form (``was_normalized=True``).
    - Unknown operator → ``UnknownOperatorError`` with a nearest-match hint.
    """
    op, param = _split(step_str)
    if not op:
        raise UnknownOperatorError(f"empty operator in step {step_str!r}")

    if op in CANONICAL_OPERATORS:
        canonical = f"{op}:{param}" if param else op
        return canonical, canonical != (step_str or "").strip()

    if op in OPERATOR_SYNONYMS:
        return OPERATOR_SYNONYMS[op](param), True

    # Unknown — fail loud with a suggestion.
    candidates = sorted(CANONICAL_OPERATORS) + sorted(OPERATOR_SYNONYMS)
    near = difflib.get_close_matches(op, candidates, n=1, cutoff=0.5)
    hint = f" Did you mean {near[0]!r}?" if near else ""
    raise UnknownOperatorError(
        f"unknown operator {op!r} (from step {step_str!r}).{hint} "
        f"Canonical operators: {sorted(CANONICAL_OPERATORS)}"
    )


def normalize_steps(steps: List[str]) -> Tuple[List[str], List[str]]:
    """Normalize a list of steps.

    Returns ``(canonical_steps, notes)`` where ``notes`` has one human-readable
    line per step that was actually rewritten (for reasoning.md logging).
    Raises ``UnknownOperatorError`` on the first unmappable step.
    """
    out: List[str] = []
    notes: List[str] = []
    for s in steps:
        canonical, changed = normalize_step(s)
        out.append(canonical)
        if changed:
            notes.append(f"normalized {s!r} → {canonical!r}")
    return out, notes
