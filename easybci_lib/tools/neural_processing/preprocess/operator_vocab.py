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
- Runnability is a separate question from vocabulary membership: many
  canonical operators are documented (``SKILL``) but have no executor.
  ``runnable_operators()`` / ``check_runnable()`` are the plan-time gate so a
  user is never asked to confirm a step the generated pipeline cannot run.
"""
from __future__ import annotations

import difflib
from typing import Any, Callable, Dict, List, Tuple

# The canonical operator vocabulary — union of engine + codegen + skill executors.
CANONICAL_OPERATORS: frozenset[str] = frozenset({
    # channel ops
    "pick_channels", "drop_bads", "drop_nondata_channels", "interpolate_bads",
    # reference
    "car", "bipolar_ref",
    # filter / transform
    "notch", "bandpass", "hilbert",
    # artifact
    "ica", "regression_eog",
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
    "respiratory_events", "plm_detect", "epoch_qc_sleep",
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
# Documentation only: the operator has a SKILL.md but NO executor implements
# it. Such a step cannot run in the generated pipeline.py — the plan-time gate
# (``check_runnable``) rejects it before the user is asked to confirm.
SKILL = "skill"
# Epoch-stage operators: run by the trial-QC block of the generated
# pipeline.py (after all continuous steps), never by ``_OPS``.
TRIAL_STAGE = "trial_stage"

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
    "regression_eog":        frozenset({CODEGEN}),
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
    # ── Extended operators (skill-documented; SKILL = no executor yet) ──
    # epoch / segmentation
    "epoch":                 frozenset({SKILL}),
    "segment":               frozenset({SKILL}),
    "baseline_correct":      frozenset({SKILL}),
    "reject_epochs":         frozenset({TRIAL_STAGE}),
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
    "reject_bad_segments":   frozenset({CODEGEN}),
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
    # psg
    "respiratory_events":    frozenset({SKILL}),
    "plm_detect":            frozenset({SKILL}),
    "epoch_qc_sleep":        frozenset({SKILL}),
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


def trial_stage_operators() -> List[str]:
    """Epoch-stage operators handled by the generated pipeline's trial-QC block."""
    return [op for op, ex in OPERATOR_EXECUTORS.items() if TRIAL_STAGE in ex]


def runnable_operators() -> List[str]:
    """Operators the generated ``pipeline.py`` can actually execute.

    This is the plan-time vocabulary: a step outside this list fails in the
    generated script no matter how well it is documented.
    """
    return [op for op, ex in OPERATOR_EXECUTORS.items()
            if CODEGEN in ex or TRIAL_STAGE in ex]


# ── Substitutes for non-runnable operators ──────────────────────────────────
# Shown to the LLM when the plan-time gate rejects a step, so it can propose a
# runnable alternative instead of discovering the gap at execution time.
_DOWNSTREAM_NOTE = (
    "downstream analysis / feature extraction, not a preprocessing step — drop "
    "it from the pipeline and run it on the preprocessed NWB or AI_ready epochs"
)
_EPOCHING_NOTE = (
    "epoching is built in: set epoch_window (event-locked) or segment_duration/"
    "stride on plan_pipeline and request the ai_ready deliverable; trial-level "
    "QC is reject_epochs"
)
_EVENTS_NOTE = "events are read automatically from the recording / sidecar; no step needed"
_DATASET_NOTE = "dataset-level operation, not a per-file pipeline step"
_MEG_NOTE = (
    "MEG hardware compensation is not available in the generated pipeline; "
    "apply it upstream before EasyBCI"
)
_FNIRS_NOTE = "fNIRS optical conversion is not available in the generated pipeline"

_SUBSTITUTES: Dict[str, Tuple[Tuple[str, ...], str]] = {
    # ocular / artifact
    "ssp_eog": (("regression_eog", "ica"), ""),
    "ssp_ecg": (("ica",), ""),
    "asr": (("reject_bad_segments", "clip"), ""),
    "wavelet_ica": (("ica",), ""),
    "ic_classify": (("ica",), ""),
    "manual_ic_selection": (("ica",), ""),
    "overlap_regression": (("regression_eog",), ""),
    "autoreject": (("reject_epochs",), ""),
    "detect_artifact_spans": (("reject_bad_segments",), ""),
    "interpolate_artifact": (("reject_bad_segments", "clip"), ""),
    # filter
    "zapline": (("notch",), ""),
    "comb_filter": (("notch",), ""),
    "detrend": (("bandpass",), "use a high-pass edge, e.g. bandpass:0.1,"),
    "high_pass_dc_removal": (("bandpass",), "use a high-pass edge, e.g. bandpass:0.1,"),
    "smooth": (("bandpass",), "use a low-pass edge, e.g. bandpass:,30"),
    "anti_alias_resample": (("resample",), "resample applies its own anti-alias filter"),
    # reference
    "bipolar_ref": (("car",), ""),
    "reref_channels": (("car",), ""),
    "laplacian_ref": (("car",), ""),
    "mastoid_reference": (("car",), ""),
    "rest_reference": (("car",), ""),
    # channel
    "interpolate_bads": (("drop_bads",), "bad channels are dropped, not interpolated"),
    "detect_bads": (("drop_bads",), "drop_bads:auto detects and drops"),
    "mark_bads": (("drop_bads",), "name channels explicitly: drop_bads:auto,<ch1>,<ch2>"),
    "pick_channels": (("drop_nondata_channels", "drop_bads"), ""),
    "minmax_scale": (("scale",), "scale:minmax"),
    "set_channel_types": ((), "channel types come from the loader; no step needed"),
    "set_montage": ((), "no montage is needed by the generated pipeline"),
    # spike
    "spike_sorting": (("threshold_spike",), ""),
    "bin_spikes": (("mua_binning",), ""),
    "firing_rate": (("mua_binning",), ""),
    "no_op": ((), "remove the step"),
}
_NOTE_GROUPS: Tuple[Tuple[frozenset, str], ...] = (
    (frozenset({
        "csp", "xdawn", "riemannian_covariance", "dss", "hilbert", "filter_bank",
        "aggregate_bands", "amplitude_modulation", "graph_metrics", "coherence",
        "dtf_pdc", "granger_causality", "pli_wpli", "plv", "cwt_morlet",
        "log_band_power", "multitaper_psd", "spectral_edge_freq", "stft",
        "dipole_fit", "lcmv_beamformer", "sloreta", "extract_psd_bands",
        "extract_csp", "extract_tfr", "extract_connectivity", "baseline_correct",
        "autocorrelogram", "cross_correlogram", "waveform_snippet",
        "peakdetect_qrs", "hfo_detector", "ied_detector", "sleep_stager",
        "respiratory_events", "plm_detect", "epoch_qc_sleep",
        "electrode_coord_validate", "marker_integrity_check",
        "derive_bipolar_channel",
    }), _DOWNSTREAM_NOTE),
    (frozenset({"epoch", "segment"}), _EPOCHING_NOTE),
    (frozenset({"define_events", "import_events", "repair_events",
                "select_events", "attach_metadata"}), _EVENTS_NOTE),
    (frozenset({"concatenate", "crop", "split_runs", "equalize_channels",
                "exclude_subjects", "sort_epochs"}), _DATASET_NOTE),
    (frozenset({"maxwell_filter", "ctf_grad_comp", "estimate_head_position",
                "align_head_position"}), _MEG_NOTE),
    (frozenset({"modified_beer_lambert", "optical_density",
                "short_separation_regression"}), _FNIRS_NOTE),
)


def substitutes_for(operator: str) -> Tuple[List[str], str]:
    """``(runnable_alternatives, note)`` for a non-runnable operator."""
    op = (operator or "").strip().lower()
    if op in _SUBSTITUTES:
        alts, note = _SUBSTITUTES[op]
        return list(alts), note
    for members, note in _NOTE_GROUPS:
        if op in members:
            return [], note
    return [], "no runnable equivalent — remove the step or ask the user"


def _param_value_str(value: Any, list_sep: str = ",") -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, (list, tuple)):
        return list_sep.join(_param_value_str(v) for v in value)
    return str(value).strip()


# Operators whose param string is ``key=value`` pairs rather than positional
# values — serializing their dict params by value alone loses every key.
_KV_PARAM_OPERATORS: frozenset = frozenset({"reject_epochs", "reject_bad_segments"})

# Multi-slot positional operators: dict params are placed by key name, never
# by dict order (``{"high": 30, "low": 8}`` must give ``bandpass:8,30``).
# Each slot lists the key aliases accepted for it.
_POSITIONAL_SLOTS: Dict[str, Tuple[Tuple[str, ...], ...]] = {
    "bandpass": (
        ("low", "l_freq", "lo", "low_hz", "lowcut", "f_low"),
        ("high", "h_freq", "hi", "high_hz", "highcut", "f_high"),
        ("route", "ch_type", "channel_type"),
    ),
}


def _positional_params(op: str, params: Dict[str, Any]) -> List[Any]:
    slots = _POSITIONAL_SLOTS.get(op)
    if slots is None:
        return list(params.values())
    lowered = {str(k).strip().lower(): v for k, v in params.items()}
    if not any(alias in lowered for slot in slots for alias in slot):
        return list(params.values())
    values = [next((lowered[a] for a in slot if a in lowered), None) for slot in slots]
    while values and values[-1] in (None, ""):
        values.pop()
    return values


def _compact_number_token(token: str) -> str:
    try:
        value = float(token)
    except ValueError:
        return token
    if value != value or value in (float("inf"), float("-inf")):
        return token
    return str(int(value)) if value.is_integer() else repr(value)


def canonical_step_key(step: Any) -> str:
    """Step string with every numeric token in compact form (``8.0`` → ``8``).

    Compares step lists written by different producers (LLM dicts, older
    script headers) without treating ``bandpass:8,30`` and
    ``bandpass:8.0,30.0`` as different plans.
    """
    s = step_to_string(step)
    if ":" not in s:
        return s
    op, _, param = s.partition(":")
    out = []
    for part in param.split(","):
        if "=" in part:
            k, _, v = part.partition("=")
            out.append(k + "=" + "~".join(_compact_number_token(t) for t in v.split("~")))
        else:
            out.append(_compact_number_token(part))
    return f"{op}:{','.join(out)}"


def step_to_string(step: Any) -> str:
    """Serialize a string- or dict-form step to ``operator[:param]``.

    Dict form: ``{"operator"|"name": op, "params"|"parameters": {...}}``.
    ``{"raw": "..."}`` params pass through verbatim; key=value operators
    (``reject_epochs`` / ``reject_bad_segments``) keep their keys, with list
    values joined by ``~`` (``window=3.5~7``); other operators join values
    positionally, keeping empty slots so ``bandpass:,40`` stays a low-pass.
    Multi-slot operators (``bandpass``) are ordered by key name, not dict
    order. Integral floats render without ``.0``.
    """
    if not isinstance(step, dict):
        return str(step or "").strip()
    op = str(step.get("operator") or step.get("name") or "").strip()
    params = step.get("params")
    if params is None:
        params = step.get("parameters")
    if isinstance(params, dict):
        if set(params) == {"raw"}:
            ps = _param_value_str(params["raw"])
        elif op.lower() in _KV_PARAM_OPERATORS:
            ps = ",".join(
                "{}={}".format(k, _param_value_str(v, "~"))
                for k, v in params.items() if v is not None and v != ""
            )
        else:
            ps = ",".join(_param_value_str(v) for v in _positional_params(op.lower(), params))
            if not ps.strip(","):
                ps = ""
    elif isinstance(params, (list, tuple)):
        ps = _param_value_str(params)
    else:
        ps = _param_value_str(params)
    return f"{op}:{ps}" if ps else op


def check_runnable(steps: List[Any]) -> List[Dict[str, Any]]:
    """Steps the generated pipeline.py cannot execute (empty list = all runnable).

    Each entry: ``index``, ``step``, ``operator``, ``reason``
    (``unknown_operator`` — not in the vocabulary at all; ``no_executor`` —
    documented but not implemented), ``substitutes`` and an optional ``note``.
    Synonyms are resolved first (``highpass`` → ``bandpass`` is runnable).
    """
    runnable = set(runnable_operators())
    out: List[Dict[str, Any]] = []
    for i, s in enumerate(steps or []):
        raw = step_to_string(s)
        op, param = _split(raw)
        if not op:
            continue
        if op in OPERATOR_SYNONYMS:
            op = _split(OPERATOR_SYNONYMS[op](param))[0]
        if op in runnable:
            continue
        entry: Dict[str, Any] = {
            "index": i,
            "step": raw,
            "operator": op,
            "reason": "no_executor" if op in CANONICAL_OPERATORS else "unknown_operator",
        }
        alts, note = substitutes_for(op)
        entry["substitutes"] = alts
        if note:
            entry["note"] = note
        if not alts and entry["reason"] == "unknown_operator":
            near = difflib.get_close_matches(op, sorted(runnable), n=1, cutoff=0.6)
            if near:
                entry["did_you_mean"] = near[0]
        out.append(entry)
    return out


def non_runnable_envelope(problems: List[Dict[str, Any]], *, tool: str) -> Dict[str, Any]:
    """Structured tool return for a plan-time rejection of non-runnable steps."""
    listed = ", ".join(p["step"] for p in problems)
    return {
        "success": False,
        "guard": "non_runnable_steps",
        "error": (
            f"{tool}: {len(problems)} step(s) cannot run in the generated "
            f"pipeline.py: {listed}"
        ),
        "non_runnable_steps": problems,
        "runnable_operators": runnable_operators(),
        "fix_hint": (
            "Replace each listed step with one of its `substitutes` (or drop it "
            "when its `note` says it is not a preprocessing step), then call "
            f"{tool} again and tell the user which step changed and why. Only "
            "operators in `runnable_operators` execute; a SKILL.md alone does "
            "not make an operator runnable."
        ),
    }


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


_ROUTED_OPERATORS: frozenset = frozenset({"bandpass"})

_ROUTE_LABEL_ALIASES: Dict[str, str] = {
    "respiratory": "resp",
    "respiration": "resp",
    "ekg": "ecg",
    "ref": "ref_meg",
}


def _normalize_routed_param(op: str, param: str) -> Tuple[str, bool]:
    """Normalize route label aliases in operators that support channel routing.

    Returns ``(normalized_param, was_changed)``.
    """
    if op not in _ROUTED_OPERATORS or not param:
        return param, False
    parts = param.split(",")
    if len(parts) < 3 or not parts[2].strip():
        return param, False
    route = parts[2].strip().lower()
    canonical = _ROUTE_LABEL_ALIASES.get(route)
    if canonical:
        parts[2] = canonical
        return ",".join(parts), True
    return param, False


def normalize_step(step_str: str) -> Tuple[str, bool]:
    """Normalize a single ``operator[:param]`` step to canonical form.

    Returns ``(canonical_step, was_normalized)``.

    - Canonical operator → returned unchanged (``was_normalized=False``), but
      lower-cased/trimmed on the operator token.
    - Known synonym → rewritten to canonical form (``was_normalized=True``).
    - Unknown operator → ``UnknownOperatorError`` with a nearest-match hint.
    """
    if isinstance(step_str, dict):
        step_str = step_to_string(step_str)
    op, param = _split(step_str)
    if not op:
        raise UnknownOperatorError(f"empty operator in step {step_str!r}")

    if op in CANONICAL_OPERATORS:
        param, route_changed = _normalize_routed_param(op, param)
        canonical = f"{op}:{param}" if param else op
        changed = canonical != (step_str or "").strip() or route_changed
        return canonical, changed

    if op in OPERATOR_SYNONYMS:
        result = OPERATOR_SYNONYMS[op](param)
        result_op, result_param = _split(result)
        result_param, _ = _normalize_routed_param(result_op, result_param)
        result = f"{result_op}:{result_param}" if result_param else result_op
        return result, True

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
