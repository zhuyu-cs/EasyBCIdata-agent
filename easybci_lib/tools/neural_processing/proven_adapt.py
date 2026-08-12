"""Adaptive reuse: anchor the enhanced skill's step *kinds/order*, recompute
numeric params from THIS recording's deep_inspect report.

Contract (design 03-adaptive-reuse.md): step kinds and order come strictly from
the skill's step list; the adapter only changes NUMBERS (which channels to drop,
which notch frequencies, the resample target, which time segments to reject). It
never adds or removes a step *kind* — the sole exception is `from_labels`, which
may skip its step when this recording has no labels (recorded explicitly).

Every slot emits a self_report row {param, strategy, gold, measured, adopted,
confidence, note} so "adapted, not copied" is auditable (tool-return-is-truth).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AdaptResult:
    steps: list[str]
    self_report: list[dict] = field(default_factory=list)
    out_of_range: bool = False
    out_of_range_reasons: list[str] = field(default_factory=list)


def _fmt(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else str(v)


_COMMON_TARGETS = [1000.0, 500.0, 512.0, 256.0, 250.0, 200.0, 128.0, 100.0]

# A file whose channel count differs from the skill's gold recording by more
# than this ratio (either direction) is flagged out_of_range and excluded from
# the batch — a very different montage (e.g. 31ch scalp vs 261ch sEEG) should
# not silently borrow a proven pipeline built for another electrode layout.
CHANNEL_COUNT_RATIO_TOLERANCE = 4.0


def _largest_safe_target(src_sfreq: float) -> float:
    """Largest common resample target strictly below src (Nyquist-safe)."""
    below = [t for t in _COMMON_TARGETS if t < src_sfreq]
    return max(below) if below else float(int(src_sfreq))


def _fp(report: dict) -> dict:
    return report.get("fingerprint", {}) if isinstance(report, dict) else {}


def _degraded(report: dict) -> bool:
    return bool(report.get("degraded")) if isinstance(report, dict) else True


def adapt_pipeline(skill_steps: list[str], slots: list[dict], report: dict,
                   *, gold_n_channels: int = 0, gold_modality: str = "",
                   reject_keywords: list[str] | None = None) -> AdaptResult:
    """Recompute numeric params for `skill_steps` from `report`, per `slots`."""
    steps = list(skill_steps)
    self_report: list[dict] = []
    slot_params = {s.get("param"): s.get("strategy") for s in (slots or [])}
    reject_keywords = [str(k).strip() for k in (reject_keywords or []) if str(k).strip()]

    # --- slot: bad_channels (from_deep_inspect) ---
    if "bad_channels" in slot_params:
        cs = report.get("channel_summary", {}) if isinstance(report, dict) else {}
        measured = list(cs.get("must_drop", []) or [])
        gold_note = "gold applied a fixed CSV bad-channel list (per-recording)"
        self_report.append({
            "param": "bad_channels", "strategy": "from_deep_inspect",
            "gold": gold_note, "measured": measured, "adopted": measured,
            "source": "deep_inspect.channel_summary.must_drop",
            "confidence": "low" if _degraded(report) else "high",
            "note": ("drop_bads:auto resolves at runtime from this recording's "
                     f"must_drop ({len(measured)} ch); gold's indices NOT copied"),
        })

    # --- slot: notch_freqs (from_powerline) ---
    if "notch_freqs" in slot_params:
        gold_notch = [_fmt(float(s.split(":", 1)[1])) for s in skill_steps
                      if s.startswith("notch:")]
        psd = report.get("psd_summary", {}) if isinstance(report, dict) else {}
        base = psd.get("power_line_peak_hz")
        harmonics = list(psd.get("harmonics_detected_hz", []) or [])
        if base:
            measured_freqs = [float(base)] + [float(h) for h in harmonics]
            new_notch = [f"notch:{_fmt(f)}" for f in measured_freqs]
            others = [s for s in steps if not s.startswith("notch:")]
            steps = new_notch + others  # notch(es) stay first, order preserved
            self_report.append({
                "param": "notch_freqs", "strategy": "from_powerline",
                "gold": gold_notch, "measured": [_fmt(f) for f in measured_freqs],
                "adopted": [_fmt(f) for f in measured_freqs],
                "source": "deep_inspect.psd_summary.power_line_peak_hz(+harmonics)",
                "confidence": "high", "note": "notch retargeted to measured mains",
            })
        else:
            self_report.append({
                "param": "notch_freqs", "strategy": "from_powerline",
                "gold": gold_notch, "measured": None, "adopted": gold_notch,
                "source": "fallback:gold (no power-line peak measured)",
                "confidence": "low", "note": "kept gold notch values",
            })

    fp = _fp(report)

    # --- slot: resample_target_hz (nyquist_bounded) ---
    if "resample_target_hz" in slot_params:
        gold_target = None
        for s in steps:
            if s.startswith("resample:"):
                gold_target = float(s.split(":", 1)[1])
                break
        src_sfreq = float(fp.get("sampling_freq_hz") or 0.0)
        if gold_target is not None:
            adopted = gold_target
            reason = "gold target feasible"
            if src_sfreq and gold_target >= src_sfreq:
                adopted = float(_largest_safe_target(src_sfreq))
                reason = f"gold target {gold_target} >= src {src_sfreq}; clamped"
            steps = [f"resample:{_fmt(adopted)}" if x.startswith("resample:") else x
                     for x in steps]
            self_report.append({
                "param": "resample_target_hz", "strategy": "nyquist_bounded",
                "gold": gold_target, "measured": src_sfreq, "adopted": adopted,
                "source": "min(gold_target, src-bounded)",
                "confidence": "low" if _degraded(report) else "high", "note": reason,
            })

    # --- slot: reject_time_segments (from_labels) ---
    # Inject a real reject_by_labels step (excise labelled windows) rather than
    # only reporting. The step is a no-op at runtime when a recording has no
    # matching labels, so it is safe to always prepend when keywords exist.
    if "reject_time_segments" in slot_params:
        n_events = int(fp.get("n_events") or 0)
        if reject_keywords:
            kw_arg = ",".join(reject_keywords)
            reject_step = f"reject_by_labels:{kw_arg}"
            # Excise BEFORE filtering/resampling so artefacts don't smear.
            steps = [reject_step] + [s for s in steps
                                     if not s.startswith("reject_by_labels:")]
            self_report.append({
                "param": "reject_time_segments", "strategy": "from_labels",
                "gold": "reject by keyword on gold labels",
                "measured": f"{n_events} events", "adopted": reject_step,
                "source": "skill.reject_keywords → reject_by_labels step",
                "confidence": "high",
                "note": ("segment rejection runs against this recording's own "
                         "labels; clean remainder is kept (whole file NOT dropped)"),
            })
        else:
            self_report.append({
                "param": "reject_time_segments", "strategy": "from_labels",
                "gold": "reject by keyword on gold labels",
                "measured": f"{n_events} events", "adopted": [],
                "source": "skipped (skill carries no reject_keywords)",
                "confidence": "low",
                "note": "no reject_keywords on skill — segment rejection skipped",
            })

    # --- out-of-range protection ---
    reasons: list[str] = []
    this_mod = str(fp.get("modality") or "").lower()
    if gold_modality and this_mod and this_mod != "auto" and this_mod != gold_modality.lower():
        reasons.append(f"modality mismatch ({this_mod} vs gold {gold_modality})")
    this_nch = int(fp.get("n_channels") or 0)
    if gold_n_channels and this_nch and (
        this_nch < gold_n_channels / CHANNEL_COUNT_RATIO_TOLERANCE
        or this_nch > gold_n_channels * CHANNEL_COUNT_RATIO_TOLERANCE
    ):
        reasons.append(f"channel count out of range ({this_nch} vs gold {gold_n_channels})")

    return AdaptResult(steps=steps, self_report=self_report,
                       out_of_range=bool(reasons), out_of_range_reasons=reasons)
