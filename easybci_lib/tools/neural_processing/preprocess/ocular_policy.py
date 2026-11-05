"""Plan-time ocular-artifact coverage check.

Pure function of the inspection report(s) and the proposed steps: were blinks
seen, which reference is available (``resolve_eog_reference`` — the same
resolver the generated ``op_ica`` / ``op_regression_eog`` use), does a step
remove them (``ica`` targeting eog / ``regression_eog``), and which steps only
look like ocular handling — ``clip`` / ``scale`` / trial rejection clip,
rescale or drop trials; the blink stays in every kept trial. The result is a
tool-return block asking for the decision to be shown to the user; it never
blocks a proposal. Decision table:
``bci/operators/adaptive_cleaning/ocular_artifact/SKILL.md``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Optional

from easybci_lib.tools.neural_processing.output.format_policy import is_invasive
from easybci_lib.tools.neural_processing.preprocess.eog_reference import resolve_eog_reference
from easybci_lib.tools.neural_processing.preprocess.operator_vocab import step_to_string


def _load_t() -> Optional[Callable[..., str]]:
    try:
        from easybci_agent.i18n import t
    except Exception:  # noqa: BLE001
        return None
    return t


_t = _load_t()

SKILL_NAME = "ocular_artifact"
# Awake adults blink ~15–20/min; below this the reference shows no blinks.
BLINK_DETECT_PER_MIN = 5.0
# ICA data sufficiency: enough recording for blink statistics, and ~25·k²
# samples at the ICA step for k components (k capped like op_ica).
ICA_MIN_DURATION_S = 60.0
ICA_SAMPLES_PER_COMPONENT_SQ = 25
ICA_MAX_COMPONENTS = 20
# A frontal proxy carries real EEG; ICA against it needs enough dimensions.
ICA_MIN_CHANNELS = 12
_NOT_OCULAR_OPS = ("clip", "scale", "reject_epochs", "reject_bad_segments")
# Paradigms whose eye movements are the signal (REM in sleep staging,
# saccade / gaze studies). A floor, matched as substrings of the paradigm.
_EOG_AS_SIGNAL = ("sleep", "psg", "eye", "saccade", "gaze", "oculo")


def _op(step: str) -> tuple[str, str]:
    op, _, param = step.partition(":")
    return op.strip().lower(), param.strip().lower()


def _handles_ocular(step: str) -> bool:
    op, param = _op(step)
    if op == "regression_eog":
        return True
    if op == "ica":
        targets = [t for t in (p.strip() for p in param.split(",")) if t in ("eog", "ecg")]
        return not targets or "eog" in targets
    return False


def _fs_at_ica(fs: float, step_strs: list[str]) -> float:
    """Sampling rate the ICA will see: the lowest of the native rate and any resample target."""
    rates = [fs] if fs > 0 else []
    for s in step_strs:
        op, param = _op(s)
        if op == "resample":
            try:
                rates.append(float(param.split(",")[0]))
            except ValueError:
                continue
    return min(rates) if rates else 0.0


def _ica_feasible(n_eff: int, duration_s: float, fs_ica: float) -> bool:
    k = max(1, min(n_eff, ICA_MAX_COMPONENTS))
    return duration_s >= ICA_MIN_DURATION_S and duration_s * fs_ica >= ICA_SAMPLES_PER_COMPONENT_SQ * k * k


def _recommend(kind: Optional[str], n_eff: int, duration_s: float, fs_ica: float,
               analysis_goal: Optional[str]) -> tuple[Optional[str], str]:
    online = analysis_goal == "online_inference"
    k = max(1, min(n_eff, ICA_MAX_COMPONENTS))
    data = (f"{n_eff} effective channels, {duration_s:.0f} s at {fs_ica:g} Hz "
            f"(ICA needs >= {ICA_MIN_DURATION_S:.0f} s and >= {ICA_SAMPLES_PER_COMPONENT_SQ * k * k} "
            f"samples for {k} components)")
    feasible = _ica_feasible(n_eff, duration_s, fs_ica)
    if kind == "eog_channel":
        if online:
            return "regression_eog", "online_inference: ICA is not used online; regress the EOG channel out"
        if not feasible:
            return "regression_eog", f"EOG channel present; {data} — too little data for a stable ICA"
        return "ica:eog", f"EOG channel present; {data}"
    if kind == "frontal_proxy":
        if not online and n_eff >= ICA_MIN_CHANNELS and feasible:
            return "ica:eog", f"{data} — enough for ICA; the frontal channels serve as the blink reference"
        return "regression_eog:auto", (
            f"{data}; ICA against a frontal proxy also needs >= {ICA_MIN_CHANNELS} channels and no "
            "online use — regression on the frontal proxy also removes real frontal EEG")
    return None, ("no EOG channel and no frontal-pole / AF channel: ocular artifacts cannot be "
                  "removed — state it in the proposal; trial rejection is the only fallback")


def _hint(b: dict[str, Any]) -> Optional[str]:
    if b["must_present"]:
        parts = [f"Blinks detected ({b['evidence']}) but no step removes them."]
        if b["not_counted"]:
            parts.append(f"{', '.join(b['not_counted'])} do not remove ocular artifacts: they clip, "
                         "rescale or drop trials and the blink stays in every kept trial.")
        parts.append(f"Recommended: {b['recommended']} — {b['recommended_reason']}.")
        parts.append("Show this to the user with the proposal: add the step (after bandpass, "
                     "before car) or record why ocular artifacts are left in. Decision table: "
                     "skill_view('ocular_artifact').")
        return " ".join(parts)
    if b["exempt"] == "eog_is_signal":
        return ("Eye movements are part of the signal for this paradigm (e.g. REM in sleep "
                "staging, saccade / gaze analysis) — do not remove them unless the user asks.")
    if b["exempt"] == "no_reference":
        if b.get("montage_recognized") is False:
            return ("No EOG or frontal channel recognised — the channel names are not 10-20 / "
                    f"10-10 labels ({', '.join(b.get('channel_names_sample') or [])}, ...). If some "
                    "electrodes sit near the eyes, ask the user which and name them explicitly: "
                    "ica:eog,<ch>,... or regression_eog:<ch>,...; otherwise state in the proposal "
                    "that ocular artifacts cannot be removed.")
        return str(b["recommended_reason"]) + "."
    if b["detected"] == "unknown" and not b["handled_by"] and b["exempt"] is None:
        rec = (f" If they must be removed: {b['recommended']} — {b['recommended_reason']}."
               if b["recommended"] else "")
        return (f"Blink rate could not be measured ({b['evidence']}) — check the frontal "
                "channels or ask the user whether ocular artifacts must be removed." + rec)
    return None


def assess_ocular(report: Optional[dict[str, Any]], steps: list[Any], *,
                  analysis_goal: Optional[str] = None, modality: Optional[str] = None,
                  paradigm: Optional[str] = None) -> dict[str, Any]:
    """Ocular-artifact coverage of ``steps`` for one inspection report (dict)."""
    step_strs = [s for s in (step_to_string(x) for x in steps or []) if s]
    block: dict[str, Any] = {
        "detected": "unknown", "blink_rate_per_min": None, "evidence": "no inspection report",
        "reference": None, "handled_by": [s for s in step_strs if _handles_ocular(s)],
        "not_counted": [s for s in step_strs if _op(s)[0] in _NOT_OCULAR_OPS],
        "recommended": None, "recommended_reason": None,
        "effective_channels": None, "duration_s": None, "montage_recognized": None,
        "exempt": None, "skill": SKILL_NAME, "must_present": False,
    }
    kind = None
    if isinstance(report, dict):
        fp = report.get("fingerprint") or {}
        must_drop = {str(c) for c in (report.get("channel_summary") or {}).get("must_drop") or []}
        names = [str(c["name"]) for c in report.get("channel_stats") or []
                 if isinstance(c, dict) and c.get("name") is not None]
        kept = [c for c in names if c not in must_drop]
        ref = resolve_eog_reference(kept)
        kind = ref["source"]
        if names:
            n_eog = len(ref["indices"]) if kind == "eog_channel" else 0
            n_eff = len(kept) - n_eog
        else:
            n_eff = max(0, int(fp.get("n_channels") or 0) - len(must_drop))
        duration = float(fp.get("duration_s") or 0.0)
        fs_ica = _fs_at_ica(float(fp.get("sampling_freq_hz") or 0.0), step_strs)
        art = report.get("artifact_summary") or {}
        rate, measured_ref = art.get("blink_rate_per_min"), art.get("blink_reference")
        rec, why = _recommend(kind, n_eff, duration, fs_ica, analysis_goal)
        label = ("eog" if kind == "eog_channel" else kind)
        block.update(reference=measured_ref or (
                         f"{label}({','.join(kept[i] for i in ref['blink_indices'])})" if kind else None),
                     recommended=rec, recommended_reason=why, effective_channels=n_eff,
                     duration_s=round(duration, 1), montage_recognized=ref["montage_recognized"])
        if not ref["montage_recognized"]:
            block["channel_names_sample"] = kept[:6]
        if isinstance(rate, (int, float)) and measured_ref:
            block["detected"] = bool(rate >= BLINK_DETECT_PER_MIN)
            block["blink_rate_per_min"] = float(rate)
            block["evidence"] = f"{measured_ref} blink events ~{float(rate):g}/min"
        else:
            block["evidence"] = ("blink rate not measured (no EOG / frontal channel, recording too "
                                 "short, or an inspection report from before event-based blink "
                                 "counting)")
    par = str(paradigm or "").lower()
    if is_invasive(modality):
        block["exempt"] = "invasive"
    elif any(k in par for k in _EOG_AS_SIGNAL):
        block["exempt"] = "eog_is_signal"
    elif analysis_goal == "online_inference":
        block["exempt"] = "online_inference"
    elif isinstance(report, dict) and kind is None:
        block["exempt"] = "no_reference"
    block["must_present"] = bool(block["detected"] is True and not block["handled_by"]
                                 and block["exempt"] is None)
    car_at = next((i for i, s in enumerate(step_strs) if _op(s)[0] == "car"), None)
    late = [s for i, s in enumerate(step_strs)
            if car_at is not None and i > car_at and _handles_ocular(s)]
    if late:
        block["order_warning"] = (f"{', '.join(late)} runs after car: the average reference has "
                                  "already spread the blink to every channel — move it before car.")
    hint = _hint(block)
    if hint:
        block["hint"] = hint
    return block


def load_inspection_reports(work_dir: Any) -> list[dict[str, Any]]:
    """Per-file reports (``middle_process/inspect/<fid>/``), else the root report."""
    if not work_dir:
        return []
    mp = Path(work_dir) / "middle_process"
    paths = sorted(mp.glob("inspect/*/inspection_report.json")) or [mp / "inspection_report.json"]
    out: list[dict[str, Any]] = []
    for p in paths:
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(d, dict):
            out.append(d)
    return out


def assess_ocular_multi(work_dir: Any, steps: list[Any], *,
                        analysis_goal: Optional[str] = None, modality: Optional[str] = None,
                        paradigm: Optional[str] = None,
                        fallback_report: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """``assess_ocular`` over every inspected file of a work_dir.

    Blinks seen in any file count. The shortest recording drives the
    recommendation: a method reliable there is reliable for every file.
    """
    reports = load_inspection_reports(work_dir)
    if not reports and isinstance(fallback_report, dict):
        reports = [fallback_report]
    kw = {"analysis_goal": analysis_goal, "modality": modality, "paradigm": paradigm}
    if len(reports) <= 1:
        return assess_ocular(reports[0] if reports else None, steps, **kw)
    per = [assess_ocular(r, steps, **kw) for r in reports]
    block = dict(min(per, key=lambda b: b.get("duration_s") or 0.0))
    det = [b["detected"] for b in per]
    block["detected"] = True if any(d is True for d in det) else (
        "unknown" if "unknown" in det else False)
    block["must_present"] = any(b["must_present"] for b in per)
    block["evidence"] = "; ".join(f"{r.get('file_id') or i}: {b['evidence']}"
                                  for i, (r, b) in enumerate(zip(reports, per)))
    block["per_file"] = [{"file_id": r.get("file_id"), "detected": b["detected"],
                          "blink_rate_per_min": b["blink_rate_per_min"],
                          "reference": b["reference"], "duration_s": b["duration_s"],
                          "recommended": b["recommended"]} for r, b in zip(reports, per)]
    block.pop("hint", None)
    hint = _hint(block)
    if hint:
        block["hint"] = hint
    return block


_FALLBACK = {
    "ocular.section": "Ocular artifacts",
    "ocular.evidence": "Evidence",
    "ocular.reference": "Blink reference",
    "ocular.none": "none (no EOG or frontal-pole channel)",
    "ocular.handled_by": "Removed by",
    "ocular.not_handled": "No step in this pipeline removes ocular artifacts",
    "ocular.not_counted": "{steps} do not remove ocular artifacts — they clip, rescale or drop trials; the blink stays in every kept trial",
    "ocular.recommended": "Recommended",
    "ocular.no_method": "Ocular artifacts cannot be removed (no EOG or frontal-pole channel); trial rejection is the only fallback",
    "ocular.exempt": "Not required for this run ({reason})",
    "ocular.order_warning": "Ocular removal runs after car — the average reference has already spread blinks to every channel; move it before car",
}


def _tr(key: str, **kw: Any) -> str:
    if _t is not None:
        try:
            s = _t(key, **kw)
            if s and s != key:
                return s
        except Exception:  # noqa: BLE001
            pass
    return _FALLBACK[key].format(**kw)


def render_reasoning_section(block: Optional[dict[str, Any]]) -> str:
    """``## Ocular artifacts`` section for reasoning.md ("" without a block)."""
    if not isinstance(block, dict):
        return ""
    lines = [f"## {_tr('ocular.section')}", "",
             f"- {_tr('ocular.evidence')}: {block.get('evidence')}",
             f"- {_tr('ocular.reference')}: {block.get('reference') or _tr('ocular.none')}"]
    if block.get("handled_by"):
        lines.append(f"- {_tr('ocular.handled_by')}: "
                     + ", ".join(f"`{s}`" for s in block["handled_by"]))
    else:
        lines.append(f"- ⚠ {_tr('ocular.not_handled')}")
    if block.get("not_counted"):
        lines.append("- " + _tr("ocular.not_counted",
                                steps=", ".join(f"`{s}`" for s in block["not_counted"])))
    if not block.get("handled_by") and block.get("recommended"):
        lines.append(f"- {_tr('ocular.recommended')}: `{block['recommended']}` — "
                     f"{block.get('recommended_reason')}")
    elif block.get("exempt") == "no_reference":
        lines.append(f"- ⚠ {_tr('ocular.no_method')}")
    if block.get("exempt") in ("invasive", "online_inference", "eog_is_signal"):
        lines.append(f"- {_tr('ocular.exempt', reason=block['exempt'])}")
    if block.get("order_warning"):
        lines.append(f"- ⚠ {_tr('ocular.order_warning')}")
    return "\n".join(lines) + "\n\n"
