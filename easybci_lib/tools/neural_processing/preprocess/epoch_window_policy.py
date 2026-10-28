"""Epoch-window decision at planning time.

The trial window is a methodological choice: for event-locked decoding it must
cover the phases whose signal differs between classes, not phases that are
identical for every class (fixation, cue, rest). This module turns the agent's
stated dataset understanding (trial timeline + sources) into a window, the
basis it rests on, and reminders. It never blocks — the decision and its
warnings are shown to the user at confirm.

Applicability depends on the data, not the modality: any recording with trial
events (scalp EEG, sEEG, ECoG, MEG, spikes, ...) gets a window decision; a
continuous recording without trial events (sleep, resting state, connectivity)
gets ``source="not_applicable"`` and no warnings.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from easybci_lib.tools.neural_processing.quality.trial_reject import (
    _TR_NON_TRIAL_RX,
    normalize_epoch_window,
)


def _load_t() -> Optional[Callable[..., str]]:
    try:
        from easybci_agent.i18n import t
    except Exception:  # noqa: BLE001
        return None
    return t


_t = _load_t()

WINDOW_SOURCES = ("dataset_doc", "metadata", "paradigm_default", "event_duration", "user", "proven_skill")
NOT_APPLICABLE = "not_applicable"
_DATA_BASIS = {"dataset_doc", "metadata", "user", "proven_skill"}
LABEL_INDEPENDENT_WARN_S = 1.0
_KIND_TO_SOURCE = {"doc": "dataset_doc", "metadata": "metadata", "user": "user",
                   "data_inspection": "event_duration"}


def trial_events_present(report: Any) -> bool | None:
    """Whether an inspection report shows trial events.

    ``None`` when the report does not say (missing / legacy); ``False`` when
    it has no events or only artifact / boundary markers.
    """
    if not isinstance(report, dict):
        return None
    if report.get("events_summary"):
        return True
    fp = report.get("fingerprint")
    if not isinstance(fp, dict) or fp.get("n_events") is None:
        return None
    try:
        n_events = int(fp.get("n_events") or 0)
    except (TypeError, ValueError):
        return None
    if n_events <= 0:
        return False
    types = [str(t).strip() for t in fp.get("event_types") or []]
    if types and all(_TR_NON_TRIAL_RX.match(t) for t in types):
        return False
    return True


def clean_timeline(du: Any) -> list[dict[str, Any]]:
    """Valid timeline phases (numeric start < end), sorted by start."""
    if not isinstance(du, dict):
        return []
    out = []
    for ph in du.get("trial_timeline") or []:
        if not isinstance(ph, dict):
            continue
        try:
            s, e = float(ph.get("start_s")), float(ph.get("end_s"))
        except (TypeError, ValueError):
            continue
        if e > s:
            out.append({"phase": str(ph.get("phase") or "?"), "start_s": s, "end_s": e,
                        "label_dependent": bool(ph.get("label_dependent"))})
    return sorted(out, key=lambda p: p["start_s"])


def _timeline_source(du: dict[str, Any]) -> str:
    kinds = [str(s.get("kind") or "") for s in du.get("sources") or [] if isinstance(s, dict)]
    for k in ("doc", "metadata", "user"):
        if k in kinds:
            return _KIND_TO_SOURCE[k]
    return "user"


def _label_dependent_span(tl: list[dict[str, Any]]) -> list[float] | None:
    """The single contiguous run of label-dependent phases, if there is one."""
    idx = [i for i, p in enumerate(tl) if p["label_dependent"]]
    if not idx or idx != list(range(idx[0], idx[-1] + 1)):
        return None
    run = tl[idx[0]:idx[-1] + 1]
    if any(abs(a["end_s"] - b["start_s"]) > 1e-6 for a, b in zip(run, run[1:])):
        return None
    return [run[0]["start_s"], run[-1]["end_s"]]


def _label_independent_overlap(window: list[float], tl: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for p in tl:
        if p["label_dependent"]:
            continue
        ov = min(window[1], p["end_s"]) - max(window[0], p["start_s"])
        if ov > 0:
            out.append({"phase": p["phase"], "seconds": round(ov, 3)})
    return out


def decide_epoch_window(explicit: Any, dataset_understanding: Any, basis: Any, *,
                        has_trial_events: bool | None = None) -> dict[str, Any]:
    """Return ``{epoch_window, source, evidence, timeline, warnings}``.

    Precedence: explicit window → single contiguous label-dependent span of the
    stated timeline → ``None`` (runtime: event duration → fallback). With
    ``has_trial_events=False`` and neither of the first two, the window is
    ``not_applicable`` (continuous recording) and nothing is warned.
    """
    du = dataset_understanding if isinstance(dataset_understanding, dict) else {}
    basis = basis if isinstance(basis, dict) else {}
    tl = clean_timeline(du)
    window = normalize_epoch_window(explicit)
    source = str(basis.get("source") or "").strip()
    evidence = str(basis.get("evidence") or "").strip()

    if window is None:
        span = _label_dependent_span(tl)
        if span is not None:
            window = span
            source = _timeline_source(du)
            phases = [p["phase"] for p in tl if p["label_dependent"]]
            refs = [str(s.get("path")) for s in du.get("sources") or [] if isinstance(s, dict) and s.get("path")]
            evidence = evidence or "label-dependent phase(s) {} {}–{} s from {}".format(
                "/".join(phases), span[0], span[1], ", ".join(refs) or "dataset_understanding")
        elif has_trial_events is False:
            return {"epoch_window": None, "source": NOT_APPLICABLE,
                    "evidence": evidence or "no trial events in the recording (continuous data)",
                    "timeline": tl, "warnings": []}
        else:
            source = "event_duration"
    if source not in WINDOW_SOURCES:
        source = "user" if window is not None else "event_duration"

    warnings: list[dict[str, Any]] = []
    if source not in _DATA_BASIS:
        warnings.append({"code": "no_data_basis",
                         "message": "The epoch window has no basis in this dataset's documentation "
                                    "or metadata — confirm it with the user."})
    if window is not None and tl:
        ov = _label_independent_overlap(window, tl)
        if sum(o["seconds"] for o in ov) >= LABEL_INDEPENDENT_WARN_S:
            warnings.append({"code": "includes_label_independent_phase", "phases": ov,
                             "message": "The window includes phases that are identical across "
                                        "classes; they add no class information and dilute features."})
    return {"epoch_window": window, "source": source, "evidence": evidence,
            "timeline": tl, "warnings": warnings}


_FALLBACK = {
    "epoch_window.section_understanding": "Dataset understanding",
    "epoch_window.section_window": "Epoch window",
    "epoch_window.no_understanding": "No dataset understanding was provided.",
    "epoch_window.unread_docs": "Dataset documentation not read before planning: {paths}",
    "epoch_window.runtime_resolved": "not set — resolved at run time from the event durations",
    "epoch_window.not_applicable": "not applicable — continuous recording without trial events; no epoching",
    "epoch_window.warn_no_data_basis": "Epoch window has no basis in this dataset's documentation or metadata — please confirm.",
    "epoch_window.warn_label_independent": "Window includes phases identical across classes: {phases}",
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


def render_reasoning_sections(dataset_understanding: Any, decision: dict[str, Any],
                              *, unread_docs: list[str]) -> str:
    """Markdown for the two sections placed before the per-step reasoning."""
    du = dataset_understanding if isinstance(dataset_understanding, dict) else {}
    lines = [f"## {_tr('epoch_window.section_understanding')}", ""]
    srcs = [s for s in du.get("sources") or [] if isinstance(s, dict)]
    if not srcs and not decision.get("timeline") and not du.get("notes"):
        lines.append(_tr("epoch_window.no_understanding"))
    for s in srcs:
        lines.append(f"- `{s.get('path', '?')}` ({s.get('kind', '?')})")
    if decision.get("timeline"):
        lines += ["", "| phase | start_s | end_s | label-dependent |", "|---|---|---|---|"]
        for p in decision["timeline"]:
            lines.append(f"| {p['phase']} | {p['start_s']} | {p['end_s']} | "
                         f"{'yes' if p['label_dependent'] else 'no'} |")
        lines.append("")
    for n in du.get("notes") or []:
        lines.append(f"- {n}")
    if unread_docs:
        lines += ["", "> ⚠ " + _tr("epoch_window.unread_docs", paths=", ".join(unread_docs))]

    lines += ["", f"## {_tr('epoch_window.section_window')}", ""]
    win = decision.get("epoch_window")
    if win:
        lines.append(f"- Window: `{win}` s")
    elif decision.get("source") == NOT_APPLICABLE:
        lines.append(f"- Window: {_tr('epoch_window.not_applicable')}")
    else:
        lines.append(f"- Window: {_tr('epoch_window.runtime_resolved')}")
    lines.append(f"- Source: `{decision.get('source')}`")
    if decision.get("evidence"):
        lines.append(f"- Evidence: {decision['evidence']}")
    for w in decision.get("warnings") or []:
        if w["code"] == "no_data_basis":
            msg = _tr("epoch_window.warn_no_data_basis")
        elif w["code"] == "includes_label_independent_phase":
            ph = ", ".join(f"{p['phase']} {p['seconds']} s" for p in w.get("phases") or [])
            msg = _tr("epoch_window.warn_label_independent", phases=ph)
        elif w.get("message"):
            msg = str(w["message"])
        else:
            continue
        lines += ["", "> ⚠ " + msg]
    return "\n".join(lines) + "\n\n"
