"""Helpers that turn QC / audit / user-feedback signals into NegativeExample."""
from __future__ import annotations

import datetime as _dt
import hashlib
from typing import Any, Dict

from . import NegativeExample
from ._step_key import canonical_step


def _make_id(*parts: str) -> str:
    raw = "|".join(p for p in parts if p)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8]


def _normalize_step(failed_step: str, failed_params: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
    """Return (operator_name, params_dict).

    Splits a possibly-parametrized failed_step (e.g. ``"ica:eog"``) into the
    operator name and a params dict augmented with the parsed tail. Caller's
    explicit ``failed_params`` win over the parsed tail. Unknown operators
    leave ``failed_step`` empty but preserve the raw string under
    ``failed_params["legacy_step_string"]`` so revoke / audit still has the
    original text.
    """
    op, params_tail = canonical_step(failed_step)
    out: Dict[str, Any] = {}
    if params_tail and op:
        out["param_string"] = params_tail
    if isinstance(failed_params, dict):
        out.update(failed_params)
    if not op and isinstance(failed_step, str) and failed_step.strip():
        out.setdefault("legacy_step_string", failed_step.strip())
    return op, out


def build_negative_from_qc_fail(
    *,
    modality: str,
    paradigm: str,
    cohort_tag: str,
    analysis_goal: str,
    failed_step: str,
    failed_params: Dict[str, Any],
    failure_evidence: str,
    fingerprint_hash: str,
    lab_id: str = "local",
) -> NegativeExample:
    op, params = _normalize_step(failed_step, failed_params)
    return NegativeExample(
        id=_make_id("qc_fail", modality, paradigm, op or failed_step, fingerprint_hash),
        modality=modality, paradigm=paradigm, cohort_tag=cohort_tag,
        analysis_goal=analysis_goal,
        failed_step=op, failed_params=params,
        failure_mode="qc_fail",
        failure_evidence=failure_evidence,
        fingerprint_hash=fingerprint_hash,
        recorded_at=_dt.datetime.utcnow().isoformat() + "Z",
        lab_id=lab_id, severity="soft",
    )


def build_negative_from_user(
    *,
    modality: str,
    paradigm: str,
    cohort_tag: str,
    analysis_goal: str,
    failed_step: str,
    failed_params: Dict[str, Any],
    user_reason: str,
    fingerprint_hash: str,
    lab_id: str = "local",
) -> NegativeExample:
    op, params = _normalize_step(failed_step, failed_params)
    return NegativeExample(
        id=_make_id("user", modality, paradigm, op or failed_step, fingerprint_hash, user_reason[:32]),
        modality=modality, paradigm=paradigm, cohort_tag=cohort_tag,
        analysis_goal=analysis_goal,
        failed_step=op, failed_params=params,
        failure_mode="user_rejected",
        failure_evidence=user_reason,
        fingerprint_hash=fingerprint_hash,
        recorded_at=_dt.datetime.utcnow().isoformat() + "Z",
        lab_id=lab_id, severity="soft",
    )


def build_negative_from_auto_flag(
    *,
    modality: str,
    paradigm: str,
    cohort_tag: str,
    analysis_goal: str,
    failed_step: str,
    failed_params: Dict[str, Any],
    auto_flag_reason: str,
    fingerprint_hash: str,
    lab_id: str = "local",
) -> NegativeExample:
    op, params = _normalize_step(failed_step, failed_params)
    return NegativeExample(
        id=_make_id("auto", modality, paradigm, op or failed_step, auto_flag_reason[:32]),
        modality=modality, paradigm=paradigm, cohort_tag=cohort_tag,
        analysis_goal=analysis_goal,
        failed_step=op, failed_params=params,
        failure_mode="auto_flagged",
        failure_evidence=auto_flag_reason,
        fingerprint_hash=fingerprint_hash,
        recorded_at=_dt.datetime.utcnow().isoformat() + "Z",
        lab_id=lab_id, severity="soft",
    )
