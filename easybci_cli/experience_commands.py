"""easybci experience subcommands.

list-negatives        — print all recorded negative examples
flag-negative         — manually inject a NegativeExample
revoke-negative       — remove a NegativeExample by id
normalize-keys        — backfill canonical failed_step on legacy entries
"""
from __future__ import annotations

import json
from typing import Any

from easybci_cli import cli_output
from easybci_lib.constants import get_easybci_home
from easybci_lib.tools.neural_processing.experience import ExperienceStore, NegativeExample
from easybci_lib.tools.neural_processing.experience._step_key import (
    canonical_step,
    known_operators,
)
from easybci_lib.tools.neural_processing.experience.negative_capturer import (
    build_negative_from_user,
)


def _store() -> ExperienceStore:
    return ExperienceStore(store_dir=str(get_easybci_home() / "experience"))


def cmd_experience_list_negatives(args: Any) -> int:
    store = _store()
    negs = store.list_negatives()
    if getattr(args, "json", False):
        print(json.dumps([n.to_dict() for n in negs], ensure_ascii=False, indent=2))
        return 0
    if not negs:
        cli_output.print_info("(no negative examples recorded)")
        return 0
    cli_output.print_info(f"{'ID':<10} {'MODALITY':<8} {'PARADIGM':<20} {'MODE':<18} {'SEVERITY':<6}")
    for n in negs:
        cli_output.print_info(
            f"{n.id:<10} {n.modality:<8} {n.paradigm:<20} {n.failure_mode:<18} {n.severity:<6}"
        )
    return 0


def cmd_experience_flag_negative(args: Any) -> int:
    try:
        params = json.loads(args.params or "{}")
    except json.JSONDecodeError:
        cli_output.print_warning(f"--params must be valid JSON; got {args.params!r}")
        return 1
    op, _ = canonical_step(args.failed_step)
    if not op:
        valid = ", ".join(sorted(known_operators()))
        cli_output.print_warning(
            f"--failed-step {args.failed_step!r} is not a recognised pipeline operator. "
            f"Valid operators: {valid}"
        )
        return 1
    neg = build_negative_from_user(
        modality=args.modality, paradigm=args.paradigm, cohort_tag=args.cohort or "",
        analysis_goal=args.analysis_goal, failed_step=args.failed_step,
        failed_params=params,
        user_reason=args.reason or "",
        fingerprint_hash="",
    )
    _store().record_negative(neg)
    cli_output.print_info(
        f"Recorded negative example {neg.id} ({neg.failure_mode}); "
        f"normalised failed_step → {neg.failed_step!r}."
    )
    return 0


def cmd_experience_revoke_negative(args: Any) -> int:
    ok = _store().revoke_negative(args.id, reason=getattr(args, "reason", "") or "")
    if not ok:
        cli_output.print_warning(f"no negative with id {args.id}")
        return 1
    cli_output.print_info(f"Revoked negative {args.id}.")
    return 0


def cmd_experience_normalize_keys(args: Any) -> int:
    """One-shot: rewrite every NegativeExample.failed_step through canonical_step.

    For each existing entry:
      * known operator (post-canonical) → set failed_step to op, lift params
        tail into failed_params.param_string (only if not already present).
      * unknown operator → preserve the raw string under
        failed_params.legacy_step_string; set failed_step="" so set-intersection
        in proven_match_negative_penalty skips it cleanly.

    Writes in-place; reports counts. Idempotent.
    """
    store = _store()
    negs = store.list_negatives()
    if not negs:
        cli_output.print_info("(no negative examples to normalise)")
        return 0
    updated: list[NegativeExample] = []
    n_changed = 0
    n_legacy = 0
    for n in negs:
        op, tail = canonical_step(n.failed_step)
        params = dict(n.failed_params) if isinstance(n.failed_params, dict) else {}
        new_failed_step = n.failed_step
        if op:
            if op != n.failed_step:
                new_failed_step = op
                n_changed += 1
            if tail and "param_string" not in params:
                params["param_string"] = tail
        else:
            # Unknown operator — preserve original under legacy_step_string.
            if n.failed_step and "legacy_step_string" not in params:
                params["legacy_step_string"] = n.failed_step
            if n.failed_step:
                new_failed_step = ""
                n_legacy += 1
        updated.append(NegativeExample(
            id=n.id, modality=n.modality, paradigm=n.paradigm,
            cohort_tag=n.cohort_tag, analysis_goal=n.analysis_goal,
            failed_step=new_failed_step, failed_params=params,
            failure_mode=n.failure_mode, failure_evidence=n.failure_evidence,
            fingerprint_hash=n.fingerprint_hash, recorded_at=n.recorded_at,
            lab_id=n.lab_id, severity=n.severity,
        ))
    # Rewrite jsonl atomically using the same helper as revoke.
    path = store._negatives_path()
    tmp = path.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for n in updated:
            f.write(json.dumps(n.to_dict(), ensure_ascii=False) + "\n")
    tmp.replace(path)
    cli_output.print_info(
        f"normalised {len(updated)} entries — {n_changed} parameter-tail stripped, "
        f"{n_legacy} flagged as legacy_unparseable."
    )
    return 0
