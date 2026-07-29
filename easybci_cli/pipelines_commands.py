"""CLI handlers for the easybci pipelines command group.

Subcommands:
  list           — table of proven pipelines + stats
  show <name>    — full details of one pipeline
  stats          — library health summary
  flag <name>    — manually flag a pipeline (downweights to ~0)
  unflag <name>  — remove manual flag
"""
from __future__ import annotations

import json
from collections import Counter
from typing import Any, Counter as CounterType, Dict, List

from easybci_cli import cli_output
from easybci_lib.constants import get_easybci_home
from easybci_lib.tools.neural_processing.proven_match import scan_proven_pipelines
from easybci_lib.tools.neural_processing.proven_tracker import ProvenPipelineTracker


def _load_tracker() -> ProvenPipelineTracker:
    return ProvenPipelineTracker(store_path=str(get_easybci_home() / "proven_tracker.json"))


def _scan_library():
    return scan_proven_pipelines()


# ---------------------------------------------------------------- list / show


def cmd_pipelines_list(args) -> int:
    entries = _scan_library()
    tracker = _load_tracker()
    json_mode = bool(getattr(args, "json", False))
    rows = []
    for e in entries:
        stats = tracker.get_stats(e.name)
        rows.append({
            "name": e.name,
            "modality": e.modality,
            "paradigm": e.paradigm,
            "lab_id": e.lab_id or "unknown",
            "cohort_tag": e.cohort_tag,
            "pass_rate": stats.pass_rate if stats else 0.0,
            "manual_flag": stats.manual_flag if stats else False,
        })
    if json_mode:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return 0
    if not rows:
        cli_output.print_info("(no proven pipelines in library)")
        return 0
    cli_output.print_info(f"{'NAME':<30} {'MODALITY':<8} {'PARADIGM':<20} {'LAB':<15} {'PASS%':>6} FLAGS")
    for r in rows:
        flag = "[flag]" if r["manual_flag"] else ""
        cli_output.print_info(
            f"{r['name']:<30} {r['modality']:<8} {r['paradigm']:<20} "
            f"{r['lab_id']:<15} {r['pass_rate']*100:>5.0f} {flag}"
        )
    return 0


def cmd_pipelines_show(args) -> int:
    name = args.name
    for e in _scan_library():
        if e.name == name:
            d = e.to_dict()
            tracker = _load_tracker()
            stats = tracker.get_stats(name)
            d["stats"] = stats.to_dict() if stats else None
            if getattr(args, "json", False):
                print(json.dumps(d, ensure_ascii=False, indent=2))
            else:
                cli_output.print_info(f"name:       {d['name']}")
                cli_output.print_info(f"modality:   {d['modality']}")
                cli_output.print_info(f"paradigm:   {d['paradigm']}")
                cli_output.print_info(f"lab_id:     {d.get('lab_id', '')}")
                cli_output.print_info(f"cohort_tag: {d.get('cohort_tag', '')}")
                cli_output.print_info(f"steps:      {', '.join(d.get('steps', []))}")
                if d.get("stats"):
                    cli_output.print_info(f"stats:      {json.dumps(d['stats'], ensure_ascii=False)}")
            return 0
    cli_output.print_warning(f"no pipeline named {name}")
    return 1


# ---------------------------------------------------------------- stats


def cmd_pipelines_stats(args) -> int:
    entries = _scan_library()
    tracker = _load_tracker()
    if not entries:
        cli_output.print_info("(library empty)")
        return 0

    total = len(entries)
    by_lab: Counter[str] = Counter((e.lab_id or "unknown") for e in entries)
    by_cohort: Counter[str] = Counter((e.cohort_tag or "(none)") for e in entries)

    auto_downweighted = 0
    manually_flagged = 0
    for e in entries:
        stats = tracker.get_stats(e.name)
        if stats is None:
            continue
        if stats.pass_rate < 0.4 and stats.total_uses > 0:
            auto_downweighted += 1
        if stats.manual_flag:
            manually_flagged += 1

    if getattr(args, "json", False):
        out = {
            "total": total,
            "auto_downweighted": auto_downweighted,
            "manually_flagged": manually_flagged,
            "by_lab": dict(by_lab.most_common()),
            "by_cohort": dict(by_cohort.most_common()),
        }
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0

    cli_output.print_info("Library summary")
    cli_output.print_info(f"  Total pipelines: {total}")
    cli_output.print_info(f"  Auto-downweighted (pass-rate < 40%): {auto_downweighted}")
    cli_output.print_info(f"  Manually flagged: {manually_flagged}")
    cli_output.print_info("  Origin distribution:")
    for lab, n in by_lab.most_common():
        pct = 100.0 * n / total
        warning = "   ← warning: dominance" if pct > 35 else ""
        cli_output.print_info(f"    lab={lab}: {n} ({pct:.0f}%){warning}")
    cli_output.print_info("  Cohort coverage:")
    for cohort, n in by_cohort.most_common():
        cli_output.print_info(f"    {cohort}: {n}")
    return 0


# ---------------------------------------------------------------- flag / unflag


def cmd_pipelines_flag(args) -> int:
    tracker = _load_tracker()
    tracker.flag_pipeline(args.name, reason=getattr(args, "reason", "") or "")
    cli_output.print_info(f"Flagged {args.name}. Will be near-zero weighted in matches.")
    return 0


def cmd_pipelines_unflag(args) -> int:
    tracker = _load_tracker()
    ok = tracker.unflag_pipeline(args.name)
    if ok:
        cli_output.print_info(f"Unflagged {args.name}.")
        return 0
    cli_output.print_warning(f"{args.name} was not flagged.")
    return 1


# ---------------------------------------------------------------- resume


def cmd_pipelines_resume(args) -> int:
    """`easybci pipelines resume <work_dir>` — check / continue an
    incomplete preprocessing run without going through the LLM.
    """
    import json as _json
    from easybci_lib.tools.neural_tools import _handle_resume_preprocessing

    payload = {
        "work_dir": args.work_dir,
        "check_only": bool(getattr(args, "check_only", False)),
        "force": bool(getattr(args, "force", False)),
    }
    timeout = getattr(args, "timeout", None)
    if timeout is not None:
        payload["timeout"] = int(timeout)

    try:
        result_str = _handle_resume_preprocessing(payload)
    except Exception as exc:
        cli_output.print_error(f"resume_preprocessing raised: {exc!r}")
        return 2

    try:
        result = _json.loads(result_str)
    except _json.JSONDecodeError as exc:
        cli_output.print_error(f"handler returned non-JSON: {exc}")
        return 2

    if getattr(args, "json", False):
        print(_json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("success") else 1

    if not result.get("success"):
        cli_output.print_error(result.get("error") or "unknown error")
        return 1

    wd = result.get("work_dir")
    goal = result.get("analysis_goal")
    before = result.get("before") or {}
    cli_output.print_info(f"work_dir: {wd}")
    cli_output.print_info(f"analysis_goal: {goal}")
    cli_output.print_info(
        f"progress: {before.get('done', 0)}/{before.get('total', 0)} done, "
        f"{before.get('pending', 0)} pending"
    )

    if not result.get("resumed"):
        reason = result.get("reason") or "unknown"
        if reason == "already_complete":
            cli_output.print_success("nothing to do — all inputs already processed.")
        elif reason == "check_only":
            pending_ids = before.get("pending_file_ids") or []
            if pending_ids:
                cli_output.print_info("pending file_ids:")
                for fid in pending_ids:
                    entry = (before.get("missing_by_entry") or {}).get(fid) or {}
                    missing = ", ".join(entry.get("missing") or []) or "?"
                    cli_output.print_info(f"  - {fid}: missing [{missing}]")
            else:
                cli_output.print_success("nothing to do — all inputs already processed.")
        else:
            cli_output.print_info(f"reason: {reason}")
        return 0

    # resumed=True path
    stages = result.get("stages_run") or []
    stage_results = result.get("stage_results") or {}
    for stage in stages:
        sr = stage_results.get(stage) or {}
        if sr.get("ok"):
            cli_output.print_success(f"stage {stage}: OK")
        else:
            cli_output.print_warning(
                f"stage {stage}: FAILED (retcode={sr.get('retcode')}); "
                f"tail:\n{sr.get('stderr_tail') or sr.get('stdout_tail') or ''}"
            )

    after = result.get("after") or {}
    cli_output.print_info(
        f"after: {after.get('done', 0)}/{after.get('total', 0)} done, "
        f"{after.get('pending', 0)} pending"
    )
    return 0 if after.get("pending", 1) == 0 else 1


def cmd_pipelines_migrate_layout(args) -> int:
    """Scan ``args.root`` for ``*_preprocess_work_dir`` directories and run
    verify_and_repair on each. Prints a summary. Exit 0 unless residuals
    remain after --apply.
    """
    import json as _json
    import sys
    from pathlib import Path

    from easybci_lib.tools.neural_processing.export.layout_repair import (
        detect_violations,
        verify_and_repair,
    )
    from easybci_agent.i18n import t

    root_path = Path(args.root).expanduser().resolve()
    dry_run = bool(getattr(args, "dry_run", False))
    json_mode = bool(getattr(args, "json_mode", False))

    if not root_path.is_dir():
        msg = t("pipelines.migrate.root_not_dir", path=str(root_path))
        if json_mode:
            sys.stdout.write(_json.dumps({"success": False, "error": msg}))
            sys.stdout.write("\n")
        else:
            cli_output.print_error(msg)
        return 2

    candidates = sorted(
        p for p in root_path.rglob("*_preprocess_work_dir") if p.is_dir()
    )
    scanned = 0
    drifted: list[dict] = []
    remaining = 0
    for wd in candidates:
        scanned += 1
        vs = detect_violations(wd)
        if not vs:
            continue
        entry: dict = {"work_dir": str(wd), "initial_violations": len(vs)}
        if not dry_run:
            report = verify_and_repair(
                wd, dry_run=False, allow_subprocess=True, write_report=True,
            )
            entry["remaining_violations"] = report.remaining_violations
            entry["rounds"] = report.rounds
            if report.remaining_violations:
                remaining += 1
        else:
            entry["remaining_violations"] = len(vs)
            remaining += 1
        drifted.append(entry)

    payload = {
        "root": str(root_path),
        "scanned": scanned,
        "drift_count": len(drifted),
        "drifted": drifted,
        "remaining_drift_count": remaining,
        "dry_run": dry_run,
    }

    if json_mode:
        sys.stdout.write(_json.dumps(payload, indent=2))
        sys.stdout.write("\n")
    else:
        cli_output.print_info(t(
            "pipelines.migrate.summary",
            scanned=scanned, drift_count=len(drifted), remaining=remaining,
        ))
        for e in drifted:
            cli_output.print_info(
                f"  {e['work_dir']}: "
                f"{e.get('initial_violations', 0)} → "
                f"{e.get('remaining_violations', 0)}"
            )

    # Dry-run: always exit 0 (reporting mode). Apply: non-zero when residual.
    if dry_run:
        return 0
    return 0 if remaining == 0 else 1
