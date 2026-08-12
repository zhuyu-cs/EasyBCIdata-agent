"""CLI handlers for the `easybci reference` command group.

Subcommands:
  import <dir>   -- ingest a gold-standard project into an enhanced
                    proven-pipeline skill (skeleton + adaptation_slots +
                    qc_baselines).
"""
from __future__ import annotations

import json as _json


def cmd_reference_import(args) -> int:
    from easybci_cli.cli_output import print_info, print_error, print_success
    from easybci_lib.tools.neural_processing.reference.ingest import ingest_reference

    try:
        res = ingest_reference(
            args.reference_dir,
            analysis_goal=getattr(args, "analysis_goal", None) or "clinical_screening",
            dry_run=bool(getattr(args, "dry_run", False)),
        )
    except Exception as exc:  # noqa: BLE001
        if getattr(args, "json", False):
            print(_json.dumps({"success": False, "error": str(exc)}))
        else:
            print_error(f"reference import failed: {exc}")
        return 1

    if getattr(args, "json", False):
        print(_json.dumps(res, default=str))
        return 0 if res.get("success") else 1

    if not res.get("success"):
        print_error(f"reference import failed: {res.get('error')}")
        return 1

    prof = res.get("profile", {})
    print_success(f"Ingested → skill '{res.get('skill_name')}'")
    print_info(f"  modality: {prof.get('modality')}  channels: {prof.get('n_channels')}")
    print_info(f"  skeleton: {prof.get('step_string')}")
    if res.get("skill_path"):
        print_info(f"  written: {res['skill_path']}")
    if res.get("unmapped"):
        print_info(f"  unmapped steps (not silently dropped): {res['unmapped']}")
    return 0
