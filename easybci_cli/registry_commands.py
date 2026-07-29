"""easybci registry check — sweep parameter_uncertainty/* for retracted citations."""
from __future__ import annotations

import json
from typing import Any

from easybci_cli import cli_output
from easybci_lib.constants import get_easybci_home
from easybci_lib.tools.neural_processing.research.citation_audit import (
    CitationAuditLog,
)
from easybci_lib.tools.neural_processing.research.citation_checker import (
    CitationChecker,
)


def cmd_registry_check(args: Any) -> int:
    checker = CitationChecker()
    results = checker.check_registry()

    audit = CitationAuditLog(path=get_easybci_home() / "citation_audit.jsonl")
    for r in results:
        audit.append(r)

    if getattr(args, "json", False):
        print(json.dumps([
            {
                "citation_id": r.citation_id,
                "status": r.status,
                "checked_at": r.checked_at,
                "detail": r.detail,
                "source": r.source,
            }
            for r in results
        ], ensure_ascii=False, indent=2))
        return 0

    if not results:
        cli_output.print_info("(no checkable citations in registry — all origins lack doi/arxiv_id/url)")
        return 0

    flagged = [r for r in results if r.status in ("retracted", "withdrawn", "revised")]
    unreachable = [r for r in results if r.status == "unreachable"]
    ok = [r for r in results if r.status == "ok"]

    cli_output.print_info(f"Checked {len(results)} citations:")
    cli_output.print_info(f"  ok:          {len(ok)}")
    cli_output.print_info(f"  unreachable: {len(unreachable)}")
    cli_output.print_info(f"  flagged:     {len(flagged)}")
    if flagged:
        cli_output.print_warning("Flagged citations:")
        for r in flagged:
            cli_output.print_warning(f"  {r.status} via {r.source}: {r.detail}")
    return 0
