"""T9 consistency checker — enforces the four-layer skill architecture contract.

Walks ``easybci_lib/skills/bci/`` and validates the layer/group/path/frontmatter
invariants defined in ``_DECISIONS.md``. Used by 1-3 / 1-4 / 1-5 migration as a
gate, and consumed by 02 G / 03 K downstream as the canonical "layer +
frontmatter" check.

Usage::

    python -m easybci_lib.tools.neural_processing._check_consistency
        --root easybci_lib/skills/bci
        [--strict]                       # exit 1 on any FAIL (default: warn-only)
        [--jsonl <path>]                 # override default audit jsonl path

Rules (R1-R9, see _DECISIONS.md):

- R1 ``layer`` field present
- R2 ``layer`` value in {L0, L1, L2, L3}
- R3 path matches layer
- R4 ``group`` field present (L0 / L2 / L3 only)
- R5 ``group`` value in the layer's predefined enum
- R6 L2 paradigm has ``analysis_goal_allowed`` or ``analysis_goal_forbidden``
- R7 ``analysis_goal_*`` values in REGISTRY 9-enum
- R8 no contradictions (same goal both allowed and forbidden)
- R9 frontmatter ``name`` equals filename stem (with ``SKILL`` mapped to parent dir)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

from easybci_lib.tools.neural_processing.skill_layers import (
    ANALYSIS_GOALS,
    GROUPS_BY_LAYER,
    LAYERS,
    infer_group_from_path,
    infer_layer_from_path,
)


# ── frontmatter parser ──────────────────────────────────────────────────────

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def _parse_frontmatter(text: str) -> Tuple[Dict[str, Any], str]:
    """Return ``(frontmatter_dict, body)`` for a markdown file.

    Returns ``({}, text)`` when no frontmatter or YAML parse fails.
    """
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    raw = m.group(1)
    try:
        fm = yaml.safe_load(raw) or {}
        if not isinstance(fm, dict):
            return {}, text[m.end() :]
        return fm, text[m.end() :]
    except yaml.YAMLError:
        return {}, text[m.end() :]


# ── target file discovery ──────────────────────────────────────────────────


_TOP_LAYER_DIRS = {"neural-io", "pipeline", "operators", "paradigms"}


def _iter_skill_files(root: Path) -> Iterable[Path]:
    """Yield every SKILL.md and L2 paradigm `.md` under ``root``.

    Skips DESCRIPTION.md, CODE_STANDARD.md, and anything under .git/.archive.
    """
    excluded_dirs = {".git", ".github", ".hub", ".archive", "scripts"}
    for path in sorted(root.rglob("*.md")):
        if any(part in excluded_dirs for part in path.parts):
            continue
        name = path.name
        if name in ("DESCRIPTION.md", "CODE_STANDARD.md"):
            continue
        # Only inspect files that live under one of the four layer roots.
        rel = path.relative_to(root)
        if not rel.parts or rel.parts[0] not in _TOP_LAYER_DIRS:
            continue
        # SKILL.md anywhere is in scope; bare `.md` files only under
        # paradigms/ are treated as L2 domain-skill files (legacy flat shape).
        if name == "SKILL.md":
            yield path
            continue
        if rel.parts[0] == "paradigms":
            yield path


# ── finding container ─────────────────────────────────────────────────────


@dataclass
class Finding:
    file: str
    rule: str
    severity: str  # "fail" | "warn"
    message: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "file": self.file,
            "rule": self.rule,
            "severity": self.severity,
            "message": self.message,
        }


# ── individual rule checks ────────────────────────────────────────────────


def _check_one(rel: Path, fm: Dict[str, Any], path: Path) -> List[Finding]:
    findings: List[Finding] = []
    rel_str = str(rel)
    layer = fm.get("layer")
    group = fm.get("group")
    name = fm.get("name")
    md = fm.get("metadata") or {}
    if not isinstance(md, dict):
        md = {}
    allowed = md.get("analysis_goal_allowed") or []
    forbidden = md.get("analysis_goal_forbidden") or []

    # ── R1: layer field present ───────────────────────────────────────────
    if layer is None:
        findings.append(
            Finding(rel_str, "R1", "fail", "missing frontmatter `layer:` field")
        )

    # ── R2: layer in enum ─────────────────────────────────────────────────
    if layer is not None and layer not in LAYERS:
        findings.append(
            Finding(
                rel_str,
                "R2",
                "fail",
                f"`layer: {layer!r}` not in {LAYERS}",
            )
        )

    # ── R3: path matches layer ────────────────────────────────────────────
    expected_layer = infer_layer_from_path(rel)
    if layer is not None and expected_layer is not None and layer != expected_layer:
        findings.append(
            Finding(
                rel_str,
                "R3",
                "fail",
                f"path implies `layer: {expected_layer}` but frontmatter says {layer!r}",
            )
        )

    # ── R4: group field present for L0/L2/L3 (skipping pre-migration flat) ─
    # We only enforce R4 once the file lives under a *group dir* path (i.e.
    # parts[1] is a known group). This protects the migration window where
    # operators are still flat under `bci/operators/<op>/`.
    inferred_group = infer_group_from_path(rel)
    if layer in ("L0", "L2", "L3"):
        # Index files (SKILL.md directly at the layer root) don't need a
        # strict group; they're navigation pages.
        is_index = path.name == "SKILL.md" and len(rel.parts) == 2
        path_already_grouped = inferred_group is not None
        if not is_index and path_already_grouped and not group:
            findings.append(
                Finding(rel_str, "R4", "fail", f"L{layer[1]} skill missing `group:` field")
            )

    # ── R5: group value in layer's enum ───────────────────────────────────
    if layer in ("L0", "L2", "L3") and group:
        enum = GROUPS_BY_LAYER[layer]
        if group not in enum:
            findings.append(
                Finding(
                    rel_str,
                    "R5",
                    "fail",
                    f"group {group!r} not in {layer} enum {enum}",
                )
            )
        elif inferred_group is not None and inferred_group != group:
            # Frontmatter group disagrees with grouped directory name.
            findings.append(
                Finding(
                    rel_str,
                    "R5",
                    "fail",
                    f"frontmatter `group: {group}` disagrees with directory `{inferred_group}`",
                )
            )

    # ── R6: L2 paradigm needs analysis_goal_* (skip index) ────────────────
    if layer == "L2" and path.name != "SKILL.md":
        if not allowed and not forbidden:
            findings.append(
                Finding(
                    rel_str,
                    "R6",
                    "fail",
                    "L2 paradigm has neither `analysis_goal_allowed` nor `analysis_goal_forbidden`",
                )
            )

    # ── R7: analysis_goal_* values in REGISTRY ────────────────────────────
    for goal in list(allowed) + list(forbidden):
        if goal not in ANALYSIS_GOALS:
            findings.append(
                Finding(
                    rel_str,
                    "R7",
                    "fail",
                    f"analysis_goal {goal!r} not in REGISTRY {ANALYSIS_GOALS}",
                )
            )

    # ── R8: no contradictions ─────────────────────────────────────────────
    overlap = set(allowed) & set(forbidden)
    if overlap:
        findings.append(
            Finding(
                rel_str,
                "R8",
                "fail",
                f"goals both allowed and forbidden: {sorted(overlap)}",
            )
        )

    # ── R9: frontmatter name matches filename ────────────────────────────
    if name:
        if path.name == "SKILL.md":
            expected = path.parent.name
        else:
            expected = path.stem
        # Index files exempt — they use a `<dir>-index` slug.
        is_index = path.name == "SKILL.md" and len(rel.parts) == 2
        if not is_index and name != expected:
            findings.append(
                Finding(
                    rel_str,
                    "R9",
                    "fail",
                    f"frontmatter `name: {name!r}` does not match filename `{expected}`",
                )
            )

    return findings


# ── driver ────────────────────────────────────────────────────────────────


def check(root: Path) -> List[Finding]:
    findings: List[Finding] = []
    for path in _iter_skill_files(root):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as e:
            findings.append(
                Finding(str(path.relative_to(root)), "IO", "fail", f"cannot read: {e}")
            )
            continue
        fm, _ = _parse_frontmatter(text)
        rel = path.relative_to(root)
        if not fm:
            findings.append(
                Finding(str(rel), "R1", "fail", "no YAML frontmatter detected")
            )
            continue
        findings.extend(_check_one(rel, fm, path))
    return findings


def _format_markdown_table(findings: List[Finding]) -> str:
    if not findings:
        return "✅ All consistency rules PASSED.\n"
    lines = ["| File | Rule | Severity | Message |", "|---|---|---|---|"]
    pipe_escape = "\\|"
    for f in findings:
        msg = f.message.replace("|", pipe_escape)
        lines.append(f"| `{f.file}` | {f.rule} | {f.severity} | {msg} |")
    return "\n".join(lines) + "\n"


def _write_jsonl(findings: List[Finding], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for finding in findings:
            f.write(json.dumps(finding.to_dict(), ensure_ascii=False) + "\n")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="_check_consistency",
        description="Validate skill four-layer architecture invariants (T9).",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("easybci_lib/skills/bci"),
        help="Root of the bci skill tree (default: easybci_lib/skills/bci)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 on any FAIL and print a markdown table to stderr",
    )
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=None,
        help="Audit jsonl output path (default: _audits/skill_consistency_<date>.jsonl)",
    )
    args = parser.parse_args(argv)

    root = args.root.resolve()
    if not root.exists():
        print(f"root not found: {root}", file=sys.stderr)
        return 2

    findings = check(root)
    fails = [f for f in findings if f.severity == "fail"]

    if args.jsonl is None:
        # Default audit path mirrors the post-phase-4 follow-up location.
        # improved_docs lives at repo root, which is the parent of the
        # EasyBCIdata-agent project directory the CLI is normally run from.
        date = datetime.utcnow().strftime("%Y%m%d")
        candidates = [
            Path("improved_docs/plans/post-phase-4-followups/_audits"),
            Path("../improved_docs/plans/post-phase-4-followups/_audits"),
        ]
        audit_dir = next((c for c in candidates if c.parent.exists()), candidates[0])
        args.jsonl = audit_dir / f"skill_consistency_{date}.jsonl"
    _write_jsonl(findings, args.jsonl)

    if args.strict:
        if fails:
            print(_format_markdown_table(fails), file=sys.stderr)
            print(
                f"\n{len(fails)} FAIL(s). See {args.jsonl} for full jsonl.",
                file=sys.stderr,
            )
            return 1
        print("✅ All consistency rules PASSED.")
        return 0

    # warn-only: print summary to stdout
    print(f"Scanned {sum(1 for _ in _iter_skill_files(root))} skill files under {root}.")
    print(f"Findings: {len(findings)} ({len(fails)} fail / {len(findings) - len(fails)} warn).")
    print(f"jsonl: {args.jsonl}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
