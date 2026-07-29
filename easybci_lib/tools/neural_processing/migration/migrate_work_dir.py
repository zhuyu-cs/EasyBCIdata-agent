"""Migration tool: bring legacy/violating *_preprocess_work_dir directories
into compliance with the mini-repo finalization contract.

Idempotent, dry-run-first, backup-first. Designed to be safe to run
repeatedly on the same directory.

Compliant layout (target):

    {subject}_preprocess_work_dir/
    ├── README.md
    ├── pipeline_record.json
    ├── plan/
    ├── code/
    │   └── pipeline.py            ← single canonical script (NO middle_process/ here)
    ├── preprocessed_output/
    │   ├── *.pkl, *.meta.json     ← top-level binary outputs moved here
    │   ├── figures/
    │   └── QC_out/
    └── middle_process/            ← root-level only; archived prior scripts live in middle_process/code/

Migration moves:

* Top-level ``pipeline_v{N}.py`` (highest N) → ``code/pipeline.py``
* Other ``pipeline_v{N}.py`` → ``middle_process/code/pipeline_v{N}.py``
* Top-level ``pipeline_final.py`` / ``pipeline_new.py`` etc. → as above
* Other top-level ``.py`` (not in {plan,code,preprocessed_output,middle_process})
  → ``middle_process/code/<name>``
* **Legacy fix-up**: any existing ``code/middle_process/`` directory is moved
  bodily into ``middle_process/code/`` (preserving filenames; ``_safe_dst``
  resolves collisions). This corrects older work_dirs that violated the
  "no middle_process under code/" invariant.
* Top-level ``*.pkl`` / ``*.npz`` / ``*.meta.json`` → ``preprocessed_output/<name>``
* If no ``pipeline_record.json`` exists → call ``finalize_work_dir(status="migrated")``
  to generate one (and ``README.md``).

Re-running on a compliant directory is a no-op.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import tarfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal


_PIPELINE_VERSION_RE = re.compile(
    r"^pipeline_(v(\d+)|final|new|backup|old|copy)\.py$",
    re.IGNORECASE,
)
_DATA_EXTS = (".pkl", ".npz", ".nwb")
_META_SUFFIX = ".meta.json"
_KNOWN_TOP_LEVEL_DIRS = {
    "plan", "code", "preprocessed_output", "middle_process", ".pipeline_cache",
}
_KNOWN_TOP_LEVEL_FILES = {"README.md", "pipeline_record.json"}


@dataclass
class Action:
    kind: Literal["mkdir", "move", "finalize"]
    src: Path | None = None
    dst: Path | None = None
    note: str = ""

    def describe(self) -> str:
        if self.kind == "mkdir":
            return f"mkdir  {self.dst}"
        if self.kind == "move":
            return f"move   {self.src.name} → {self.dst}"
        if self.kind == "finalize":
            return f"finalize  {self.note}"
        return f"{self.kind}  {self.src} → {self.dst}"


@dataclass
class Result:
    work_dir: Path
    actions: List[Action] = field(default_factory=list)
    applied: bool = False
    backup_path: Path | None = None
    skipped_reason: str | None = None


def is_work_dir(path: Path) -> bool:
    return path.is_dir() and path.name.endswith("_preprocess_work_dir")


def _find_pipeline_versions(work_dir: Path) -> list[tuple[int, Path]]:
    """Return [(rank, path)] for all pipeline_v{N}/_final/_new/... files at root.

    Numeric suffixes get their N as rank; named suffixes use heuristic ranks
    so the "winner" is well-defined (final > new > vN > backup > old > copy).
    """
    name_rank = {"final": 10_000, "new": 5_000, "backup": -10, "old": -20, "copy": -30}
    out: list[tuple[int, Path]] = []
    for entry in work_dir.iterdir():
        if not entry.is_file():
            continue
        m = _PIPELINE_VERSION_RE.match(entry.name)
        if not m:
            continue
        suffix = m.group(1)
        if suffix.lower().startswith("v"):
            try:
                rank = int(m.group(2))
            except (TypeError, ValueError):
                rank = 0
        else:
            rank = name_rank.get(suffix.lower(), 0)
        out.append((rank, entry))
    return out


def _find_other_top_level_py(work_dir: Path) -> list[Path]:
    """Top-level .py that aren't pipeline_*.py."""
    out: list[Path] = []
    for entry in work_dir.iterdir():
        if not entry.is_file() or not entry.name.endswith(".py"):
            continue
        if _PIPELINE_VERSION_RE.match(entry.name):
            continue
        out.append(entry)
    return out


def _find_top_level_data(work_dir: Path) -> list[Path]:
    out: list[Path] = []
    for entry in work_dir.iterdir():
        if not entry.is_file():
            continue
        name = entry.name
        if name.endswith(_DATA_EXTS) or name.endswith(_META_SUFFIX):
            out.append(entry)
    return out


def _safe_dst(dst: Path) -> Path:
    """If ``dst`` already exists, append ``_dup<N>`` before the suffix."""
    if not dst.exists():
        return dst
    n = 1
    stem = dst.stem
    suffix = dst.suffix
    parent = dst.parent
    while True:
        candidate = parent / f"{stem}_dup{n}{suffix}"
        if not candidate.exists():
            return candidate
        n += 1


def plan_migration(work_dir: Path) -> List[Action]:
    """Return the list of actions needed to migrate ``work_dir`` to compliance."""
    actions: List[Action] = []
    wd = work_dir
    if not is_work_dir(wd):
        return actions

    # ---- pipeline versions ----
    versions = _find_pipeline_versions(wd)
    code_dir = wd / "code"
    # Archive scripts live under <work_dir>/middle_process/code/, NOT under
    # <work_dir>/code/middle_process/ (the latter is a contract violation —
    # see plans/multi-session-routing/00-overview.md "no middle_process under
    # code/" invariant).
    archive_dir = wd / "middle_process" / "code"
    output_dir = wd / "preprocessed_output"

    # ---- LEGACY FIX-UP: code/middle_process/ → middle_process/code/ ----
    legacy_archive = code_dir / "middle_process"
    if legacy_archive.is_dir():
        actions.append(Action(
            kind="mkdir", dst=archive_dir, note="ensure middle_process/code/",
        ))
        for entry in sorted(legacy_archive.iterdir()):
            if not entry.is_file():
                continue
            dst = _safe_dst(archive_dir / entry.name)
            actions.append(Action(
                kind="move", src=entry, dst=dst,
                note="relocate legacy code/middle_process/ → middle_process/code/",
            ))

    if versions:
        # Pick the highest-rank as the canonical pipeline.py.
        versions.sort(key=lambda t: t[0])
        winner_rank, winner_path = versions[-1]
        losers = [p for r, p in versions[:-1]]
        target = code_dir / "pipeline.py"
        # Only move winner if its destination is safe.
        if not target.exists():
            actions.append(Action(kind="mkdir", dst=code_dir, note="ensure code/"))
            actions.append(Action(
                kind="move", src=winner_path, dst=target,
                note=f"canonical (rank={winner_rank})",
            ))
        else:
            # code/pipeline.py already canonical — archive winner with the rest.
            losers.insert(0, winner_path)
        if losers:
            actions.append(Action(
                kind="mkdir", dst=archive_dir, note="ensure middle_process/code/",
            ))
            for p in losers:
                dst = _safe_dst(archive_dir / p.name)
                actions.append(Action(
                    kind="move", src=p, dst=dst, note="archive prior version",
                ))

    # ---- other top-level .py ----
    other_py = _find_other_top_level_py(wd)
    if other_py:
        actions.append(Action(
            kind="mkdir", dst=archive_dir, note="ensure middle_process/code/",
        ))
        for p in other_py:
            dst = _safe_dst(archive_dir / p.name)
            actions.append(Action(
                kind="move", src=p, dst=dst, note="archive scatter script",
            ))

    # ---- top-level binaries → preprocessed_output/ ----
    binaries = _find_top_level_data(wd)
    if binaries:
        actions.append(Action(
            kind="mkdir", dst=output_dir, note="ensure preprocessed_output/",
        ))
        for p in binaries:
            dst = _safe_dst(output_dir / p.name)
            actions.append(Action(
                kind="move", src=p, dst=dst, note="binary output to canonical dir",
            ))

    # ---- finalize: generate pipeline_record.json + README.md if absent ----
    # The contract places pipeline_record.json under plan/ (per repo_builder).
    needs_finalize = (
        not (wd / "plan" / "pipeline_record.json").exists()
        or not (wd / "README.md").exists()
    )
    if needs_finalize:
        actions.append(Action(kind="finalize", note="status=migrated"))

    return actions


def _make_backup(work_dir: Path) -> Path:
    """tar.gz the work_dir into its parent. Returns the backup path."""
    ts = time.strftime("%Y%m%d_%H%M%S")
    backup = work_dir.parent / f".migration_backup_{work_dir.name}_{ts}.tar.gz"
    with tarfile.open(backup, "w:gz") as tar:
        tar.add(str(work_dir), arcname=work_dir.name)
    return backup


def apply_migration(
    work_dir: Path,
    *,
    dry_run: bool = True,
    no_backup: bool = False,
    log_fn=None,
) -> Result:
    """Plan and (optionally) execute migration."""
    log = log_fn or (lambda s: None)
    res = Result(work_dir=work_dir)

    if not is_work_dir(work_dir):
        res.skipped_reason = f"not a *_preprocess_work_dir: {work_dir}"
        log(res.skipped_reason)
        return res

    if (work_dir / ".git").exists():
        res.skipped_reason = (
            f"refusing to migrate {work_dir}: contains .git/ (treat as git "
            f"repo and migrate manually)"
        )
        log(res.skipped_reason)
        return res

    actions = plan_migration(work_dir)
    res.actions = actions

    if not actions:
        log(f"  [{work_dir.name}] already compliant — no actions needed")
        return res

    log(f"\n  [{work_dir.name}] {len(actions)} action(s):")
    for a in actions:
        log(f"    - {a.describe()}")

    if dry_run:
        log("  (dry-run — pass --apply to execute)")
        return res

    # Backup before any write.
    if not no_backup:
        log("  Creating backup tarball...")
        try:
            res.backup_path = _make_backup(work_dir)
            log(f"  Backup: {res.backup_path}")
        except Exception as exc:
            log(f"  WARNING: backup failed ({exc}); aborting migration")
            res.skipped_reason = f"backup failed: {exc}"
            return res

    # Execute actions in order.
    log_path = work_dir / ".migration_log"
    with log_path.open("a", encoding="utf-8") as logf:
        logf.write(f"\n# Migration run: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

        for a in actions:
            try:
                if a.kind == "mkdir":
                    a.dst.mkdir(parents=True, exist_ok=True)
                    logf.write(f"mkdir {a.dst}\n")
                elif a.kind == "move":
                    a.dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(a.src), str(a.dst))
                    logf.write(f"move {a.src} -> {a.dst}\n")
                elif a.kind == "finalize":
                    from easybci_lib.tools.neural_processing.export.finalize import (
                        finalize_work_dir,
                    )
                    finalize_work_dir(
                        work_dir, status="migrated",
                        partial_reason="legacy work_dir migrated by easybci migrate-work-dir",
                    )
                    logf.write(f"finalize status=migrated\n")
            except Exception as exc:
                logf.write(f"# FAILED: {a.describe()}: {exc}\n")
                log(f"    ERROR: {a.describe()} — {exc}")

    res.applied = True
    log(f"  Migration applied. Log: {log_path}")
    return res


def find_work_dirs_recursive(parent: Path, max_depth: int = 3) -> list[Path]:
    """Find all *_preprocess_work_dir under ``parent`` (bounded depth)."""
    out: list[Path] = []
    if is_work_dir(parent):
        out.append(parent)
        return out
    if not parent.is_dir() or max_depth <= 0:
        return out
    try:
        for entry in parent.iterdir():
            if entry.is_dir() and not entry.name.startswith("."):
                if is_work_dir(entry):
                    out.append(entry)
                else:
                    out.extend(find_work_dirs_recursive(entry, max_depth - 1))
    except PermissionError:
        pass
    return out
