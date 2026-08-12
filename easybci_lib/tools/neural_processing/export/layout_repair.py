"""Repair primitives that bring a work_dir back into contract.

Each ``_fix_*`` function is a pure, idempotent operation:

  - Takes ``(wd: Path, ...spec_args, *, dry_run: bool, allow_subprocess: bool)``.
  - Returns a ``FixResult`` describing what it did (or would have done).
  - Only touches paths inside ``wd``; never mutates user source data.
  - Never raises for expected drift — encodes failure in ``FixResult.residual``.
  - Raises ``Unrepairable`` when the fix genuinely requires an outside
    subprocess and the caller passed ``allow_subprocess=False``.

The public entry-point ``verify_and_repair(wd, ...)`` is added by Phase 2
(orchestrator loop). This module intentionally exports the primitives +
data classes + ``detect_violations`` so it stays testable in isolation.
"""
from __future__ import annotations

import json
import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from easybci_lib.tools.neural_processing.export.layout_spec import (
    CANONICAL,
    LayoutSpec,
    ResolvedLayoutSpec,
    resolve_for_goal,
)

logger = logging.getLogger(__name__)


# --- Data classes -----------------------------------------------------------

@dataclass(frozen=True)
class Violation:
    kind: str
    detail: str
    path: str = ""
    severity: str = "error"


@dataclass
class FixResult:
    kind: str
    applied: bool
    notes: list[str] = field(default_factory=list)
    residual: bool = False
    elapsed_s: float = 0.0


class Unrepairable(RuntimeError):
    """Raised when a fix legitimately needs a subprocess and none is allowed."""


# --- Primitive: missing required directory ---------------------------------

def _fix_missing_dir(wd: Path, dir_name: str, *, dry_run: bool) -> FixResult:
    """Create ``wd / dir_name`` if missing. Idempotent."""
    started = time.monotonic()
    target = wd / dir_name
    if target.is_dir():
        return FixResult(
            kind=f"missing_dir:{dir_name}", applied=False, notes=[], residual=False,
            elapsed_s=time.monotonic() - started,
        )
    if dry_run:
        return FixResult(
            kind=f"missing_dir:{dir_name}", applied=False,
            notes=[f"would create directory {dir_name}/"], residual=True,
            elapsed_s=time.monotonic() - started,
        )
    try:
        target.mkdir(parents=True, exist_ok=True)
        return FixResult(
            kind=f"missing_dir:{dir_name}", applied=True,
            notes=[f"mkdir {dir_name}/"], residual=False,
            elapsed_s=time.monotonic() - started,
        )
    except OSError as exc:
        logger.warning("_fix_missing_dir: mkdir %s failed: %s", target, exc)
        return FixResult(
            kind=f"missing_dir:{dir_name}", applied=False,
            notes=[f"mkdir failed: {exc}"], residual=True,
            elapsed_s=time.monotonic() - started,
        )


# --- Primitive: forbidden code/middle_process ------------------------------

def _fix_forbidden_code_middle_process(wd: Path, *, dry_run: bool) -> FixResult:
    """Relocate <wd>/code/middle_process/ contents to <wd>/middle_process/code/.

    The forbidden directory is a hard contract violation (Step 14 cleanup
    depends on middle_process/ living exclusively at wd root; anything
    under code/middle_process/ survives the cleanup and pollutes the mini-repo).
    """
    started = time.monotonic()
    forbidden = wd / "code" / "middle_process"
    if not forbidden.exists():
        return FixResult(
            kind="forbidden:code/middle_process", applied=False,
            notes=[], residual=False, elapsed_s=time.monotonic() - started,
        )

    dest = wd / "middle_process" / "code"
    if dry_run:
        return FixResult(
            kind="forbidden:code/middle_process", applied=False,
            notes=[f"would move code/middle_process/* → {dest.relative_to(wd)}/"],
            residual=True, elapsed_s=time.monotonic() - started,
        )

    notes: list[str] = []
    try:
        dest.mkdir(parents=True, exist_ok=True)
        for child in list(forbidden.iterdir()):
            target = dest / child.name
            if target.exists():
                # Suffix to avoid clobber; preserve evidence.
                stem, dot, ext = child.name.rpartition(".")
                target = dest / (
                    f"{stem}_conflict.{ext}" if dot else f"{child.name}_conflict"
                )
            shutil.move(str(child), str(target))
            notes.append(
                f"moved code/middle_process/{child.name} → middle_process/code/{target.name}"
            )
        forbidden.rmdir()
        notes.append("removed empty code/middle_process/")
    except OSError as exc:
        logger.warning("_fix_forbidden_code_middle_process failed: %s", exc)
        return FixResult(
            kind="forbidden:code/middle_process", applied=len(notes) > 0,
            notes=notes + [f"error: {exc}"], residual=True,
            elapsed_s=time.monotonic() - started,
        )
    return FixResult(
        kind="forbidden:code/middle_process", applied=True,
        notes=notes, residual=False, elapsed_s=time.monotonic() - started,
    )


# --- Primitive: missing README.md ------------------------------------------

def _fix_missing_readme(wd: Path, *, dry_run: bool) -> FixResult:
    """Write a minimal auto-generated README.md if absent.

    The stub is intentionally sparse — build_mini_repo's _write_readme is
    the authoritative writer during normal finalize. This is only a
    contract-satisfying husk for degenerate recovery paths.
    """
    started = time.monotonic()
    target = wd / "README.md"
    if target.exists():
        return FixResult(
            kind="missing_file:README.md", applied=False,
            notes=[], residual=False, elapsed_s=time.monotonic() - started,
        )
    body = (
        "# Auto-generated stub\n\n"
        "This README was synthesized by layout_repair because the run\n"
        "finalized without producing one. See plan/reasoning.md for\n"
        "pipeline provenance and plan/pipeline_record.json for the record.\n"
    )
    if dry_run:
        return FixResult(
            kind="missing_file:README.md", applied=False,
            notes=["would write README.md stub"], residual=True,
            elapsed_s=time.monotonic() - started,
        )
    try:
        target.write_text(body, encoding="utf-8")
    except OSError as exc:
        return FixResult(
            kind="missing_file:README.md", applied=False,
            notes=[f"write failed: {exc}"], residual=True,
            elapsed_s=time.monotonic() - started,
        )
    return FixResult(
        kind="missing_file:README.md", applied=True,
        notes=["wrote README.md stub"], residual=False,
        elapsed_s=time.monotonic() - started,
    )


# --- Primitive: orphan / illegal-ext / space-in-name -----------------------

_SWEEP_PARENT = "middle_process"


def _sweep_dir(wd: Path, ts: str) -> Path:
    return wd / _SWEEP_PARENT / f"sweep_{ts}"


def _fix_orphan_files(wd: Path, *, dry_run: bool) -> FixResult:
    """Handle: (a) illegal extensions in preprocessed/, (b) spaces in filenames,
    (c) files not covered by any routing entry.

    Actions:
      - Rename space → underscore in place (safe: no data loss, routing key
        is ``stem_safe`` which is already the underscored form).
      - Move illegal-ext / no-routing files to ``middle_process/sweep_<ts>/``.
    """
    started = time.monotonic()
    routing_path = wd / "middle_process" / "inputs_routing.json"
    if not routing_path.is_file():
        return FixResult(
            kind="orphan:no_routing", applied=False,
            notes=[], residual=False, elapsed_s=time.monotonic() - started,
        )

    try:
        table = json.loads(routing_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return FixResult(
            kind="orphan:routing_unreadable", applied=False,
            notes=[f"routing table unreadable: {exc}"], residual=True,
            elapsed_s=time.monotonic() - started,
        )
    valid_triples = {
        (i.get("subject_id"), i.get("session_id"), i.get("stem_safe"))
        for i in (table.get("inputs") or [])
    }

    notes: list[str] = []
    ts = time.strftime("%Y%m%d_%H%M%S")
    swept_to = _sweep_dir(wd, ts)

    def _sweep(src: Path, reason: str) -> None:
        if dry_run:
            notes.append(f"would sweep {src.relative_to(wd)} ({reason})")
            return
        swept_to.mkdir(parents=True, exist_ok=True)
        dest = swept_to / src.name
        if dest.exists():
            dest = swept_to / f"{src.stem}_dup{src.suffix}"
        shutil.move(str(src), str(dest))
        notes.append(f"swept {src.relative_to(wd)} → sweep_{ts}/{dest.name} ({reason})")

    def _rename_space(src: Path) -> None:
        fixed_name = src.name.replace(" ", "_")
        target = src.with_name(fixed_name)
        if dry_run:
            notes.append(f"would rename {src.name} → {fixed_name}")
            return
        if target.exists():
            _sweep(src, "space-in-name conflict")
            return
        src.rename(target)
        notes.append(f"renamed {src.relative_to(wd)} → {fixed_name}")

    pre_base = wd / "preprocessed_output" / "preprocessed"
    if pre_base.is_dir():
        for sub_dir in pre_base.glob("sub-*"):
            sub = sub_dir.name.removeprefix("sub-")
            for ses_dir in sub_dir.iterdir():
                if not ses_dir.is_dir():
                    continue
                ses = ses_dir.name.removeprefix("ses-")
                for entry in list(ses_dir.iterdir()):
                    if not entry.is_file():
                        continue
                    if " " in entry.name:
                        _rename_space(entry)
                        continue
                    if entry.suffix not in CANONICAL.preprocessed_ext_allowed:
                        _sweep(entry, "illegal ext in preprocessed/")
                        continue
                    stem = entry.stem.removesuffix("_preprocessed")
                    if (sub, ses, stem) not in valid_triples:
                        _sweep(entry, "no matching routing entry")

    fig_base = wd / "preprocessed_output" / "figures"
    if fig_base.is_dir():
        for png in list(fig_base.rglob("*.png")):
            if " " in png.name:
                _rename_space(png)
                continue
            parts = png.relative_to(fig_base).parts
            if len(parts) >= 3:
                sub = parts[0].removeprefix("sub-")
                ses = parts[1].removeprefix("ses-")
                if not any(
                    s == sub and se == ses and png.name.startswith(stem + "_")
                    for (s, se, stem) in valid_triples
                ):
                    _sweep(png, "figure not covered by routing")

    applied = any(not n.startswith("would") for n in notes)
    return FixResult(
        kind="orphan:preprocessed_or_figures",
        applied=applied,
        notes=notes,
        residual=(dry_run and bool(notes)),
        elapsed_s=time.monotonic() - started,
    )


# --- Primitive: missing code script (codegen) ------------------------------

CodegenGenerator = Callable[[str, Path], None]


def _fix_missing_code_script(
    wd: Path,
    script_name: str,
    *,
    dry_run: bool,
    generator: Optional[CodegenGenerator] = None,
) -> FixResult:
    """Regenerate ``code/<script_name>`` when missing.

    ``generator`` is a callable ``(script_name, target_path) -> None`` that
    writes the file. Passing ``None`` raises Unrepairable — this primitive
    is the wrapper for a codegen call, and without a generator we cannot
    recover the file. The default production generator (added by Phase 2's
    integration layer) dispatches to ``codegen.generator.generate_*_script``
    based on ``script_name``.
    """
    started = time.monotonic()
    target = wd / "code" / script_name
    if target.is_file():
        return FixResult(
            kind=f"missing_file:code/{script_name}", applied=False,
            notes=[], residual=False, elapsed_s=time.monotonic() - started,
        )
    if generator is None:
        raise Unrepairable(
            f"code/{script_name} missing and no generator provided"
        )
    if dry_run:
        return FixResult(
            kind=f"missing_file:code/{script_name}", applied=False,
            notes=[f"would regenerate code/{script_name} via codegen"], residual=True,
            elapsed_s=time.monotonic() - started,
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        generator(script_name, target)
    except Exception as exc:  # noqa: BLE001 - generator may fail for any reason
        logger.warning("code regen for %s failed: %s", script_name, exc)
        return FixResult(
            kind=f"missing_file:code/{script_name}", applied=False,
            notes=[f"codegen failed: {exc}"], residual=True,
            elapsed_s=time.monotonic() - started,
        )
    ok = target.is_file()
    return FixResult(
        kind=f"missing_file:code/{script_name}", applied=ok,
        notes=(
            [f"regenerated code/{script_name}"] if ok
            else ["codegen returned but no file written"]
        ),
        residual=not ok, elapsed_s=time.monotonic() - started,
    )


# --- Primitive: missing artifact (subprocess re-run) -----------------------

ScriptRunner = Callable[[Path, str], bool]

_STAGE_TO_SCRIPT = {
    "pipeline": "pipeline.py",
    "qc": "qc.py",
    "vis": "vis.py",
    "build_ai_ready": "build_ai_ready.py",
}


def _fix_missing_artifact(
    wd: Path,
    *,
    stage: str,
    dry_run: bool,
    allow_subprocess: bool,
    runner: Optional[ScriptRunner],
) -> FixResult:
    """Re-run ``code/<stage>.py`` as subprocess to regenerate its artifacts.

    ``allow_subprocess=False`` (safe default for read-only diagnostic
    contexts) raises Unrepairable — the caller must escalate to a manual
    ``easybci doctor layout --apply`` or return the violation upstream.
    """
    started = time.monotonic()
    script_name = _STAGE_TO_SCRIPT.get(stage)
    if not script_name:
        return FixResult(
            kind=f"missing_artifact:{stage}", applied=False,
            notes=[f"unknown stage {stage!r}"], residual=True,
            elapsed_s=time.monotonic() - started,
        )
    script = wd / "code" / script_name
    if not script.is_file():
        return FixResult(
            kind=f"missing_artifact:{stage}", applied=False,
            notes=[f"code/{script_name} absent — regenerate the script first"],
            residual=True, elapsed_s=time.monotonic() - started,
        )

    if not allow_subprocess:
        raise Unrepairable(
            f"missing_artifact:{stage} needs a subprocess re-run but "
            f"allow_subprocess=False"
        )
    if runner is None:
        raise Unrepairable(
            f"missing_artifact:{stage} needs a runner injection"
        )
    if dry_run:
        return FixResult(
            kind=f"missing_artifact:{stage}", applied=False,
            notes=[f"would re-run code/{script_name} <work_dir>"], residual=True,
            elapsed_s=time.monotonic() - started,
        )
    try:
        ok = bool(runner(wd, stage))
    except Exception as exc:  # noqa: BLE001
        logger.warning("script_runner for %s failed: %s", stage, exc)
        return FixResult(
            kind=f"missing_artifact:{stage}", applied=False,
            notes=[f"runner raised: {exc}"], residual=True,
            elapsed_s=time.monotonic() - started,
        )
    return FixResult(
        kind=f"missing_artifact:{stage}", applied=ok,
        notes=[f"re-ran code/{script_name} (success={ok})"],
        residual=not ok, elapsed_s=time.monotonic() - started,
    )


# --- Primitive: husk proposal fields ---------------------------------------

def _fix_husk_proposal(wd: Path, *, dry_run: bool) -> FixResult:
    """Fill husk fields (modality/paradigm/analysis_goal == "unknown") from
    middle_process/inspection_report.json when available."""
    started = time.monotonic()
    proposal_path = wd / "plan" / "proposal.json"
    if not proposal_path.is_file():
        return FixResult(
            kind="husk:proposal.missing", applied=False,
            notes=["plan/proposal.json missing"], residual=True,
            elapsed_s=time.monotonic() - started,
        )
    try:
        payload = json.loads(proposal_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return FixResult(
            kind="husk:proposal.unreadable", applied=False,
            notes=[str(exc)], residual=True,
            elapsed_s=time.monotonic() - started,
        )
    husk_fields = [
        f for f in ("modality", "paradigm", "analysis_goal")
        if isinstance(payload.get(f), str) and payload[f].strip().lower() == "unknown"
    ]
    if not husk_fields:
        return FixResult(
            kind="husk:proposal", applied=False, notes=[], residual=False,
            elapsed_s=time.monotonic() - started,
        )

    src = wd / "middle_process" / "inspection_report.json"
    if not src.is_file():
        return FixResult(
            kind="husk:proposal", applied=False,
            notes=[f"husk fields {husk_fields} — no inspection_report.json to recover"],
            residual=True, elapsed_s=time.monotonic() - started,
        )
    try:
        report = json.loads(src.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return FixResult(
            kind="husk:proposal", applied=False,
            notes=[f"inspection unreadable: {exc}"], residual=True,
            elapsed_s=time.monotonic() - started,
        )
    fp = report.get("fingerprint") or {}
    notes: list[str] = []
    for field_name in husk_fields:
        candidate = fp.get(field_name)
        if isinstance(candidate, str) and candidate.strip() and candidate.lower() != "unknown":
            if dry_run:
                notes.append(f"would set proposal.{field_name}={candidate!r}")
            else:
                payload[field_name] = candidate
                notes.append(f"set proposal.{field_name}={candidate!r}")
    if not notes:
        return FixResult(
            kind="husk:proposal", applied=False,
            notes=["inspection_report.json has no usable values"], residual=True,
            elapsed_s=time.monotonic() - started,
        )
    if dry_run:
        return FixResult(
            kind="husk:proposal", applied=False, notes=notes, residual=True,
            elapsed_s=time.monotonic() - started,
        )
    proposal_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    residual = any(
        isinstance(payload.get(f), str) and payload[f].strip().lower() == "unknown"
        for f in ("modality", "paradigm", "analysis_goal")
    )
    return FixResult(
        kind="husk:proposal", applied=True, notes=notes, residual=residual,
        elapsed_s=time.monotonic() - started,
    )


# --- Detector ---------------------------------------------------------------

def detect_violations(
    wd: Path,
    *,
    resolved: Optional[ResolvedLayoutSpec] = None,
) -> list[Violation]:
    """Enumerate contract violations for ``wd`` against ``resolved`` (or the
    generic default). No side effects — purely a diff between what should
    exist and what does. Callers pass the result to a repair loop or render
    it into a report."""
    if resolved is None:
        resolved = resolve_for_goal(None)
    violations: list[Violation] = []
    for d in resolved.base.required_dirs:
        if not (wd / d).is_dir():
            violations.append(
                Violation(kind=f"missing_dir:{d}", detail=f"{d}/ absent", path=str(wd / d))
            )
    for f in resolved.base.required_files:
        if not (wd / f).is_file():
            violations.append(
                Violation(kind=f"missing_file:{f}", detail=f"{f} absent", path=str(wd / f))
            )
    for forb in resolved.base.forbidden_paths:
        if (wd / forb).exists():
            violations.append(
                Violation(kind=f"forbidden:{forb}", detail=f"{forb} must not exist",
                          path=str(wd / forb))
            )
    # code/ scripts required for this goal
    if (wd / "code").is_dir():
        for script in resolved.code_files_required:
            if not (wd / "code" / script).is_file():
                violations.append(
                    Violation(
                        kind=f"missing_file:code/{script}",
                        detail=f"code/{script} absent",
                        path=str(wd / "code" / script),
                    )
                )
    # husk check on plan/proposal.json
    proposal_path = wd / "plan" / "proposal.json"
    if proposal_path.is_file():
        try:
            payload = json.loads(proposal_path.read_text(encoding="utf-8"))
            for f in ("modality", "paradigm", "analysis_goal"):
                val = payload.get(f)
                if isinstance(val, str) and val.strip().lower() == "unknown":
                    violations.append(
                        Violation(
                            kind=f"husk:proposal.{f}",
                            detail=f"proposal.{f} == 'unknown'",
                            path=str(proposal_path),
                        )
                    )
        except (OSError, json.JSONDecodeError):
            violations.append(
                Violation(
                    kind="husk:proposal.unreadable",
                    detail="plan/proposal.json unreadable",
                    path=str(proposal_path),
                )
            )
    # Disallowed extensions in output leaves (auto-move targets). One violation
    # per bucket is enough to trigger the fix (the primitive sweeps all matches).
    for _kind, _top, _allowed in (
        ("disallowed_ext:preprocessed",
         wd / "preprocessed_output" / "preprocessed",
         set(resolved.base.preprocessed_ext_allowed)),
        ("disallowed_ext:ai_ready",
         wd / "preprocessed_output" / "AI_ready",
         set(resolved.base.ai_ready_ext_allowed)),
    ):
        if _top.is_dir():
            for _f in _top.rglob("*"):
                if (_f.is_file()
                        and len(_f.relative_to(_top).parts) >= 2
                        and _f.suffix not in _allowed):
                    violations.append(Violation(
                        kind=_kind,
                        detail=f"illegal extension {_f.suffix} under {_top.name}/",
                        path=str(_f),
                    ))
                    break  # one violation per bucket is enough to trigger the fix
    return violations


# --- Orchestrator (Phase 2) -------------------------------------------------

@dataclass
class RepairReport:
    work_dir: str
    initial_violations: int
    remaining_violations: int
    rounds: int
    fixes: list[FixResult]
    wall_clock_s: float
    unrepairable: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "work_dir": self.work_dir,
            "initial_violations": self.initial_violations,
            "remaining_violations": self.remaining_violations,
            "rounds": self.rounds,
            "wall_clock_s": round(self.wall_clock_s, 4),
            "unrepairable": list(self.unrepairable),
            "fixes": [
                {
                    "kind": f.kind, "applied": f.applied, "notes": f.notes,
                    "residual": f.residual, "elapsed_s": round(f.elapsed_s, 4),
                }
                for f in self.fixes
            ],
        }


def _dispatch_violation(
    wd: Path, v: Violation, *, dry_run: bool, allow_subprocess: bool,
    generator: Optional[CodegenGenerator], runner: Optional[ScriptRunner],
) -> FixResult:
    if v.kind.startswith("missing_dir:"):
        return _fix_missing_dir(wd, v.kind.split(":", 1)[1], dry_run=dry_run)
    if v.kind == "missing_file:README.md":
        return _fix_missing_readme(wd, dry_run=dry_run)
    if v.kind.startswith("missing_file:code/"):
        script = v.kind.split(":code/", 1)[1]
        return _fix_missing_code_script(wd, script, dry_run=dry_run, generator=generator)
    if v.kind.startswith("forbidden:"):
        # Only code/middle_process supported today; extend if forbidden_paths grows.
        return _fix_forbidden_code_middle_process(wd, dry_run=dry_run)
    if v.kind.startswith("husk:proposal"):
        return _fix_husk_proposal(wd, dry_run=dry_run)
    if v.kind.startswith("orphan:"):
        return _fix_orphan_files(wd, dry_run=dry_run)
    if v.kind == "disallowed_ext:preprocessed":
        return _fix_disallowed_ext_in_preprocessed(wd, dry_run=dry_run)
    if v.kind == "disallowed_ext:ai_ready":
        return _fix_disallowed_ext_in_ai_ready(wd, dry_run=dry_run)
    if v.kind.startswith("missing_artifact:"):
        stage = v.kind.split(":", 1)[1]
        return _fix_missing_artifact(
            wd, stage=stage, dry_run=dry_run,
            allow_subprocess=allow_subprocess, runner=runner,
        )
    return FixResult(
        kind=v.kind, applied=False,
        notes=[f"no dispatcher for {v.kind}"], residual=True,
    )


def _default_codegen_generator(
    wd: Path, resolved: ResolvedLayoutSpec
) -> CodegenGenerator:
    """Bind a codegen adapter to the current work_dir / goal.

    Returns a callable ``(script_name, target_path) -> None`` that writes a
    minimal contract-satisfying stub for ``script_name``. Full metadata
    reconstruction lives in ``neural_tools.py:_do_handle_generate_code``;
    this last-ditch primitive only makes sure the file exists so the
    contract check passes — the user must re-run ``generate_code`` (or the
    LLM's next tool call) to fill it.
    """
    def _gen(script_name: str, target: Path) -> None:
        stub = (
            f"# Auto-generated stub for {script_name}\n"
            f"# analysis_goal = {resolved.analysis_goal}\n"
            f"# This is a contract-satisfying husk. Re-run generate_code to fill it.\n"
            f"if __name__ == '__main__':\n    import sys; sys.exit(0)\n"
        )
        target.write_text(stub, encoding="utf-8")
        logger.info("layout_repair: wrote stub for %s (regenerate to fill)", script_name)
    return _gen


def _default_script_runner() -> ScriptRunner:
    """Wrap ``codegen.script_runner.run_script`` into the ``(wd, stage) → bool`` shape."""
    def _run(wd: Path, stage: str) -> bool:
        from easybci_lib.tools.neural_processing.codegen.script_runner import run_script
        result = run_script(work_dir=str(wd), stage=stage)
        if isinstance(result, dict):
            return bool(result.get("ok"))
        return bool(result)
    return _run


def _read_deliverables_for_layout(wd: Path) -> "list[str] | None":
    """Resolve the confirmed deliverables for layout verification.

    Priority: plan/proposal.json → middle_process/proposal.confirmed marker →
    on-disk AI_ready probe (via resolve_deliverables). Returns None only when
    the deliverables module itself is unavailable, in which case the caller
    falls back to the goal's legacy produces_ai_ready hint (unchanged
    behaviour). Never raises.
    """
    try:
        from easybci_lib.tools.neural_processing.preprocess.deliverables import (
            resolve_deliverables,
        )
    except Exception:  # noqa: BLE001
        return None
    import json as _json
    record = None
    for rel in ("plan/proposal.json", "middle_process/proposal.confirmed"):
        p = wd / rel
        if not p.is_file():
            continue
        try:
            obj = _json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if isinstance(obj, dict) and isinstance(obj.get("deliverables"), list):
            record = {"deliverables": obj["deliverables"]}
            break
    try:
        return resolve_deliverables(record, work_dir=wd)
    except Exception:  # noqa: BLE001
        return None


def verify_and_repair(
    work_dir: Path | str,
    *,
    analysis_goal: Optional[str] = None,
    max_rounds: int = 3,
    dry_run: bool = False,
    allow_subprocess: bool = True,
    generator: Optional[CodegenGenerator] = None,
    runner: Optional[ScriptRunner] = None,
    write_report: bool = False,
) -> RepairReport:
    """Run detect → dispatch → apply → re-detect until convergence or ``max_rounds``.

    Default injections (production adapters):
      - ``generator`` → ``_default_codegen_generator`` (writes stubs)
      - ``runner``    → ``_default_script_runner``    (dispatches to script_runner.run_script)

    Set both to ``None`` and ``dry_run=True`` for read-only diagnostic mode.
    """
    started = time.monotonic()
    wd = Path(work_dir)
    # deliverables (from plan/proposal.json → confirm marker → on-disk probe)
    # overrides the goal's legacy produces_ai_ready hint so a run that produced
    # NWB-only (default) is not false-flagged as missing code/build_ai_ready.py.
    _deliverables = _read_deliverables_for_layout(wd)
    resolved = resolve_for_goal(analysis_goal, _deliverables)

    if generator is None and not dry_run:
        generator = _default_codegen_generator(wd, resolved)
    if runner is None and allow_subprocess and not dry_run:
        runner = _default_script_runner()

    initial = detect_violations(wd, resolved=resolved)
    fixes: list[FixResult] = []
    unrepairable: list[str] = []
    violations = initial
    rounds = 0
    while violations and rounds < max_rounds:
        rounds += 1
        made_progress = False
        for v in violations:
            try:
                r = _dispatch_violation(
                    wd, v, dry_run=dry_run, allow_subprocess=allow_subprocess,
                    generator=generator, runner=runner,
                )
            except Unrepairable as exc:
                unrepairable.append(f"{v.kind}: {exc}")
                fixes.append(FixResult(
                    kind=v.kind, applied=False,
                    notes=[str(exc)], residual=True,
                ))
                continue
            fixes.append(r)
            if r.applied:
                made_progress = True
        # Recompute — a fix may have unlocked or exposed other violations.
        violations = detect_violations(wd, resolved=resolved)
        if not made_progress:
            break  # loop diverges — bail early

    remaining = len(detect_violations(wd, resolved=resolved))

    report = RepairReport(
        work_dir=str(wd),
        initial_violations=len(initial),
        remaining_violations=remaining,
        rounds=rounds,
        fixes=fixes,
        wall_clock_s=time.monotonic() - started,
        unrepairable=unrepairable,
    )
    if write_report and not dry_run and (wd / "plan").is_dir():
        try:
            (wd / "plan" / "repair_report.json").write_text(
                json.dumps(report.to_dict(), indent=2), encoding="utf-8",
            )
        except OSError as exc:
            logger.warning("could not write plan/repair_report.json: %s", exc)
    return report


# ---- L1 pipeline hygiene primitives (extends strict-layout-enforcement) ----
#
# These primitives close the last mechanical-hygiene gaps that SKILL.md used to
# ask the LLM to perform by hand (see improved_docs/plans/pipeline-skill-to-code).
# They are added here so they share the strict-layout-enforcement primitive
# family (FixResult contract, sweep-dir convention, plan/repair_report.json
# audit) rather than living in a new module.


# Map handler stage name -> (source dirs to sweep, failed-side subdir name).
# Source dirs are relative to work_dir.
_SWEEP_MAP: dict[str, tuple[tuple[str, ...], str]] = {
    "preprocess_neural": (
        ("preprocessed_output/preprocessed",),
        "failed_outputs",
    ),
    "save_processed": (
        ("preprocessed_output/AI_ready",),
        "failed_outputs",
    ),
    "quality_check": (
        (
            "preprocessed_output/figures",
            "preprocessed_output/QC_out",
        ),
        "failed_qc",
    ),
}


def _uniquify_if_exists(p: Path) -> Path:
    """If ``p`` already exists, append -1 / -2 / ... to the stem until unique.
    Used by sweep to survive rare collisions where two sweeps hit the same second.
    """
    if not p.exists():
        return p
    stem = p.stem
    suffix = p.suffix
    parent = p.parent
    i = 1
    while True:
        candidate = parent / f"{stem}-{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def _append_hygiene_event(work_dir: Path, event: dict) -> None:
    """Append a hygiene event to plan/repair_report.json without clobbering the
    verify_and_repair-written body. No-op if plan/ absent. Best-effort."""
    plan = work_dir / "plan"
    if not plan.is_dir():
        return
    report = plan / "repair_report.json"
    try:
        data = json.loads(report.read_text(encoding="utf-8")) if report.exists() else {}
        if not isinstance(data, dict):
            data = {}
        data.setdefault("hygiene_events", []).append(event)
        report.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    except (OSError, ValueError) as exc:
        logger.debug("could not append hygiene event to %s: %s", report, exc)


def sweep_failed_partials(
    work_dir: Path,
    stage: str,
    *,
    dry_run: bool = False,
) -> dict:
    """Move partial outputs from a previously failed retry into
    middle_process/failed_*/<ts>/.

    Called by _handle_preprocess_neural / _handle_save_processed /
    _handle_quality_check at entry, ONLY when the stage's autofix_state shows
    attempts > 0 (retry #2/#3, not the first attempt). Explicit-call primitive
    — not part of the verify_and_repair dispatch (it maps to no Violation.kind).

    Returns a dict (primitive/stage/moved_files/target_dir/success/dry_run) the
    handler can log.
    """
    if stage not in _SWEEP_MAP:
        return {
            "primitive": "sweep_failed_partials",
            "stage": stage,
            "moved_files": [],
            "target_dir": None,
            "success": True,
            "dry_run": dry_run,
            "note": f"unknown stage: {stage} (no-op)",
        }

    source_rels, failed_subdir = _SWEEP_MAP[stage]
    ts = time.strftime("%Y%m%d_%H%M%S")
    target_root = work_dir / "middle_process" / failed_subdir / ts

    moved: list[dict[str, str]] = []
    for src_rel in source_rels:
        src_root = work_dir / src_rel
        if not src_root.exists():
            continue
        for f in src_root.rglob("*"):
            if not f.is_file():
                continue
            rel_from_wd = f.relative_to(work_dir)
            dst = target_root / rel_from_wd
            if not dry_run:
                dst.parent.mkdir(parents=True, exist_ok=True)
                # If the destination already exists (previous sweep collision), suffix.
                final_dst = _uniquify_if_exists(dst)
                shutil.move(str(f), str(final_dst))
                moved.append({"src": str(rel_from_wd), "dst": str(final_dst.relative_to(work_dir))})
            else:
                moved.append({"src": str(rel_from_wd), "dst": str(dst.relative_to(work_dir))})

    result = {
        "primitive": "sweep_failed_partials",
        "stage": stage,
        "moved_files": moved,
        "target_dir": str(target_root.relative_to(work_dir)) if moved else None,
        "success": True,
        "dry_run": dry_run,
    }
    if not dry_run and moved:
        _append_hygiene_event(work_dir, result)
    return result


@dataclass(frozen=True)
class ReuseSignal:
    """Result of reading the reuse-mode signal from disk.

    active   — True when the current work_dir runs in Reuse Mode.
    source   — proven skill name (meaningful only when active).
    read_from — which file provided the signal, for audit trace.
    """
    active: bool
    source: str | None
    read_from: str | None


def reuse_mode_guard(work_dir: Path) -> ReuseSignal:
    """Read the reuse-mode signal from plan/proposal.json:reuse_source.

    Called by _handle_research_preprocessing / _handle_research_parameter at
    entry. When active=True, the handler returns a soft-suppressed envelope
    instead of the network call (see 00-overview.md).

    Reads plan/proposal.json:reuse_source (materialized by
    _handle_propose_pipeline_evidence in Reuse Mode, see P3 Task 1).
    Fail-open: unreadable / missing / absent field -> New-Plan Mode.
    """
    proposal_path = work_dir / "plan" / "proposal.json"
    if proposal_path.exists():
        try:
            with proposal_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            src = data.get("reuse_source") if isinstance(data, dict) else None
            if src:
                return ReuseSignal(active=True, source=str(src),
                                   read_from="plan/proposal.json")
        except Exception:
            logger.exception(
                "reuse_mode_guard: failed to read %s — treating as new-plan",
                proposal_path,
            )
    return ReuseSignal(active=False, source=None, read_from=None)


def _fix_disallowed_ext_in_preprocessed(wd: Path, *, dry_run: bool) -> FixResult:
    """Move files under preprocessed/sub-*/ses-*/ whose extension is not in
    CANONICAL.preprocessed_ext_allowed to middle_process/sweep_<ts>/preprocessed/…"""
    return _move_disallowed_ext(
        wd=wd,
        source_top=wd / "preprocessed_output" / "preprocessed",
        allowed_exts=set(CANONICAL.preprocessed_ext_allowed),
        target_kind="preprocessed",
        kind="disallowed_ext:preprocessed",
        dry_run=dry_run,
    )


def _fix_disallowed_ext_in_ai_ready(wd: Path, *, dry_run: bool) -> FixResult:
    """Mirror for AI_ready/ (CANONICAL.ai_ready_ext_allowed)."""
    return _move_disallowed_ext(
        wd=wd,
        source_top=wd / "preprocessed_output" / "AI_ready",
        allowed_exts=set(CANONICAL.ai_ready_ext_allowed),
        target_kind="AI_ready",
        kind="disallowed_ext:ai_ready",
        dry_run=dry_run,
    )


def _move_disallowed_ext(
    *, wd: Path, source_top: Path, allowed_exts: set[str],
    target_kind: str, kind: str, dry_run: bool,
) -> FixResult:
    """Move files with a non-allowlist extension under source_top's
    sub-*/ses-*/ leaves into middle_process/sweep_<ts>/<target_kind>/…,
    preserving relative sub-path. Top-level BIDS metadata is left alone.
    Reuses the sweep-dir convention from _fix_orphan_files (_sweep_dir)."""
    started = time.monotonic()
    if not source_top.is_dir():
        return FixResult(kind=kind, applied=False, notes=[], residual=False,
                         elapsed_s=time.monotonic() - started)
    ts = time.strftime("%Y%m%d_%H%M%S")
    target_root = _sweep_dir(wd, ts) / target_kind
    notes: list[str] = []
    for f in source_top.rglob("*"):
        if not f.is_file():
            continue
        rel = f.relative_to(source_top)
        # Only leaf files under sub-*/ses-*/ — top-level metadata is legal.
        # (preprocessed uses "sub-" prefix; AI_ready drops it — accept both.)
        if len(rel.parts) < 2:
            continue
        if f.suffix in allowed_exts:
            continue
        dst = target_root / rel
        if dry_run:
            notes.append(f"would sweep {source_top.name}/{rel} (illegal ext {f.suffix})")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        final_dst = _uniquify_if_exists(dst)
        shutil.move(str(f), str(final_dst))
        notes.append(f"swept {source_top.name}/{rel} → sweep_{ts}/{target_kind}/{rel} (illegal ext {f.suffix})")
    applied = any(not n.startswith("would") for n in notes)
    return FixResult(kind=kind, applied=applied, notes=notes,
                     residual=(dry_run and bool(notes)),
                     elapsed_s=time.monotonic() - started)
