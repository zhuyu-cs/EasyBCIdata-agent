"""Runtime path guard that forces agents through the pipeline
skill's 14-step flow.

The pipeline skill defines a contract: ``pipeline.py / qc.py /
run.py`` under ``code/``, ``plan/proposal.json / goal.json`` under
``plan/``, BIDS ``sub-*/ses-*`` data files under ``preprocessed_output/``
— all produced by *specific* tools (generate_code / propose_pipeline /
preprocess_neural / export_repo). Agents that bypass those tools and
write the same paths directly via ``write_file / terminal /
execute_code`` produce broken mini-repos that drift from the contract
(no plan/, no QC, wrong layout, modality=unknown).

This guard sits at the bypass-prone tool entrypoints (file_operations,
terminal_tool, code_execution_tool) and refuses writes to those paths
until ``<work_dir>/middle_process/proposal.confirmed`` exists (i.e. the
agent went through ``propose_pipeline`` + ``mark_proposal_confirmed``).
After the marker exists, all writes are released — so the agent can
still patch a generated ``pipeline.py`` to fix bugs.

Outside any ``*_preprocess_work_dir/``, the guard never fires.
"""
from __future__ import annotations

import fnmatch
import logging
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


# Paths under work_dir that may only be written by their owning standard
# tool. Used by ``write_file`` / ``terminal`` (catches direct shell
# redirects too). Patterns are matched against the path *relative to the
# work_dir* in posix form.
_SENSITIVE_STRICT: tuple[str, ...] = (
    # code/ — exclusive to generate_code
    "code/pipeline.py",
    "code/run.py",
    "code/qc.py",
    "code/vis.py",
    "code/build_ai_ready.py",
    "code/requirements.txt",
    # plan/ — exclusive to propose_pipeline / export_repo
    "plan/proposal.json",
    "plan/goal.json",
    "plan/pipeline_record.json",
    "plan/reasoning.md",
    "plan/config.yaml",
    "plan/web_evidence.json",
    "plan/input_ref.json",
    # preprocessed_output/* — preprocess_neural / save_processed /
    # quality_check / export_repo only. fnmatch's * doesn't cross /, so
    # enumerate depths up to BIDS sub/ses/file (3 deep is enough for
    # the canonical layout).
    "preprocessed_output/preprocessed/*",
    "preprocessed_output/preprocessed/*/*",
    "preprocessed_output/preprocessed/*/*/*",
    "preprocessed_output/AI_ready/*",
    "preprocessed_output/AI_ready/*/*",
    "preprocessed_output/AI_ready/*/*/*",
    "preprocessed_output/QC_out/*",
    "preprocessed_output/QC_out/*/*",
    "preprocessed_output/QC_out/*/*/*",
    "preprocessed_output/figures/*",
    "preprocessed_output/figures/*/*",
    "preprocessed_output/figures/*/*/*",
    # README — exclusive to export_repo
    "README.md",
)

# Looser set for execute_code: only the "real preprocessed data product"
# paths trip the guard. Sandbox scripts may freely write figures,
# scratch files, JSON sidecars, etc. — the goal is to stop bulk-data
# bypass, not to chill exploratory code.
_SENSITIVE_LOOSE: tuple[str, ...] = (
    "preprocessed_output/preprocessed/*/*.pkl",
    "preprocessed_output/preprocessed/*/*.fif",
    "preprocessed_output/preprocessed/*/*.npz",
    "preprocessed_output/preprocessed/*/*.h5",
    "preprocessed_output/preprocessed/*/*.hdf5",
    "preprocessed_output/preprocessed/*/*.mat",
    "preprocessed_output/preprocessed/*/*.nwb",
    "preprocessed_output/preprocessed/*/*/*.pkl",
    "preprocessed_output/preprocessed/*/*/*.fif",
    "preprocessed_output/preprocessed/*/*/*.npz",
    "preprocessed_output/preprocessed/*/*/*.h5",
    "preprocessed_output/preprocessed/*/*/*.hdf5",
    "preprocessed_output/preprocessed/*/*/*.mat",
    "preprocessed_output/preprocessed/*/*/*.nwb",
    "preprocessed_output/AI_ready/*/*.pkl",
    "preprocessed_output/AI_ready/*/*.fif",
    "preprocessed_output/AI_ready/*/*.npz",
    "preprocessed_output/AI_ready/*/*.h5",
    "preprocessed_output/AI_ready/*/*.hdf5",
    "preprocessed_output/AI_ready/*/*.mat",
    "preprocessed_output/AI_ready/*/*/*.pkl",
    "preprocessed_output/AI_ready/*/*/*.fif",
    "preprocessed_output/AI_ready/*/*/*.npz",
    "preprocessed_output/AI_ready/*/*/*.h5",
    "preprocessed_output/AI_ready/*/*/*.hdf5",
    "preprocessed_output/AI_ready/*/*/*.mat",
)


def detect_work_dir(abs_path: str) -> Optional[Path]:
    """Return the nearest ancestor whose name ends with
    ``_preprocess_work_dir``, or None if abs_path is outside any such
    tree. ``abs_path`` may name a file or directory; need not exist.
    """
    try:
        p = Path(abs_path).resolve(strict=False)
    except (OSError, RuntimeError):
        return None
    for parent in (p, *p.parents):
        if parent.name.endswith("_preprocess_work_dir"):
            return parent
    return None


def _match_any(rel_posix: str, patterns: tuple[str, ...]) -> Optional[str]:
    for pat in patterns:
        if fnmatch.fnmatchcase(rel_posix, pat):
            return pat
    return None


def is_sensitive_write(
    abs_path: str, *, source: str,
) -> Tuple[bool, Optional[str], Optional[Path]]:
    """Return ``(is_sensitive, matched_pattern_or_None, work_dir_or_None)``.

    ``source`` ∈ ``{'write_file', 'terminal', 'execute_code'}``.
    ``execute_code`` uses the looser pattern set (bulk data files only);
    the others use strict (full contract surface).
    """
    wd = detect_work_dir(abs_path)
    if wd is None:
        return False, None, None
    try:
        rel = Path(abs_path).resolve(strict=False).relative_to(wd).as_posix()
    except ValueError:
        return False, None, wd
    # middle_process/ is always free — it's the scratch area.
    if rel == "middle_process" or rel.startswith("middle_process/"):
        return False, None, wd
    patterns = _SENSITIVE_LOOSE if source == "execute_code" else _SENSITIVE_STRICT
    matched = _match_any(rel, patterns)
    return (matched is not None), matched, wd


def proposal_confirmed(work_dir: Path) -> bool:
    return (work_dir / "middle_process" / "proposal.confirmed").is_file()


def evaluate(abs_path: str, *, source: str) -> Tuple[bool, Optional[dict]]:
    """Return ``(allowed, denial_payload_or_None)``.

    Allow when:

    - path is outside any ``*_preprocess_work_dir/``
    - path is under ``middle_process/`` in the work_dir
    - path doesn't match any sensitive pattern for this source
    - ``middle_process/proposal.confirmed`` exists in the work_dir

    Otherwise return ``(False, denial_payload)`` where ``denial_payload``
    is a json-serializable dict the caller can return verbatim as a
    tool error.
    """
    sensitive, pattern, wd = is_sensitive_write(abs_path, source=source)
    if not sensitive:
        return True, None
    assert wd is not None  # is_sensitive_write guarantees this when sensitive
    if proposal_confirmed(wd):
        return True, None
    rel = Path(abs_path).resolve(strict=False).relative_to(wd).as_posix()
    denial = {
        "error": (
            f"skill_compliance: refusing to write '{rel}' under "
            f"'{wd}' via {source}."
        ),
        "matched_pattern": pattern,
        "work_dir": str(wd),
        "source": source,
        "remediation": [
            "This file is produced by the standard preprocessing tools, "
            "not by direct file writes. The pipeline skill "
            "enforces a 14-step contract. Required sequence:",
            "1. inspect_data(data_path=...) — fingerprint.",
            "2. deep_inspect(data_path=..., work_dir=...) — writes "
            "middle_process/inspection_report.json.",
            "3. propose_pipeline(data_path=..., analysis_goal=..., "
            "steps=[...], rationale=[...], modality=..., paradigm=..., "
            "output_path=<work_dir>, inspection_report_path=<step 2>) "
            "— stages the proposal in middle_process/proposal.staged.json. "
            "Nothing lands in plan/ yet.",
            "4. Present the proposal to the user. On user confirm, call "
            "mark_proposal_confirmed(work_dir=<work_dir>, "
            "user_decision='confirm', proposal_summary='<one line>') "
            "— THIS step materializes plan/proposal.json + plan/goal.json "
            "(+ plan/reasoning.md when the evidence-driven step form was "
            "used + plan/web_evidence.json) and writes the "
            "middle_process/proposal.confirmed marker.",
            "5. generate_code(steps=[...], data_info=..., modality=..., "
            "analysis_goal=..., reasoning={...}, work_dir=..., "
            "inspection_report_path=..., proposal_confirmed=True) — "
            "writes code/pipeline.py, qc.py, run.py CORRECTLY.",
            "Once middle_process/proposal.confirmed exists, this guard "
            "releases all writes in the work_dir (so you can patch "
            "pipeline.py to fix bugs).",
            "If your write target lives OUTSIDE *_preprocess_work_dir/, "
            "no guard applies — write freely.",
            "Call skill_view('pipeline') to refresh the "
            "contract if you've forgotten any details.",
        ],
    }
    return False, denial
