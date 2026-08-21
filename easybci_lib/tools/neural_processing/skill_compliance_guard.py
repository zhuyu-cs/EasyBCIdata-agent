"""Runtime path guard — seals work_dir as a tool-chain-only container.

The preprocessing work directory (*_preprocess_work_dir / *_preprocess_work_dirs)
is a sealed container: ALL content inside it is produced exclusively by
EasyBCI's standard tools (deep_inspect, generate_code, preprocess_neural,
batch_process_adaptive, export_repo, etc.). Direct writes from agent-facing
tools (write_file, terminal, execute_code) are DEFAULT-DENIED.

Allowed exceptions (the only four):
1. ``middle_process/`` — scratch area, always free for any source.
2. Outside any work_dir — guard does not fire.
3. After ``proposal.confirmed`` exists: patch (not create) existing
   ``code/*.py`` files — agent fixing bugs in generated code.
4. Terminal re-running a tool-generated script that carries the
   ``EASYBCI_STEPS:`` header marker (checked by the terminal guard,
   not by this module).

This guard sits at the bypass-prone tool entrypoints (file_tools,
terminal_tool, code_execution_tool). It is unconditional — no model,
no prompt, no --yolo flag can override it.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def detect_work_dir(abs_path: str) -> Optional[Path]:
    """Return the nearest ancestor whose name ends with
    ``_preprocess_work_dir`` or ``_preprocess_work_dirs`` (batch container),
    or None if abs_path is outside any such tree.
    ``abs_path`` may name a file or directory; need not exist.
    """
    try:
        p = Path(abs_path).resolve(strict=False)
    except (OSError, RuntimeError):
        return None
    for parent in (p, *p.parents):
        name = parent.name
        if name.endswith("_preprocess_work_dir") or name.endswith("_preprocess_work_dirs"):
            return parent
    return None


def proposal_confirmed(work_dir: Path) -> bool:
    return (work_dir / "middle_process" / "proposal.confirmed").is_file()


def _is_existing_code_patch(abs_path: str, rel_posix: str) -> bool:
    """True when `rel_posix` targets an existing .py file under code/.

    This allows the agent to patch tool-generated code to fix bugs,
    but NOT to create new files (which would be "reinventing the wheel").
    """
    if not rel_posix.startswith("code/") or not rel_posix.endswith(".py"):
        return False
    return Path(abs_path).exists()


def evaluate(abs_path: str, *, source: str) -> Tuple[bool, Optional[dict]]:
    """Return ``(allowed, denial_payload_or_None)``.

    DEFAULT-DENY model: block ALL writes inside work_dir trees except:
    - ``middle_process/`` (scratch area, always free)
    - After proposal.confirmed: patches to EXISTING ``code/*.py``
    - Outside any work_dir (guard never fires)

    ``source`` ∈ ``{'write_file', 'terminal', 'execute_code', 'patch'}``.
    """
    wd = detect_work_dir(abs_path)
    if wd is None:
        return True, None

    try:
        rel = Path(abs_path).resolve(strict=False).relative_to(wd).as_posix()
    except ValueError:
        return True, None

    # middle_process/ is always free — scratch area for tool internals
    if rel == "middle_process" or rel.startswith("middle_process/"):
        return True, None

    # After proposal.confirmed: allow patches to EXISTING code/*.py
    if proposal_confirmed(wd) and _is_existing_code_patch(abs_path, rel):
        return True, None

    # Everything else: DENIED
    denial = {
        "error": (
            f"workflow_compliance: refusing to write '{rel}' inside "
            f"work_dir '{wd}' via {source}. "
            "The work directory is a sealed container — only EasyBCI's "
            "standard tools can produce content here."
        ),
        "work_dir": str(wd),
        "target": rel,
        "source": source,
        "remediation": [
            "Do NOT write custom scripts, configs, or data files here. "
            "All content is produced by the standard tool chain.",
            "Two workflows exist (load via skill_view('pipeline')):",
            "A) No proven skill → new_pipeline workflow: "
            "inspect_data → deep_inspect → plan_pipeline → propose_pipeline "
            "→ user confirm → generate_code → preprocess_neural → "
            "quality_check → export_repo.",
            "B) Proven skill exists → reference_adaptive workflow: "
            "import_reference (if needed) → batch_process_adaptive "
            "(confirm=false → preview → confirm=true → execute).",
            "middle_process/ is the only scratch area you can write to freely.",
            "After proposal.confirmed, you may patch EXISTING code/*.py "
            "to fix bugs (not create new files).",
        ],
    }
    return False, denial


# Legacy API — kept for callers that use the old three-return-value signature
def is_sensitive_write(
    abs_path: str, *, source: str,
) -> Tuple[bool, Optional[str], Optional[Path]]:
    """Legacy compatibility: returns (is_blocked, reason, work_dir).

    Under default-deny, everything inside work_dir (except middle_process/
    and permitted patches) is sensitive.
    """
    wd = detect_work_dir(abs_path)
    if wd is None:
        return False, None, None
    try:
        rel = Path(abs_path).resolve(strict=False).relative_to(wd).as_posix()
    except ValueError:
        return False, None, wd
    if rel == "middle_process" or rel.startswith("middle_process/"):
        return False, None, wd
    if proposal_confirmed(wd) and _is_existing_code_patch(abs_path, rel):
        return False, None, wd
    return True, f"default-deny: {rel}", wd
