"""Best-effort ruff lint pass on agent-generated files.

Runs `ruff check --fix --select I,PLW1514` on a single generated file
(typically the mini-repo's `pipeline.py`) so the exported code has sorted
imports and explicit text-mode `encoding="utf-8"`.

`I` is enforced *only* on agent-generated files (not on the existing
codebase); see `pyproject.toml` `[tool.ruff.lint]` for the rationale.

Linting must NEVER break export — every error path returns ``None`` and
the caller is expected to ignore failures silently. ruff is a dev-only
extra; if absent, this is a no-op.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Optional


def ruff_fix(path: str | Path) -> Optional[str]:
    """Run `ruff check --fix --select I,PLW1514 <path>` best-effort.

    Returns ruff's textual diagnostic (remaining issues after auto-fix) when
    non-empty, or ``None`` when the file is clean, ruff is unavailable, or
    the subprocess fails for any reason.

    Auto-fixable findings (e.g. unsorted imports) are corrected in place;
    only un-fixable remainders surface in the return value.
    """
    ruff_bin = shutil.which("ruff")
    if ruff_bin is None:
        return None
    try:
        proc = subprocess.run(
            [ruff_bin, "check", "--fix", "--select", "I,PLW1514", str(path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    diagnostic = (proc.stdout or "").strip()
    return diagnostic or None
