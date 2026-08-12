"""Subprocess driver for generated pipeline / build_ai_ready / qc scripts.

Private helper for ``_handle_preprocess_neural`` / ``_handle_save_processed`` /
``_handle_quality_check``. Not registered as an agent-callable tool.

Contract:

* ``run_script(work_dir, stage, input_path=None, timeout)`` runs the generated
  ``code/<stage>.py``.
    - When ``input_path`` is None (multi-input mode) → ``python <stage>.py <work_dir>``.
      The script reads ``middle_process/inputs_routing.json`` and loops over inputs.
    - When ``input_path`` is given (legacy single-file mode) → ``python <stage>.py
      <input_path> <work_dir>``. Script processes that one file.
* On retcode == 0, the script writes either
  ``<work_dir>/middle_process/<stage>_status.json`` (single) or
  ``<stage>_status_aggregate.json`` (multi). Both are loaded into ``result["status"]``;
  the multi-input aggregate is preferred when present.
* On retcode != 0 (including timeout), parse the tail of stderr into a
  structured traceback dict and archive the offending script to
  ``<work_dir>/middle_process/code/<stage>_failed_<ts>.py``. The live script at
  ``code/<stage>.py`` is NOT removed — the agent edits it and we re-run.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Literal, Optional

logger = logging.getLogger(__name__)

Stage = Literal["pipeline", "ai_ready", "build_ai_ready", "qc", "vis"]
TAIL_BYTES = 4096


def _subprocess_env() -> Dict[str, str]:
    """Build the env passed to generated-script subprocesses.

    Generated ``pipeline.py`` / ``qc.py`` import ``from easybci_lib...``.
    When the agent's interpreter has ``easybci_lib`` installed only via an
    editable .pth (or via PYTHONPATH from the parent shell) and the
    subprocess inherits a different PYTHONPATH or none at all, those
    imports raise ``ModuleNotFoundError``. We resolve ``easybci_lib``'s
    parent directory at runtime (it must be importable here — the agent
    is using it) and prepend it to ``PYTHONPATH`` so the child sees it
    regardless of how its interpreter was originally provisioned.
    """
    env = os.environ.copy()
    try:
        import easybci_lib
        pkg_parent = str(Path(easybci_lib.__file__).resolve().parent.parent)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Could not locate easybci_lib for PYTHONPATH propagation: %s", exc)
        return env
    existing = env.get("PYTHONPATH", "")
    parts = [pkg_parent]
    if existing:
        parts.extend(p for p in existing.split(os.pathsep) if p and p != pkg_parent)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


_TRACEBACK_LINE = re.compile(
    r'^\s*File "(?P<file>[^"]+)", line (?P<line>\d+)', re.MULTILINE
)
_EXCEPTION_LINE = re.compile(
    r'^(?P<type>[A-Za-z_][A-Za-z0-9_.]*):\s*(?P<msg>.*)$', re.MULTILINE
)


def _tail(s: str, n: int = TAIL_BYTES) -> str:
    return s[-n:] if len(s) > n else s


def _classify(error_type: str, message: str) -> str:
    et = error_type or ""
    msg = (message or "").lower()
    if et in ("ImportError", "ModuleNotFoundError"):
        return "import_error"
    if et == "AttributeError":
        return "attribute_error"
    if "could not broadcast" in msg:
        return "shape_mismatch"
    if "shape" in msg and "mismatch" in msg:
        return "shape_mismatch"
    if "shape" in msg and ("into shape" in msg or "expected" in msg):
        return "shape_mismatch"
    if et == "FileNotFoundError" or "no such file" in msg:
        return "dependency_missing"
    if et in ("ValueError", "TypeError", "KeyError", "IndexError"):
        return "value_error"
    return "other"


def _parse_traceback(stderr: str, stage_script: Path) -> Dict[str, Any]:
    """Best-effort Python traceback parser. Returns last-frame info."""
    last_frame = None
    for m in _TRACEBACK_LINE.finditer(stderr):
        last_frame = m
    exc_match = None
    for m in _EXCEPTION_LINE.finditer(stderr):
        # Filter out "warnings:" / "Note:" style lines — only `XxxError:` matches.
        type_lower = m.group("type").lower()
        if "error" in type_lower or "exception" in type_lower:
            exc_match = m

    error_type = exc_match.group("type") if exc_match else "Unknown"
    if exc_match:
        error_message = exc_match.group("msg").strip()
    elif stderr.strip():
        error_message = stderr.splitlines()[-1]
    else:
        error_message = ""

    if last_frame:
        file_path = last_frame.group("file")
        line_no = int(last_frame.group("line"))
    else:
        file_path = str(stage_script)
        line_no = 0

    snippet = ""
    try:
        if Path(file_path).is_file():
            lines = Path(file_path).read_text(encoding="utf-8").splitlines()
            if 1 <= line_no <= len(lines):
                snippet = lines[line_no - 1]
    except OSError:
        pass

    return {
        "file": file_path,
        "line": line_no,
        "error_type": error_type,
        "error_message": error_message,
        "snippet": snippet,
        "suggestion_kind": _classify(error_type, error_message),
    }


def _archive_failed(work_dir: Path, stage: Stage) -> Optional[Path]:
    src = work_dir / "code" / f"{stage}.py"
    if not src.exists():
        return None
    dest_dir = work_dir / "middle_process" / "code"
    dest_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = dest_dir / f"{stage}_failed_{ts}.py"
    try:
        shutil.copy2(str(src), str(dest))
        return dest
    except OSError as exc:
        logger.warning("Failed to archive %s: %s", src, exc)
        return None


def _load_status(work_dir: Path, stage: Stage) -> Optional[Dict[str, Any]]:
    """Read the stage's status sidecar.

    Multi-input runs write ``<stage>_status_aggregate.json`` (preferred when
    present); single-input runs write ``<stage>_status.json``. When both exist
    the aggregate wins — it carries the per-file breakdown.
    """
    mp = work_dir / "middle_process"
    aggregate_name = {
        "pipeline": "pipeline_status_aggregate.json",
        "qc": "qc_status.json",                          # qc.py aggregates into the canonical file
        "build_ai_ready": "build_ai_ready_status.json",
        "ai_ready": "build_ai_ready_status.json",
        "vis": "vis_status.json",
    }.get(stage)
    candidates = []
    if aggregate_name:
        candidates.append(mp / aggregate_name)
    candidates.append(mp / f"{stage}_status.json")

    for sidecar in candidates:
        if not sidecar.is_file():
            continue
        try:
            return json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to read status sidecar %s: %s", sidecar, exc)
    return None


def run_script(
    *,
    work_dir: str,
    stage: Stage,
    input_path: Optional[str] = None,
    timeout: Optional[int] = None,
) -> Dict[str, Any]:
    """Run the generated stage script and return a structured result.

    Multi-input mode: ``input_path=None`` → ``python <stage>.py <work_dir>``.
    Script loops over ``middle_process/inputs_routing.json``.

    Legacy single-file mode: ``input_path=<raw>`` → ``python <stage>.py
    <raw> <work_dir>``. Script processes that one file.

    ``timeout`` is a wall-clock cap on the subprocess. ``None`` / ``<=0`` →
    unlimited (aligns with the gateway's ``EASYBCI_AGENT_TIMEOUT=0`` default;
    long BCI batches must not be hard-killed). Set ``EASYBCI_SCRIPT_TIMEOUT_MAX``
    for an environment-wide ceiling.
    """
    if timeout is not None and timeout <= 0:
        timeout = None
    _env_cap = os.environ.get("EASYBCI_SCRIPT_TIMEOUT_MAX")
    if _env_cap:
        try:
            cap = int(_env_cap)
            if cap > 0 and (timeout is None or timeout > cap):
                timeout = cap
        except ValueError:
            pass

    wd = Path(work_dir)
    script = wd / "code" / f"{stage}.py"

    if not script.exists():
        return {
            "ok": False,
            "stage": stage,
            "retcode": -1,
            "stdout_tail": "",
            "stderr_tail": f"Script not found: {script}",
            "traceback": {
                "file": str(script),
                "line": 0,
                "error_type": "FileNotFoundError",
                "error_message": f"Script not found: {script}",
                "snippet": "",
                "suggestion_kind": "dependency_missing",
            },
            "archived_to": None,
        }

    # Pre-execution AST safety scan: catch writes targeting protected source.
    # Re-raises CodegenSafetyViolation so the agent can rewrite the script.
    try:
        from easybci_lib.tools.neural_processing.codegen.safety_scan import (
            scan_script as _safety_scan_script,
        )
        _safety_scan_script(str(script), work_dir=str(wd))
    except ImportError:
        pass

    if input_path is None:
        cmd = [sys.executable, str(script), str(wd)]
    else:
        cmd = [sys.executable, str(script), str(input_path), str(wd)]
    timed_out = False
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            cwd=str(wd), check=False, env=_subprocess_env(),
        )
        stdout, stderr, retcode = proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        stdout = exc.stdout or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        stderr = exc.stderr or ""
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        stderr += f"\n[script_runner] Timed out after {timeout}s"
        retcode = -9

    if retcode == 0 and not timed_out:
        return {
            "ok": True,
            "stage": stage,
            "retcode": 0,
            "stdout_tail": _tail(stdout),
            "stderr_tail": _tail(stderr),
            "status": _load_status(wd, stage),
        }

    tb = _parse_traceback(stderr, script)
    if timed_out:
        tb["error_type"] = "TimeoutError"
        tb["error_message"] = f"Script exceeded {timeout}s wall time"
        tb["suggestion_kind"] = "timeout"
    archived = _archive_failed(wd, stage)
    return {
        "ok": False,
        "stage": stage,
        "retcode": retcode,
        "stdout_tail": _tail(stdout),
        "stderr_tail": _tail(stderr),
        "traceback": tb,
        "archived_to": str(archived) if archived else None,
    }


__all__ = ["run_script", "Stage"]
