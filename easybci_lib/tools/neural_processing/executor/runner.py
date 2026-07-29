"""Code executor — subprocess-based Python execution with output capture.

Runs generated code (pipeline.py, new skill drafts) in a subprocess and
captures stdout/stderr for validation. Streams output lines via callback
for real-time display in the Web UI.

Security model:
- Runs on the same machine as the agent (research tool, not multi-tenant SaaS)
- Subprocess isolation via Python's subprocess module
- Timeout enforced (default 120s, configurable)
- Working directory isolated to output dir
- No network access restriction (needed for pip imports)
- Source data integrity verified after execution (mtime/size guard)

For production multi-user deployment, this should be replaced with Docker
container execution or a proper sandbox (gVisor, nsjail).
"""

import logging
import os
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

_MEMORY_BUDGET_RATIO = 0.8  # Use 80% of available memory


def _get_available_memory_bytes() -> int:
    """Get available system memory in bytes, accounting for cgroup limits."""
    available = None
    # Try /proc/meminfo first
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    available = int(line.split()[1]) * 1024  # kB → bytes
                    break
    except (OSError, ValueError, IndexError):
        pass

    # Check cgroup memory limit (may be lower than physical)
    cgroup_limit = None
    for cg_path in (
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
        "/sys/fs/cgroup/memory.max",
    ):
        try:
            with open(cg_path, encoding="utf-8") as f:
                val = f.read().strip()
                if val != "max" and val.isdigit():
                    cgroup_limit = int(val)
                    break
        except (OSError, ValueError):
            continue

    if available is None:
        available = 8 * 1024 * 1024 * 1024  # fallback 8GB

    # Effective limit is the lesser of available and cgroup cap
    if cgroup_limit and cgroup_limit < available:
        available = cgroup_limit

    return available


def _get_memory_budget_bytes() -> int:
    """Return the memory budget (80% of available) for subprocess execution."""
    return int(_get_available_memory_bytes() * _MEMORY_BUDGET_RATIO)


def _make_preexec_fn(memory_limit_bytes: int):
    """Create a preexec_fn that sets RLIMIT_AS on the child process."""
    def _set_limits():
        try:
            resource.setrlimit(resource.RLIMIT_AS, (memory_limit_bytes, memory_limit_bytes))
        except (ValueError, resource.error):
            pass  # Non-fatal: some systems don't support RLIMIT_AS
    return _set_limits


def execute_code(
    code: str,
    timeout: float = 120.0,
    working_dir: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    output_callback: Optional[Callable[[str, str], None]] = None,
) -> Dict[str, Any]:
    """Execute Python code in a subprocess.

    Parameters
    ----------
    code : str
        Python source code to execute.
    timeout : float
        Maximum execution time in seconds (default: 120).
    working_dir : str, optional
        Working directory for the subprocess. If None, uses a temp dir.
    env : dict, optional
        Extra environment variables to set.
    output_callback : callable, optional
        Called with (stream, line) for each output line.
        stream is "stdout" or "stderr".

    Returns
    -------
    dict with keys:
        success : bool
        stdout : str
        stderr : str
        returncode : int
        duration : float (seconds)
        error : str (if exception)
    """
    use_tempdir = working_dir is None
    if use_tempdir:
        tmpdir = tempfile.mkdtemp(prefix="easybci_exec_")
        working_dir = tmpdir

    Path(working_dir).mkdir(parents=True, exist_ok=True)

    # Write code to a temp file in the working directory
    script_path = os.path.join(working_dir, "_exec_script.py")
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(code)

    return execute_file(
        script_path,
        timeout=timeout,
        working_dir=working_dir,
        env=env,
        output_callback=output_callback,
    )


def execute_file(
    filepath: str,
    args: Optional[List[str]] = None,
    timeout: float = 120.0,
    working_dir: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    output_callback: Optional[Callable[[str, str], None]] = None,
) -> Dict[str, Any]:
    """Execute a Python file in a subprocess.

    Parameters
    ----------
    filepath : str
        Path to the Python script to execute.
    args : list of str, optional
        Command-line arguments to pass to the script.
    timeout : float
        Maximum execution time in seconds.
    working_dir : str, optional
        Working directory. Defaults to the script's parent directory.
    env : dict, optional
        Extra environment variables.
    output_callback : callable, optional
        Called with (stream, line) for each output line.

    Returns
    -------
    dict with keys: success, stdout, stderr, returncode, duration, error
    """
    if not os.path.exists(filepath):
        return {
            "success": False,
            "stdout": "",
            "stderr": "",
            "returncode": -1,
            "duration": 0.0,
            "error": f"File not found: {filepath}",
        }

    if working_dir is None:
        working_dir = str(Path(filepath).parent)

    cmd = [sys.executable, filepath] + (args or [])

    proc_env = os.environ.copy()
    if env:
        proc_env.update(env)

    # Inject memory budget as environment variable for subprocess awareness
    memory_budget = _get_memory_budget_bytes()
    memory_budget_mb = memory_budget // (1024 * 1024)
    proc_env["EASYBCI_MEMORY_BUDGET_MB"] = str(memory_budget_mb)

    stdout_lines = []
    stderr_lines = []
    start_time = time.time()

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=working_dir,
            env=proc_env,
            text=True,
            bufsize=1,
            preexec_fn=_make_preexec_fn(memory_budget),
        )

        # Read output with timeout
        import selectors
        sel = selectors.DefaultSelector()
        sel.register(proc.stdout, selectors.EVENT_READ)
        sel.register(proc.stderr, selectors.EVENT_READ)

        while proc.poll() is None:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                proc.kill()
                proc.wait()
                return {
                    "success": False,
                    "stdout": "".join(stdout_lines),
                    "stderr": "".join(stderr_lines) + f"\n[TIMEOUT after {timeout}s]",
                    "returncode": -9,
                    "duration": elapsed,
                    "error": f"Execution timed out after {timeout}s",
                }

            remaining = timeout - elapsed
            events = sel.select(timeout=min(0.1, remaining))
            for key, _ in events:
                line = key.fileobj.readline()
                if not line:
                    continue
                if key.fileobj is proc.stdout:
                    stdout_lines.append(line)
                    if output_callback:
                        output_callback("stdout", line.rstrip("\n"))
                else:
                    stderr_lines.append(line)
                    if output_callback:
                        output_callback("stderr", line.rstrip("\n"))

        # Read remaining output after process exits
        remaining_stdout = proc.stdout.read()
        remaining_stderr = proc.stderr.read()
        if remaining_stdout:
            for line in remaining_stdout.splitlines(keepends=True):
                stdout_lines.append(line)
                if output_callback:
                    output_callback("stdout", line.rstrip("\n"))
        if remaining_stderr:
            for line in remaining_stderr.splitlines(keepends=True):
                stderr_lines.append(line)
                if output_callback:
                    output_callback("stderr", line.rstrip("\n"))

        sel.close()

        duration = time.time() - start_time
        returncode = proc.returncode

        # Detect OOM kill (SIGKILL = 137, or -9 from signal)
        if returncode in (137, -9):
            available_mb = _get_available_memory_bytes() // (1024 * 1024)
            oom_msg = (
                f"Process killed (exit code {returncode}) — likely out of memory. "
                f"Memory budget was {memory_budget_mb} MB "
                f"(80% of {available_mb} MB available). "
                f"Consider using chunked processing for large datasets, "
                f"or reducing the number of data copies held simultaneously."
            )
            return {
                "success": False,
                "stdout": "".join(stdout_lines),
                "stderr": "".join(stderr_lines) + f"\n[OOM] {oom_msg}",
                "returncode": returncode,
                "duration": round(duration, 2),
                "error": oom_msg,
                "oom": True,
                "memory_budget_mb": memory_budget_mb,
            }

        return {
            "success": returncode == 0,
            "stdout": "".join(stdout_lines),
            "stderr": "".join(stderr_lines),
            "returncode": returncode,
            "duration": round(duration, 2),
            "error": "" if returncode == 0 else f"Process exited with code {returncode}",
        }

    except Exception as e:
        duration = time.time() - start_time
        return {
            "success": False,
            "stdout": "".join(stdout_lines),
            "stderr": "".join(stderr_lines),
            "returncode": -1,
            "duration": round(duration, 2),
            "error": str(e),
        }


def execute_code_guarded(
    code: str,
    timeout: float = 120.0,
    working_dir: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    output_callback: Optional[Callable[[str, str], None]] = None,
) -> Dict[str, Any]:
    """Execute code with source data integrity verification.

    Wraps execute_code with pre/post mtime checks on all registered source
    data files. If any source file is modified during execution, the result
    is marked as a critical integrity violation.
    """
    from easybci_agent.source_data_guard import snapshot_source_files, verify_source_integrity

    before = snapshot_source_files()

    result = execute_code(
        code=code,
        timeout=timeout,
        working_dir=working_dir,
        env=env,
        output_callback=output_callback,
    )

    if before:
        violations = verify_source_integrity(before)
        if violations:
            violation_msg = "\n".join(violations)
            logger.critical(
                "SOURCE DATA INTEGRITY VIOLATION during code execution:\n%s",
                violation_msg,
            )
            result["source_data_violations"] = violations
            result["stderr"] = (
                (result.get("stderr") or "")
                + f"\n\n*** CRITICAL: SOURCE DATA INTEGRITY VIOLATION ***\n{violation_msg}\n"
            )

    return result
