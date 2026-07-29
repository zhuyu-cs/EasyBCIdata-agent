"""Memory-aware execution strategy — decides serial vs parallel processing.

Given a set of data files and their estimated memory footprints, determines
whether they can be processed in parallel (multiple files concurrently) or
must be processed sequentially (one at a time with explicit memory release).

This is used by:
- The batch processor (tools/neural_processing/batch/processor.py) for multi-subject runs
- The codegen module to generate memory-safe pipeline code
- The orchestrator to plan multi-session workflows
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


_MEMORY_BUDGET_RATIO = 0.7
_PIPELINE_OVERHEAD_FACTOR = 8.0  # raw + MNE copy + ICA decomposition + intermediate peaks


@dataclass
class ExecutionStrategy:
    """Result of strategy computation."""

    mode: str  # "parallel", "sequential", or "chunked_sequential"
    max_workers: int
    memory_budget_mb: int
    estimated_per_file_mb: float
    total_estimated_mb: float
    available_mb: float
    reason: str


def _get_available_memory_mb() -> float:
    """Get available system memory in MB, respecting cgroup limits and env override."""
    env_budget = os.environ.get("EASYBCI_MEMORY_BUDGET_MB")
    if env_budget:
        try:
            budget = float(env_budget)
            if budget > 0:
                return budget
        except ValueError:
            pass

    available = None
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    available = int(line.split()[1]) / 1024  # kB -> MB
                    break
    except (OSError, ValueError, IndexError):
        pass

    cgroup_limit_mb = None
    for cg_path in (
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
        "/sys/fs/cgroup/memory.max",
    ):
        try:
            with open(cg_path, encoding="utf-8") as f:
                val = f.read().strip()
                if val != "max" and val.isdigit():
                    cgroup_limit_mb = int(val) / (1024 * 1024)
                    break
        except (OSError, ValueError):
            continue

    if available is None:
        available = 8000.0

    if cgroup_limit_mb and cgroup_limit_mb < available:
        available = cgroup_limit_mb

    return available


def estimate_file_memory_mb(
    filepath: str,
    n_channels: int = 0,
    frequency: float = 0.0,
    duration_s: float = 0.0,
) -> float:
    """Estimate peak memory footprint for processing one file.

    Accounts for the full pipeline overhead: loading + copies for
    before/after comparison + ICA decomposition peaks.
    """
    if n_channels > 0 and frequency > 0 and duration_s > 0:
        raw_bytes = n_channels * frequency * duration_s * 8  # float64
        return (raw_bytes * _PIPELINE_OVERHEAD_FACTOR) / (1024 * 1024)

    try:
        file_size = os.path.getsize(filepath)
        return (file_size * _PIPELINE_OVERHEAD_FACTOR) / (1024 * 1024)
    except OSError:
        return 500.0


def compute_execution_strategy(
    files: List[str],
    n_channels: int = 0,
    frequency: float = 0.0,
    duration_s: float = 0.0,
    max_workers: int = 4,
) -> ExecutionStrategy:
    """Decide whether to process files in parallel or sequentially.

    Decision logic:
    - If one file alone exceeds the budget -> chunked_sequential
    - If all files fit concurrently -> parallel (capped at max_workers)
    - Otherwise -> sequential (one at a time, release memory between)

    Parameters
    ----------
    files : list of str
        Paths to data files to process.
    n_channels, frequency, duration_s :
        If known, used for more accurate memory estimation.
        If 0, falls back to file-size heuristic.
    max_workers : int
        Maximum desired parallelism.

    Returns
    -------
    ExecutionStrategy with mode, max_workers, and reasoning.
    """
    available_mb = _get_available_memory_mb()
    budget_mb = int(available_mb * _MEMORY_BUDGET_RATIO)

    if not files:
        return ExecutionStrategy(
            mode="sequential",
            max_workers=1,
            memory_budget_mb=budget_mb,
            estimated_per_file_mb=0,
            total_estimated_mb=0,
            available_mb=available_mb,
            reason="No files to process",
        )

    per_file_estimates = []
    for f in files:
        est = estimate_file_memory_mb(f, n_channels, frequency, duration_s)
        per_file_estimates.append(est)

    max_per_file = max(per_file_estimates)
    avg_per_file = sum(per_file_estimates) / len(per_file_estimates)
    total_estimated = sum(per_file_estimates)

    # Case 1: single file exceeds budget -> needs chunked processing
    if max_per_file > budget_mb:
        return ExecutionStrategy(
            mode="chunked_sequential",
            max_workers=1,
            memory_budget_mb=budget_mb,
            estimated_per_file_mb=avg_per_file,
            total_estimated_mb=total_estimated,
            available_mb=available_mb,
            reason=(
                f"Single file needs ~{max_per_file:.0f} MB but budget is {budget_mb} MB. "
                f"Must use chunked processing."
            ),
        )

    # Case 2: can we fit N files in parallel?
    safe_workers = max(1, int(budget_mb / max(max_per_file, 1)))
    safe_workers = min(safe_workers, max_workers, len(files), 8)

    if safe_workers >= 2:
        return ExecutionStrategy(
            mode="parallel",
            max_workers=safe_workers,
            memory_budget_mb=budget_mb,
            estimated_per_file_mb=avg_per_file,
            total_estimated_mb=total_estimated,
            available_mb=available_mb,
            reason=(
                f"Each file needs ~{avg_per_file:.0f} MB, budget {budget_mb} MB "
                f"allows {safe_workers} concurrent workers."
            ),
        )

    # Case 3: only one file at a time fits
    return ExecutionStrategy(
        mode="sequential",
        max_workers=1,
        memory_budget_mb=budget_mb,
        estimated_per_file_mb=avg_per_file,
        total_estimated_mb=total_estimated,
        available_mb=available_mb,
        reason=(
            f"Each file needs ~{avg_per_file:.0f} MB, budget {budget_mb} MB "
            f"only fits 1 file at a time. Processing sequentially."
        ),
    )


def format_strategy_report(strategy: ExecutionStrategy) -> str:
    """Format strategy as a human-readable string for agent output."""
    lines = [
        f"Memory Strategy: {strategy.mode.upper()}",
        f"  Available memory: {strategy.available_mb:.0f} MB",
        f"  Budget (80%):     {strategy.memory_budget_mb} MB",
        f"  Per-file estimate: {strategy.estimated_per_file_mb:.0f} MB",
        f"  Max workers:      {strategy.max_workers}",
        f"  Reason: {strategy.reason}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Sampling decision — used by inspect to avoid OOM on large directories
# ---------------------------------------------------------------------------

_SAMPLING_FILE_THRESHOLD = 10


def should_use_sampling(
    files: List[str],
    n_channels: int = 0,
    frequency: float = 0.0,
    duration_s: float = 0.0,
) -> bool:
    """Decide whether inspect should use sampling mode for a file list.

    Returns True when inspecting all files would likely exceed memory or
    when the file count alone exceeds the threshold (even conservative
    header reads at scale cause excessive I/O).
    """
    if len(files) <= _SAMPLING_FILE_THRESHOLD:
        return False

    # Quick check: many files → always sample regardless of memory
    if len(files) > 50:
        return True

    # Memory-based check
    available = _get_available_memory_mb()
    budget = available * _MEMORY_BUDGET_RATIO

    # For inspect, overhead is lower (header only ≈ 0.5x file peak)
    _INSPECT_OVERHEAD = 0.5
    total_est = 0.0
    for f in files[:20]:  # Only estimate first 20 to avoid slow stat calls
        est = estimate_file_memory_mb(f, n_channels, frequency, duration_s)
        total_est += est * _INSPECT_OVERHEAD

    # Extrapolate for remaining files
    if len(files) > 20:
        avg_est = total_est / 20
        total_est = avg_est * len(files)

    return total_est > budget
