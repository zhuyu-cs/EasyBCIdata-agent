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
# Recipes WITHOUT ICA only ever hold ~2 working copies (float32 base + a float64
# mne transient); 8x falsely excludes files that measurably fit. Drop to 3x when
# no ICA step is present. Mirrors batch/orchestrate.py:_oom_excluded's measured
# calibration (261ch/2000Hz/4h decimated peaks ~14 GB, real ~1.9x).
_NO_ICA_OVERHEAD_FACTOR = 3.0


def _available_cpu_count() -> int:
    """Usable CPU cores, cgroup/cpuset-aware.

    Order: EASYBCI_MAX_WORKERS env override → os.sched_getaffinity(0) (respects
    cpuset/cgroup pinning) → os.cpu_count(). Never returns < 1. No CPU detection
    existed anywhere in the repo before this; keep it the single source.
    """
    env_override = os.environ.get("EASYBCI_MAX_WORKERS")
    if env_override:
        try:
            n = int(env_override)
            if n > 0:
                return n
        except ValueError:
            pass
    try:
        # sched_getaffinity is Linux-only but reflects cpuset/taskset pinning.
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def estimate_peak_mb(
    *,
    n_channels: int,
    frequency: float,
    duration_s: float,
    has_ica: bool,
    target_hz: Optional[float] = None,
) -> float:
    """Authoritative peak-processing-footprint estimate (MB) for one recording.

    Single source of truth shared by the single-file admission guard
    (batch/orchestrate.py:_oom_excluded), the batch scheduler
    (compute_strategy_from_peaks), and the in-pipeline global gate injected into
    generated code. The pipeline is float32 end to end (4 bytes); a transient
    float64 mne copy plus ICA's eigendecomposition drive the overhead factor.

    ``target_hz`` reflects load-time decimation: the loader decimates on the fly
    (io/nk_backend._read_decimated_uV), so a recording resampled below its native
    rate peaks proportionally lower. When target_hz < frequency, budget for the
    decimated rate.

    Returns 0.0 when metadata is insufficient (caller falls back to a size-based
    or preload_full_mb estimate).
    """
    n_ch = int(n_channels or 0)
    fs = float(frequency or 0.0)
    dur = float(duration_s or 0.0)
    if n_ch <= 0 or fs <= 0 or dur <= 0:
        return 0.0
    eff_fs = fs
    if target_hz and target_hz > 0 and target_hz < fs:
        eff_fs = float(target_hz)
    overhead = _PIPELINE_OVERHEAD_FACTOR if has_ica else _NO_ICA_OVERHEAD_FACTOR
    return (n_ch * eff_fs * dur * 4 * overhead) / (1024 * 1024)  # float32


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
    """Estimate peak memory footprint for processing one file (ICA worst case).

    Back-compat wrapper over ``estimate_peak_mb`` — assumes ICA is present so
    existing callers keep the conservative 8x budget. Callers that know the
    recipe should prefer ``estimate_peak_mb(has_ica=...)`` directly.
    """
    if n_channels > 0 and frequency > 0 and duration_s > 0:
        return estimate_peak_mb(
            n_channels=n_channels, frequency=frequency,
            duration_s=duration_s, has_ica=True,
        )

    try:
        file_size = os.path.getsize(filepath)
        return (file_size * _PIPELINE_OVERHEAD_FACTOR) / (1024 * 1024)
    except OSError:
        return 500.0


def safe_max_duration_s(
    n_channels: int,
    frequency: float,
    total_duration_s: float,
    overhead_factor: float = _PIPELINE_OVERHEAD_FACTOR,
    budget_ratio: float = _MEMORY_BUDGET_RATIO,
) -> Optional[float]:
    """How many seconds of a recording fit within the memory budget.

    Returns None when the full recording fits (no crop needed). Otherwise
    returns the number of seconds (from the start) whose peak processing
    footprint stays within ``available_memory * budget_ratio``. Used by the
    adaptive batch path to refuse/crop a recording before it OOMs the host.

    A degenerate result (missing metadata, or budget too small for even 1 s)
    returns None so callers fall back to their own guard instead of loading 0
    samples.
    """
    if n_channels <= 0 or frequency <= 0 or total_duration_s <= 0:
        return None

    budget_mb = _get_available_memory_mb() * budget_ratio
    bytes_per_second = n_channels * frequency * 8 * overhead_factor  # float64 peak
    peak_full_mb = (bytes_per_second * total_duration_s) / (1024 * 1024)
    if peak_full_mb <= budget_mb:
        return None  # whole file fits

    budget_bytes = budget_mb * 1024 * 1024
    seconds = budget_bytes / bytes_per_second if bytes_per_second else 0.0
    if seconds < 1.0:
        return None  # can't even fit 1 s — let caller skip, not load nothing
    return min(seconds, total_duration_s)



def compute_strategy_from_peaks(
    peaks_mb: List[float],
    budget_mb: Optional[float] = None,
    cpu_count: Optional[int] = None,
    available_mb: Optional[float] = None,
) -> ExecutionStrategy:
    """Decide parallel vs serial from per-file peak footprints (HETEROGENEOUS).

    This is the data-review-driven, hardware-aware core: instead of assuming all
    files share one footprint, it takes each file's own recipe-aware
    ``estimate_peak_mb`` (recorded on its routing entry) and finds the largest
    worker count that is safe under *any* concurrent scheduling.

    Safe-worker rule
    ----------------
    The worst case for N concurrent workers is the N *largest* files running at
    once. So the safe worker count is the largest N whose top-N peaks still sum
    within ``budget_mb`` — a big file collapses N toward 1 (serial) while a batch
    of small files keeps N high (parallel). N is then capped by ``cpu_count``
    (hardware) and the file count. This makes serial-vs-parallel an emergent
    property of the actual data + hardware, never a hardcoded single path.

    Parameters
    ----------
    peaks_mb : per-file peak-processing estimates (MB), any order. Zeros/negatives
        (thin metadata) are treated as an unknown-but-nonzero footprint by
        substituting the batch's max known peak, so they never appear "free".
    budget_mb : memory ceiling; defaults to ``available_mb * _MEMORY_BUDGET_RATIO``.
    cpu_count : hardware parallelism cap; defaults to ``_available_cpu_count()``.
    """
    if available_mb is None:
        available_mb = _get_available_memory_mb()
    if budget_mb is None:
        budget_mb = available_mb * _MEMORY_BUDGET_RATIO
    if cpu_count is None:
        cpu_count = _available_cpu_count()
    budget_mb = float(budget_mb)
    cpu_count = max(1, int(cpu_count))
    budget_int = int(budget_mb)

    if not peaks_mb:
        return ExecutionStrategy(
            mode="sequential", max_workers=1, memory_budget_mb=budget_int,
            estimated_per_file_mb=0, total_estimated_mb=0,
            available_mb=available_mb, reason="No files to process",
        )

    # Unknown footprints (<=0) are pessimistically treated as the batch max so a
    # thin-metadata file can never inflate the safe worker count.
    known_max = max((p for p in peaks_mb if p and p > 0), default=0.0)
    peaks = sorted(((p if p and p > 0 else known_max) for p in peaks_mb),
                   reverse=True)
    n_files = len(peaks)
    max_per_file = peaks[0]
    total_estimated = float(sum(peaks))
    avg_per_file = total_estimated / n_files

    # Case 1: the single largest file alone blows the budget. Layer A
    # (_oom_excluded) should have dropped it; be defensive and force chunking.
    if max_per_file > budget_mb:
        return ExecutionStrategy(
            mode="chunked_sequential", max_workers=1, memory_budget_mb=budget_int,
            estimated_per_file_mb=avg_per_file, total_estimated_mb=total_estimated,
            available_mb=available_mb,
            reason=(
                f"Largest file needs ~{max_per_file:.0f} MB but budget is "
                f"{budget_int} MB. Must chunk within the file."
            ),
        )

    # Largest N whose top-N peaks sum within budget (prefix sum on the
    # descending list). This is the max concurrency safe for ANY subset.
    safe_n, running = 0, 0.0
    for p in peaks:
        if running + p > budget_mb:
            break
        running += p
        safe_n += 1
    safe_workers = max(1, min(safe_n, cpu_count, n_files))

    if safe_workers >= 2:
        return ExecutionStrategy(
            mode="parallel", max_workers=safe_workers, memory_budget_mb=budget_int,
            estimated_per_file_mb=avg_per_file, total_estimated_mb=total_estimated,
            available_mb=available_mb,
            reason=(
                f"peaks up to ~{max_per_file:.0f} MB, budget {budget_int} MB, "
                f"{cpu_count} CPU → top-{safe_workers} fit concurrently."
            ),
        )

    return ExecutionStrategy(
        mode="sequential", max_workers=1, memory_budget_mb=budget_int,
        estimated_per_file_mb=avg_per_file, total_estimated_mb=total_estimated,
        available_mb=available_mb,
        reason=(
            f"largest file ~{max_per_file:.0f} MB vs budget {budget_int} MB "
            f"only fits 1 at a time — serial."
        ),
    )


def compute_execution_strategy(
    files: List[str],
    n_channels: int = 0,
    frequency: float = 0.0,
    duration_s: float = 0.0,
    max_workers: int = 4,
) -> ExecutionStrategy:
    """Homogeneous back-compat wrapper over :func:`compute_strategy_from_peaks`.

    Assumes every file shares the same ``(n_channels, frequency, duration_s)``
    footprint (ICA worst case via ``estimate_file_memory_mb``). Prefer
    ``compute_strategy_from_peaks`` with per-file peaks when they are known.
    ``max_workers`` is folded into the CPU cap.
    """
    if not files:
        return compute_strategy_from_peaks([])
    peaks = [estimate_file_memory_mb(f, n_channels, frequency, duration_s)
             for f in files]
    cpu_cap = min(_available_cpu_count(), max(1, max_workers))
    return compute_strategy_from_peaks(peaks, cpu_count=cpu_cap)


def format_strategy_report(strategy: ExecutionStrategy) -> str:
    """Format strategy as a human-readable string for agent output."""
    lines = [
        f"Memory Strategy: {strategy.mode.upper()}",
        f"  Available memory: {strategy.available_mb:.0f} MB",
        f"  Budget ({_MEMORY_BUDGET_RATIO:.0%}):     {strategy.memory_budget_mb} MB",
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
