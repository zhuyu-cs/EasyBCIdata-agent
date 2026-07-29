"""Environment isolation advisor — recommends execution environment.

Analyzes data characteristics and available resources to recommend:
- Local: default for small/medium data on capable machines
- Docker/Singularity: for untrusted code or when isolation is needed
- HPC (SLURM/PBS): for multi-subject batch on cluster
- GPU: when CUDA is available and operations benefit from acceleration

Also provides SLURM/PBS job script generation for HPC batch submission.
"""

import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class EnvironmentRecommendation:
    """Recommended execution environment with reasoning."""
    recommended: str = "local"  # "local", "docker", "singularity", "ssh_hpc", "gpu"
    reason: str = ""
    alternatives: List[str] = field(default_factory=list)
    gpu_available: bool = False
    docker_available: bool = False
    singularity_available: bool = False
    hpc_detected: bool = False
    estimated_memory_mb: float = 0.0
    data_size_mb: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recommended": self.recommended,
            "reason": self.reason,
            "alternatives": self.alternatives,
            "capabilities": {
                "gpu_available": self.gpu_available,
                "docker_available": self.docker_available,
                "singularity_available": self.singularity_available,
                "hpc_detected": self.hpc_detected,
            },
            "data_size_mb": round(self.data_size_mb, 1),
            "estimated_memory_mb": round(self.estimated_memory_mb, 1),
        }


def recommend_environment(
    data_size_mb: float = 0.0,
    n_subjects: int = 1,
    has_custom_code: bool = False,
    operation: str = "preprocess",
) -> EnvironmentRecommendation:
    """Recommend the best execution environment based on context.

    Parameters
    ----------
    data_size_mb : float
        Total data size in MB.
    n_subjects : int
        Number of subjects (for batch processing).
    has_custom_code : bool
        Whether user-supplied code will be executed.
    operation : str
        Type of operation: "preprocess", "feature_extract", "train", "batch".

    Returns
    -------
    EnvironmentRecommendation with reasoning.
    """
    gpu = detect_gpu()
    docker = detect_docker()
    singularity = detect_singularity()
    hpc = detect_hpc_scheduler()
    available_mem = _get_available_memory_mb()

    estimated_mem = data_size_mb * 3  # processing overhead ~3x

    rec = EnvironmentRecommendation(
        gpu_available=gpu,
        docker_available=docker,
        singularity_available=singularity,
        hpc_detected=hpc,
        estimated_memory_mb=estimated_mem,
        data_size_mb=data_size_mb,
    )

    # Decision logic

    # 1. HPC batch: many subjects + cluster available
    if n_subjects >= 5 and hpc:
        rec.recommended = "ssh_hpc"
        rec.reason = (
            f"Batch processing {n_subjects} subjects — HPC scheduler detected. "
            f"Recommend submitting as a SLURM/PBS array job for parallel execution."
        )
        rec.alternatives = ["local", "docker"] if docker else ["local"]
        return rec

    # 2. Large data + insufficient memory → container or cluster
    if estimated_mem > available_mem * 0.7:
        if hpc:
            rec.recommended = "ssh_hpc"
            rec.reason = (
                f"Data requires ~{estimated_mem:.0f} MB but only {available_mem:.0f} MB available. "
                f"HPC node likely has more memory."
            )
        elif singularity:
            rec.recommended = "singularity"
            rec.reason = (
                f"Data requires ~{estimated_mem:.0f} MB — using Singularity for memory isolation."
            )
        else:
            rec.recommended = "local"
            rec.reason = (
                f"Data is large (~{data_size_mb:.0f} MB) but no container/HPC available. "
                f"Consider chunked processing mode."
            )
        rec.alternatives = ["docker"] if docker else []
        return rec

    # 3. Untrusted custom code → container
    if has_custom_code:
        if docker:
            rec.recommended = "docker"
            rec.reason = "Custom user code detected — using Docker for isolation."
            rec.alternatives = ["singularity"] if singularity else ["local"]
        elif singularity:
            rec.recommended = "singularity"
            rec.reason = "Custom user code detected — using Singularity for isolation."
            rec.alternatives = ["local"]
        else:
            rec.recommended = "local"
            rec.reason = "Custom code but no container runtime available. Proceeding locally."
        return rec

    # 4. GPU-accelerated operations
    if gpu and operation in ("train", "feature_extract"):
        rec.recommended = "gpu"
        rec.reason = (
            f"CUDA GPU detected — using GPU acceleration for {operation}."
        )
        rec.alternatives = ["local"]
        return rec

    # 5. Default: local
    rec.recommended = "local"
    rec.reason = "Standard processing — local environment is sufficient."
    if docker:
        rec.alternatives.append("docker")
    if gpu:
        rec.alternatives.append("gpu")

    return rec


# --- Detection functions ---

def detect_gpu() -> bool:
    """Check if CUDA GPU is available."""
    # Check nvidia-smi
    if shutil.which("nvidia-smi"):
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                return True
        except (subprocess.TimeoutExpired, OSError):
            pass

    # Check CUDA_VISIBLE_DEVICES
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
        return True

    # Check for /dev/nvidia*
    if Path("/dev/nvidia0").exists():
        return True

    return False


def detect_docker() -> bool:
    """Check if Docker is available and running."""
    if not shutil.which("docker"):
        return False
    try:
        import subprocess
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True, text=True, timeout=5,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def detect_singularity() -> bool:
    """Check if Singularity/Apptainer is available."""
    return bool(shutil.which("singularity") or shutil.which("apptainer"))


def detect_hpc_scheduler() -> str:
    """Detect HPC job scheduler type. Returns 'slurm', 'pbs', or ''."""
    if shutil.which("sbatch") or os.environ.get("SLURM_JOB_ID"):
        return "slurm"
    if shutil.which("qsub") or os.environ.get("PBS_JOBID"):
        return "pbs"
    return ""


# --- HPC Job Script Generation ---

def generate_slurm_script(
    data_pattern: str,
    steps: List[str],
    output_dir: str,
    n_subjects: int = 1,
    partition: str = "default",
    time_limit: str = "02:00:00",
    memory_gb: int = 16,
    cpus_per_task: int = 4,
    conda_env: str = "agent",
    module_loads: Optional[List[str]] = None,
) -> str:
    """Generate a SLURM array job script for batch neural processing.

    Parameters
    ----------
    data_pattern : str
        Glob pattern for input files.
    steps : list of str
        Pipeline steps to apply.
    output_dir : str
        Output directory path.
    n_subjects : int
        Number of subjects (determines array size).
    partition : str
        SLURM partition name.
    time_limit : str
        Wall time limit (HH:MM:SS).
    memory_gb : int
        Memory per task in GB.
    cpus_per_task : int
        CPUs per task.
    conda_env : str
        Conda environment to activate.
    module_loads : list of str, optional
        Module load commands.

    Returns
    -------
    SLURM batch script as string.
    """
    steps_str = json.dumps(steps)
    modules_section = ""
    if module_loads:
        modules_section = "\n".join(f"module load {m}" for m in module_loads) + "\n"

    script = f"""#!/bin/bash
#SBATCH --job-name=easybci_batch
#SBATCH --partition={partition}
#SBATCH --time={time_limit}
#SBATCH --mem={memory_gb}G
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --array=1-{n_subjects}
#SBATCH --output={output_dir}/logs/slurm_%A_%a.out
#SBATCH --error={output_dir}/logs/slurm_%A_%a.err

# EasyBCI Batch Processing — SLURM Array Job
# Generated automatically. Each array task processes one subject.

{modules_section}# Activate environment
source activate {conda_env} 2>/dev/null || conda activate {conda_env}

# Get subject file for this array task
FILES=({data_pattern})
SUBJECT_FILE="${{FILES[$SLURM_ARRAY_TASK_ID-1]}}"

if [ -z "$SUBJECT_FILE" ] || [ ! -f "$SUBJECT_FILE" ]; then
    echo "ERROR: No file found for task $SLURM_ARRAY_TASK_ID"
    exit 1
fi

echo "Processing: $SUBJECT_FILE (task $SLURM_ARRAY_TASK_ID of {n_subjects})"

# Run preprocessing
python -c "
import sys
sys.path.insert(0, '.')
from easybci_lib.tools.neural_processing.io.loader import load_neural
from easybci_lib.tools.neural_processing.preprocess.pipeline import preprocess
from easybci_lib.tools.neural_processing.quality.validators import validate_signal
import json, os, numpy as np

data_dict = load_neural('$SUBJECT_FILE')
result = preprocess(data_dict, steps={steps_str})
qc = validate_signal(result['data'], frequency=result.get('frequency'))

subject_id = os.path.basename('$SUBJECT_FILE').split('.')[0]
out_dir = os.path.join('{output_dir}', subject_id)
os.makedirs(out_dir, exist_ok=True)

np.savez_compressed(os.path.join(out_dir, 'processed.npz'), data=result['data'])
with open(os.path.join(out_dir, 'qc_result.json'), 'w') as f:
    json.dump(qc, f, indent=2, default=str)

print(f'Done: {{subject_id}} — QC passed: {{qc.get(\"passed\", False)}}')
"

echo "Task $SLURM_ARRAY_TASK_ID complete."
"""
    return script


def generate_pbs_script(
    data_pattern: str,
    steps: List[str],
    output_dir: str,
    n_subjects: int = 1,
    queue: str = "batch",
    walltime: str = "02:00:00",
    memory_gb: int = 16,
    ncpus: int = 4,
    conda_env: str = "agent",
) -> str:
    """Generate a PBS/Torque array job script for batch processing.

    Parameters
    ----------
    data_pattern : str
        Glob pattern for input files.
    steps : list of str
        Pipeline steps to apply.
    output_dir : str
        Output directory path.
    n_subjects : int
        Number of subjects (array range).
    queue : str
        PBS queue name.
    walltime : str
        Wall time limit.
    memory_gb : int
        Memory per task in GB.
    ncpus : int
        CPUs per task.
    conda_env : str
        Conda environment name.

    Returns
    -------
    PBS batch script as string.
    """
    steps_str = json.dumps(steps)

    script = f"""#!/bin/bash
#PBS -N easybci_batch
#PBS -q {queue}
#PBS -l walltime={walltime}
#PBS -l mem={memory_gb}gb
#PBS -l ncpus={ncpus}
#PBS -J 1-{n_subjects}
#PBS -o {output_dir}/logs/
#PBS -e {output_dir}/logs/

# EasyBCI Batch Processing — PBS Array Job

source activate {conda_env} 2>/dev/null || conda activate {conda_env}

FILES=({data_pattern})
SUBJECT_FILE="${{FILES[$PBS_ARRAY_INDEX-1]}}"

if [ -z "$SUBJECT_FILE" ] || [ ! -f "$SUBJECT_FILE" ]; then
    echo "ERROR: No file for index $PBS_ARRAY_INDEX"
    exit 1
fi

echo "Processing: $SUBJECT_FILE (index $PBS_ARRAY_INDEX of {n_subjects})"

python -c "
import sys
sys.path.insert(0, '.')
from easybci_lib.tools.neural_processing.io.loader import load_neural
from easybci_lib.tools.neural_processing.preprocess.pipeline import preprocess
from easybci_lib.tools.neural_processing.quality.validators import validate_signal
import json, os, numpy as np

data_dict = load_neural('$SUBJECT_FILE')
result = preprocess(data_dict, steps={steps_str})
qc = validate_signal(result['data'], frequency=result.get('frequency'))

subject_id = os.path.basename('$SUBJECT_FILE').split('.')[0]
out_dir = os.path.join('{output_dir}', subject_id)
os.makedirs(out_dir, exist_ok=True)

np.savez_compressed(os.path.join(out_dir, 'processed.npz'), data=result['data'])
with open(os.path.join(out_dir, 'qc_result.json'), 'w') as f:
    json.dump(qc, f, indent=2, default=str)

print(f'Done: {{subject_id}} — QC passed: {{qc.get(\"passed\", False)}}')
"

echo "Task $PBS_ARRAY_INDEX complete."
"""
    return script


# --- GPU acceleration helpers ---

_GPU_ACCELERATED_OPS = {
    "bandpass": "cusignal.firwin + cupy.convolve",
    "notch": "cusignal.iirnotch + cupy.sosfilt",
    "ica": "cuml.decomposition.FastICA",
    "resample": "cusignal.resample_poly",
    "extract_csp": "cupy.linalg.eigh",
    "extract_tfr": "cupy.fft.rfft (batch)",
}


def get_gpu_acceleration_info() -> Dict[str, Any]:
    """Check GPU availability and list accelerable operations."""
    gpu = detect_gpu()
    cupy_available = False
    cusignal_available = False

    if gpu:
        try:
            import importlib
            cupy_available = importlib.util.find_spec("cupy") is not None
        except (ImportError, ValueError):
            pass
        try:
            import importlib
            cusignal_available = importlib.util.find_spec("cusignal") is not None
        except (ImportError, ValueError):
            pass

    return {
        "gpu_detected": gpu,
        "cupy_available": cupy_available,
        "cusignal_available": cusignal_available,
        "accelerable_operations": list(_GPU_ACCELERATED_OPS.keys()) if gpu else [],
        "acceleration_backends": _GPU_ACCELERATED_OPS if gpu else {},
        "recommendation": (
            "GPU acceleration available for signal processing operations."
            if gpu and (cupy_available or cusignal_available)
            else "Install cupy and cusignal for GPU-accelerated processing."
            if gpu
            else "No GPU detected. Using CPU-based processing."
        ),
    }


def _get_available_memory_mb() -> float:
    """Get available system memory in MB."""
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024
    except (OSError, ValueError, IndexError):
        pass
    return 8000.0


# Need json for script generation
import json
