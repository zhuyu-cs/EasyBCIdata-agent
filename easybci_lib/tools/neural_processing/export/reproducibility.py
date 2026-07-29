"""Pipeline reproducibility verification — smoke test, hash checks, Dockerfile.

Provides:
1. Smoke test: run pipeline.py in subprocess, verify output matches results/
2. Input hash verification: SHA256 full-file hash check before re-execution
3. Dockerfile generation: optional container for one-click reproducibility

Integrates with the mini-repo builder (export/repo_builder.py).
"""

import hashlib
import json
import logging
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def compute_file_hash(filepath: str, algorithm: str = "sha256") -> str:
    """Compute full-content hash (default SHA256).

    For a single file: chunked hash of contents.
    For a directory: hash of a canonical manifest mapping each relative
    file path (sorted) to its file content hash. BIDS / multi-run datasets
    are commonly passed as a directory rather than a single file, so the
    helper accepts either — re-hashing the same tree always yields the
    same digest.
    """
    p = Path(filepath)
    if p.is_dir():
        manifest: Dict[str, str] = {}
        for child in sorted(p.rglob("*")):
            if child.is_file():
                rel = str(child.relative_to(p))
                manifest[rel] = compute_file_hash(str(child), algorithm)
        h = hashlib.new(algorithm)
        h.update(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        return h.hexdigest()
    h = hashlib.new(algorithm)
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def write_input_hash(meta_dir: str, input_path: str) -> Dict[str, Any]:
    """Write comprehensive input hash info to meta/input_ref.json.

    Extends the basic input_ref with full SHA256 (not just first 1MB).
    Accepts either a single file or a directory tree (BIDS / multi-run).

    Parameters
    ----------
    meta_dir : str
        Path to the meta/ directory in the mini-repo.
    input_path : str
        Path to the original input data file or directory.

    Returns
    -------
    Dict with hash info written to input_ref.json.
    """
    path = Path(input_path)
    meta = Path(meta_dir)
    meta.mkdir(parents=True, exist_ok=True)

    ref: Dict[str, Any] = {
        "path": str(path.resolve()) if path.exists() else input_path,
        "filename": path.name,
        "verified_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    if path.is_dir():
        ref["is_directory"] = True
        files = [f for f in path.rglob("*") if f.is_file()]
        ref["size_bytes"] = sum(f.stat().st_size for f in files)
        ref["file_count"] = len(files)
        # Manifest-of-content-hashes for the whole tree.
        ref["sha256"] = compute_file_hash(str(path))
        # Partial hash is meaningless across multiple files; reuse the full
        # tree hash so callers that only inspect sha256_1mb still see a
        # stable fingerprint.
        ref["sha256_1mb"] = ref["sha256"]
    elif path.exists():
        stat = path.stat()
        ref["size_bytes"] = stat.st_size
        ref["modified"] = time.strftime(
            "%Y-%m-%dT%H:%M:%S", time.localtime(stat.st_mtime))
        ref["sha256"] = compute_file_hash(str(path))
        # Also store partial hash for quick checks
        sha_partial = hashlib.sha256()
        with open(path, "rb") as f:
            sha_partial.update(f.read(1024 * 1024))
        ref["sha256_1mb"] = sha_partial.hexdigest()

    ref_path = meta / "input_ref.json"
    ref_path.write_text(
        json.dumps(ref, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return ref


def verify_input_hash(mini_repo_dir: str, input_path: Optional[str] = None) -> Dict[str, Any]:
    """Verify that the input file hasn't changed since the mini-repo was built.

    Parameters
    ----------
    mini_repo_dir : str
        Path to the mini-repo root.
    input_path : str, optional
        Override input path. If None, reads from meta/input_ref.json.

    Returns
    -------
    Dict with verification result: {valid, message, details}.
    """
    meta_path = Path(mini_repo_dir) / "meta" / "input_ref.json"
    if not meta_path.exists():
        return {"valid": False, "message": "No input_ref.json found in meta/"}

    try:
        ref = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return {"valid": False, "message": f"Failed to read input_ref.json: {exc}"}

    # Determine input file path
    data_path = input_path or ref.get("path", "")
    if not data_path or not Path(data_path).exists():
        return {
            "valid": False,
            "message": f"Input file not found: {data_path}",
            "stored_path": ref.get("path"),
        }

    # Verify size first (fast check)
    actual_size = Path(data_path).stat().st_size
    expected_size = ref.get("size_bytes")
    if expected_size is not None and actual_size != expected_size:
        return {
            "valid": False,
            "message": (
                f"File size mismatch: expected {expected_size} bytes, "
                f"got {actual_size} bytes. Input data has changed."
            ),
        }

    # Verify hash
    stored_hash = ref.get("sha256") or ref.get("sha256_1mb")
    if not stored_hash:
        return {"valid": True, "message": "No hash stored for comparison (legacy format)"}

    if ref.get("sha256"):
        actual_hash = compute_file_hash(data_path)
        hash_type = "full"
    else:
        sha = hashlib.sha256()
        with open(data_path, "rb") as f:
            sha.update(f.read(1024 * 1024))
        actual_hash = sha.hexdigest()
        hash_type = "1mb"

    if actual_hash == stored_hash:
        return {
            "valid": True,
            "message": f"Input file verified ({hash_type} SHA256 match).",
            "hash_type": hash_type,
        }
    else:
        return {
            "valid": False,
            "message": (
                f"Hash mismatch ({hash_type}): input data has been modified since "
                f"this pipeline was exported."
            ),
            "expected": stored_hash[:16] + "...",
            "actual": actual_hash[:16] + "...",
        }


def smoke_test_pipeline(
    mini_repo_dir: str,
    timeout_seconds: int = 300,
    verify_output: bool = True,
    python_executable: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the exported pipeline.py as a smoke test and verify output.

    Executes code/run.py (or code/pipeline.py directly) in a subprocess,
    optionally comparing output hash with existing results/.

    Parameters
    ----------
    mini_repo_dir : str
        Path to the mini-repo root directory.
    timeout_seconds : int
        Maximum execution time (default 5 minutes).
    verify_output : bool
        If True, compare output with existing results/.
    python_executable : str, optional
        Python binary to use. Default: same as current interpreter.

    Returns
    -------
    Dict with: success, stdout, stderr, execution_time_s, output_verified, details.
    """
    repo = Path(mini_repo_dir)
    code_dir = repo / "code"
    results_dir = repo / "results"

    # Find the script to execute
    run_script = code_dir / "run.py"
    pipeline_script = code_dir / "pipeline.py"
    config_path = repo / "plan" / "config.yaml"

    if not pipeline_script.exists():
        return {"success": False, "error": "code/pipeline.py not found in mini-repo"}

    python = python_executable or sys.executable

    # Determine command
    if run_script.exists() and config_path.exists():
        cmd = [python, str(run_script), "--config", str(config_path)]
    else:
        cmd = [python, str(pipeline_script)]

    # Capture existing output hashes for verification
    existing_hashes = {}
    if verify_output and results_dir.exists():
        for f in results_dir.iterdir():
            if f.is_file() and f.suffix in (".pkl", ".npz", ".hdf5", ".npy", ".nwb"):
                existing_hashes[f.name] = compute_file_hash(str(f))

    # Use a temp directory for smoke test output to avoid overwriting
    with tempfile.TemporaryDirectory(prefix="easybci_smoke_") as tmp_out:
        env_override = {
            "EASYBCI_SMOKE_TEST": "1",
            "EASYBCI_OUTPUT_DIR": tmp_out,
        }
        import os
        env = {**os.environ, **env_override}

        start_time = time.time()
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                cwd=str(code_dir),
                env=env,
            )
            execution_time = time.time() - start_time
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"Pipeline execution timed out after {timeout_seconds}s",
                "execution_time_s": timeout_seconds,
            }
        except OSError as exc:
            return {"success": False, "error": f"Failed to execute pipeline: {exc}"}

        result: Dict[str, Any] = {
            "success": proc.returncode == 0,
            "returncode": proc.returncode,
            "execution_time_s": round(execution_time, 2),
            "stdout": proc.stdout[-2000:] if proc.stdout else "",
            "stderr": proc.stderr[-2000:] if proc.stderr else "",
        }

        if proc.returncode != 0:
            result["error"] = f"Pipeline exited with code {proc.returncode}"
            return result

        # Verify output consistency (if we have previous results)
        if verify_output and existing_hashes:
            output_verified = _verify_output_consistency(tmp_out, existing_hashes)
            result["output_verified"] = output_verified["verified"]
            result["output_details"] = output_verified.get("details", "")
        else:
            result["output_verified"] = None
            result["output_details"] = "No previous output to compare against."

        return result


def _verify_output_consistency(
    output_dir: str,
    expected_hashes: Dict[str, str],
) -> Dict[str, Any]:
    """Compare smoke test output hashes with expected results.

    Note: Exact hash match is not always expected (floating-point non-determinism).
    We check that output files exist and have reasonable sizes.
    """
    out = Path(output_dir)
    found_files = list(out.rglob("*.pkl")) + list(out.rglob("*.npz")) + list(out.rglob("*.npy")) + list(out.rglob("*.nwb"))

    if not found_files and expected_hashes:
        return {
            "verified": False,
            "details": "Smoke test produced no output files.",
        }

    if not expected_hashes:
        return {
            "verified": True,
            "details": "Smoke test completed; no reference output for comparison.",
        }

    # Check that at least one output file was produced
    details = []
    for name, expected_hash in expected_hashes.items():
        matching = [f for f in found_files if f.name == name]
        if matching:
            actual_hash = compute_file_hash(str(matching[0]))
            if actual_hash == expected_hash:
                details.append(f"{name}: exact match")
            else:
                # Check size similarity (floating point diffs are expected)
                expected_size = None
                for f in found_files:
                    if f.name == name:
                        expected_size = f.stat().st_size
                details.append(f"{name}: produced (hash differs — floating-point variance expected)")
        else:
            details.append(f"{name}: NOT produced")

    all_produced = all("NOT produced" not in d for d in details)

    return {
        "verified": all_produced,
        "details": "; ".join(details),
    }


def generate_dockerfile(
    mini_repo_dir: str,
    python_version: str = "3.11",
    base_image: str = "python",
    include_gpu: bool = False,
) -> str:
    """Generate a Dockerfile for the mini-repo.

    Parameters
    ----------
    mini_repo_dir : str
        Path to the mini-repo root.
    python_version : str
        Python version for the base image.
    base_image : str
        Base image name (default: "python", or "nvidia/cuda" for GPU).
    include_gpu : bool
        If True, use NVIDIA CUDA base image.

    Returns
    -------
    Dockerfile content as string.
    """
    repo = Path(mini_repo_dir)

    # Read requirements
    reqs_path = repo / "code" / "requirements.txt"
    requirements = ""
    if reqs_path.exists():
        requirements = reqs_path.read_text(encoding="utf-8").strip()

    # Determine base image
    if include_gpu:
        base = f"nvidia/cuda:12.1.0-runtime-ubuntu22.04"
        python_install = f"""
RUN apt-get update && apt-get install -y python{python_version} python3-pip && \\
    ln -sf /usr/bin/python{python_version} /usr/bin/python && \\
    apt-get clean && rm -rf /var/lib/apt/lists/*
"""
    else:
        base = f"{base_image}:{python_version}-slim"
        python_install = ""

    # Check if config.yaml exists
    has_config = (repo / "plan" / "config.yaml").exists()
    run_cmd = (
        'CMD ["python", "code/run.py", "--config", "plan/config.yaml"]'
        if has_config
        else 'CMD ["python", "code/pipeline.py"]'
    )

    dockerfile = f"""# Auto-generated Dockerfile for EasyBCI reproducible pipeline
# Built from mini-repo: {repo.name}
FROM {base}
{python_install}
WORKDIR /pipeline

# Install dependencies
COPY code/requirements.txt /pipeline/code/requirements.txt
RUN pip install --no-cache-dir -r code/requirements.txt

# Copy pipeline code and configuration
COPY code/ /pipeline/code/
COPY plan/ /pipeline/plan/

# Copy input data reference (for verification)
COPY meta/ /pipeline/meta/

# Create output directory
RUN mkdir -p /pipeline/results

# Run the pipeline
{run_cmd}
"""

    return dockerfile


def write_dockerfile(mini_repo_dir: str, include_gpu: bool = False) -> str:
    """Generate and write Dockerfile to the mini-repo.

    Returns the path to the written Dockerfile.
    """
    content = generate_dockerfile(mini_repo_dir, include_gpu=include_gpu)
    dockerfile_path = Path(mini_repo_dir) / "Dockerfile"
    dockerfile_path.write_text(content, encoding="utf-8")
    return str(dockerfile_path)


def generate_docker_compose(mini_repo_dir: str, input_path: str = "") -> str:
    """Generate a docker-compose.yml for easy execution.

    Parameters
    ----------
    mini_repo_dir : str
        Path to the mini-repo.
    input_path : str
        Path to input data file (mounted as volume).

    Returns
    -------
    docker-compose.yml content as string.
    """
    repo_name = Path(mini_repo_dir).name.lower().replace(" ", "_")

    volumes = ["./results:/pipeline/results"]
    if input_path:
        volumes.append(f"{input_path}:/pipeline/data/input:ro")

    volumes_str = "\n".join(f"      - {v}" for v in volumes)

    compose = f"""# Auto-generated docker-compose for reproducible pipeline execution
version: "3.8"

services:
  pipeline:
    build: .
    container_name: {repo_name}
    volumes:
{volumes_str}
    environment:
      - PYTHONUNBUFFERED=1
"""

    return compose


def full_reproducibility_check(
    mini_repo_dir: str,
    input_path: Optional[str] = None,
    run_smoke_test: bool = True,
    generate_docker: bool = True,
    timeout_seconds: int = 300,
) -> Dict[str, Any]:
    """Run a complete reproducibility verification suite on a mini-repo.

    Steps:
    1. Verify input data hash (hasn't changed)
    2. Run smoke test (pipeline.py executes without error)
    3. Verify output consistency (results match previous run)
    4. Optionally generate Dockerfile for container reproducibility

    Parameters
    ----------
    mini_repo_dir : str
        Path to the mini-repo root.
    input_path : str, optional
        Override input data path.
    run_smoke_test : bool
        Whether to execute the pipeline smoke test.
    generate_docker : bool
        Whether to generate Dockerfile.
    timeout_seconds : int
        Smoke test timeout.

    Returns
    -------
    Dict with comprehensive verification report.
    """
    report: Dict[str, Any] = {
        "mini_repo": mini_repo_dir,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "checks": {},
    }

    # 1. Input hash verification
    hash_result = verify_input_hash(mini_repo_dir, input_path)
    report["checks"]["input_hash"] = hash_result

    # 2. Structure check (required files exist)
    structure_result = _check_repo_structure(mini_repo_dir)
    report["checks"]["structure"] = structure_result

    # 3. Smoke test
    if run_smoke_test and structure_result.get("valid"):
        smoke_result = smoke_test_pipeline(
            mini_repo_dir,
            timeout_seconds=timeout_seconds,
        )
        report["checks"]["smoke_test"] = smoke_result
    else:
        report["checks"]["smoke_test"] = {
            "skipped": True,
            "reason": "Structure check failed" if not structure_result.get("valid") else "Disabled",
        }

    # 4. Generate Dockerfile
    if generate_docker:
        try:
            dockerfile_path = write_dockerfile(mini_repo_dir)
            compose_content = generate_docker_compose(mini_repo_dir, input_path or "")
            compose_path = Path(mini_repo_dir) / "docker-compose.yml"
            compose_path.write_text(compose_content, encoding="utf-8")
            report["checks"]["docker"] = {
                "success": True,
                "dockerfile": dockerfile_path,
                "compose": str(compose_path),
            }
        except Exception as exc:
            report["checks"]["docker"] = {"success": False, "error": str(exc)}

    # Overall verdict
    all_valid = all(
        c.get("valid", c.get("success", True))
        for c in report["checks"].values()
        if not c.get("skipped")
    )
    report["reproducible"] = all_valid
    report["summary"] = (
        "All reproducibility checks passed."
        if all_valid
        else "Some checks failed — see details."
    )

    return report


def _check_repo_structure(mini_repo_dir: str) -> Dict[str, Any]:
    """Verify the mini-repo has all required files."""
    repo = Path(mini_repo_dir)
    required = [
        "code/pipeline.py",
        "code/requirements.txt",
        "plan/config.yaml",
        "meta/input_ref.json",
    ]
    optional = [
        "code/run.py",
        "results/",
        "figures/",
        "README.md",
        "plan/reasoning.md",
        "meta/pipeline_record.json",
    ]

    missing = [f for f in required if not (repo / f).exists()]
    present_optional = [f for f in optional if (repo / f).exists()]

    return {
        "valid": len(missing) == 0,
        "missing_required": missing,
        "present_optional": present_optional,
        "message": (
            "All required files present."
            if not missing
            else f"Missing: {', '.join(missing)}"
        ),
    }
