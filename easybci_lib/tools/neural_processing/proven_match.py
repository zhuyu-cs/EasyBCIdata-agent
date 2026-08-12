"""Proven pipeline matching — find similar validated pipelines for new data.

Scans the proven-pipelines skill directory, extracts metadata from each file's
YAML frontmatter, encodes as a feature vector, and retrieves the top-N most
similar pipelines for a given data profile.

No external dependencies beyond numpy and yaml (already in core deps).
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .proven_match_dimensions import (
    DEFAULT_DIMENSIONS,
    SimilarityDimension,
    compute_similarity_via_dimensions,
)
from .proven_match_diversity import apply_diversity_penalty
from .proven_match_negative_penalty import apply_negative_penalty

try:
    from easybci_cli import config as _ebci_config  # type: ignore
except Exception:  # noqa: BLE001
    _ebci_config = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


@dataclass
class ProvenPipelineEntry:
    """Parsed metadata from a proven pipeline skill file."""
    name: str
    file_path: str
    modality: str = ""
    paradigm: str = ""
    n_channels: int = 0
    frequency_hz: float = 0.0
    duration_s: float = 0.0
    source_format: str = ""
    steps: List[str] = field(default_factory=list)
    proven_date: str = ""
    qc_passed: bool = True
    description: str = ""
    # Provenance / cohort metadata (back-compat: optional with empty defaults)
    lab_id: str = ""
    origin_set: str = ""
    contributed_at: str = ""
    cohort_tag: str = ""
    # Analysis goal — empty on legacy/auto-crystallized entries that pre-date
    # the field; used as a hard filter at the Reuse gate to prevent goal
    # mismatch (e.g. a feature_extraction pipeline being reused for
    # online_inference). See proven_match_dimensions._score_analysis_goal.
    analysis_goal: str = ""
    # Reference-import extension blocks (P2 writes them into frontmatter
    # metadata; empty on all non-reference / legacy skills). Surfaced here so
    # the P3 adaptive-reuse layer can recompute numeric params per recording.
    source_kind: str = ""
    reference_origin: str = ""
    adaptation_slots: List[Dict[str, Any]] = field(default_factory=list)
    qc_baselines: Dict[str, Any] = field(default_factory=dict)
    # Gold reject keywords (reference_import only) — drive the reject_by_labels
    # step so labelled windows are excised per recording instead of dropping the
    # whole file. Empty on non-reference / legacy skills.
    reject_keywords: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "modality": self.modality,
            "paradigm": self.paradigm,
            "n_channels": self.n_channels,
            "frequency_hz": self.frequency_hz,
            "duration_s": self.duration_s,
            "steps": self.steps,
            "proven_date": self.proven_date,
            "description": self.description,
            "lab_id": self.lab_id,
            "origin_set": self.origin_set,
            "contributed_at": self.contributed_at,
            "cohort_tag": self.cohort_tag,
            "analysis_goal": self.analysis_goal,
            "source_kind": self.source_kind,
            "reference_origin": self.reference_origin,
            "adaptation_slots": self.adaptation_slots,
            "qc_baselines": self.qc_baselines,
        }


@dataclass
class MatchResult:
    """A matched proven pipeline with similarity score."""
    entry: ProvenPipelineEntry
    similarity: float
    match_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = self.entry.to_dict()
        d["similarity"] = round(self.similarity, 3)
        d["match_reasons"] = self.match_reasons
        return d


def find_all_proven_pipelines_dirs() -> List[Path]:
    """Return every existing proven-pipelines directory, in scan order.

    Crystallized skills are written by ``skill_manager_tool`` with
    ``category="proven-pipelines"`` to ``$EASYBCI_HOME/skills/proven-pipelines``
    (see ``_resolve_skill_dir``). The repo also bundles a seed directory at
    ``easybci_lib/skills/proven-pipelines`` (holds only ``DESCRIPTION.md``).

    Historically this function returned the *first* existing candidate, so the
    always-present repo seed dir shadowed the user-space directory and 100% of
    crystallized skills were invisible to ``scan_proven_pipelines`` /
    ``match_proven_pipelines`` / ``batch_process_adaptive``. We now collect
    *all* existing candidates and scan the union.
    """
    candidates: List[Path] = []

    # 1. User-space crystallize target — where import_reference / crystallize
    #    actually write (get_skills_dir()/proven-pipelines, no "bci/" segment).
    try:
        from easybci_lib.constants import get_skills_dir
        candidates.append(get_skills_dir() / "proven-pipelines")
        # Legacy/alternate layout kept for back-compat with older writes.
        candidates.append(get_skills_dir() / "bci" / "proven-pipelines")
    except ImportError:
        pass

    # 2. Repo-bundled seed directory (relative to this file).
    repo_root = Path(__file__).resolve().parent.parent.parent / "skills"
    candidates.append(repo_root / "proven-pipelines")
    candidates.append(repo_root / "bci" / "proven-pipelines")

    seen: set = set()
    dirs: List[Path] = []
    for cand in candidates:
        try:
            resolved = cand.resolve()
        except OSError:
            resolved = cand
        if resolved in seen:
            continue
        seen.add(resolved)
        if cand.is_dir():
            dirs.append(cand)
    return dirs


def find_proven_pipelines_dir() -> Optional[Path]:
    """Locate a proven-pipelines directory (first existing candidate).

    Prefer :func:`find_all_proven_pipelines_dirs` — this single-dir variant is
    kept for back-compat. Ordering now puts the user-space crystallize target
    first so crystallized skills are no longer shadowed by the repo seed dir.
    """
    dirs = find_all_proven_pipelines_dirs()
    return dirs[0] if dirs else None


def scan_proven_pipelines(directory: Optional[Path] = None) -> List[ProvenPipelineEntry]:
    """Scan the proven-pipelines directory/directories and parse all entries.

    Parameters
    ----------
    directory : Path, optional
        Override directory to scan. If None, auto-discovers and scans *every*
        existing proven-pipelines location (see
        :func:`find_all_proven_pipelines_dirs`), de-duplicating by skill name
        (first occurrence wins).

    Returns
    -------
    List of ProvenPipelineEntry with metadata extracted from YAML frontmatter.
    """
    if directory is not None:
        scan_dirs = [directory] if directory.is_dir() else []
    else:
        scan_dirs = find_all_proven_pipelines_dirs()
    if not scan_dirs:
        return []

    entries = []
    seen_names: set = set()
    for scan_dir in scan_dirs:
        for md_file in sorted(scan_dir.glob("**/*.md")):
            if md_file.name in ("DESCRIPTION.md", "README.md"):
                continue
            entry = _parse_proven_file(md_file)
            if entry is None:
                continue
            if entry.name in seen_names:
                continue
            seen_names.add(entry.name)
            entries.append(entry)

    logger.debug(
        "Scanned %d proven pipeline entries from %s",
        len(entries), [str(d) for d in scan_dirs],
    )
    return entries


def _parse_proven_file(filepath: Path) -> Optional[ProvenPipelineEntry]:
    """Parse a proven pipeline Markdown file with YAML frontmatter."""
    try:
        content = filepath.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None

    # Extract YAML frontmatter
    frontmatter, body = _split_frontmatter(content)
    if not frontmatter:
        return None

    name = frontmatter.get("name", filepath.stem)
    metadata = frontmatter.get("metadata", {})

    entry = ProvenPipelineEntry(
        name=name,
        file_path=str(filepath),
        description=frontmatter.get("description", ""),
        modality=_first_item(metadata.get("modalities", [])),
        paradigm=_first_item(metadata.get("paradigm") or metadata.get("paradigms", [])),
        proven_date=str(metadata.get("proven_date", "")),
        source_format=str(metadata.get("source_format", "")),
        qc_passed=bool(metadata.get("qc_passed", True)),
        # Optional provenance/cohort fields (default empty for legacy files)
        lab_id=str(metadata.get("lab_id", "")),
        origin_set=str(metadata.get("origin_set", "")),
        contributed_at=str(metadata.get("contributed_at", "")),
        cohort_tag=str(metadata.get("cohort_tag", "")),
        analysis_goal=str(metadata.get("analysis_goal", "")).strip(),
        source_kind=str(metadata.get("source_kind", "")).strip(),
        reference_origin=str(metadata.get("reference_origin", "")).strip(),
        adaptation_slots=list(metadata.get("adaptation_slots", []) or []),
        qc_baselines=dict(metadata.get("qc_baselines", {}) or {}),
        reject_keywords=[str(k) for k in (metadata.get("reject_keywords", []) or [])],
    )

    # Extract n_channels, frequency from body
    entry.n_channels = _extract_int_from_body(body, r"Channels:\s*\*{0,2}(\d+)")
    entry.frequency_hz = _extract_float_from_body(
        body, r"Sampling(?:\s*rate)?:\s*\*{0,2}([\d.]+)\s*Hz"
    )
    entry.duration_s = _extract_float_from_body(body, r"Duration:\s*([\d.]+)\s*s")

    # Extract pipeline steps from body
    entry.steps = _extract_steps_from_body(body)

    # Fallback: parse channel/freq from name (eeg-motor_imagery-64ch-256hz-20260527)
    if entry.n_channels == 0:
        match = re.search(r"(\d+)ch", name)
        if match:
            entry.n_channels = int(match.group(1))
    if entry.frequency_hz == 0:
        match = re.search(r"(\d+)hz", name)
        if match:
            entry.frequency_hz = float(match.group(1))

    return entry


def match_proven_pipelines(
    modality: str,
    paradigm: str = "default",
    n_channels: int = 0,
    frequency_hz: float = 0.0,
    duration_s: float = 0.0,
    top_n: int = 3,
    entries: Optional[List[ProvenPipelineEntry]] = None,
    *,
    cohort_tag: str = "",
    analysis_goal: str = "generic",
) -> List[MatchResult]:
    """Find the most similar proven pipelines for given data characteristics.

    Parameters
    ----------
    modality : str
        Data modality (eeg, seeg, ecog, meg, spike, fnirs).
    paradigm : str
        Processing paradigm.
    n_channels : int
        Number of channels in the new data.
    frequency_hz : float
        Sampling rate of the new data.
    duration_s : float
        Recording duration of the new data.
    top_n : int
        Number of matches to return.
    entries : list, optional
        Pre-scanned entries (avoids rescanning directory).

    Returns
    -------
    List of MatchResult sorted by similarity (highest first).
    """
    if entries is None:
        entries = scan_proven_pipelines()

    if not entries:
        return []

    # Load tracker for success-rate weighting
    try:
        from easybci_lib.tools.neural_processing.proven_tracker import ProvenPipelineTracker
        tracker = ProvenPipelineTracker()
        deprecated = set(tracker.get_deprecated_pipelines())
    except Exception:
        tracker = None
        deprecated = set()

    results = []
    for entry in entries:
        # Skip deprecated pipelines
        if entry.name in deprecated:
            continue

        similarity, reasons = _compute_similarity(
            entry, modality, paradigm, n_channels, frequency_hz, duration_s,
            cohort_tag=cohort_tag, analysis_goal=analysis_goal,
        )
        if similarity > 0.0:
            # Adjust by success rate weight
            if tracker:
                weight = tracker.get_similarity_weight(entry.name)
                if weight < 1.0:
                    similarity *= weight
                    reasons.append(f"weighted by pass rate ({weight:.2f})")

            results.append(MatchResult(
                entry=entry,
                similarity=similarity,
                match_reasons=reasons,
            ))

    results.sort(key=lambda r: r.similarity, reverse=True)

    # Diversity penalty by lab_id — keep top-N from drifting toward one origin.
    try:
        cfg: Dict[str, Any] = {}
        if _ebci_config is not None:
            try:
                cfg = (_ebci_config.load_config() or {}).get("proven_pipelines") or {}
            except Exception:  # noqa: BLE001
                cfg = {}
        max_per_origin = int(cfg.get("max_per_origin", 2))
        library_lab_count = len({(e.lab_id or "unknown") for e in entries if e.lab_id})
        apply_diversity_penalty(
            results,
            max_per_origin=max_per_origin,
            library_lab_count=library_lab_count or None,
        )
        results.sort(key=lambda r: r.similarity, reverse=True)
    except Exception:  # noqa: BLE001
        logger.debug("diversity penalty failed", exc_info=True)

    # Negative-experience penalty: pipelines whose steps overlap a known
    # failure mode get downweighted (hard ×0.3, soft ×0.7).  Best-effort —
    # never raises into match_proven_pipelines callers.
    try:
        apply_negative_penalty(
            results,
            modality=modality,
            paradigm=paradigm,
            cohort_tag=cohort_tag,
            analysis_goal=analysis_goal,
        )
        results.sort(key=lambda r: r.similarity, reverse=True)
    except Exception:  # noqa: BLE001
        logger.debug("negative penalty failed", exc_info=True)
    return results[:top_n]


def _compute_similarity(
    entry: ProvenPipelineEntry,
    modality: str,
    paradigm: str,
    n_channels: int,
    frequency_hz: float,
    duration_s: float,
    cohort_tag: str = "",
    analysis_goal: str = "",
) -> Tuple[float, List[str]]:
    """Back-compat wrapper. Delegates to the pluggable dimension framework.

    cohort_tag is optional and defaults empty so all existing call sites
    continue to work without changes. When ``proven_pipelines.use_cohort`` is
    true in config, similarity-with-cohort weights are loaded instead of
    defaults. analysis_goal participates as a hard filter — mismatch (entry
    has a goal and it differs from the query) short-circuits to 0.
    """
    dimensions = _load_active_dimensions()
    return compute_similarity_via_dimensions(
        entry,
        modality=modality,
        paradigm=paradigm,
        n_channels=n_channels,
        frequency_hz=frequency_hz,
        duration_s=duration_s,
        cohort_tag=cohort_tag,
        analysis_goal=analysis_goal,
        dimensions=dimensions,
    )


def _load_active_dimensions():
    """Return DEFAULT_DIMENSIONS or a cohort-enabled rebalanced set, per config."""
    cfg: Dict[str, Any] = {}
    if _ebci_config is not None:
        try:
            cfg = _ebci_config.load_config() or {}
        except Exception:  # noqa: BLE001
            cfg = {}
    use_cohort = bool(((cfg.get("proven_pipelines") or {}).get("use_cohort", False)))
    if not use_cohort:
        return DEFAULT_DIMENSIONS
    # Cohort enabled — rebalance to keep sum = 1.0
    return [
        SimilarityDimension("modality",      0.30, is_hard_filter=True, compute_fn=DEFAULT_DIMENSIONS[0].compute_fn),
        SimilarityDimension("paradigm",      0.20, compute_fn=DEFAULT_DIMENSIONS[1].compute_fn),
        SimilarityDimension("n_channels",    0.10, compute_fn=DEFAULT_DIMENSIONS[2].compute_fn),
        SimilarityDimension("frequency_hz",  0.10, compute_fn=DEFAULT_DIMENSIONS[3].compute_fn),
        SimilarityDimension("duration_s",    0.05, compute_fn=DEFAULT_DIMENSIONS[4].compute_fn),
        SimilarityDimension("cohort",        0.15, compute_fn=DEFAULT_DIMENSIONS[5].compute_fn),
        SimilarityDimension("analysis_goal", 0.10, is_hard_filter=True, compute_fn=DEFAULT_DIMENSIONS[6].compute_fn),
    ]


# --- Internal helpers ---

def _split_frontmatter(content: str) -> Tuple[Dict[str, Any], str]:
    """Split YAML frontmatter from Markdown body."""
    if not content.startswith("---"):
        return {}, content

    end_idx = content.find("---", 3)
    if end_idx < 0:
        return {}, content

    yaml_str = content[3:end_idx].strip()
    body = content[end_idx + 3:].strip()

    try:
        import yaml
        frontmatter = yaml.safe_load(yaml_str) or {}
    except Exception:
        frontmatter = {}

    return frontmatter, body


def _first_item(lst: Any) -> str:
    if isinstance(lst, list) and lst:
        return str(lst[0])
    if isinstance(lst, str):
        return lst
    return ""


def _extract_int_from_body(body: str, pattern: str) -> int:
    match = re.search(pattern, body)
    if match:
        try:
            return int(match.group(1))
        except (ValueError, IndexError):
            pass
    return 0


def _extract_float_from_body(body: str, pattern: str) -> float:
    match = re.search(pattern, body)
    if match:
        try:
            return float(match.group(1))
        except (ValueError, IndexError):
            pass
    return 0.0


def _extract_steps_from_body(body: str) -> List[str]:
    """Extract pipeline steps from Markdown body.

    Looks for patterns like:
      notch:50 → bandpass:0.5,40 → resample:256
    or code blocks with step lists.
    """
    # Pattern 1: arrow-separated steps
    arrow_match = re.search(r"([\w:.,]+(?:\s*→\s*[\w:.,]+)+)", body)
    if arrow_match:
        steps_str = arrow_match.group(1)
        steps = [s.strip() for s in re.split(r"\s*→\s*", steps_str)]
        if all(_is_valid_step(s) for s in steps):
            return steps

    # Pattern 2: lines starting with step-like patterns in a code block
    code_match = re.search(r"```[^\n]*\n(.*?)```", body, re.DOTALL)
    if code_match:
        lines = code_match.group(1).strip().split("\n")
        steps = []
        for line in lines:
            line = line.strip().strip("-").strip()
            if _is_valid_step(line):
                steps.append(line)
        if steps:
            return steps

    return []


_KNOWN_STEP_PREFIXES = {
    "notch", "bandpass", "resample", "scale", "car", "bipolar_ref",
    "drop_bads", "interpolate_bads", "ica", "hilbert", "clip",
    "fill_nan", "pick_channels",
}


def _is_valid_step(s: str) -> bool:
    """Check if a string looks like a valid pipeline step."""
    name = s.split(":")[0].strip()
    return name in _KNOWN_STEP_PREFIXES
