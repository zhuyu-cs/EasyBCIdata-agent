"""Proven pipeline evolution — version tracking and success rate statistics.

Extends the proven pipeline system with:
1. Version tracking: when a pipeline is refined, the new version references
   its predecessor, forming an evolution chain (v1 → v2 → v3).
2. Success rate tracking: each time a proven pipeline is reused, record
   whether QC passed. Low pass-rate pipelines are auto-downweighted.

Storage: ~/.easybci/proven_stats.json (persistent across sessions)
"""

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from easybci_lib.constants import get_easybci_home as _get_easybci_home
except Exception:  # noqa: BLE001
    _get_easybci_home = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


@dataclass
class PipelineVersion:
    """A single version in a pipeline's evolution chain."""
    version: int
    name: str
    steps: List[str]
    created_at: float = 0.0
    parent_version: int = 0
    change_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "name": self.name,
            "steps": self.steps,
            "created_at": self.created_at,
            "parent_version": self.parent_version,
            "change_reason": self.change_reason,
        }


@dataclass
class PipelineStats:
    """Usage statistics for a proven pipeline."""
    pipeline_name: str
    total_uses: int = 0
    qc_passed: int = 0
    qc_failed: int = 0
    last_used: float = 0.0
    pass_rate: float = 1.0
    is_deprecated: bool = False
    deprecation_reason: str = ""
    # Manual flag (administrator override)
    manual_flag: bool = False
    manual_flag_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pipeline_name": self.pipeline_name,
            "total_uses": self.total_uses,
            "qc_passed": self.qc_passed,
            "qc_failed": self.qc_failed,
            "last_used": self.last_used,
            "pass_rate": round(self.pass_rate, 3),
            "is_deprecated": self.is_deprecated,
            "deprecation_reason": self.deprecation_reason,
            "manual_flag": self.manual_flag,
            "manual_flag_reason": self.manual_flag_reason,
        }


class ProvenPipelineTracker:
    """Manages version history and usage statistics for proven pipelines.

    Persistence: all data stored in a single JSON file at
    ~/.easybci/proven_stats.json.
    """

    def __init__(self, store_path: Optional[str] = None):
        if store_path:
            self.store_path = Path(store_path)
        else:
            self.store_path = Path.home() / ".easybci" / "proven_stats.json"
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self._data: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        if self.store_path.exists():
            try:
                with open(self.store_path, encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        return {"versions": {}, "stats": {}}

    def _save(self) -> None:
        tmp = self.store_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)
        tmp.rename(self.store_path)

    # --- Version Tracking ---

    def register_version(
        self,
        base_name: str,
        steps: List[str],
        parent_name: Optional[str] = None,
        change_reason: str = "",
    ) -> PipelineVersion:
        """Register a new version of a proven pipeline.

        Parameters
        ----------
        base_name : str
            Base pipeline name (without version suffix).
        steps : list of str
            Pipeline steps for this version.
        parent_name : str, optional
            Name of the parent version this was derived from.
        change_reason : str
            Why this new version was created.

        Returns
        -------
        PipelineVersion with assigned version number.
        """
        versions = self._data.setdefault("versions", {})
        chain = versions.setdefault(base_name, [])

        version_num = len(chain) + 1
        parent_ver = 0
        if parent_name and chain:
            for v in chain:
                if v.get("name") == parent_name:
                    parent_ver = v["version"]
                    break
            if parent_ver == 0:
                parent_ver = len(chain)

        new_version = PipelineVersion(
            version=version_num,
            name=f"{base_name}-v{version_num}",
            steps=steps,
            created_at=time.time(),
            parent_version=parent_ver,
            change_reason=change_reason,
        )

        chain.append(new_version.to_dict())
        self._save()

        logger.debug("Registered %s v%d (parent: v%d)", base_name, version_num, parent_ver)
        return new_version

    def get_version_chain(self, base_name: str) -> List[Dict[str, Any]]:
        """Get the full version history for a pipeline."""
        return self._data.get("versions", {}).get(base_name, [])

    def get_latest_version(self, base_name: str) -> Optional[Dict[str, Any]]:
        """Get the most recent version of a pipeline."""
        chain = self.get_version_chain(base_name)
        return chain[-1] if chain else None

    # --- Success Rate Tracking ---

    def record_usage(self, pipeline_name: str, qc_passed: bool) -> PipelineStats:
        """Record a usage of a proven pipeline and update stats.

        Parameters
        ----------
        pipeline_name : str
            Name of the pipeline that was used.
        qc_passed : bool
            Whether QC passed for this usage.

        Returns
        -------
        Updated PipelineStats.
        """
        stats_dict = self._data.setdefault("stats", {})
        entry = stats_dict.setdefault(pipeline_name, {
            "pipeline_name": pipeline_name,
            "total_uses": 0,
            "qc_passed": 0,
            "qc_failed": 0,
            "last_used": 0.0,
            "is_deprecated": False,
            "deprecation_reason": "",
        })

        entry["total_uses"] += 1
        if qc_passed:
            entry["qc_passed"] += 1
        else:
            entry["qc_failed"] += 1
        entry["last_used"] = time.time()

        total = entry["total_uses"]
        entry["pass_rate"] = entry["qc_passed"] / total if total > 0 else 1.0

        # Auto-deprecate if pass rate drops below threshold after enough uses
        if total >= 5 and entry["pass_rate"] < 0.4 and not entry["is_deprecated"]:
            entry["is_deprecated"] = True
            entry["deprecation_reason"] = (
                f"Auto-deprecated: QC pass rate {entry['pass_rate']:.0%} "
                f"after {total} uses (below 40% threshold)"
            )
            logger.info("Auto-deprecated pipeline '%s': pass rate %.0f%%",
                        pipeline_name, entry["pass_rate"] * 100)

        self._save()

        return PipelineStats(
            pipeline_name=pipeline_name,
            total_uses=entry["total_uses"],
            qc_passed=entry["qc_passed"],
            qc_failed=entry["qc_failed"],
            last_used=entry["last_used"],
            pass_rate=entry["pass_rate"],
            is_deprecated=entry["is_deprecated"],
            deprecation_reason=entry["deprecation_reason"],
        )

    def get_stats(self, pipeline_name: str) -> Optional[PipelineStats]:
        """Get usage statistics for a pipeline."""
        entry = self._data.get("stats", {}).get(pipeline_name)
        if entry is None:
            return None
        return PipelineStats(**{k: v for k, v in entry.items() if k != "pass_rate"},
                             pass_rate=entry.get("pass_rate", 1.0))

    def get_all_stats(self) -> List[PipelineStats]:
        """Get stats for all tracked pipelines."""
        results = []
        for name, entry in self._data.get("stats", {}).items():
            results.append(PipelineStats(
                pipeline_name=name,
                total_uses=entry.get("total_uses", 0),
                qc_passed=entry.get("qc_passed", 0),
                qc_failed=entry.get("qc_failed", 0),
                last_used=entry.get("last_used", 0),
                pass_rate=entry.get("pass_rate", 1.0),
                is_deprecated=entry.get("is_deprecated", False),
                deprecation_reason=entry.get("deprecation_reason", ""),
            ))
        return results

    def get_similarity_weight(self, pipeline_name: str) -> float:
        """Get a weight factor based on success rate.

        Used by match_proven_pipelines to adjust similarity scores.
        Returns 1.0 for pipelines with no usage data or good pass rates.
        Returns 0.0 for deprecated pipelines.
        Returns proportional value for pipelines with declining pass rates.
        """
        entry = self._data.get("stats", {}).get(pipeline_name)
        if entry is None:
            return 1.0
        if entry.get("is_deprecated"):
            return 0.0
        # Manual flag → near-zero weight (kept in library, ignored in matches)
        if entry.get("manual_flag"):
            return 0.05
        total = entry.get("total_uses", 0)
        if total < 3:
            return 1.0  # Not enough data to penalize
        pass_rate = entry.get("pass_rate", 1.0)
        if pass_rate >= 0.7:
            return 1.0
        return max(0.2, pass_rate / 0.7)

    def flag_pipeline(self, pipeline_name: str, *, reason: str = "") -> None:
        """Manually flag a pipeline as suspicious; near-zero weight in matches."""
        stats = self._data.setdefault("stats", {}).setdefault(pipeline_name, {})
        stats.setdefault("pipeline_name", pipeline_name)
        stats["manual_flag"] = True
        stats["manual_flag_reason"] = reason
        self._save()

    def unflag_pipeline(self, pipeline_name: str) -> bool:
        """Remove manual flag. Returns True if a flag was present, False otherwise."""
        stats_all = self._data.get("stats", {})
        if pipeline_name not in stats_all:
            return False
        if not stats_all[pipeline_name].get("manual_flag"):
            return False
        stats_all[pipeline_name]["manual_flag"] = False
        stats_all[pipeline_name]["manual_flag_reason"] = ""
        self._save()
        return True

    def get_deprecated_pipelines(self) -> List[str]:
        """Get list of auto-deprecated pipeline names."""
        return [
            name for name, entry in self._data.get("stats", {}).items()
            if entry.get("is_deprecated")
        ]


    def get_summary(self) -> Dict[str, Any]:
        """Get overall summary of proven pipeline tracking."""
        stats = self._data.get("stats", {})
        versions = self._data.get("versions", {})

        total_pipelines = len(stats)
        deprecated = sum(1 for s in stats.values() if s.get("is_deprecated"))
        total_uses = sum(s.get("total_uses", 0) for s in stats.values())
        total_versions = sum(len(v) for v in versions.values())

        return {
            "total_tracked_pipelines": total_pipelines,
            "total_versions": total_versions,
            "total_uses_recorded": total_uses,
            "deprecated_pipelines": deprecated,
            "active_pipelines": total_pipelines - deprecated,
            "version_chains": len(versions),
        }


def extract_base_name(pipeline_name: str) -> str:
    """Extract base name from a versioned pipeline name.

    Examples:
        "eeg-motor_imagery-64ch-256hz-v2" → "eeg-motor_imagery-64ch-256hz"
        "my-pipeline-v3" → "my-pipeline"
        "simple-pipeline" → "simple-pipeline"
    """
    match = re.match(r"^(.+?)-v(\d+)$", pipeline_name)
    if match:
        return match.group(1)
    return pipeline_name
