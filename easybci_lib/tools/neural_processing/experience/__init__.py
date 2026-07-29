"""Experience distillation — structured processing logs and knowledge extraction.

Records detailed processing history (retries, QC failures, parameter adjustments)
and distills reusable knowledge from accumulated experience.

Components:
1. ProcessingRecord — structured log of a single preprocessing run
2. ExperienceStore — persistent storage of processing records
3. NegativeExample — structured "what NOT to do" entry
4. Knowledge extraction — identify patterns from accumulated records
"""

import datetime as _dt
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

try:
    from easybci_lib.tools.neural_processing.research.internalization_audit import (
        AuditEvent as _AuditEvent,
        InternalizationAuditLog as _InternalizationAuditLog,
    )
except Exception:  # noqa: BLE001
    _AuditEvent = None  # type: ignore[assignment]
    _InternalizationAuditLog = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


_VALID_FAILURE_MODES: tuple[str, ...] = ("qc_fail", "user_rejected", "auto_flagged", "audit_downgraded")
_VALID_SEVERITY: tuple[str, ...] = ("soft", "hard")


@dataclass
class NegativeExample:
    """Structured negative example — what NOT to do for similar data."""
    id: str
    modality: str
    paradigm: str
    cohort_tag: str
    analysis_goal: str
    failed_step: str
    failed_params: Dict[str, Any]
    failure_mode: str
    failure_evidence: str
    fingerprint_hash: str
    recorded_at: str  # ISO 8601 UTC
    lab_id: str
    severity: str

    def __post_init__(self) -> None:
        if self.failure_mode not in _VALID_FAILURE_MODES:
            raise ValueError(
                f"failure_mode {self.failure_mode!r} not in {_VALID_FAILURE_MODES}"
            )
        if self.severity not in _VALID_SEVERITY:
            raise ValueError(f"severity {self.severity!r} not in {_VALID_SEVERITY}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "modality": self.modality,
            "paradigm": self.paradigm,
            "cohort_tag": self.cohort_tag,
            "analysis_goal": self.analysis_goal,
            "failed_step": self.failed_step,
            "failed_params": self.failed_params,
            "failure_mode": self.failure_mode,
            "failure_evidence": self.failure_evidence,
            "fingerprint_hash": self.fingerprint_hash,
            "recorded_at": self.recorded_at,
            "lab_id": self.lab_id,
            "severity": self.severity,
        }


@dataclass
class RetryEvent:
    """Records a single retry attempt during preprocessing."""
    attempt: int
    trigger: str  # what caused the retry (qc_failure, error, user_request)
    error_message: str = ""
    qc_issues: List[str] = field(default_factory=list)
    steps_before: List[str] = field(default_factory=list)
    steps_after: List[str] = field(default_factory=list)
    parameter_changes: Dict[str, str] = field(default_factory=dict)
    result: str = ""  # "success", "partial", "failed"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attempt": self.attempt,
            "trigger": self.trigger,
            "error_message": self.error_message,
            "qc_issues": self.qc_issues,
            "steps_before": self.steps_before,
            "steps_after": self.steps_after,
            "parameter_changes": self.parameter_changes,
            "result": self.result,
        }


@dataclass
class ProcessingRecord:
    """Complete record of a preprocessing session."""
    # --- Identity ---
    record_id: str = ""
    timestamp: float = 0.0
    data_path: str = ""
    modality: str = ""
    paradigm: str = ""

    # --- Data characteristics ---
    n_channels: int = 0
    frequency: float = 0.0
    duration_s: float = 0.0
    data_profile_summary: Dict[str, Any] = field(default_factory=dict)

    # --- Pipeline ---
    initial_steps: List[str] = field(default_factory=list)
    final_steps: List[str] = field(default_factory=list)
    steps_added: List[str] = field(default_factory=list)
    steps_removed: List[str] = field(default_factory=list)
    steps_modified: Dict[str, str] = field(default_factory=dict)

    # --- Retries ---
    n_retries: int = 0
    retry_events: List[RetryEvent] = field(default_factory=list)

    # --- QC ---
    qc_passed: bool = False
    qc_grade: str = ""
    qc_score: float = 0.0
    qc_issues_final: List[str] = field(default_factory=list)

    # --- Outcome ---
    success: bool = False
    duration_elapsed_s: float = 0.0
    stage: str = ""  # "preprocessed" | "exported" | "split" — tracks pipeline completion stage
    notes: str = ""

    # --- Negative experience ---
    negative_lessons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "timestamp": self.timestamp,
            "data_path": self.data_path,
            "modality": self.modality,
            "paradigm": self.paradigm,
            "data": {
                "n_channels": self.n_channels,
                "frequency": self.frequency,
                "duration_s": self.duration_s,
                "profile_summary": self.data_profile_summary,
            },
            "pipeline": {
                "initial": self.initial_steps,
                "final": self.final_steps,
                "added": self.steps_added,
                "removed": self.steps_removed,
                "modified": self.steps_modified,
            },
            "retries": {
                "count": self.n_retries,
                "events": [r.to_dict() for r in self.retry_events],
            },
            "qc": {
                "passed": self.qc_passed,
                "grade": self.qc_grade,
                "score": self.qc_score,
                "issues": self.qc_issues_final,
            },
            "outcome": {
                "success": self.success,
                "elapsed_s": round(self.duration_elapsed_s, 1),
                "stage": self.stage,
                "notes": self.notes,
            },
            "negative_lessons": self.negative_lessons,
        }


class ExperienceStore:
    """Persistent storage for processing records.

    Stores records as JSON lines in ~/.easybci/experience/records.jsonl
    and maintains an index for fast querying.
    """

    def __init__(self, store_dir: Optional[str] = None):
        if store_dir:
            self.store_dir = Path(store_dir)
        else:
            home = Path.home() / ".easybci" / "experience"
            self.store_dir = home
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.records_path = self.store_dir / "records.jsonl"
        self.lessons_path = self.store_dir / "negative_lessons.json"

    def save_record(self, record: ProcessingRecord) -> None:
        """Append a processing record to the store."""
        if not record.record_id:
            record.record_id = f"{int(record.timestamp)}_{record.modality}_{record.paradigm}"

        with open(self.records_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record.to_dict(), default=str, ensure_ascii=False) + "\n")

        # Extract and persist negative lessons
        if record.negative_lessons:
            self._append_lessons(record.negative_lessons, record)

        logger.debug("Saved processing record: %s", record.record_id)

    def load_records(
        self,
        modality: Optional[str] = None,
        paradigm: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Load records, optionally filtered by modality/paradigm."""
        if not self.records_path.exists():
            return []

        records = []
        with open(self.records_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if modality and rec.get("modality", "").lower() != modality.lower():
                        continue
                    if paradigm and rec.get("paradigm", "").lower() != paradigm.lower():
                        continue
                    records.append(rec)
                except json.JSONDecodeError:
                    continue

        return records[-limit:]

    def get_negative_lessons(
        self,
        modality: Optional[str] = None,
        paradigm: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve accumulated negative lessons (what NOT to do)."""
        if not self.lessons_path.exists():
            return []

        try:
            with open(self.lessons_path, encoding="utf-8") as f:
                all_lessons = json.load(f)
        except (json.JSONDecodeError, OSError):
            return []

        if modality or paradigm:
            filtered = []
            for lesson in all_lessons:
                if modality and lesson.get("modality", "").lower() != modality.lower():
                    continue
                if paradigm and lesson.get("paradigm") and paradigm.lower() not in lesson["paradigm"].lower():
                    continue
                filtered.append(lesson)
            return filtered

        return all_lessons

    # ---------------------------------------------------------------- Negative examples

    def _negatives_path(self) -> Path:
        return self.store_dir / "negatives.jsonl"

    def record_negative(self, example: NegativeExample) -> None:
        """Append a structured negative example to ~/.easybci/experience/negatives.jsonl."""
        path = self._negatives_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(example.to_dict(), ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.warning("record_negative failed: %s", exc)

    def list_negatives(self) -> List[NegativeExample]:
        path = self._negatives_path()
        if not path.exists():
            return []
        out: List[NegativeExample] = []
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        out.append(NegativeExample(**d))
                    except (json.JSONDecodeError, TypeError, ValueError) as exc:
                        logger.debug("skipping corrupted negative: %s", exc)
        except OSError as exc:
            logger.warning("list_negatives read failed: %s", exc)
        return out

    def find_relevant_negatives(
        self,
        *,
        modality: str,
        paradigm: str,
        cohort_tag: str,
        analysis_goal: str,
    ) -> List[NegativeExample]:
        """Modality hard filter + OR-matching on (paradigm, cohort_tag, analysis_goal).

        Modality is required to match exactly when the query supplies one —
        without this, a single ``(eeg, motor_imagery, ica)`` negative would
        pollute every modality's query through the OR clauses. When the query
        modality is empty the modality gate is open (caller hasn't classified
        the data yet); the OR fallback then still requires at least one of
        the remaining three fields to match.
        """
        all_neg = self.list_negatives()
        out: List[NegativeExample] = []
        for e in all_neg:
            if modality and e.modality != modality:
                continue  # modality hard filter
            if (
                (paradigm and e.paradigm == paradigm)
                or (cohort_tag and e.cohort_tag == cohort_tag)
                or (analysis_goal and e.analysis_goal == analysis_goal)
            ):
                out.append(e)
        return out

    def revoke_negative(self, negative_id: str, *, reason: str) -> bool:
        """Remove a negative example by id. Writes a revoke event to audit log.
        Returns True if found-and-removed; False if not found.
        """
        all_neg = self.list_negatives()
        match = [e for e in all_neg if e.id == negative_id]
        if not match:
            return False
        remaining = [e for e in all_neg if e.id != negative_id]
        path = self._negatives_path()
        try:
            tmp = path.with_suffix(".jsonl.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                for e in remaining:
                    f.write(json.dumps(e.to_dict(), ensure_ascii=False) + "\n")
            tmp.replace(path)
        except OSError as exc:
            logger.warning("revoke_negative write failed: %s", exc)
            return False
        # Audit revoke event — reuse the internalization audit schema, but
        # write to a sibling file so events stay properly sourced.
        if _AuditEvent is not None and _InternalizationAuditLog is not None:
            try:
                audit = _InternalizationAuditLog(path=self.store_dir / "negative_audit.jsonl")
                audit.append(_AuditEvent(
                    event="revoke",
                    internalization_id=negative_id,
                    skill_path=str(path),
                    source_url="",
                    content_excerpt=match[0].failure_evidence[:200],
                    timestamp_iso=_dt.datetime.utcnow().isoformat() + "Z",
                    confidence="",
                    revoke_reason=reason,
                ))
            except Exception:  # noqa: BLE001
                logger.debug("negative audit write failed", exc_info=True)
        return True

    def get_stats(self) -> Dict[str, Any]:
        """Return summary statistics of accumulated experience."""
        records = self.load_records(limit=10000)
        if not records:
            return {"n_records": 0}

        modalities = set()
        paradigms = set()
        success_count = 0
        total_retries = 0

        for rec in records:
            modalities.add(rec.get("modality", "unknown"))
            paradigms.add(rec.get("paradigm", "unknown"))
            if rec.get("outcome", {}).get("success"):
                success_count += 1
            total_retries += rec.get("retries", {}).get("count", 0)

        lessons = self.get_negative_lessons()

        return {
            "n_records": len(records),
            "modalities": sorted(modalities),
            "paradigms": sorted(paradigms),
            "success_rate": round(success_count / len(records), 3) if records else 0,
            "avg_retries": round(total_retries / len(records), 1) if records else 0,
            "n_negative_lessons": len(lessons),
        }

    def distill_knowledge(
        self, modality: Optional[str] = None, paradigm: Optional[str] = None
    ) -> Dict[str, Any]:
        """Extract actionable knowledge from accumulated records.

        Returns patterns like:
        - Most common successful pipeline sequences
        - Most common failure modes and their fixes
        - Steps that frequently get added during retries
        - Parameter values that work best for this data type
        """
        records = self.load_records(modality=modality, paradigm=paradigm, limit=500)
        if len(records) < 3:
            return {"insufficient_data": True, "n_records": len(records)}

        # Most common successful pipelines
        successful_pipelines = []
        retry_patterns = {}
        step_frequency = {}

        for rec in records:
            final_steps = rec.get("pipeline", {}).get("final", [])
            outcome = rec.get("outcome", {})

            if outcome.get("success"):
                steps_key = " → ".join(final_steps)
                successful_pipelines.append(steps_key)

            # Track which steps get added during retries
            added = rec.get("pipeline", {}).get("added", [])
            for step in added:
                step_name = step.split(":")[0]
                retry_patterns[step_name] = retry_patterns.get(step_name, 0) + 1

            for step in final_steps:
                step_name = step.split(":")[0]
                step_frequency[step_name] = step_frequency.get(step_name, 0) + 1

        # Find most common pipelines
        from collections import Counter
        pipeline_counts = Counter(successful_pipelines)
        top_pipelines = pipeline_counts.most_common(5)

        # Common retry additions
        top_retry_steps = sorted(retry_patterns.items(), key=lambda x: x[1], reverse=True)[:5]

        # QC failure patterns
        qc_issues_freq = {}
        for rec in records:
            for issue in rec.get("qc", {}).get("issues", []):
                qc_issues_freq[issue] = qc_issues_freq.get(issue, 0) + 1
        top_qc_issues = sorted(qc_issues_freq.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            "n_records_analyzed": len(records),
            "top_successful_pipelines": [
                {"pipeline": p, "count": c} for p, c in top_pipelines
            ],
            "common_retry_additions": [
                {"step": s, "count": c} for s, c in top_retry_steps
            ],
            "common_qc_issues": [
                {"issue": i, "count": c} for i, c in top_qc_issues
            ],
            "step_usage_frequency": dict(sorted(step_frequency.items(), key=lambda x: x[1], reverse=True)[:10]),
            "negative_lessons": self.get_negative_lessons(modality=modality, paradigm=paradigm),
        }

    def _append_lessons(self, lessons: List[str], record: ProcessingRecord) -> None:
        """Persist negative lessons with context."""
        existing = []
        if self.lessons_path.exists():
            try:
                with open(self.lessons_path, encoding="utf-8") as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, OSError):
                existing = []

        for lesson_text in lessons:
            existing.append({
                "lesson": lesson_text,
                "modality": record.modality,
                "paradigm": record.paradigm,
                "context": {
                    "n_channels": record.n_channels,
                    "frequency": record.frequency,
                    "duration_s": record.duration_s,
                },
                "timestamp": record.timestamp,
            })

        with open(self.lessons_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)


def create_processing_record(
    data_path: str,
    modality: str,
    paradigm: str,
    initial_steps: List[str],
    final_steps: List[str],
    qc_result: Optional[Dict[str, Any]] = None,
    qc_metrics: Optional[Dict[str, Any]] = None,
    data_profile: Optional[Dict[str, Any]] = None,
    retry_events: Optional[List[Dict[str, Any]]] = None,
    start_time: Optional[float] = None,
    success: bool = True,
    n_channels: int = 0,
    frequency: float = 0.0,
    duration_s: float = 0.0,
    stage: str = "",
) -> ProcessingRecord:
    """Convenience constructor for ProcessingRecord from tool outputs.

    Automatically extracts: steps_added, steps_removed, QC grade/score,
    negative lessons from retry patterns.
    """
    now = time.time()
    record = ProcessingRecord(
        timestamp=now,
        data_path=data_path,
        modality=modality,
        paradigm=paradigm,
        n_channels=n_channels,
        frequency=frequency,
        duration_s=duration_s,
        initial_steps=initial_steps,
        final_steps=final_steps,
        success=success,
        stage=stage,
        duration_elapsed_s=(now - start_time) if start_time else 0,
    )

    # Compute steps diff
    initial_set = set(initial_steps)
    final_set = set(final_steps)
    record.steps_added = [s for s in final_steps if s not in initial_set]
    record.steps_removed = [s for s in initial_steps if s not in final_set]

    # QC info
    if qc_metrics:
        overall = qc_metrics.get("overall", {})
        record.qc_grade = overall.get("grade", "")
        record.qc_score = overall.get("score", 0)
        record.qc_issues_final = overall.get("warnings", [])
        record.qc_passed = record.qc_grade in ("A", "B")

    if qc_result:
        record.qc_passed = qc_result.get("passed", False)
        issues = qc_result.get("issues", [])
        record.qc_issues_final = [i.get("detail", "") for i in issues]

    # Data profile summary
    if data_profile:
        record.data_profile_summary = {
            "quality_score": data_profile.get("scores", {}).get("quality", 0),
            "noise_score": data_profile.get("scores", {}).get("noise", 0),
            "powerline_present": data_profile.get("powerline", {}).get("present", False),
            "n_bad_channels": data_profile.get("channels", {}).get("n_bad", 0),
        }

    # Retry events
    if retry_events:
        record.n_retries = len(retry_events)
        for evt in retry_events:
            record.retry_events.append(RetryEvent(
                attempt=evt.get("attempt", 0),
                trigger=evt.get("trigger", "unknown"),
                error_message=evt.get("error", ""),
                qc_issues=evt.get("qc_issues", []),
                steps_before=evt.get("steps_before", []),
                steps_after=evt.get("steps_after", []),
                parameter_changes=evt.get("parameter_changes", {}),
                result=evt.get("result", ""),
            ))

    # Extract negative lessons from failure patterns
    record.negative_lessons = _extract_negative_lessons(record)

    return record


def _extract_negative_lessons(record: ProcessingRecord) -> List[str]:
    """Identify negative lessons from a processing record."""
    lessons = []

    # Lesson: step was removed because it caused problems
    for step in record.steps_removed:
        step_name = step.split(":")[0]
        lessons.append(
            f"Step '{step}' was removed for {record.modality}/{record.paradigm} data "
            f"({record.n_channels}ch, {record.frequency}Hz)"
        )

    # Lesson: multiple retries suggest initial pipeline was wrong
    if record.n_retries >= 3 and not record.success:
        lessons.append(
            f"Pipeline failed after {record.n_retries} retries for "
            f"{record.modality}/{record.paradigm} ({record.n_channels}ch). "
            f"Initial steps: {record.initial_steps}"
        )

    # Lesson: low QC despite success (marginal quality)
    if record.success and record.qc_score < 0.5:
        lessons.append(
            f"Pipeline '{' → '.join(record.final_steps)}' produced low quality "
            f"(score={record.qc_score:.2f}) for {record.modality}/{record.paradigm}"
        )

    return lessons
