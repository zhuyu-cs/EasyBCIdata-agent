"""Auto-finalize helper — the safety net behind the contract guarantee.

The user-facing contract (pipeline skill, plan/03-phase2-auto-finalize.md)
says *every* neural preprocessing run leaves behind a mini-repo at
``{subject}_preprocess_work_dir/`` that satisfies the canonical layout
(``preprocessed_output/``, ``code/``, ``plan/``, ``README.md``).

The LLM-orchestrated path achieves this by calling the ``export_repo`` tool
as Step 8 of the pipeline skill. But when the LLM stops early
(debug-and-stop), the user hits Ctrl-C, or the run fails on an exception, no
such tool call happens and the work_dir is left in a contract-violating state.

This module is invoked from finally blocks in:
- ``run_agent.py:main`` (CLI entry — wraps ``agent.run_conversation``)
- ``gateway/platforms/api_server.py:_maybe_finalize_neural_run`` (WebUI hook
  that already fires after every ``run_conversation`` return).

Two responsibilities:

1. Track which work_dir the current LLM run is touching, so the finally hook
   knows where to point ``build_mini_repo``. Tool handlers
   (``_handle_propose_pipeline``, ``_handle_export_repo``, ...) call
   :func:`set_current_work_dir`; the finalize hook calls
   :func:`get_current_work_dir`.

2. Run ``build_mini_repo`` with the right ``status`` and swallow any
   exception — finalize failures must never cascade into the user-visible
   run failure.

The registry is thread-local because the Gateway runs each request on its own
thread (``loop.run_in_executor``) and there can be multiple concurrent runs
across sessions.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_state = threading.local()

# Optional callback invoked by ``set_current_work_dir`` so layers above (the
# gateway / CLI session store) can persist the path against the active session
# without this module needing to import any DB or runtime types.  The callback
# receives the resolved path string; exceptions are swallowed.
_persist_callback: Optional[callable] = None  # type: ignore[valid-type]


def register_work_dir_persist_callback(cb) -> None:
    """Install a callback that fires whenever set_current_work_dir is called.

    Designed for the gateway/CLI to persist the active mini-repo work_dir on
    the session row so the WebUI ``/artifacts`` endpoint can recover it after
    a restart. Pass ``None`` to clear.
    """
    global _persist_callback
    _persist_callback = cb


def set_current_work_dir(work_dir: str | Path | None) -> None:
    """Record the most recent neural work_dir touched by the active run.

    Called by tool handlers that create or operate on a mini-repo
    (``propose_pipeline``, ``export_repo``, ``finalize``). Safe to call with
    ``None`` to clear (e.g. on a chat-only session reset).
    """
    if work_dir is None:
        _state.work_dir = None
        if _persist_callback is not None:
            try:
                _persist_callback(None)
            except Exception:
                logger.debug("work_dir persist callback failed for None", exc_info=True)
        return
    try:
        resolved = str(Path(work_dir))
        _state.work_dir = resolved
    except Exception:
        # Defensive: any path conversion failure shouldn't break tool handlers.
        logger.debug("set_current_work_dir: failed to record %r", work_dir)
        return
    if _persist_callback is not None:
        try:
            _persist_callback(resolved)
        except Exception:
            logger.debug("work_dir persist callback failed for %s", resolved, exc_info=True)


def get_current_work_dir() -> Optional[str]:
    """Return the most recent work_dir tracked for the current thread.

    Returns ``None`` when no neural preprocessing tool ran in this thread.
    """
    return getattr(_state, "work_dir", None)


def clear_current_work_dir() -> None:
    """Reset the thread-local work_dir. Call between independent runs."""
    _state.work_dir = None


# Lazy import of SessionDB inside maybe_archive_completed_work_dir avoids
# pulling the SQLite layer at finalize import time. Tests monkeypatch this
# module-level attribute to inject an isolated instance.
try:
    from easybci_lib.state import SessionDB  # noqa: F401
except Exception:  # noqa: BLE001
    SessionDB = None  # type: ignore[assignment]


def _next_run_suffix(work_dir: Path) -> Path:
    """Return the first unused ``{work_dir.name}_runN`` sibling path."""
    n = 1
    while True:
        cand = work_dir.with_name(work_dir.name + f"_run{n}")
        if not cand.exists():
            return cand
        n += 1


def maybe_archive_completed_work_dir(
    work_dir: str | Path, session_id: Optional[str]
) -> Optional[Path]:
    """Archive *work_dir* to ``{work_dir}_runN`` iff it is a completed
    (``status="ok"``) run not yet archived in this session.

    Returns the archived path, or None when no archive was performed.

    Session-scoped idempotent: same ``(session_id, work_dir)`` pair archives
    only once. Pass ``session_id=None`` to bypass the idempotency check
    (e.g. CLI debug path).
    """
    wd = Path(work_dir)
    record_path = wd / "plan" / "pipeline_record.json"
    if not record_path.exists():
        return None
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if record.get("status") != "ok":
        return None

    if session_id and SessionDB is not None:
        try:
            db = SessionDB()
            if db.has_archived_in_session(session_id, str(wd)):
                return None
        except Exception as exc:
            logger.debug("archive idempotency check failed: %s", exc)
            # Fall through — better to archive than skip silently.

    new_path = _next_run_suffix(wd)
    try:
        wd.rename(new_path)
    except OSError:
        # Cross-volume rename fails → fall back to shutil.move.
        shutil.move(str(wd), str(new_path))

    if session_id and SessionDB is not None:
        try:
            SessionDB().record_archived(
                session_id, original=str(wd), archived_to=str(new_path)
            )
        except Exception as exc:
            logger.debug("archive record persistence failed: %s", exc)

    logger.info("Archived completed work_dir %s → %s", wd, new_path)
    return new_path


_PHYSIO_TOKENS = ("EOG", "ECG", "EMG", "RESP", "GSR")
_MARKER_TOKENS = ("TRIGGER", "STIM", "STATUS", "MARKER", "EVENT")


def _proposal_steps_as_strings(proposal: dict) -> list[str]:
    """Convert proposal.json's structured ``steps`` array to the
    ``operator:k=v;k=v`` string form expected by build_mini_repo
    (matches the shape codegen / config.yaml / reasoning.md consume).
    """
    out: list[str] = []
    for s in proposal.get("steps") or []:
        if not isinstance(s, dict):
            continue
        op = s.get("operator", "")
        if not op:
            continue
        params = s.get("params") or {}
        # `propose_pipeline` stores params under a "raw" key for the common
        # single-value case (e.g. ``{"raw": "50"}`` for notch:50). Unwrap so
        # the rendered string is ``notch:50`` not ``notch:raw=50``.
        if isinstance(params, dict) and set(params.keys()) == {"raw"}:
            value = params["raw"]
            out.append(f"{op}:{value}" if value not in (None, "") else op)
            continue
        if params:
            pstr = ";".join(f"{k}={v}" for k, v in params.items())
            out.append(f"{op}:{pstr}")
        else:
            out.append(op)
    return out


def _load_proposal(wd: Path) -> Optional[dict]:
    """Read ``<wd>/plan/proposal.json`` (the canonical record of what
    propose_pipeline + the user agreed to run). Returns None if missing
    or unparseable."""
    path = wd / "plan" / "proposal.json"
    if not path.is_file():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _load_inspection_fingerprint(wd: Path) -> tuple[dict, str]:
    """Read fingerprint + source data_path from
    ``<wd>/middle_process/inspection_report.json`` first, then fall back to
    ``<wd>/plan/proposal.json``'s embedded ``data_info`` / ``input_path``
    fields (propose_pipeline copies the fingerprint there for survival
    when middle_process/ is later cleaned up by Step 14). Returns
    ``({}, "")`` if both sources are unreadable / empty.

    Maps schema → build_mini_repo's data_info shape:
      - ``fingerprint.n_channels``        → ``n_channels``
      - ``fingerprint.sampling_freq_hz``  → ``frequency_hz``
      - ``fingerprint.duration_s``        → ``duration_seconds``

    Used by finalize_work_dir to backfill README Source-file / Channels /
    Sampling-rate / Duration rows when the agent stops before
    export_repo could capture them.
    """
    # Primary: inspection_report.json (current data, written by deep_inspect)
    path = wd / "middle_process" / "inspection_report.json"
    if path.is_file():
        try:
            with open(path, "r", encoding="utf-8") as f:
                report = json.load(f)
            if isinstance(report, dict):
                fp = report.get("fingerprint") or {}
                src = str(report.get("data_path") or "")
                if isinstance(fp, dict):
                    data_info = {
                        "n_channels": fp.get("n_channels"),
                        "frequency_hz": fp.get("sampling_freq_hz"),
                        "duration_seconds": fp.get("duration_s"),
                    }
                    # ``file`` = basename so reasoning.md's Data Fingerprint
                    # section ("**File**: ...") renders the source name instead
                    # of "unknown" when input_ref.json isn't on disk yet.
                    if src:
                        data_info["file"] = Path(src).name
                    data_info = {
                        k: v for k, v in data_info.items()
                        if v not in (None, "")
                    }
                    if data_info or src:
                        return data_info, src
        except (OSError, json.JSONDecodeError):
            pass

    # Fallback: plan/proposal.json["data_info"] + ["input_path"]
    # propose_pipeline embeds the fingerprint there so finalize survives
    # the case where middle_process/ has been cleaned up.
    proposal_path = wd / "plan" / "proposal.json"
    if proposal_path.is_file():
        try:
            with open(proposal_path, "r", encoding="utf-8") as f:
                proposal = json.load(f)
            if isinstance(proposal, dict):
                p_info_raw = proposal.get("data_info") or {}
                p_src = str(proposal.get("input_path") or "")
                p_info = {
                    k: v for k, v in p_info_raw.items()
                    if v not in (None, "")
                } if isinstance(p_info_raw, dict) else {}
                # Mirror the inspection branch — surface basename as "file"
                # so reasoning.md's Data Fingerprint row is populated.
                if p_src and "file" not in p_info:
                    p_info["file"] = Path(p_src).name
                return p_info, p_src
        except (OSError, json.JSONDecodeError):
            pass

    return {}, ""


def _infer_modality_from_channels(channels: list) -> str:
    """Heuristic: pick a modality string from raw channel names.

    Used only by the middle_process recovery path when the recorded
    pipeline_record lacks an explicit modality. Conservative — returns
    "unknown" rather than guessing wildly.
    """
    if not channels:
        return "unknown"
    upper = [str(c).upper() for c in channels]
    eeg_landmarks = ("FP1", "FPZ", "FP2", "FZ", "CZ", "PZ", "OZ", "O1", "O2")
    if any(name in upper for name in eeg_landmarks):
        return "eeg"
    if any(name.startswith("MEG") for name in upper):
        return "meg"
    if any(name.startswith("ECOG") for name in upper):
        return "ecog"
    if any(name.startswith("SEEG") or name.startswith("DEPTH") for name in upper):
        return "seeg"
    return "unknown"


def _recover_from_middle_process(wd: Path) -> dict:
    """Build a plan/-like record from middle_process/session*_pipeline_record.json.

    Used when finalize_work_dir is called but the LLM-driven path never wrote
    plan/pipeline_record.json (early stop, Ctrl-C, SSE drop). Picks the
    newest middle_process record by mtime and reconstructs the canonical
    field set so downstream build_mini_repo writes a real plan/config.yaml
    instead of an "unknown" husk.

    Returns ``{}`` when no usable middle_process record is found; finalize
    then falls back to the original husk path.

    Field mapping:
      - steps: ``record["steps_applied"]`` or stitched from ``step_states[*].step``
      - modality: ``record["modality"]``, else inferred from before-channel names
      - paradigm: ``record["paradigm"]``, else "unknown" (banner added later)
      - data_info: built from ``record["step_states"][0]["before"]``
      - input_path: ``record["input_path"]`` (plan/input_ref.json wins upstream)
      - analysis_goal: ``record["analysis_goal"]``
    """
    mp = wd / "middle_process"
    if not mp.is_dir():
        return {}

    candidates = sorted(
        mp.glob("session*_pipeline_record.json"),
        key=lambda p: p.stat().st_mtime if p.exists() else 0.0,
        reverse=True,
    )
    for src in candidates:
        try:
            record = json.loads(src.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.debug("middle_process record %s unreadable: %s", src, exc)
            continue
        if not isinstance(record, dict):
            continue

        steps = list(record.get("steps_applied") or [])
        if not steps:
            for entry in record.get("step_states") or []:
                if isinstance(entry, dict) and entry.get("step"):
                    steps.append(entry["step"])

        before = {}
        states = record.get("step_states") or []
        if states and isinstance(states[0], dict):
            before = states[0].get("before") or {}

        channels = list(before.get("channels") or record.get("channels") or [])
        modality = record.get("modality") or _infer_modality_from_channels(channels)
        paradigm = record.get("paradigm") or "unknown"

        # Hard constraint — refuse to write out a husk that downstream tools
        # would silently treat as a real result. Raise instead so the caller
        # (gateway → SSE error; CLI → non-zero exit + diagnostic) can surface
        # the failure clearly.
        if modality == "unknown" and paradigm == "unknown":
            from easybci_lib.tools.neural_processing.export.errors import (
                IncompleteRunError,
            )
            raise IncompleteRunError(
                "middle_process/session*_pipeline_record.json lacks both "
                "modality and paradigm (both resolved to 'unknown') — "
                "refusing to write a husk plan/.",
                work_dir=str(wd),
            )

        # Build data_info from step_states[0].before (the pre-pipeline snapshot).
        data_info = {
            "n_channels": before.get("n_channels"),
            "frequency_hz": before.get("frequency") or record.get("frequency"),
            "duration_seconds": before.get("duration_s"),
            "channels": channels,
        }
        # Drop empty fields so build_mini_repo's data_info renderer doesn't
        # write null placeholders into config.yaml.
        data_info = {k: v for k, v in data_info.items() if v not in (None, "")}

        result: dict = {
            "steps": steps,
            "modality": modality,
            "paradigm": paradigm,
            "data_info": data_info,
            "input_path": record.get("input_path") or "",
            "existing_record": record,
        }
        # Optional fields populated by later stages. Threaded through so
        # those writers don't need to re-touch finalize when they start writing
        # these into the middle_process record.
        if record.get("analysis_goal"):
            result["analysis_goal"] = record["analysis_goal"]
        if "web_evidence_used" in record:
            result["web_evidence_used"] = record.get("web_evidence_used")
        if record.get("web_evidence_provider"):
            result["web_evidence_provider"] = record.get("web_evidence_provider")

        logger.info(
            "_recover_from_middle_process: %s → modality=%s paradigm=%s steps=%d",
            src.name, modality, paradigm, len(steps),
        )
        return result

    return {}


def _should_keep_middle_process() -> bool:
    """User-facing knobs that preserve middle_process/ after a successful run."""
    val = os.environ.get("EASYBCI_KEEP_MIDDLE_PROCESS", "").strip().lower()
    return val in {"1", "true", "yes", "on"}


def _cleanup_middle_process(wd: Path, status: str) -> None:
    """Remove ``<wd>/middle_process/`` after a successful finalize.

    Only runs when ``status == "ok"`` — partial / migrated / failed runs keep
    middle_process so the user can inspect crashed-pipeline artifacts.
    Honours ``EASYBCI_KEEP_MIDDLE_PROCESS=1`` for forensic / opt-out cases.
    Best-effort: any error is logged and swallowed, never re-raised.
    """
    if status != "ok":
        return
    if _should_keep_middle_process():
        logger.debug(
            "EASYBCI_KEEP_MIDDLE_PROCESS=1; preserving %s/middle_process/", wd,
        )
        return
    mp = wd / "middle_process"
    if not mp.exists() or not mp.is_dir():
        return
    try:
        shutil.rmtree(mp)
        logger.info("finalize: removed %s after successful run", mp)
    except OSError as exc:  # noqa: BLE001
        logger.warning(
            "finalize: failed to remove %s (%s) — leaving in place", mp, exc,
        )


def _cleanup_orphaned_subject_debris(wd: Path, status: str) -> None:
    """Sweep obviously-orphaned subject directories from a successful run.

    Only removes ``sub-data/`` — the canonical 'no identity, no fallback'
    debris dir name from pre-identity pipeline runs. Other sub-* dirs are
    left alone because they might be real subjects from earlier runs the
    user explicitly wants to keep around.
    """
    if status != "ok":
        return
    if _should_keep_middle_process():
        return
    preproc = wd / "preprocessed_output" / "preprocessed"
    orphan = preproc / "sub-data"
    if orphan.is_dir():
        try:
            shutil.rmtree(orphan)
            logger.info("finalize: removed orphan %s from pre-identity run", orphan)
        except OSError as exc:
            logger.warning("finalize: failed to remove %s: %s", orphan, exc)


def finalize_work_dir(
    work_dir: str | Path | None,
    status: str = "ok",
    partial_reason: Optional[str] = None,
) -> Optional[dict]:
    """Best-effort wrapper around ``build_mini_repo`` for finally hooks.

    - Returns ``None`` (silently) if ``work_dir`` is falsy / doesn't exist —
      a chat-only session that never touched preprocessing must not produce a
      spurious mini-repo.
    - Calls ``build_mini_repo`` with whatever metadata can be reconstructed
      from existing artifacts. Passes ``status`` and ``partial_reason``
      through so the resulting record is correctly tagged.
    - Catches every exception and downgrades to ``logger.warning`` — the
      user's run must not fail because finalize failed.
    """
    if not work_dir:
        return None
    wd = Path(work_dir)
    if not wd.exists() or not wd.is_dir():
        return None

    try:
        from easybci_lib.tools.neural_processing.export.repo_builder import build_mini_repo

        # Try to reconstruct steps + pipeline_record from any pre-existing
        # plan/pipeline_record.json the LLM may have produced via export_repo
        # earlier in the run. If nothing is there, fall back to empty values —
        # build_mini_repo with status="partial" still writes a minimal record.
        existing_record: dict = {}
        steps: list = []
        # Default to "unknown" (not "eeg") so partial/auto-finalize on a
        # non-EEG run (MEG/sEEG/spike) doesn't stamp a wrong modality into
        # the README and pipeline_record. Real value, when known, is
        # restored below from any prior pipeline_record.json the LLM wrote.
        modality = "unknown"
        paradigm = "unknown"
        analysis_goal = ""
        record_path = wd / "plan" / "pipeline_record.json"
        if record_path.exists():
            try:
                existing_record = json.loads(record_path.read_text(encoding="utf-8"))
                steps = list(existing_record.get("steps") or [])
                modality = existing_record.get("modality") or modality
                paradigm = existing_record.get("paradigm") or paradigm
                analysis_goal = existing_record.get("analysis_goal") or ""
            except Exception:
                existing_record = {}

        # Try to recover data_info / input_path from plan/input_ref.json.
        input_path = ""
        data_info: dict = {}
        input_ref_path = wd / "plan" / "input_ref.json"
        if input_ref_path.exists():
            try:
                ref = json.loads(input_ref_path.read_text(encoding="utf-8"))
                input_path = ref.get("path", "")
            except Exception:
                pass

        # mark_proposal_confirmed materializes plan/goal.json
        # from the staged envelope as the canonical post-confirm carrier of
        # analysis_goal. Read it as a second source of truth (after
        # pipeline_record) so finalize can surface the goal even when only a
        # husk pipeline_record exists. Pre-confirm runs have no goal.json
        # here — finalize then falls through to "generic" further below.
        if not analysis_goal:
            goal_path = wd / "plan" / "goal.json"
            if goal_path.exists():
                try:
                    payload = json.loads(goal_path.read_text(encoding="utf-8"))
                    analysis_goal = (payload.get("analysis_goal") or "").strip()
                    # mark_proposal_confirmed also stores modality/paradigm
                    # here — use them as a third-tier fallback so finalize
                    # produce a self-describing plan/ when middle_process
                    # has no record.
                    if modality == "unknown":
                        modality = payload.get("modality") or modality
                    if paradigm == "unknown":
                        paradigm = payload.get("paradigm") or paradigm
                except (OSError, json.JSONDecodeError):
                    pass

        # When plan/web_evidence.json exists (written by
        # _handle_propose_pipeline), buffer it for build_mini_repo. We DO
        # NOT mutate existing_record yet — that would short-circuit the
        # `if not existing_record` middle_process recovery below.
        deferred_web_evidence: Optional[dict] = None
        ev_path = wd / "plan" / "web_evidence.json"
        if ev_path.exists():
            try:
                evidence_payload = json.loads(ev_path.read_text(encoding="utf-8"))
                if isinstance(evidence_payload, dict):
                    deferred_web_evidence = evidence_payload
            except (OSError, json.JSONDecodeError):
                logger.debug("plan/web_evidence.json unreadable; skipping")

        # When plan/pipeline_record.json was never written (LLM early stop /
        # Ctrl-C / SSE drop), fall back to middle_process/session*_pipeline_record.json
        # so the rebuilt plan/ carries real modality/paradigm/steps/data_info
        # instead of the "unknown" husk. The fallback ONLY fires when the
        # primary path produced an empty existing_record — never override a
        # successful read.
        if not existing_record:
            recovered = _recover_from_middle_process(wd)
            if recovered:
                existing_record = recovered.get("existing_record") or {}
                steps = recovered.get("steps") or steps
                # Only adopt recovered modality/paradigm when the value is
                # a real signal — _recover_from_middle_process returns
                # "unknown" placeholders when the source record lacks the
                # field, and we don't want those to override anything we
                # already learned from goal.json / pipeline_record.
                _rec_modality = recovered.get("modality")
                if _rec_modality and _rec_modality != "unknown" and modality == "unknown":
                    modality = _rec_modality
                _rec_paradigm = recovered.get("paradigm")
                if _rec_paradigm and _rec_paradigm != "unknown" and paradigm == "unknown":
                    paradigm = _rec_paradigm
                if not data_info and recovered.get("data_info"):
                    data_info = recovered["data_info"]
                if not input_path and recovered.get("input_path"):
                    input_path = recovered["input_path"]
                if not analysis_goal and recovered.get("analysis_goal"):
                    analysis_goal = recovered["analysis_goal"]
                # Stamp recovery-source fields into the record we hand to
                # build_mini_repo so downstream tooling can tell that plan/
                # was reconstructed (rather than written by a clean export_repo).
                existing_record.setdefault("recovered_from", "middle_process")
                if "steps" not in existing_record:
                    existing_record["steps"] = steps
                # build_mini_repo writes pipeline_record.json verbatim and
                # does NOT merge top-level metadata in. Stamp modality /
                # paradigm into the record itself so the regenerated plan/
                # is self-describing (otherwise rec.modality stays None).
                if not existing_record.get("modality"):
                    existing_record["modality"] = modality
                if not existing_record.get("paradigm"):
                    existing_record["paradigm"] = paradigm
                if recovered.get("analysis_goal"):
                    existing_record["analysis_goal"] = recovered["analysis_goal"]
                if "web_evidence_used" in recovered:
                    existing_record["web_evidence_used"] = recovered["web_evidence_used"]
                if recovered.get("web_evidence_provider"):
                    existing_record["web_evidence_provider"] = recovered[
                        "web_evidence_provider"
                    ]

        if not analysis_goal:
            analysis_goal = "generic"

        # Now that recovery has run, fold the deferred web_evidence (if any)
        # into existing_record so build_mini_repo has the full picture.
        if deferred_web_evidence and "web_evidence" not in existing_record:
            existing_record["web_evidence"] = deferred_web_evidence

        # === plan/proposal.json — the authoritative steps + rationale ===
        # propose_pipeline writes the full operator list + per-step rationale
        # the user already confirmed (mark_proposal_confirmed). It MUST win
        # over finalize's generic_defaults husk (the 2-step "drop_bads /
        # drop_nondata_channels" fallback). Without this, README's pipeline
        # table and reasoning.md both lose the real 9-step plan whenever
        # preprocess_neural fails and finalize falls through to bootstrap.
        proposal = _load_proposal(wd)
        force_rebuild = False
        if proposal:
            proposal_steps = _proposal_steps_as_strings(proposal)
            # Authoritative replacement: when the proposal has real steps
            # and what we computed so far is shorter (i.e. came from
            # generic_defaults), use the proposal. Equal-length-or-longer
            # cases trust the recovered record (e.g. middle_process had
            # the post-execution step list with auto-injected cleanups).
            if proposal_steps and len(proposal_steps) > len(steps):
                steps = proposal_steps
                existing_record["steps"] = proposal_steps
                # The on-disk record may be a stale "ok-status with 2-step
                # generic_defaults" husk that wasn't tagged with
                # ``auto_synthesized``. We need build_mini_repo to actually
                # rewrite it instead of short-circuiting. ``force=True``
                # tells the build to skip both the status no-op and the
                # manifest cache.
                force_rebuild = True
            # Per-step rationale → reasoning dict for repo_builder's
            # reasoning.md fallback to render real explanations rather
            # than generic boilerplate. Store under BOTH the full
            # "operator:params" string AND the bare operator name —
            # repo_builder looks up ``reasoning.get(step) or
            # reasoning.get(step_name)`` so codegen-injected param
            # rewrites (e.g. drop_nondata_channels:markers_only →
            # drop_nondata_channels:data_only via _enforce_clean_output)
            # still hit the right rationale via the operator-name key.
            rationale = proposal.get("rationale")
            if isinstance(rationale, list) and rationale:
                # Build {operator: rationale_text} from the ORIGINAL proposal
                # ordering (not the post-codegen `steps` list), so the index
                # alignment is preserved.
                proposal_step_dicts = proposal.get("steps") or []
                reasoning_map: dict[str, str] = {}
                for i, sdict in enumerate(proposal_step_dicts):
                    if i >= len(rationale):
                        break
                    if not isinstance(sdict, dict):
                        continue
                    op = sdict.get("operator", "")
                    text = rationale[i] if isinstance(rationale[i], str) else ""
                    if not op or not text:
                        continue
                    # Full "operator:params" key — exact match path
                    params = sdict.get("params") or {}
                    if isinstance(params, dict) and set(params.keys()) == {"raw"}:
                        value = params["raw"]
                        full = f"{op}:{value}" if value not in (None, "") else op
                    elif params:
                        full = f"{op}:" + ";".join(f"{k}={v}" for k, v in params.items())
                    else:
                        full = op
                    reasoning_map[full] = text
                    # Operator-only key as fallback, set only if not already
                    # taken — preserves the FIRST occurrence's rationale
                    # when the proposal lists the same operator twice
                    # (e.g. drop_nondata_channels at start and end).
                    reasoning_map.setdefault(op, text)
                if reasoning_map and "reasoning" not in existing_record:
                    existing_record["reasoning"] = reasoning_map
            # proposal carries modality/paradigm too — adopt if still
            # missing or "unknown".
            if modality in ("", "unknown"):
                modality = proposal.get("modality") or modality
            if paradigm in ("", "unknown"):
                paradigm = proposal.get("paradigm") or paradigm

        # === middle_process/inspection_report.json — data_info source ===
        # When the agent did inspect_data + deep_inspect but stopped before
        # export_repo, the fingerprint is on disk but data_info hasn't been
        # threaded through. Fill it so README's "Input Data" table shows
        # real channels / sampling rate / duration instead of "?". Falls
        # back to plan/proposal.json["data_info"] when middle_process/ has
        # been cleaned by Step 14.
        if not data_info or not input_path:
            insp_data_info, insp_src = _load_inspection_fingerprint(wd)
            if insp_data_info and not data_info:
                data_info = insp_data_info
                # data_info recovery is a meaningful improvement over what
                # the on-disk pipeline_record.json holds — force a rebuild
                # so README / reasoning.md "Data Fingerprint" rows reflect
                # it instead of staying at "?". (Otherwise build_mini_repo's
                # status-ok no-op would short-circuit the recovery.)
                force_rebuild = True
            if insp_src and not input_path:
                input_path = insp_src
                # input_path recovery enables plan/input_ref.json generation
                # via write_input_hash — same force-rebuild rationale.
                force_rebuild = True

        result = build_mini_repo(
            output_dir=str(wd),
            steps=steps,
            data_info=data_info,
            pipeline_record=existing_record,
            input_path=input_path,
            modality=modality,
            paradigm=paradigm,
            status=status,
            partial_reason=partial_reason,
            analysis_goal=analysis_goal,
            force=force_rebuild,
        )
        logger.info(
            "finalize_work_dir: %s finalized (status=%s, reason=%s)",
            wd, status, partial_reason,
        )
        # After a successful build, sweep transient debris so the final
        # mini-repo matches the user-facing contract (plan/, code/,
        # preprocessed_output/, README.md). Failures here are best-effort
        # — never re-raise, the run is already considered successful.
        try:
            # Trust the requested status — when caller passed status="ok"
            # and build_mini_repo returned without raising, the run is
            # considered successful for cleanup purposes. build_mini_repo's
            # own ``status`` field in the result reflects DIRECTORY-level
            # status (was the mini-repo populated?), not run-level success.
            if status == "ok" and result is not None:
                _cleanup_middle_process(wd, "ok")
                _cleanup_orphaned_subject_debris(wd, "ok")
        except Exception as cleanup_exc:  # noqa: BLE001
            logger.warning(
                "finalize_work_dir: post-build cleanup failed for %s: %s",
                wd, cleanup_exc,
            )
        # === Final layout enforcement (defense in depth) ===
        # build_mini_repo has already run _salvage_layout at its start. The
        # verify_and_repair sweep here handles drift introduced BY
        # build_mini_repo itself (unlikely) or by outside code paths that
        # bypassed the tool path. allow_subprocess=True — finalize is not
        # on a hot tool-return path; we can afford to re-run vis.py / qc.py
        # if that's what it takes to satisfy the contract.
        try:
            from easybci_lib.tools.neural_processing.export.layout_repair import (
                verify_and_repair,
            )
            repair_report = verify_and_repair(
                wd, analysis_goal=analysis_goal,
                allow_subprocess=True, dry_run=False, write_report=True,
            )
            # Stamp a compact summary into pipeline_record.json for observability.
            rec_path = wd / "plan" / "pipeline_record.json"
            if rec_path.is_file():
                try:
                    rec = json.loads(rec_path.read_text(encoding="utf-8"))
                    rec["layout_repair"] = {
                        "initial_violations": repair_report.initial_violations,
                        "remaining_violations": repair_report.remaining_violations,
                        "rounds": repair_report.rounds,
                        "unrepairable": repair_report.unrepairable,
                    }
                    rec_path.write_text(json.dumps(rec, indent=2), encoding="utf-8")
                except (OSError, json.JSONDecodeError) as _stamp_exc:
                    logger.debug(
                        "stamping layout_repair into pipeline_record failed: %s",
                        _stamp_exc,
                    )
        except Exception as _repair_exc:  # noqa: BLE001
            logger.warning(
                "finalize_work_dir: layout verify_and_repair failed for %s: %s",
                wd, _repair_exc,
            )
        return result
    except Exception as exc:
        # IncompleteRunError must propagate so the gateway/CLI exit handler
        # can surface "no real run" to the user — swallowing it would
        # silently write out the husk plan/ we are trying to prevent.
        from easybci_lib.tools.neural_processing.export.errors import (
            IncompleteRunError,
        )
        if isinstance(exc, IncompleteRunError):
            raise
        # CRITICAL: never re-raise. The user's run must not crash because
        # the safety net itself crashed.
        logger.warning(
            "finalize_work_dir: build_mini_repo failed for %s (status=%s): %s",
            wd, status, exc, exc_info=True,
        )
        return None
