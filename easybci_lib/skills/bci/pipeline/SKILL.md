---
name: pipeline
description: "BCI data preprocessing workflow router — routes to new_pipeline or reference_adaptive based on proven skill availability"
layer: L1
metadata:
  tags: [orchestrator, preprocessing, pipeline, bci, high-priority]
  modalities: [eeg, seeg, ecog, meg, spike, fnirs]
  analysis_goal_allowed: [classification, source_localization, feature_extraction, clinical_screening, exploratory, generic, connectivity, phase_amplitude_coupling, online_inference, sleep_staging]
  analysis_goal_forbidden: []
---

# BCI Data Preprocessing — Workflow Router

Triggers on ANY data preprocessing request. Routes to one of two workflows.

## Route Decision (execute IN THIS SKILL before loading any workflow)

You MUST complete steps 1–3 below BEFORE calling `skill_view` on any workflow. Do NOT load a workflow speculatively.

**Step 1 — LIGHT INSPECT:** `inspect_data(data_path=<path>)` — confirm the file is readable, get modality/channels/fs.

**Step 2 — DEEP INSPECT + WORK DIR:** `mkdir -p "<work_dir>"` then `deep_inspect(data_path=<path>, work_dir=<work_dir>)` — writes inspection_report + routing entry.

**Step 3 — ROUTE:**

| Condition | Workflow |
|-----------|----------|
| User provides reference project path | **reference_adaptive** → `skill_view('pipeline/workflows/reference_adaptive')` |
| `plan_pipeline(inspection_report_path=...)` returns `proven_recommendation` (similarity ≥ 0.6) | **reference_adaptive** → `skill_view('pipeline/workflows/reference_adaptive')` |
| No proven match / `proven_reuse_rejected` / similarity < 0.6 | **new_pipeline** → `skill_view('pipeline/workflows/new_pipeline')` |

**Step 4 — LOAD WORKFLOW:** Only NOW call `skill_view` for the chosen workflow. Follow it step-by-step from its own Step 1 onward (skip steps you already did here — inspect/deep_inspect/work_dir are done). Do NOT mix steps across workflows.

## ABSOLUTE PROHIBITIONS (both workflows)

1. **Source data is READ-ONLY** — never write/modify/move/rename input files. `source_data_guard.py` enforces 4 layers.
2. **work_dir is sealed** — only tool-chain writes allowed inside `*_preprocess_work_dir/`. `skill_compliance_guard.py` default-deny.
3. **`middle_process/` lives at `<work_dir>/middle_process/`, NEVER under `<work_dir>/code/`** — Step 14 cleanup depends on this.
4. **Identity from routing table, NEVER from `Path(raw).stem`** — `identity_resolver` runs once in `deep_inspect`; downstream MUST read `inputs_routing.json`.
5. **One canonical script per stage** — `pipeline.py`/`qc.py`/`vis.py`/`build_ai_ready.py`/`run.py` each written ONCE, loop internally. No per-session variants unless user EXPLICITLY asks.
6. **Output format: `preprocessed/` = NWB-only; `AI_ready/` = pkl-only** — other extensions auto-swept.
7. **Skill library is READ-ONLY** except `proven-pipelines/` category — no patching/writing to `bci/pipeline/`, `bci/operators/`, `bci/paradigms/`, etc.
8. **Deterministic seed 42** — all random operations pinned.
9. **Layout contract is highest-priority (equal to correctness)** — use `repair_layout`, never manual `mv`/`rm`/`mkdir`.
10. **Path safety** — always wrap `work_dir` and data paths in double quotes in terminal commands.
11. **`ai_ready` is intent-driven, NOT auto-inferred** — add to deliverables ONLY when user explicitly asks. Presence of events alone is NOT a request.
12. **Phase 2 NEVER re-does Phase 1** — no calling `inspect_data`/`deep_inspect`/`suggest_pipeline`/`plan_pipeline`/`propose_pipeline`/`mark_proposal_confirmed` after confirmation.
13. **`pipeline.py` is standalone** — does NOT import easybci_lib (CODE_STANDARD.md Rule 15).

## Workflow Files

- `pipeline/workflows/new_pipeline.md` — 12-step from-scratch flow (goal → plan → propose → confirm → generate → execute → QC → export → crystallize)
- `pipeline/workflows/reference_adaptive.md` — proven skill execution (import_reference if needed → batch_process_adaptive)
