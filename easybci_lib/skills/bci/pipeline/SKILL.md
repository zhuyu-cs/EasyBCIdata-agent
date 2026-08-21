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

## Invariants (both workflows share these)

- **work_dir is sealed** — only tool-chain writes allowed (harness-enforced, unconditional)
- **Source data is read-only** — never write/modify input files
- **Output format:** `preprocessed/` = `.nwb` only; `AI_ready/` = `.pkl` only
- **Deterministic seed 42** — all random operations pinned
- **Layout contract** is highest priority (equal to correctness)
- `middle_process/` is scratch — always writable; wiped on clean export

## Route Decision

**Single question: does a matching proven-pipeline skill exist?**

1. If user points to a reference project → `import_reference` first (creates proven skill), then route to **reference_adaptive**
2. Call `plan_pipeline` (or `suggest_pipeline`) with `inspection_report_path`
3. Check the return:

| Condition | Workflow | Action |
|-----------|----------|--------|
| Return contains `proven_recommendation` | **reference_adaptive** | `skill_view('pipeline/workflows/reference_adaptive')` |
| User provides reference project path | **reference_adaptive** | `import_reference` first, then load workflow |
| No proven match (or `proven_reuse_rejected`) | **new_pipeline** | `skill_view('pipeline/workflows/new_pipeline')` |

4. Load the workflow via `skill_view` and follow it step-by-step. Do NOT mix steps across workflows.

## Proven Skill Sources (all route to reference_adaptive)

- `import_reference` — ingests a gold-standard reference project
- Previous session's crystallize (Step 13 of new_pipeline)
- Third-party registration (`register_analysis_goal` / `install_skill`)
- Manual placement in `~/.easybci/skills/proven-pipelines/`

## When a Proven Skill Exists but Doesn't Match

- `proven_reuse_out_of_range` → data too different from reference → **new_pipeline**
- `proven_reuse_rejected` (goal mismatch) → **new_pipeline**
- Similarity < 0.6 → **new_pipeline**

## Workflow Files

- `pipeline/workflows/new_pipeline.md` — build pipeline from scratch (14-step: inspect → plan → propose → confirm → generate → execute → QC → export → crystallize)
- `pipeline/workflows/reference_adaptive.md` — execute with proven skill (import_reference if needed → batch_process_adaptive)
