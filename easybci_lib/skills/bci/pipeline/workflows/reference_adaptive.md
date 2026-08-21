---
name: pipeline/workflows/reference_adaptive
description: "Execute preprocessing using an existing proven-pipeline skill via batch_process_adaptive"
layer: L1
metadata:
  tags: [workflow, reference, adaptive, batch, proven-pipeline]
---

# Reference Adaptive Workflow

Use when a **proven-pipeline skill exists** — from `import_reference`, previous crystallization, or manual install. The proven skill locks the step skeleton; `batch_process_adaptive` adapts per-file parameters from each recording's deep_inspect.

---

## Step 1 — IMPORT REFERENCE (conditional)

**Only if the user provides a reference project AND no proven skill exists yet.**

`import_reference(reference_dir=<path>, analysis_goal=<goal>, modality=<modality>)`

Returns: `{success, skill_name, skill_path, ...}`

If a matching proven skill already exists, skip this step.

## Step 2 — PREVIEW BATCH

`batch_process_adaptive(pattern=<glob>, skill_name=<name>, output_dir=<work_dir>, confirm=false)`

Key parameters:
- `pattern` — glob with file extension (e.g. `'/data/patient/**/*.EEG'`)
- `skill_name` — from `import_reference` return or `suggest_pipeline`'s `proven_recommendation.name`
- `output_dir` — the work_dir path
- `source_root` — top-level data directory (recommended: ensures no files are missed)
- `confirm=false` — PREVIEW only

Returns: `{presentation_block, n_matched, steps, excluded, awaiting_confirmation, ...}`

## Step 3 — PRESENT TO USER

Paste `presentation_block` **VERBATIM** — every numbered step, n_routed count, every included/excluded file.

Do NOT summarize or abbreviate it. Do NOT use the `clarify` tool.

Wait for user reply:
- Approves → Step 4
- Names files to skip → note for `exclude_paths` in Step 4
- Rejects → terminate

## Step 4 — EXECUTE

`batch_process_adaptive(pattern=<same>, skill_name=<same>, output_dir=<same>, confirm=true)`

If user named files to skip: add `exclude_paths=[...]` (NEVER narrow the pattern).

Executes full pipeline: deep_inspect each file → adapt parameters → generate code → run → QC → vis → finalize. Memory-aware serial execution built in.

## Step 5 — REPORT COMPLETION

Paste `completion_block` **VERBATIM** — including Storage Footprint (raw → preprocessed size + reduction).

If `label_diagnostics.suspicious_count > 0`: surface suspicious labels, suggest `extra_reject_keywords` for re-run.

## Step 6 — EXPERIENCE (Reuse Mode)

`skill_manage(action="patch", name=<existing proven name>, ...)` — append ONE row to Reuse History table.

**Do NOT `action="create"`** — that clones the skill and breaks the flywheel.

---

## Notes

- **Serial execution built-in** — respect memory plan and global gate. Never override with custom parallelism.
- **Single file works too** — no separate "single-file" path needed.
- **QC baselines are soft** — advisory warnings, never hard failures. One file's failure never aborts the batch.
- **Adaptation is per-file** — bad channels, notch frequencies, resample targets, reject segments recomputed from each file's inspection. Only step skeleton (operators + order) is locked.
- **`research_preprocessing`/`research_parameter` auto-suppressed** under Reuse Mode (returns `suppressed=true`). Not an error.
- **Per-Step Rationale from proven skill must be copied verbatim** — do NOT paraphrase. They pass unchanged to `export_repo`.
