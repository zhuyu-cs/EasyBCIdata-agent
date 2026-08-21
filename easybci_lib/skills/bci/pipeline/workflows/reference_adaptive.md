---
name: pipeline/workflows/reference_adaptive
description: "Execute preprocessing using an existing proven-pipeline skill via batch_process_adaptive"
layer: L1
metadata:
  tags: [workflow, reference, adaptive, batch, proven-pipeline]
---

# Reference Adaptive Workflow

Use when a **proven-pipeline skill exists** for this data — whether from `import_reference`, a previous crystallization, or manual installation. Covers ALL scenarios: single file, multiple files, reference imitation, scientific replication.

The proven skill locks the step skeleton; `batch_process_adaptive` adapts per-file parameters (bad channels, notch frequencies, etc.) from each recording's deep_inspect.

---

## Step 1 — IMPORT REFERENCE (conditional)

**Only if the user provides a reference project AND no proven skill exists yet.**

`import_reference(reference_dir=<path>, analysis_goal=<goal>, modality=<modality>)`

Returns: `{success, skill_name, skill_path, ...}`

If a matching proven skill already exists (from a previous session or manual install), skip this step entirely.

## Step 2 — PREVIEW BATCH

`batch_process_adaptive(pattern=<glob>, skill_name=<name>, output_dir=<work_dir>, confirm=false)`

Key parameters:
- `pattern` — glob with file extension (e.g. `'/data/patient/**/*.EEG'`)
- `skill_name` — the proven-pipeline skill name (from `import_reference` return, or from `suggest_pipeline`'s `proven_recommendation.name`)
- `output_dir` — the work_dir path
- `source_root` — top-level data directory (recommended: ensures no files are missed)
- `confirm=false` — PREVIEW only, computes plan without executing

Returns: `{presentation_block, n_matched, steps, excluded, awaiting_confirmation, ...}`

## Step 3 — PRESENT TO USER

Paste `presentation_block` **VERBATIM** to the user in chat — every numbered step, n_routed count, every included/excluded file.

Do NOT summarize or abbreviate it. Do NOT use the `clarify` tool.

Wait for the user's plain-text reply:
- User approves → proceed to Step 4
- User names files to skip → note them for `exclude_paths` in Step 4
- User rejects → terminate

## Step 4 — EXECUTE

`batch_process_adaptive(pattern=<same>, skill_name=<same>, output_dir=<same>, confirm=true)`

If user named files to skip: add `exclude_paths=[...]` (NEVER narrow the pattern).

This executes the full pipeline: deep_inspect each file → adapt parameters → generate code → run pipeline → QC → vis → finalize. Memory-aware serial execution is built in.

## Step 5 — REPORT COMPLETION

Paste `completion_block` **VERBATIM** in chat — including the Storage Footprint line (raw → preprocessed size + reduction).

If `label_diagnostics.suspicious_count > 0`: surface the suspicious labels and suggest `extra_reject_keywords` for a re-run.

---

## Notes

- **Serial execution is built-in** — `batch_process_adaptive` respects the memory plan and global gate. Never override with custom parallelism.
- **Single file works too** — `batch_process_adaptive` handles 1 file the same as 100. No separate "single-file" path needed.
- **QC baselines are soft** — the skill's baselines produce advisory warnings, never hard failures. One file's failure never aborts the batch.
- **Adaptation is per-file** — bad channels, notch frequencies, resample targets, reject segments are all recomputed from each file's own inspection. Only the step skeleton (operators + order) is locked.
