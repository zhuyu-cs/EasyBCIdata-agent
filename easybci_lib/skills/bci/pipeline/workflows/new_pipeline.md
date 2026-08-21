---
name: pipeline/workflows/new_pipeline
description: "Build a new preprocessing pipeline from scratch — 14-step flow from inspection to crystallization"
layer: L1
metadata:
  tags: [workflow, new-pipeline, preprocessing]
---

# New Pipeline Workflow

Use when **no proven-pipeline skill exists** for this data. Builds a pipeline from scratch through inspection, planning, code generation, execution, and crystallization.

After successful completion (Step 13), the pipeline is crystallized into a proven skill — subsequent similar data automatically routes to `reference_adaptive`.

---

## Phase 1: Inspection & Proposal (Steps 0–7)

Step 7 is the ONLY human gate. Steps 0–6: no questions. Phase 2: no questions.

### Step 0 — ESTABLISH WORK DIR

`terminal(command="mkdir -p <work_dir>")`

### Step 1 — LIGHT INSPECT

`inspect_data(data_path=<path>)` — header + 1s sample.

Use `channel_summary` to spot marker/trigger channels (must-drop). Format handling lives in neural-io layer — non-built-in formats dispatched to L0 loader skills via `skill_view(name='neural-io-index')`.

### Step 2 — INFER ANALYSIS GOAL

- User stated downstream purpose → pick matching enum from `analysis_goals.py:REGISTRY`
- No purpose stated → `generic`
- `scenario`: `research`(default) | `clinical` | `deployment`
- `deliverables`: `["preprocessed"]` default. Add `ai_ready` ONLY when user explicitly asks.

### Step 3 — DEEP INSPECT

`deep_inspect(data_path=<path>, work_dir=<work_dir>)` — full scan.

Writes `inspection_report.json` + routing entry. If `degraded=true`, proceed.

Pin critical values: `PINNED: modality=sEEG, fs=2000Hz, n_channels=128, line_noise=50Hz`

### Step 4 — PROVEN REUSE CHECK

Call `plan_pipeline(inspection_report_path=<from Step 3>)`.

| Result | Action |
|--------|--------|
| `proven_recommendation` returned | **STOP this workflow.** Load `skill_view('pipeline/workflows/reference_adaptive')` and follow it instead. |
| `proven_reuse_rejected` or no match | Continue to Step 5 (New-Plan Mode). |

### Step 5 — PLAN

Based on: analysis_goal + modality/paradigm + inspection_report.

Skill loading: L1(this) → L2(paradigm via `skill_view`) → L3(operators via `skill_view`).

`research_preprocessing` — call when scenario exceeds domain-skill coverage (non-standard paradigm, >256ch, >5kHz, TMS-EEG, user asks "what's best approach"). Skip when domain skill suffices.

Draft: list operators with param NAMES. Leave dataset-dependent values as `<TBD>`.

### Step 6 — PROPOSE

`plan_pipeline(data_path=..., analysis_goal=..., scenario=..., deliverables=..., steps=[...], rationale=[...], modality=..., paradigm=..., output_path=<work_dir>, inspection_report_path=<from Step 3>)`

Use evidence-driven `steps` form (operator + params + param_evidence). For `<TBD>` params: call `research_parameter(...)`.

Rationale per step: reference real numbers from inspection_report. 3+ sentences, 80+ words.

### Step 7 — CONFIRM (only human gate)

Present FULL pipeline from propose return. Required format:

> Based on inspection (modality=…, fs=…, n_bad=…, line_freq=…):
>
> 1. notch:50 — [rationale citing data evidence]
> 2. bandpass:1,40 — [rationale]
> ...
>
> **Scenario:** research · **Deliverables:** preprocessed (NWB)
>
> Confirm (y) / modify (m) / abort (n)?

Call: `mark_proposal_confirmed(work_dir=..., user_decision=..., proposal_summary=..., presented_steps=[...])`

`presented_steps` REQUIRED on confirm — must match staged proposal exactly.

---

## Phase 2: Execution (automated — no user gates)

Phase 2 is linear — no path back to Phase 1. Autofix budget: 10 attempts per work_dir.

### Step 8 — GENERATE CODE

`generate_code(work_dir=..., steps=..., data_info=..., modality=..., analysis_goal=..., reasoning=...)`

On `success=true`: proceed IMMEDIATELY to Step 9. Do NOT `read_file` on generated scripts.

### Step 9 — EXECUTE PIPELINE

`preprocess_neural(data_path=<any_input>, steps=[...], modality=..., analysis_goal=..., output_path=<work_dir>)`

Call ONCE. On failure: patch code, re-invoke with SAME args. Hard cap: 10 attempts.

### Step 10 — BUILD AI_READY (conditional)

Only if `has_build_ai_ready=true` from Step 8. `save_processed(...)`.

### Step 11 — QC + VISUALIZATION

`quality_check(data_path=<raw>, modality=..., output_path=<work_dir>)`

### Step 12 — EXPORT MINI-REPO

1. `repair_layout(work_dir=..., dry_run=false)`
2. `export_repo(output_dir=<work_dir>, steps=..., data_info=..., pipeline_record=..., ...)`

### Step 13 — CRYSTALLIZE (save proven skill)

**New-Plan Mode** (NOT generic/exploratory goals):

`skill_manage(action="create", category="proven-pipelines", name=<modality>-<paradigm>-<N>ch-<freq>hz-<YYYYMMDD>, content=<full skill text>)`

After crystallization: subsequent similar data will match this skill and route to `reference_adaptive` automatically.

Skip crystallization if: `analysis_goal ∈ {generic, exploratory}`, missing per-step rationale, or missing data_profile.

### Step 14 — CLEANUP

Automatic. `export_repo` removes `middle_process/` on clean completion.

---

## Error Recovery

| Failure | Recovery |
|---------|----------|
| File not found (Step 1) | Ask user to verify path |
| `degraded=true` (Step 3) | Proceed — still valid |
| Phase 2 stage failed (under cap) | Patch code + re-invoke same tool |
| `recovery_exhausted=true` | Stop and report to user |

## Output Path Convention

Default: `{data_parent_parent}/{data_parent_name}_preprocess_work_dir/`.

User specifies location → pass as `output_base_dir` to `deep_inspect` at Step 3.
