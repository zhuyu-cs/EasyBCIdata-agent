---
name: pipeline
description: "Two-phase BCI data preprocessing pipeline: deep-inspect → plan → propose → user confirm → automated code/execute/QC/export"
layer: L1
metadata:
  tags: [orchestrator, preprocessing, pipeline, bci, high-priority]
  modalities: [eeg, seeg, ecog, meg, spike, fnirs]
  analysis_goal_allowed: [classification, source_localization, feature_extraction, clinical_screening, exploratory, generic, connectivity, phase_amplitude_coupling, online_inference, sleep_staging]
  analysis_goal_forbidden: []
---

# BCI Data Preprocessing Orchestrator

Two-phase flow: **Phase 1** = inspect → plan → propose → user confirm. **Phase 2** = generate → execute → QC → export (fully automated, no user gates).

## Invariants

- `work_dir = {data_parent_parent}/{data_parent_name}_preprocess_work_dir/` — all writes inside it.
- `middle_process/` at `<work_dir>/middle_process/`, NEVER under `code/`.
- Step 7 is the ONLY human gate. Phase 1 Steps 0–6: no questions. Phase 2: no questions.
- **Source data is read-only.** Do NOT write to any path returned by `inspect_data`/`deep_inspect` as `data_path`. All output goes to `work_dir`.
- Layout contract is highest priority (equal to correctness). `verify_and_repair` runs at every tool boundary.
- Phase 2 is linear — no path back to Phase 1. `recovery_exhausted=true` → terminate.
- Multi-input: identity from routing table (`inputs_routing.json`), NEVER from `Path.stem`. One script per stage loops internally over the routing table. Do NOT generate per-session script variants.
- Output buckets follow `sub-{subject_id}/ses-{session_id}/{stem_safe}_*`. `stem_safe = Path(raw).stem.replace(" ", "_")`.
- Narrow question exception: ONLY when Step 1 returns "unknown" for a structurally critical field AND all fallbacks exhausted. One question max.

---

## Phase 1: Inspection & Proposal

### Step 0 — ESTABLISH WORK DIR

`terminal(command="mkdir -p <work_dir>")`

### Step 1 — LIGHT INSPECT

`inspect_data(data_path=<path>)` — header + 1s sample. Multi-input: call once per file.

Use `channel_summary` to spot marker/trigger channels (must-drop). Format handling lives in neural-io layer — non-built-in formats dispatched to L0 loader skills via `skill_view(name='neural-io-index')`.

### Step 2 — INFER ANALYSIS GOAL

- User stated downstream purpose → pick matching enum from `analysis_goals.py:REGISTRY`
- No purpose stated → `generic`
- Quote triggering phrase in reasoning: `analysis_goal=classification; user said "train a motor imagery classifier"`
- `scenario`: `research`(default) | `clinical` | `deployment` — infer from user's delivery context words
- `deliverables`: `["preprocessed"]` default. Add `ai_ready` ONLY when user explicitly asks for training data / epochs. Events file presence alone is NOT a request.

### Step 3 — DEEP INSPECT

`deep_inspect(data_path=<path>, work_dir=<work_dir>)` — full scan. Returns `{success, report_path, report, degraded, elapsed_s}`.

Writes `inspection_report.json` (schema v4, includes `events_summary` for sidecar CSV/TSV) + routing entry. If `degraded=true`, proceed — degraded report still valid for codegen.

**Source data safety:** `deep_inspect` registers input paths as protected. After this point, `read_file` on these paths is blocked — all data info comes from the inspection report's structured fields (`fingerprint`, `channel_stats`, `psd_summary`, `events_summary`).

Multi-input: call once per file. After all calls, verify routing table has all inputs with valid `(subject_id, session_id)`. If any `identity_confidence < 0.5`, surface at Step 7 so user can override.

Pin critical values with `PINNED:` lines (survives context compaction):
```
PINNED: sub-03 ses-01 — modality=sEEG, fs=2000Hz, n_channels=128, line_noise=50Hz
PINNED: bad channels = [C3, C17] (flat); plan notch=50Hz + resample→250Hz
```

### Step 4 — PROVEN REUSE CHECK

Call `plan_pipeline(inspection_report_path=<from Step 3>)`.

| Result | Action |
|--------|--------|
| `proven_recommendation.similarity ≥ 0.6` + `reuse_contract == "full_flow_required"` + goal matches | **Reuse Mode**: load skill via `skill_view`, verify modality/channel/freq within 2×, copy steps/params/reasoning verbatim. Pass `reuse_source=<name>` to propose. |
| similarity ≥ 0.6 but goal mismatch (`proven_reuse_rejected`) | **New-Plan Mode**. Quote rejection reason verbatim in reasoning.md. |
| Otherwise | **New-Plan Mode**. |

Reuse Mode specifics: research tools auto-suppressed by runtime. Per-Step Rationale strings pass through unchanged (no paraphrasing). Step 13 uses `skill_manage(action="patch")` not "create".

### Step 5 — PLAN

Based on: analysis_goal + modality/paradigm + inspection_report (channel_summary, psd_summary, warnings).

Skill loading order: L1(this) → L2(paradigm via `skill_view`) → L3(operators via `skill_view`) → L0(loader, only if needed).

`research_preprocessing` — call when scenario exceeds domain-skill coverage (non-standard paradigm, >256ch, >5kHz, TMS-EEG, user asks "what's best approach"). Skip when domain skill suffices.

Draft: list operators with param NAMES. Leave dataset-dependent values as `<TBD>`. If `channel_summary.must_drop` non-empty, put `drop_nondata_channels:markers_only` early. Goal-conditional final cleanup is auto-injected by codegen — do NOT add a manual final `data_only` step.

### Step 6 — PROPOSE

`plan_pipeline(data_path=..., analysis_goal=..., scenario=..., deliverables=..., steps=[...], rationale=[...], modality=..., paradigm=..., output_path=<work_dir>, inspection_report_path=<from Step 3>)`

`inspection_report_path` is REQUIRED — handler rejects without it.

**Use evidence-driven `steps` form** (NOT legacy string form):

```json
[{
  "operator": "notch",
  "params": {"freq": 50},
  "param_evidence": {
    "freq": {"source": "inspection_report", "value": 50,
             "confidence": 0.95,
             "rationale": "PSD peak at 50 Hz, 12 dB above neighbours"}
  }
}, {
  "operator": "bandpass",
  "params": {"low": 1.0, "high": 40.0},
  "param_evidence": {
    "low":  {"source": "research_parameter", "value": 1.0, "confidence": 0.8, "rationale": "..."},
    "high": {"source": "research_parameter", "value": 40.0, "confidence": 0.8, "rationale": "..."}
  }
}]
```

For `<TBD>` params: call `research_parameter(operator=..., parameter=..., modality=..., paradigm=..., context={fingerprint:..., channel_summary:..., psd_summary:...})`. Copy returned `source/value/confidence/default_origin` verbatim into `param_evidence`.

Rationale per step: reference real numbers from inspection_report ("Channel P7 variance 8× higher than median" not "remove noisy channels"). 3+ sentences, 80+ words.

Tool stages envelope at `middle_process/proposal.staged.json`. `plan/` does NOT exist yet — do not check for it.

### Step 7 — CONFIRM

Present FULL pipeline from propose return value (`proposal`/`viz`/`reasoning_preview` fields). Do NOT read disk. Required format:

> Based on inspection (modality=…, fs=…, n_bad=…, line_freq=…):
>
> 1. notch:50 — [rationale citing data evidence]
> 2. bandpass:1,40 — [rationale]
> ...
>
> **Scenario:** research · **Deliverables:** preprocessed (NWB)
> _(Need AI-ready / epochs? Say so now. Wrong scenario? Tell me.)_
>
> Confirm (y) / modify (m) / abort (n)?

Call: `mark_proposal_confirmed(work_dir=..., user_decision=..., proposal_summary=..., presented_steps=[ordered operator list])`

`presented_steps` is REQUIRED on confirm — pass the exact ordered operator list you showed (matches `presented_steps_expected` from propose return). Handler rejects if missing/mismatched with `guard:"presentation_required"` + `rendered_pipeline` + `expected_steps`. On rejection: show `rendered_pipeline` verbatim to user, re-ask, call again with `presented_steps=expected_steps`.

| Decision | Effect |
|----------|--------|
| confirm | Materializes `plan/` (proposal.json + goal.json + web_evidence.json + reasoning.md). Writes `proposal.confirmed` marker. Proceed to Phase 2. |
| modify | Keeps staged envelope. Call `plan_pipeline` again with changes to overwrite staged, re-present. Same-stage — do NOT re-inspect. |
| abort | Deletes staged + marker. Terminate. |

After confirm: verify with `terminal(command="ls <work_dir>/plan/")` — must list proposal.json, goal.json, web_evidence.json, reasoning.md.

---

## Phase 2: Code Generation & Execution (automated — execute, don't review)

Phase 2 is **execution-only**. The confirmed plan is on disk; your job is to plumb args through to tool calls and run them — not to re-derive, verify, or review generated code. Everything needed already exists:

- `plan/proposal.json` — confirmed steps, params, modality, paradigm, web_evidence
- `plan/goal.json` — confirmed analysis_goal
- `plan/reasoning.md` — per-step rationale (pass through verbatim to export)
- `middle_process/inspection_report.json` — channel stats, PSD, fingerprint
- `middle_process/inputs_routing.json` — per-input (subject_id, session_id) table
- `middle_process/proposal.confirmed` — Phase 1 → Phase 2 gate marker

**Forbidden in Phase 2:**
- `inspect_data`, `deep_inspect`, `plan_pipeline`, `suggest_pipeline`, `propose_pipeline`, `mark_proposal_confirmed` — re-doing Phase 1 work wastes tokens and risks divergence.
- `read_file` on ANY file under `code/` UNLESS a preceding execution step returned `success=false` and you need to patch a specific failure. The generated code is correct-by-construction from the confirmed plan; reading it back "to verify" is the #1 token waste pattern.
- Re-deriving `steps`, `analysis_goal`, `modality`, or `reasoning` from disk — pass values through from what you already hold from Step 6/7.

**Execution tempo:** each step calls ONE tool and immediately proceeds to the next on success. The sequence is: generate → execute → build_ai_ready → QC → export → save → cleanup. No pause, no read-back, no verification detour between steps.

**Source data safety (reminder):** Generated scripts (`pipeline.py`, `build_ai_ready.py`) read source files directly via Python I/O in the sandbox — this is correct and unblocked. The `read_file` tool barrier only prevents LLM context pollution; it does not affect subprocess execution.

Autofix budget: 10 attempts per work_dir (`middle_process/autofix_state.json`). `recovery_exhausted=true` → stop and report to user. Do NOT return to Phase 1 or re-plan.

### Step 8 — GENERATE CODE

`generate_code(work_dir=..., steps=..., data_info=..., modality=..., analysis_goal=..., reasoning=...)`

Handler auto-reads `proposal.confirmed` marker + `inspection_report.json` from work_dir. Pass `steps`/`data_info`/`modality`/`analysis_goal`/`reasoning` verbatim from Step 6 propose return.

Writes: `code/pipeline.py`, `qc.py`, `vis.py`, `run.py`, `requirements.txt` + conditional `build_ai_ready.py` (only if `ai_ready ∈ deliverables` from confirmed marker AND events/labels present). If `ai_ready ∈ deliverables` but no events → returns `success=false` with `fix_hint`.

`pipeline.py` is standalone (no easybci_lib import). Multi-input: call ONCE — script loops over routing table internally.

**On `success=true`: proceed IMMEDIATELY to Step 9.** Do NOT `read_file` on any generated script. The tool return carries all metadata needed for the next step (file list, has_build_ai_ready, deliverables). Trust the generator — it was fed the confirmed plan.

### Step 9 — EXECUTE PIPELINE

`preprocess_neural(data_path=<any_input>, steps=[...], modality=..., analysis_goal=..., output_path=<work_dir>)`

Multi-input: call ONCE. Dispatcher detects routing table, invokes `python code/pipeline.py <work_dir>` (script iterates internally).

**On success:** proceed to Step 10. Do NOT read `pipeline.py` after a successful run.

**On failure:** read ONLY the traceback/error section from the tool return's `suggestion_kind` + `traceback` fields. If you need to see the failing line, `read_file` with a narrow offset (±20 lines around the error line number). Patch via `write_file`, re-invoke with SAME args. Hard cap: 10 attempts.

### Step 10 — BUILD AI_READY (conditional)

Only if `code/build_ai_ready.py` exists (check `has_build_ai_ready` from Step 8 return — do NOT `read_file` to check). `save_processed(data_path=..., output_path=<work_dir>, modality=..., analysis_goal=...)`.

Absent/skipped → clean skip (not an error). `ai_ready ∉ deliverables` → script intentionally not generated, `save_processed` returns `{skipped: true, reason: "not_requested"}`.

Multi-input: call ONCE — script loops over routing table, reads `events_path` from each routing entry.

**On success/skip:** proceed to Step 11. On failure: same narrow-read + patch pattern as Step 9.

### Step 11 — QC + VISUALIZATION

`quality_check(data_path=<raw>, modality=..., output_path=<work_dir>)` — chains `qc.py` then `vis.py`.

- vis skipped when goal opts out (`produces_figures=False`, e.g. `online_inference`)
- vis-only failure (`success:false, stage:"vis", qc_ok:true`): patch `code/vis.py` (narrow read around error line) + re-invoke
- Report grade + reference figures path to user
- Multi-input: call ONCE — both scripts loop over routing table

### Step 12 — EXPORT MINI-REPO

1. Pre-export: `repair_layout(work_dir=..., dry_run=false)` — if `unrepairable` non-empty → stop and surface to user.
2. `export_repo(output_dir=<work_dir>, steps=..., data_info=..., pipeline_record=..., input_path=..., modality=..., paradigm=..., reasoning=..., step_states=...)`

`reasoning` (dict: step→rationale text from plan/reasoning.md) + `step_states` (array from Step 9 `preprocess_neural` result) are MANDATORY — pass through verbatim, do NOT rephrase.

### Step 13 — SAVE EXPERIENCE

- **New-Plan Mode** (specialized goals only, NOT generic/exploratory):
  `skill_manage(action="create", category="proven-pipelines", name=<modality>-<paradigm>-<N>ch-<freq>hz-<YYYYMMDD>, content=<full skill text>)`
  Content must have: frontmatter (`metadata.analysis_goal`, `metadata.data_profile.{channels, sfreq_hz, duration_s, cohort_tag}`, `metadata.qc_grade`, etc.) + 8 body sections:
  1. When to Reuse
  2. Data Profile
  3. Pipeline Steps
  4. Per-Step Rationale
  5. Parameters Used
  6. QC Result
  7. When NOT to Reuse
  8. References

- **Reuse Mode**: `skill_manage(action="patch", name=..., ...)` — append one Reuse History row. Do NOT "create".

- **Skip crystallization** if: `analysis_goal ∈ {generic, exploratory}`, missing per-step rationale from `plan/reasoning.md`, or missing `data_profile.channels`/`sfreq_hz`.

### Step 14 — CLEANUP

Automatic. `export_repo` removes `middle_process/` on clean completion. Pin with `EASYBCI_KEEP_MIDDLE_PROCESS=1`.

---

## Error Recovery

| Failure | Recovery |
|---------|----------|
| File not found (Step 1) | Ask user to verify path (narrow exception) |
| `degraded=true` (Step 3) | Proceed — degraded report still valid |
| Phase 2 stage failed (under cap) | Patch code via `write_file` + re-invoke same tool with SAME args |
| `recovery_exhausted=true` | **Stop. Report failure + inspection summary + last traceback to user. No re-plan.** |
| QC FAIL (not exhausted) | Report grade + figures. Run ends (linear — no re-plan). |

## Output Path Convention

Default: `{data_parent_parent}/{data_parent_name}_preprocess_work_dir/`.

User specifies location → pass as `output_base_dir` to `deep_inspect` at Step 3 (once; Phase 2 reuses work_dir from disk). User names no location → omit, `deep_inspect` derives default next to data.

## Format Contract

- `preprocessed/` = `.nwb` only; `AI_ready/` = `.pkl` only. Other extensions auto-swept to `middle_process/sweep_<ts>/`.
- No spaces in filenames (hard failure at `verify_layout_strict_multi`).
- No `code/middle_process/` (must be at `<work_dir>/middle_process/`).
- **Source data immutable** — never write/modify any input file. All derived output goes to work_dir.
- Skill library read-only except `proven-pipelines/` (create + patch only).
