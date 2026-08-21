---
name: pipeline/workflows/new_pipeline
description: "Build a new preprocessing pipeline from scratch — 12-step flow from goal inference to crystallization (router handles inspect)"
layer: L1
metadata:
  tags: [workflow, new-pipeline, preprocessing]
---

# New Pipeline Workflow

Use when **no proven-pipeline skill exists** for this data. After success (Step 11), crystallizes a proven skill → subsequent similar data auto-routes to `reference_adaptive`.

---

## Autonomy Contract

- **Steps 1–4: MUST NOT ask clarifying questions.** Call tools, infer goal, propose pipeline yourself.
- **Step 5: the ONLY human gate.** Present proposal, wait for user's decision.
- **Steps 6–12: fully automated.** No further user questions. Except: `recovery_exhausted=true` → return to Step 4.

### Narrow exception (router inspect only)
The router may ask ONE question ONLY when ALL hold:
1. INSPECT failed or returned "unknown" for an unrecoverable field
2. The missing field changes pipeline *structure* (not just a parameter)
3. You tried at least one fallback (custom loader, filename heuristic, domain-skill default)

---

## Phase 1: Planning & Proposal (Steps 1–5)

> **Note:** The router (SKILL.md) already completed: work_dir creation, `inspect_data`, `deep_inspect`, and `plan_pipeline` routing check. Start from Step 1 using the data you already have.

### Step 1 — INFER ANALYSIS GOAL

- User stated downstream purpose → pick matching enum from `analysis_goals.py:REGISTRY`
- No purpose stated → `generic` (cheaply skips downstream-purpose artifacts; false-specialization is much harder to undo)
- Quote triggering phrase in `plan/reasoning.md`
- `scenario`: `research`(default) | `clinical` | `deployment` — infer from user's words
- `deliverables`: `["preprocessed"]` default. Add `ai_ready` ONLY when user explicitly asks.

### Step 2 — VERIFY ROUTING TABLE (multi-input only)

If multiple inputs, verify `inputs_routing.json`:
```
terminal(command='cat "<work_dir>/middle_process/inputs_routing.json"')
```
Confirm every input has non-empty `(subject_id, session_id)`. If `identity_source="fallback"` AND `identity_confidence<0.5`, surface at Step 5.

Pin critical values: `PINNED: modality=<X>, fs=<Y>Hz, n_channels=<N>, line_noise=<Z>Hz`

### Step 3 — PLAN

Based on: analysis_goal + modality/paradigm + inspection_report.

**Skill loading order — L1 → L2 → L3 → L0:**
1. L1 orchestrator (this) — already loaded
2. L2 paradigm — `skill_view(name='<paradigm>')` once you know modality+task
3. L3 operators — `skill_view(name='<operator>')` per step
4. L0 loader — only if Step 1 flagged non-built-in format

`research_preprocessing` — call when scenario exceeds domain-skill coverage (non-standard paradigm, >256ch, >5kHz, TMS-EEG, user asks "what's best approach"). Skip when domain skill suffices.

Draft: list operators with param NAMES. Leave dataset-dependent values as `<TBD>`.

### Step 4 — PROPOSE

`propose_pipeline(data_path=..., analysis_goal=..., scenario=..., deliverables=..., steps=[...], rationale=[...], modality=..., paradigm=..., output_path=<work_dir>, inspection_report_path=<from router>)`

**`inspection_report_path` is REQUIRED.**

**Use the evidence-driven `steps` form (NEVER the legacy string form):**

```json
steps = [
  {
    "operator": "notch",
    "params": {"freq": 50},
    "param_evidence": {
      "freq": {"source": "inspection_report", "value": 50,
               "confidence": 0.95,
               "rationale": "PSD peak at 50 Hz, 12 dB above neighbours"}
    }
  },
  {
    "operator": "bandpass",
    "params": {"low": 1.0, "high": 40.0},
    "param_evidence": {
      "low":  {"source": "research_parameter", "value": 1.0, "confidence": 0.8, "rationale": "..."},
      "high": {"source": "research_parameter", "value": 40.0, "confidence": 0.8, "rationale": "..."}
    }
  }
]
```

For each `<TBD>` param: call `research_parameter(operator=..., parameter=..., modality=..., paradigm=..., context={...})`.

**Rationale: 3+ complete sentences, 80+ words per step. MUST reference real numbers from inspection_report** — "Channel P7 variance 8× higher than median" beats "remove noisy channels".

**`propose_pipeline` does NOT write `plan/`.** It stages to `middle_process/proposal.staged.json`. `plan/` materializes only at Step 5 confirm. Iterative modify cycles (propose → user changes → propose again → confirm) never leave stale drafts.

### Step 5 — CONFIRM (the ONE human gate)

Present from `propose_pipeline` return value (`presentation_block` / `proposal` / `viz` / `web_evidence` / `reasoning_preview`). Do NOT read disk — `plan/` does not exist yet.

Required format:

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

**`presented_steps` REQUIRED on confirm** — must match staged proposal exactly.

---

## Phase 2: Execution (fully automated — no user gates)

**Phase 2 tool args are pass-through, not re-derivation.** Use exact values from the confirmed proposal. Do NOT recompose or rephrase rationale strings.

**Forbidden in Phase 2:** `inspect_data` / `deep_inspect` / `suggest_pipeline` / `plan_pipeline` / `propose_pipeline` / `mark_proposal_confirmed`.

**Autofix budget:** 3 attempts per stage (shared counter at `middle_process/autofix_state.json`). On 3rd failure → `recovery_exhausted=true` → STOP, return to Step 4.

### Step 6 — GENERATE CODE

`generate_code(work_dir=<work_dir>, steps=<from confirmed proposal>, data_info=<from inspect_data>, modality=<from proposal>, analysis_goal=<from plan/goal.json>, reasoning=<from confirmed proposal>)`

The handler reads `proposal.confirmed` and locates `inspection_report.json` automatically.

Writes: `code/pipeline.py + qc.py + vis.py + run.py + requirements.txt` (+ `build_ai_ready.py` iff `ai_ready ∈ deliverables` AND events/labels present).

On `success=true`: proceed IMMEDIATELY to Step 7. Do NOT `read_file` on generated scripts.

**Multi-input:** call ONCE per work_dir. Scripts loop over routing table internally. `run_routing_safety_check` rejects stem-based derivation patterns.

**Vis script selection:** non-invasive → 5 figures (PSD/variance/amplitude/timeseries + before/after); invasive (sEEG/ECoG/iEEG/DBS/spike/unit) → 4 single-state figures only.

### Step 7 — EXECUTE PIPELINE

`preprocess_neural(data_path=<any_input>, steps=[...], modality=..., analysis_goal=..., output_path=<work_dir>)`

Call ONCE. On failure: patch `code/pipeline.py` via `write_file`, re-invoke with SAME args. Hard cap: 3 attempts.

**Multi-input:** pass any one input as `data_path` (dispatcher uses routing table internally). Do NOT loop the tool call per input.

### Step 8 — BUILD AI_READY (conditional)

Only if `code/build_ai_ready.py` exists (from Step 6). `save_processed(...)`.

When absent or `ai_ready ∉ deliverables`: `{success: false, skipped: true}` — treat as clean skip.

### Step 9 — QC + VISUALIZATION

`quality_check(data_path=<raw>, modality=..., output_path=<work_dir>)`

Runs `qc.py` (→ `QC_out/`) then chains `vis.py` (→ `figures/`). Separate autofix counters.

On vis-only failure: repair `code/vis.py`, re-invoke `quality_check`.

**Multi-input:** call ONCE — scripts loop over routing table.

### Step 10 — EXPORT MINI-REPO

1. `repair_layout(work_dir=<work_dir>, dry_run=false)` — if `unrepairable` non-empty, surface and STOP.
2. `export_repo(output_dir=<work_dir>, steps=..., data_info=..., pipeline_record=..., input_path=..., modality=..., paradigm=..., reasoning=..., step_states=...)`

**`reasoning` and `step_states` are MANDATORY** — pass through verbatim from earlier steps. Without them, `reasoning.md` falls back to boilerplate.

### Step 11 — CRYSTALLIZE (save proven skill)

`skill_manage(action="create", category="proven-pipelines", name=<modality>-<paradigm>-<N>ch-<freq>hz-<YYYYMMDD>, content=<full skill text>)`

**Skip crystallization when:**
- `analysis_goal ∈ {generic, exploratory}` (`crystallize_eligible=False`)
- Missing per-step rationale in `plan/reasoning.md`
- Missing `data_profile.channels` or `sfreq_hz`

**Required skill content (matches `_render_skill_md` format):**
- Frontmatter: `name`, `description`, `layer: L1`, `group: proven-pipelines`, `metadata.analysis_goal`, `metadata.data_profile`, `metadata.qc_grade`, `metadata.version`, `metadata.proven_date`
- Body sections: When to Reuse / Data Profile / Pipeline Steps / Per-Step Rationale / Parameters Used / QC Result / When NOT to Reuse / References

### Step 12 — CLEANUP

Automatic. `export_repo` removes `middle_process/` on clean completion. Preserve with `EASYBCI_KEEP_MIDDLE_PROCESS=1`.

---

## Error Recovery

| Failure | Recovery |
|---------|----------|
| `degraded=true` (from router) | Proceed — still valid |
| Phase 2 stage failed (under cap) | Patch code + re-invoke same tool with SAME args |
| `recovery_exhausted=true` | STOP. Return to Step 4. Present failure + traceback to user. |
| QC FAIL (not exhausted) | Report grade + figures; if user retries → Step 4 |

## Multi-Input Routing Table Contract

When multiple inputs exist, **`<work_dir>/middle_process/inputs_routing.json`** governs all routing:

1. **Identity from routing table, never file stem** — `identity_resolver` runs in `deep_inspect`; codegen safety check rejects stem-based patterns.
2. **One canonical script per stage** — loop internally over routing table. No per-session variants unless user explicitly asks.
3. **Output buckets: `sub-{subject_id}/ses-{session_id}/{stem_safe}_*`** — spaces in filenames = hard failure.
4. **Call each Phase 2 tool ONCE** (not per-input) — scripts iterate internally.

## Output Path Convention

Default: `{data_parent_parent}/{data_parent_name}_preprocess_work_dir/`.

User specifies location → pass as `output_base_dir` to `deep_inspect` in the router (only once; Phase 2 reuses work_dir on disk).
