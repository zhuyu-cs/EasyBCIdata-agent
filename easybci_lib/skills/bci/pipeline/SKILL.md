---
name: pipeline
description: "Two-phase BCI data preprocessing pipeline: deep-inspect → plan → propose → user confirm → automated code/execute/QC/export"
layer: L1
metadata:
  tags: [orchestrator, preprocessing, pipeline, bci, high-priority]
  modalities: [eeg, seeg, ecog, meg, spike, fnirs]
  analysis_goal_allowed: [classification, source_localization, feature_extraction, clinical_screening, exploratory, generic, connectivity, phase_amplitude_coupling, online_inference]
  analysis_goal_forbidden: []
---

# BCI Data Preprocessing Orchestrator (Two-Phase)

When a researcher provides a data path + a preprocessing goal, you run a **two-phase** flow:

- **Phase 1** = data inspection + analysis-goal inference + reuse check + pipeline proposal + **one human confirmation gate**.
- **Phase 2** = code generation + execution + AI-ready build + QC + mini-repo export + experience save + cleanup. Fully automated. No further user gates.

The only handover between phases is a trio of stable artifacts under `<work_dir>/middle_process/`:
- `inspection_report.json` — written by `deep_inspect`, consumed by `propose_pipeline` + `generate_code`
- `proposal.staged.json` — written by `propose_pipeline` (envelope carrying everything that will become `plan/` on confirm); each new propose call overwrites it so iterative modify cycles never leak drafts to `plan/`
- `proposal.confirmed` — written by `mark_proposal_confirmed` after the user accepts; the same call also materializes `plan/` from the staged envelope. `generate_code` checks this marker as ground truth.

## Working Directory Constraint (ABSOLUTE)

Before writing ANY file, compute and enter `work_dir = {data_parent_parent}/{data_parent_name}_preprocess_work_dir/`. ALL writes use absolute paths inside `work_dir`. NEVER write outside it. (If the user specified an output location, pass it verbatim as `output_base_dir` to `deep_inspect` instead of computing the path yourself — see "Output Path Convention".)

## Routing Table Contract (ABSOLUTE — multi-input runs)

When the user gives MORE THAN ONE input file (multi-session, multi-subject, or
batch processing), all routing in this work_dir is governed by
**`<work_dir>/middle_process/inputs_routing.json`** — written by `deep_inspect`,
consumed by every downstream stage script and tool dispatcher.

Three absolute rules (no exceptions):

1. **Identity comes from the routing table, never the file stem.** The
   `identity_resolver` policy (BIDS path → sibling BIDS → participants.tsv →
   manifest → same-directory heuristic) runs ONCE per input inside
   `deep_inspect`. Downstream code MUST NOT re-derive `(subject_id, session_id)`
   from `Path(raw).stem` — that's the regression mode that produces
   "1623 file landed in ses-1842 bucket". The codegen safety check
   (`run_routing_safety_check`) refuses to emit pipeline.py / qc.py with
   stem-based derivation patterns.

2. **One canonical script per stage.** `code/pipeline.py`, `code/qc.py`,
   `code/build_ai_ready.py`, and `code/run.py` are each written ONCE — they
   loop over the routing table internally. Do NOT generate per-session script
   variants unless the user EXPLICITLY asks for a session deviation; that
   path produces `code/pipeline_subX_sesY.py` AND registers
   `override_script="pipeline_subX_sesY.py"` in the routing entry. Without
   an explicit user ask, you MUST NOT create deviation scripts — the
   "one pipeline reuses across all subjects/sessions" property is the whole
   point.

3. **Output buckets follow `sub-{subject_id}/ses-{session_id}/{stem_safe}_*`.**
   `stem_safe = Path(raw).stem.replace(" ", "_")` is the single source of
   truth for filename normalization. A file with a space in its name landing
   in a `_preprocessed.nwb` (or `_preprocessed.pkl` legacy) bucket is a
   contract violation —
   `verify_layout_strict_multi` (Step 12) treats it as a hard failure.

`middle_process/` lives at `<work_dir>/middle_process/` and **never** under
`<work_dir>/code/`. The Step 14 cleanup (`rm -rf "<work_dir>/middle_process/"`)
relies on this — if anything leaks under `code/middle_process/` it survives
cleanup and pollutes the user's mini-repo. Both the contract checker and
the migration tool enforce this.

## Fault Tolerance Contract (ABSOLUTE)

The FINAL `<work_dir>/` MUST conform to the mini-repo layout: required
directories `plan/`, `code/`, `preprocessed_output/`; required files
`README.md`, `plan/pipeline_record.json`; `preprocessed/` is NWB-only,
`AI_ready/` is pkl-only, no spaces in filenames, no `code/middle_process/`.

Enforcement is code-driven — every tool return runs `verify_and_repair`
which auto-repairs drift and auto-sweeps disallowed-extension files into
`middle_process/sweep_<ts>/`. Partial outputs from failed retries are
auto-swept into `middle_process/failed_{outputs,qc}/<ts>/` at the next
retry's entry. `middle_process/` itself is auto-removed after a successful
export (pin with `EASYBCI_KEEP_MIDDLE_PROCESS=1`).

**The layout is the highest-priority contract in EasyBCI, equal in weight
to correctness itself.**

**Path safety.** When you do issue `terminal()` calls for anything (never for layout drift though — use `repair_layout` for that), always wrap paths in double quotes: both `work_dir` and the user's data path MAY contain spaces or shell-special characters; unquoted interpolation silently breaks the move.

## Autonomy Contract

- During Phase 1 Steps 0–6 you MUST NOT ask the user any clarifying question — call `inspect_data` + `deep_inspect`, infer `analysis_goal`, and propose a concrete pipeline yourself.
- At **Phase 1 Step 7 only**, present the proposal and wait for the user's verbatim decision. This is the ONLY human gate.
- After confirmation, **Phase 2 runs to completion without further user input** — except: a Phase 2 stage that hits the `recovery_exhausted=true` signal MUST return to Phase 1 Step 6 (re-propose), not loop on broken code.

### Narrow exception — when may you ask one question?
ONLY when ALL hold:
1. Step 1 INSPECT failed or returned "unknown" for a field that is genuinely unrecoverable from the file (modality cannot be inferred from headers / channel count / sampling rate / filename, AND no domain-skill default applies)
2. The missing field would change the *structure* of the pipeline (not just a parameter value)
3. You have tried at least one fallback (custom loader via `terminal`, filename heuristic, domain-skill default)

Even then: ask ONE specific, narrow question, then resume the flow. Never a list of questions.

---

## ═══ PHASE 1: INSPECTION & PROPOSAL (until user confirmation) ═══

### Step 0 — ESTABLISH WORK DIRECTORY

`terminal(command="mkdir -p <work_dir>")`, then use absolute paths inside it for the rest of the flow.

### Step 1 — LIGHT INSPECT

Tool: `inspect_data(data_path=<path>)`.

Reads file header + 1 s sample only. Produces a Data Fingerprint (modality, channels, sampling freq, duration, events, basic stats). If the loader fails or returns "unknown" fields, proceed with partial info — `deep_inspect` will fill the rest at Step 3, or degrade gracefully.

Use the `channel_summary` field to spot pure marker/trigger channels (must-drop) and physio refs (suggest-drop).

**Multi-input runs**: when the user references multiple files (a directory, an explicit list, or a glob), call `inspect_data` on EACH file. The first call's output is the "primary" fingerprint shown to the user; subsequent calls inform the routing table at Step 3.

**Format handling lives in the neural-io layer.** Built-in loaders cover EDF / FIF / BrainVision / EEGLAB SET / XDF / NPY / MAT / CSV. Any other format (NWB, BIDS-iEEG, MEF3, SpikeGLX, Open Ephys, Blackrock, Plexon, Neuralynx, SNIRF, LSL, …) is dispatched to an L0 loader skill: `skill_view(name='neural-io-index')` to locate the family, then `skill_view(name='<format>')` for the loader contract. This orchestrator does not enumerate formats — the canonical list lives in `easybci_lib/skills/bci/neural-io/SKILL.md`. The preprocessing principles below are modality-agnostic; format-specific quirks stay in the neural-io layer.

### Step 2 — INFER ANALYSIS GOAL

The decision here is simpler than picking from a 9-value enum. The primary question is: **did the user state a concrete downstream use purpose for the preprocessed data?**

- **No use purpose stated** → `analysis_goal = generic`. The user wants the data cleaned but hasn't told you what they'll do with it next; you have no basis to invent one. Downstream-purpose artifacts (`build_ai_ready.py`, proven-pipeline crystallization) are skipped — producing them would mean guessing the purpose, which is exactly what the user did not give you.
- **Use purpose stated** → pick the enum value that matches the stated purpose: `classification | source_localization | feature_extraction | clinical_screening | exploratory | connectivity | phase_amplitude_coupling | online_inference`. The canonical description of each value lives in `easybci_lib/tools/neural_processing/preprocess/analysis_goals.py:REGISTRY` — consult each entry's `description` / `notes` when the mapping is ambiguous.

A "use purpose" is a phrase from the user that names a downstream artifact: a target, label, decoder, feature, anatomical model, clinical readout, connectivity measure, real-time constraint, or an explicit "just exploring" intent. Data shape is NOT a use purpose — "EEG file with events.tsv" tells you nothing about whether the user wants a classifier, a connectivity analysis, or just a cleaned signal. Only the user's words about *what comes next* count.

Quote the triggering phrase verbatim in `plan/reasoning.md`, e.g. `analysis_goal=classification; user said "train a motor imagery classifier"`. When no use-purpose phrase is present, record `analysis_goal=generic; user stated no downstream use purpose`.

Asymmetric cost: defaulting to `generic` cheaply skips downstream-purpose artifacts. False-specialization bakes a silent assumption into pipeline parameters — much harder to undo than re-running with a specialized goal once the user clarifies.

The value flows into Steps 5/6/7 unchanged and into `plan/goal.json` + `plan/proposal.json` + `pipeline_record.json` as the single source of truth.

**Also capture `scenario` (delivery context) here.** Orthogonal to `analysis_goal`: `research` (default) | `clinical` | `deployment`. Infer from the user's words about who the output is for — "clinical / 临床 / diagnosis / 诊断" → `clinical`; "online / real-time / 实时 / deploy / 部署" → `deployment`; otherwise `research`. Encourage the user to state it up front when they give the data path + goal. If unstated, default to `research` and mark it "(inferred)" at Step 7 so they can override. scenario biases recommended parameters (conservative for clinical, low-latency for deployment) but forces NO pipeline branching — every step still lands in `proposal.json` for review.

**Capture AI-ready intent, do NOT auto-infer it.** `deliverables` defaults to `["preprocessed"]` (NWB only). Add `ai_ready` (epochs.pkl training data) ONLY when the user explicitly asks for AI-ready / training data / epochs. The mere presence of an events file is NOT a request for AI-ready — many clinical/research users only want the cleaned signal. The final AI-ready decision is confirmed at Step 7 before any beyond-NWB artefact is generated.

### Step 3 — DEEP INSPECT (mandatory in New-Plan Mode)

Tool: `deep_inspect(data_path=<path>, work_dir=<work_dir>)`.

Full-data scan: per-channel variance / NaN / flat / spike stats, welch PSD with 50/60 Hz peak detection, sample-based artifact rate, bad-channel candidates, memory estimate. Writes `<work_dir>/middle_process/inspect/<file_id>/inspection_report.json` (schema v3) AND mirrors to the legacy `<work_dir>/middle_process/inspection_report.json` path (back-compat). Also upserts a `RoutingEntry` into `<work_dir>/middle_process/inputs_routing.json`.

The handler returns `{success, report_path, report, degraded, elapsed_s}`. The `report` field is the full dict for quick reading; downstream tools re-read from disk (single source of truth).

If `degraded=true`, the report still has a valid fingerprint but `channel_stats / psd_summary / artifact_summary` may be empty. You MAY still proceed — Step 6 PROPOSE will surface a warning in `reasoning.md`, and the generated `pipeline.py` will carry a "DEGRADED inspection" comment block.

**Multi-input runs**: call `deep_inspect` ONCE PER input file. Calls are idempotent — re-running for the same file (matched by `file_id = sha256_1mb[:8]`) replaces its entry, new files append. After all calls, verify the routing table:

```
terminal(command='cat <work_dir>/middle_process/inputs_routing.json')
```

Confirm every input file is present with a non-empty `(subject_id, session_id)`. If any entry has `identity_source="fallback"` AND `identity_confidence<0.5`, surface this to the user in the Step 7 confirmation message — they may want to pass `--subject-id`/`--session-id` explicitly.

**Pin key findings so long runs survive context compaction.** On a long
multi-input session the conversation may be auto-compacted (older turns
summarized) before you reach Step 6/7 — and lossy summarization can blur exact
values you still need. When `deep_inspect` (or any step) surfaces a value you
must carry forward verbatim — modality, sampling rate, channel count, detected
line-noise (50/60 Hz), bad-channel candidates, resolved `(subject_id,
session_id)`, resample/notch decisions — emit a line beginning with `PINNED:`
in your reply, one fact per line, e.g.:

```
PINNED: sub-03 ses-01 — modality=sEEG, fs=2000Hz, n_channels=128, line_noise=50Hz
PINNED: bad channels = [C3, C17] (flat); plan notch=50Hz + resample→250Hz
```

Compaction preserves every `PINNED:` line byte-for-byte in a dedicated
"Pinned Findings" section and carries it across repeated compactions, so these
values are never lost. Pin sparingly (the essentials, not whole reports) — the
full report always lives on disk in `inspection_report.json`.


### Step 4 — PROVEN-PIPELINE REUSE CHECK

Call `suggest_pipeline` / `plan_pipeline` (passing `inspection_report_path` from Step 3 — required). If `proven_recommendation.similarity ≥ 0.6` AND `reuse_contract == "full_flow_required"` is returned → **Reuse Mode**:

1. Load the matched skill with `skill_view(name=<...>)`; verify `metadata.reuse_contract_version=="1"` + required sections present (Reuse Contract / Pipeline Steps / Per-Step Rationale / Web Evidence Summary / QC Targets / Reuse History). Missing any → fall back to New-Plan Mode.
2. Verify fingerprint sanity: if the new data's modality differs from the skill, OR channels/freq differ by more than 2×, fall back to New-Plan Mode with a clear message.
3. Verify `metadata.analysis_goal` equals the current goal from Step 2 (the gateway already filters at the strict Reuse gate; this is the orchestrator-side reciprocal check). On missing or mismatched goal, fall back to New-Plan Mode and surface `proven_reuse_rejected.reason` from the suggest_pipeline payload in `plan/reasoning.md`.
4. In Reuse Mode you MUST copy the proven skill's Per-Step Rationale strings
   verbatim into your response — do NOT paraphrase (they pass unchanged to
   `export_repo` later). The `research_preprocessing` / `research_parameter`
   tools are automatically suppressed by the runtime under Reuse Mode; no
   need to remember not to call them.
   - When you call `propose_pipeline` in Reuse Mode, pass
     `reuse_source="<the proven skill name from suggest_pipeline's
     proven_recommendation.name>"`. This persists the reuse marker into
     plan/proposal.json so the runtime can recognize Reuse Mode (and auto-suppress
     the research tools). Omit `reuse_source` entirely in New-Plan Mode.
5. Step 5 PLAN + Step 6 PROPOSE in Reuse Mode = copy steps/params/reasoning from the proven skill verbatim. Step 7 CONFIRM simplifies to a one-line question but **still waits for the user** — no timeout-auto-confirm.
6. Step 13 SAVE EXPERIENCE in Reuse Mode = `skill_manage(action="patch", ...)` to append one Reuse History row, NOT `action="create"`.

If `suggest_pipeline` returns `proven_reuse_rejected` (similarity ≥ 0.6 but goal mismatch / missing) → **New-Plan Mode** (proceed to Step 5); the rejection reason MUST be quoted verbatim in `plan/reasoning.md` so the user can see why the top match was not reused.

Otherwise → **New-Plan Mode** (proceed to Step 5).

### Step 5 — PLAN

Pick the operator sequence and rough parameters based on:
- `analysis_goal` from Step 2
- modality + paradigm from Step 1
- `inspection_report.channel_summary` + `psd_summary` + `warnings`  ← real data evidence
- Web research (`research_preprocessing`) IS optional — call it when the scenario exceeds domain-skill coverage (non-standard paradigm, unusual data characteristics like >256 ch / >5 kHz / TMS-EEG, user explicitly asks "what's the best approach"). Skip when domain skill suffices.

Load the relevant Domain Skill via `skill_view`: motor_imagery / p300_erp / ssvep / seeg_epilepsy / sleep_staging / eeg_general / connectivity / etc.

**Dispatch order — L1 → L2 → L3 → L0.** Skill loading follows this exact sequence:

1. **L1 orchestrator** (this file) — already loaded.
2. **L2 paradigm** — once you know modality+task, call `skill_view(name='<paradigm>')` *once*. Browse the available paradigms with `skill_view(name='paradigms-index')` if uncertain.
3. **L3 operators** — for each step you intend to write into `pipeline.py`, call `skill_view(name='<operator>')` to fetch parameter defaults / modality constraints / ordering rules. Browse with `skill_view(name='operators-index')`.
4. **L0 loader (on demand)** — only if Step 1 INSPECT flagged a non-built-in format. `skill_view(name='neural-io-index')` then `skill_view(name='<format>')`. Otherwise skip L0 entirely — most runs never touch it.

Never invert this order. Loading L3 operators before fixing the paradigm makes parameter choices arbitrary; loading L0 first is wasted unless you've already hit a custom-format wall.

Compose pipeline draft: list operators with parameter NAMES; leave dataset-dependent values as `<TBD>`. Filter-order / method-style defaults (registry `research_trigger: never`) may be filled directly.

Channel cleanup: if `channel_summary.must_drop` is non-empty, put `drop_nondata_channels:markers_only` as one of the first steps. Default to `markers_only`. Escalate to `data_only` only with a concrete reason. **Goal-conditional final cleanup is auto-injected by codegen** (you do NOT need to append a final `data_only` step manually).

### Step 6 — PROPOSE (stages the deliverable; nothing lands in `plan/` yet)

Tool: `propose_pipeline(data_path=..., analysis_goal=<from Step 2>, scenario=<from Step 2, default research>, deliverables=<["preprocessed"] unless the user explicitly asked for AI-ready>, steps=[...], rationale=[...], modality=..., paradigm=..., output_path=<work_dir>, inspection_report_path=<from Step 3>)`.

**`inspection_report_path` is REQUIRED** — the handler rejects calls without it.

**propose_pipeline does NOT write `plan/`.** It stages the entire deliverable inside `<work_dir>/middle_process/proposal.staged.json` and returns the proposal contents in the tool result. `plan/` materializes only when the user confirms via `mark_proposal_confirmed` at Step 7. This ordering means iterative modify cycles (propose → user says "change step 3" → propose again → confirm) never leave half-finished or wrong-version files on disk. The user's final `plan/` is exactly the version they approved — no earlier draft survives anywhere.

**Use the evidence-driven `steps` form. Never the legacy string form.** The legacy string form `["notch:50", "bandpass:1,40"]` stages a husk proposal (`params:{raw:"50"}`, `param_evidence:{}` empty) and produces NO `reasoning.md` at confirm time — only the export-time fallback boilerplate later. With the evidence-driven form, confirm materializes a rich `plan/reasoning.md` with per-parameter evidence tables.

Per-step shape:

```
steps = [
  {
    "operator": "notch",
    "params": {"freq": 50},
    "param_evidence": {
      "freq": {"source": "inspection_report", "value": 50,
               "confidence": 0.95,
               "rationale": "PSD peak at 50 Hz, 12 dB above neighbours"}
    },
  },
  {
    "operator": "bandpass",
    "params": {"low": 1.0, "high": 40.0},
    "param_evidence": {
      "low":  {"source": "research_parameter", "value": 1.0,  "confidence": 0.8, ...},
      "high": {"source": "research_parameter", "value": 40.0, "confidence": 0.8, ...},
    },
  },
  ...
]
```

For each `<TBD>` from Step 5, call `research_parameter(operator=..., parameter=..., modality=..., paradigm=..., context={fingerprint: ..., channel_summary: ..., psd_summary: ...})`. The tool falls back to registry `empirical_default` when no provider is configured. Copy each returned default's `source` / `value` / `confidence` / `default_origin` verbatim into the corresponding `param_evidence[<pname>]` entry — missing entries get auto-filled with `empirical_default` but a warning is emitted.

**Rationale must reference real numbers from inspection_report** — "Channel P7 variance 8× higher than median" beats "remove noisy channels". Three complete sentences minimum (Observation / Strategy / Implementation), 80+ words per step. Do NOT use colons or dashes in rationale strings (YAML/JSON parsing).

**Web evidence rides into the staged envelope.** When `research_preprocessing` ran in Step 5 (a web search provider was active), the result is captured into the envelope and will materialize at confirm time as:

- `plan/web_evidence.json` — raw payload (`provider`, `recommendations`, `applied_to_steps`, `confidence`, citations, `status`)
- `plan/proposal.json:web_evidence` — same dict embedded in the proposal so the record is self-contained
- `plan/reasoning.md` opens with a `> **Web evidence:** queried <provider> for SOTA preprocessing · confidence X.XX applied to <steps>. See plan/web_evidence.json.` banner (evidence-driven form only)

You do not pass these explicitly — the dispatcher captures the `research_preprocessing` result and forwards it into the envelope. Your job is to ACTUALLY CALL `research_preprocessing` when the scenario warrants it (Step 5 lists when). When no web evidence is available (no provider configured, all backends failed, no recommendations returned), the same files materialize at confirm time but carry `status="unavailable"` and a `reason` — that is the correct shape; never delete them, never paper over them.

**What the tool returns:** the staged envelope path (`staged_path`) plus a `proposal` dict, `web_evidence` payload, `reasoning_preview` (evidence-driven form), `viz` block, and a one-line `summary`. These are what you present to the user at Step 7 — do NOT read disk to compose the chat message. Optionally verify the staging file exists:

```
terminal(command="ls <work_dir>/middle_process/proposal.staged.json")
```

`plan/` does NOT exist at this point — checking for it now is a confusion of the lifecycle. Proceed to Step 7.

### Step 7 — CONFIRM (the ONE human gate; materializes `plan/`)

Present the proposal to the user **from the `propose_pipeline` return value** (`presentation_block` / `proposal` / `viz` / `web_evidence` / `reasoning_preview` fields). Do NOT read disk — `plan/` does not exist yet.

**You MUST show the FULL pipeline before asking for confirmation.** Paste the `presentation_block` (or enumerate every step from `viz.steps` / `proposal.steps`) as a numbered list — each with its operator, params, and rationale. NEVER collapse this to a bare "confirm this pipeline?" — the expert cannot judge a pipeline they cannot see. Required form:

> "Based on inspection (modality=…, fs=…, paradigm=…, n_bad_candidates=…, line_freq=…), I propose the pipeline below.
>
> 1. notch:50 — [rationale]
> 2. bandpass:1,40 — [rationale]
> …
>
> **Scenario:** research (inferred) · **Deliverables:** preprocessed signal (NWB) only
> _(If you need AI-ready training data / epochs, say so now and I'll add it. If the scenario is wrong — clinical or deployment — tell me and I'll re-tune the parameters.)_
>
> Confirm (y) to run end-to-end, modify (m) to revise a step, or abort (n)."

After the user responds, immediately call:

`mark_proposal_confirmed(work_dir=<work_dir>, user_decision="confirm"|"modify"|"abort", proposal_summary=<one-line>, presented_steps=<ordered operator list you showed>)`

**`presented_steps` is REQUIRED on `confirm` and is code-enforced.** Pass the exact ordered list of operators you presented (it equals the `presented_steps_expected` field in the propose return, e.g. `["drop_nondata_channels","notch","bandpass",...]`). The handler compares it against the staged proposal's real steps; if it is missing, empty, or doesn't match (count/order/operators), **confirmation is REJECTED** with `guard:"presentation_required"` and the return carries a `rendered_pipeline` string + `expected_steps`. When rejected: show the user that `rendered_pipeline` verbatim, get their decision, then call again with `presented_steps=expected_steps`. This guarantees the expert always sees the full pipeline — do not try to bypass it by guessing values.


- `confirm` → reads `middle_process/proposal.staged.json` and MATERIALIZES the post-confirmation deliverable: `pipeline.yaml` at work_dir root (legacy form only) + `plan/proposal.json` + `plan/goal.json` + `plan/web_evidence.json` (+ `plan/reasoning.md` for evidence-driven form). Writes `middle_process/proposal.confirmed` marker. Resets the autofix counter. The tool's return value includes `materialized: [...]` — the list of files it created. The confirmed scenario/deliverables are persisted into the `proposal.confirmed` marker; Step 8 codegen reads deliverables from that marker — beyond-NWB artefacts are produced ONLY when confirmed here. **Proceed to Phase 2.**
- `modify` → marker cleared (if present); the staged envelope is KEPT so the next `propose_pipeline` overwrites it with the revised proposal. Return to Step 5 PLAN. Keep `inspection_report.json`; do NOT re-inspect.
- `abort` → marker AND staged envelope both deleted; terminate.

**Self-check after `confirm`** — verify `plan/` actually materialized:

```
terminal(command="ls <work_dir>/plan/")
# evidence-driven form: must list proposal.json  goal.json  web_evidence.json  reasoning.md
# legacy string form:   must list proposal.json  goal.json  web_evidence.json
```

If a file is missing, inspect `mark_proposal_confirmed`'s return `materialized` list and the corresponding `proposal.staged.json` envelope to see what was supposed to be written. If `web_evidence.json` shows `status="unavailable"` AND you expected web search to run → `research_preprocessing` was not actually invoked at Step 5; you cannot fix this post-confirm. Go back to Step 5, call `research_preprocessing`, re-propose, then re-confirm.

**Do NOT call `generate_code` yourself** until `mark_proposal_confirmed` succeeded with `confirm`. The tool layer will reject it (three checks: `proposal_confirmed=True` param + valid `inspection_report_path` + marker file present).

---

## ═══ PHASE 2: CODE GENERATION & EXECUTION (fully automated) ═══

**Reuse, don't repeat.** Phase 2 is execution-only — it materializes the pipeline the user confirmed at Step 7 and runs it. Everything Phase 2 needs is already on disk after Step 7:

- `plan/proposal.json` — the confirmed `steps`, `params`, `modality`, `paradigm`, `web_evidence`
- `plan/goal.json` — the confirmed `analysis_goal`
- `plan/reasoning.md` — the per-step rationale
- `middle_process/inspection_report.json` — the deep-inspection report (channel stats, PSD, bad channels, fingerprint)
- `middle_process/inputs_routing.json` — per-input `(subject_id, session_id)` table for multi-input runs
- `middle_process/proposal.confirmed` — the Phase 1 → Phase 2 gate marker

**Forbidden in Phase 2** (re-doing Phase 1 work wastes turns and risks divergence from the confirmed plan):
- Do NOT call `inspect_data` / `deep_inspect` / `suggest_pipeline` / `plan_pipeline` / `propose_pipeline` / `mark_proposal_confirmed` again.
- The `research_preprocessing` / `research_parameter` tools remain callable in
  Phase 2, but under Reuse Mode the runtime auto-suppresses them (returns
  `suppressed=true`). Do not treat that as an error — Reuse Mode's parameters
  are the skill's, not the web's.
- Do NOT re-derive `steps`, `analysis_goal`, `modality`, or `reasoning` — pass through the values you already saw in Step 6's `propose_pipeline` return (which equal what's in `plan/`).

**Phase 2 tool args are pass-through, not re-derivation.** When a Step below shows `steps=[...]` or `reasoning={...}` in a tool signature, use the exact values from the confirmed proposal — the LLM's job is plumbing them through to the executor, not recomputing them. Args the handler auto-discovers from `work_dir` (the `proposal.confirmed` marker, `inspection_report.json` path, routing table) need not be passed explicitly.

The ONLY legitimate path back to Phase 1 in Phase 2 is the `recovery_exhausted=true` bail-out (3rd autofix failure on a stage) — surface the failure and return to Phase 1 Step 6.

Phase 2 stages share a counter at `<work_dir>/middle_process/autofix_state.json`. Each stage gets up to **3** AutoFixer attempts. On the 3rd failure, the tool returns `recovery_exhausted=true` — **do NOT call write_file again**; surface the failure + `inspection_report` summary to the user and return to Phase 1 Step 6.

### Step 8 — GENERATE CODE

Tool: `generate_code(work_dir=<work_dir>, steps=<from confirmed proposal>, data_info=<from inspect_data>, modality=<from proposal>, analysis_goal=<from plan/goal.json>, reasoning=<from confirmed proposal>, label_config=<optional>)`.

The handler reads the `proposal.confirmed` marker and locates `inspection_report.json` from `work_dir` automatically — you do NOT pass `proposal_confirmed=True` or `inspection_report_path`. Pass `steps` / `data_info` / `modality` / `analysis_goal` / `reasoning` through verbatim from what `propose_pipeline` returned at Step 6 (these equal what was written to `plan/`); do not re-derive them.

Writes `code/pipeline.py + qc.py + vis.py + run.py + requirements.txt` (+ `build_ai_ready.py` iff `ai_ready ∈ deliverables` from the confirm marker **AND** events or `label_config` present). Inspection-driven hints are baked in: notch frequency uses the detected power-line peak, the "Inspection-driven hints" comment block records bad-channel candidates so the human reader sees why these parameters were chosen.

`vis.py` is a standalone matplotlib-only script. **Non-invasive modalities (EEG/MEG/fNIRS/...)** get 5 figures (PSD / channel variance / amplitude distribution / timeseries + a `before_after_timeseries` panel that re-loads the raw input via MNE / pickle / npz). **Invasive modalities (sEEG/ECoG/iEEG/DBS/spike/spikes/unit/units, gated by `format_policy.is_invasive`)** get the 4 single-state figures only — no before/after, because the raw NWB-shaped input is expensive to reload and the 4 single-state figs on the processed signal are sufficient evidence. Per-figure failures don't abort; status flows through `middle_process/vis_status.json`. The selection rule lives in `codegen/generator.py:generate_vis_script`.

`qc.py` writes only `preprocessed_output/QC_out/sub-<id>/ses-<ses>/qc_report.{json,md}` (no figures — those moved to `vis.py`).

`ai_ready ∉ deliverables` (the default) skips `build_ai_ready.py` even when events exist — this is the new default (NWB only). The result payload's `ai_ready_skipped_reason="not_requested"` signals this; `plan/reasoning.md` should mention it. If `ai_ready ∈ deliverables` but the data has no events/labels, `generate_code` returns `success=false` with a `fix_hint` (supply labels or re-confirm without ai_ready) rather than silently emitting empty epochs.

**NWB is the universal default for `preprocessed/`.** `format_policy.resolve_default_format(modality, "auto")` returns `"nwb"` across every modality. `preprocessed/` is NWB-only regardless of deliverables. The generated `pipeline.py` hard-fails on NWB write errors (no silent pkl fallback) so corrupted exports surface immediately.

**Spike algorithm coverage in code-emit layer.** `_OPS` includes `threshold_spike` (MAD-based extracellular spike detection — `meta["spike_times"]`) and `mua_binning` (firing-rate matrix from spike times — `meta["mua_train"]`). Both honour Rule 5: continuous data array is never modified by spike ops.

`pipeline.py` is **standalone** — it does NOT import easybci_lib (per CODE_STANDARD.md Rule 15) so the mini-repo runs on a machine where easybci is not installed.

**Multi-input awareness**: Generated scripts read `<work_dir>/middle_process/inputs_routing.json` and loop over the entries internally. You call `generate_code` ONCE per work_dir regardless of how many inputs are in the routing table — DO NOT pass an explicit "list of inputs" to it. After codegen, the handler runs a static safety check (`run_routing_safety_check`) that rejects scripts containing stem-based subject_id derivation; a `success=False` return with `routing_violations` means the templates regressed.

Any pre-existing scripts are archived to `<work_dir>/middle_process/code/<stage>_<ts>.py` — **never** to `<work_dir>/code/middle_process/`.

### Step 9 — EXECUTE PIPELINE

Tool: `preprocess_neural(data_path=<raw>, steps=[...], modality=..., analysis_goal=..., output_path=<work_dir>)`.

Runs `code/pipeline.py` as subprocess (600 s timeout, override via `timeout`). On failure: read traceback's `suggestion_kind` (`import_error` / `attribute_error` / `value_error` / `shape_mismatch` / `dependency_missing` / `timeout` / `other`), `write_file` to patch `code/pipeline.py`, re-invoke same args. **Hard cap: 3 attempts per work_dir.** On the 3rd failure → `recovery_exhausted=true` → return to Phase 1 Step 6.

**Multi-input runs**: call `preprocess_neural` ONCE — pass any one input as `data_path` (the dispatcher uses it for source-data protection, but the actual loop happens inside `pipeline.py` reading the routing table). The dispatcher detects the routing table and invokes `python code/pipeline.py <work_dir>` (no input_path argument); the script iterates internally and writes per-(sub, ses) buckets. Do NOT loop the `preprocess_neural` tool call per input — that breaks status aggregation and trips the autofix counter. If any single input fails, the aggregate sidecar at `middle_process/pipeline_status_aggregate.json` records which `file_id`s succeeded vs failed.

### Step 10 — BUILD AI_READY (conditional)

Run only if `code/build_ai_ready.py` exists. Tool: `save_processed(data_path=<preprocessed.pkl>, output_path=<work_dir>, modality=..., analysis_goal=...)`. Same 3-attempt cap, same recovery path.

When `build_ai_ready.py` is absent and no events / label_config are available, the tool returns `{success: false, skipped: true, reason: ...}` — treat this as a clean skip, not an error.

When `ai_ready ∉ deliverables` (the default), `code/build_ai_ready.py` is intentionally not generated even when events / label_config are present — `save_processed` returns `{success: false, skipped: true, reason: "not_requested"}`. Treat this as a clean skip identical to the events-absent case.

**Multi-input runs**: call `save_processed` ONCE — the script loops over the routing table. Each entry's `events_path` field tells `build_ai_ready.py` where to find the sidecar events CSV for that specific recording.

### Step 11 — RUN QC

Tool: `quality_check(data_path=<raw>, modality=..., output_path=<work_dir>)`. Runs `code/qc.py` (writes the QC report to `preprocessed_output/QC_out/sub-<id>/ses-<id>/qc_report.{json,md}`) **and then chains `code/vis.py`** (writes figures to `preprocessed_output/figures/sub-<id>/ses-<id>/`). qc and vis have independent 3-attempt autofix counters (`quality_check` / `quality_check_vis`).

When `code/vis.py` does not exist (the goal opted out via `REGISTRY[goal].produces_figures=False`, e.g. `online_inference`), `quality_check` skips the vis sub-step silently and the response carries `vis.skipped: true`. `contract_check.figures_missing` is already goal-conditional, so this is consistent end-to-end.

On a vis-only failure (`success: false, stage: "vis", qc_ok: true`), repair `code/vis.py` via `write_file` and re-invoke `quality_check`. qc.py re-runs idempotently — the re-run cost is small.

Report grade + reference the figures + QC_out path to the user.

**Multi-input runs**: call `quality_check` ONCE — `qc.py` loops over the routing table and produces a per-`(sub, ses)` report set, then `vis.py` loops over the same routing table for figures. The aggregates at `middle_process/qc_status.json` and `middle_process/vis_status.json` carry `n_success`/`n_failed`/per-file detail.

### Step 12 — EXPORT MINI-REPO

**Pre-export self-check.** Run `repair_layout(work_dir="<work_dir>",
dry_run=false)`. If the returned `unrepairable` list is non-empty, surface
those items to the user and STOP — they require human judgment. Otherwise
proceed to `export_repo`. Manual `mv`/`rm`/`mkdir` against `<work_dir>` for
layout drift is not supported — the audit trail lives in
`plan/repair_report.json`.

Then:

Tool: `export_repo(output_dir=<work_dir>, steps=..., data_info=..., pipeline_record=..., input_path=..., modality=..., paradigm=..., reasoning=..., step_states=...)`.

`reasoning` (dict: step → rationale text from the confirmed Step 6 proposal — same values now in `plan/reasoning.md`) and `step_states` (array from `preprocess_neural` Step 9 result) are MANDATORY. Pass them through from what you already have — do NOT recompose or rephrase the rationale strings. Without them, reasoning.md falls back to generic boilerplate.

This assembles the final mini-repo (README + plan/ + code/ + preprocessed_output/ + middle_process/) per the layout in `improved_docs/`. `verify_and_repair` runs at the tool boundary regardless — belt-and-braces.

**Multi-input addendum**: when `<work_dir>/middle_process/inputs_routing.json`
exists, `repair_layout` reads the routing table via
`verify_layout_strict_multi` under the hood. You do not need to loop over
`(sub, ses)` triples manually.

### Step 13 — SAVE EXPERIENCE

- **New-Plan Mode** → `skill_manage(action="create", category="proven-pipelines", name=<modality>-<paradigm>-<N>ch-<freq>hz-<YYYYMMDD>, content=<full skill text>)`. Set `metadata.reuse_contract_version: "1"`. Initialize `Reuse History` with row 1.
- **Reuse Mode** → `skill_manage(action="patch", name=<existing proven name>, ...)` to append exactly one new row to the skill's `Reuse History` table. **Do NOT** call `action="create"` — that clones the skill and breaks the flywheel.

#### Eligibility & Format Requirements

**Skip crystallization entirely** when `analysis_goal ∈ {generic, exploratory}` — these are non-specialized runs and would pollute the proven-pipeline library. The auto-crystallize safety net (`contract_check.maybe_crystallize_proven`) refuses these goals; you must do the same in New-Plan Mode (do NOT call `skill_manage(action="create", category="proven-pipelines", ...)` for them).

When you DO crystallize (specialized goals only), the SKILL.md content MUST mirror the format produced by `contract_check._render_skill_md`:

**Frontmatter (required keys):** `name`, `description`, `layer: L1`, `group: proven-pipelines`, `metadata.analysis_goal` (a specialized goal), `metadata.analysis_goal_allowed`, `metadata.modalities`, `metadata.paradigm`, `metadata.step_string`, `metadata.data_profile.{channels, sfreq_hz, duration_s, cohort_tag}`, `metadata.qc_grade`, `metadata.qc_metrics`, `metadata.source_run`, `metadata.web_evidence_used`, `metadata.version`, `metadata.auto_crystallized`, `metadata.proven_date`.

**Required body sections (in order):**
1. `## When to Reuse` — modality / paradigm / channel-count / sfreq compatibility envelope
2. `## Data Profile` — concrete numbers from the source run (channels, sfreq, duration, cohort)
3. `## Pipeline Steps` — operator chain in a fenced code block
4. `### Per-Step Rationale` — one paragraph per step, each citing the reasoning from `plan/reasoning.md`
5. `## Parameters Used` — markdown table of `(step, parameter, value, notes)`
6. `## QC Result` — grade + key metrics
7. `## When NOT to Reuse` — disqualifying conditions
8. `## References` — `source_run` path + reasoning.md + web_evidence refs if any

If you cannot extract per-step rationale from `plan/reasoning.md` OR `data_profile.channels` / `sfreq_hz` is missing, **do not crystallize** — log "skipping crystallization: incomplete provenance" and proceed to Step 14.

### Step 14 — CLEANUP

Automatic. `export_repo` deletes `<work_dir>/middle_process/` when the
export chain completed cleanly. Preserve with `EASYBCI_KEEP_MIDDLE_PROCESS=1`.
The result payload's `middle_process_cleaned` / `cleanup_skipped_reason`
fields report what happened.

---

## Error Recovery (Phase-aware)

| Failure | Phase | Recovery |
|---------|-------|----------|
| File not found at Step 1 | 1 | Ask user to verify path (the Step 1 narrow exception) |
| `deep_inspect` failed → `degraded=true` | 1 | Proceed; degraded report still valid for Phase 2 |
| Phase 2 stage failed once / twice | 2 | `write_file` to patch + re-invoke same tool with SAME args |
| Phase 2 stage returns `recovery_exhausted=true` | 2 | **Stop. Return to Phase 1 Step 6.** Present failure + inspection summary + last traceback to user. Ask them to revise. The next `mark_proposal_confirmed(confirm)` resets all counters. |
| QC FAIL but not `recovery_exhausted` | 2 | Report Grade + figures, ask user; if they retry, go to Phase 1 Step 6 |
| `proposal_confirmed=False` accidentally passed to `generate_code` | gate | Tool rejects; call `mark_proposal_confirmed` first |

## Output Path Convention

Default `work_dir = {data_parent_parent}/{data_parent_name}_preprocess_work_dir/`. Examples:
- Input `/data/study/raw/eeg.edf` → `work_dir = /data/study/raw_preprocess_work_dir/`
- Input `/home/user/study/session1/raw.fif` → `work_dir = /home/user/study/session1_preprocess_work_dir/`

**If the user names a storage location** (e.g. "把结果存到 /data/preprocessed/ 下面",
"save the output under D:/out"), do NOT compute the work_dir path yourself. Pass the raw
location string as `output_base_dir` to `deep_inspect` at Step 2. Code creates
`{data_parent_name}_preprocess_work_dir/` inside it (the `_preprocess_work_dir` folder is
appended automatically) — this may live on a different disk than the raw data. You only need
to specify the location ONCE, at `deep_inspect`; Phase 2 tools reuse the work_dir already on disk.

If the user names no location, omit both `work_dir` and `output_base_dir` — `deep_inspect`
derives the default next to the data automatically.

## Visualization & Output Contract (unchanged)

`preprocessed_output/` layout, `FinalDataView` figure contract, `Reuse Contract` skill template, and the Learning Loop are unchanged from prior versions. See `improved_docs/` for the canonical layout reference.

### Final output format — AI-ready only

`preprocessed/` is `.nwb`-only; `AI_ready/` is `.pkl`-only. Any other
extension in these two directories is auto-swept to `middle_process/sweep_<ts>/`
by `verify_and_repair`. The single-source-of-truth allowlist is
`easybci_lib/tools/neural_processing/export/layout_spec.py:CANONICAL`.

### Skill library — read-only except `proven-pipelines/`

EasyBCI's built-in skill library (everything under `easybci_lib/skills/` and its installed mirror at `~/.easybci/skills/`) is READ-ONLY from the agent's perspective. The ONLY permitted `skill_manage` writes are inside `category="proven-pipelines"`:

- `action="create"` to crystallise a successful pipeline (Step 13 New-Plan Mode).
- `action="patch"` to append exactly one row to an existing proven-pipeline's `Reuse History` (Step 13 Reuse Mode).

Patching, write_file, delete, rename, or attaching new files (e.g. `references/*.md`) to ANY non-`proven-pipelines/` skill — `bci/pipeline/`, `bci/operators/*`, `bci/neural-io/*`, `bci/paradigms/*`, `data-science/*`, `mlops/*`, etc. — is FORBIDDEN. Ad-hoc skill edits contaminate the global library and propagate into every future session. If you find a real gap, report it to the user verbally; do NOT self-patch.

## Communication Style

- Lead with action, explain after.
- Use Data Fingerprint format consistently.
- Format pipeline proposals as numbered steps with inline rationale.
- Keep QC reports structured: status first, then details.
- If multiple valid approaches exist, present top 2 options with trade-offs (Phase 1 only; never during Phase 2 automation).
