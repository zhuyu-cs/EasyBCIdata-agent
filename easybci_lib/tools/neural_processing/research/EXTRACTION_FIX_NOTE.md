# Web-search extraction fix — A/B verification NOTE

Follow-up to commit `ff0e2c8` ("web-search: top-5 gating + parallel search/distill
+ LLM query planner"). That refactor regressed per-citation evidence extraction:
`plan/web_evidence.json` came back with empty `recommendations`.

Fixed in commits `dd62f0f` → `e461df8` (6 commits). Full design writeup:
`improved_docs/plans/web-search-extraction-fix/PLAN.md` (repo-root sibling).

## Root causes fixed
1. **Snippet-shadowing** (`evidence_synthesizer.py`): a truthy-but-useless per-citation
   summary (e.g. "Not relevant to the question.") shadowed the good raw snippet via an
   `or` chain, starving the aggregate synthesizer. Fix: raw snippet is ALWAYS the floor;
   the summary is appended as a hint only, gated on non-empty `key_params`.
2. **`key_params` discarded** when aggregate synthesis returned empty. Fix: always salvage
   per-citation `key_params` into `recommendations` (deduped).
3. **Cache poisoning**: empty extractions were cached for 7 days. Fix: only cache when
   `key_information or key_params`.
4. **Model-facing "distill" wording**: `citation_distiller.py`→`citation_extractor.py`,
   JSON field `"distilled"`→`"key_information"`, all model-facing verbs → plain "extract";
   extraction concurrency 5→2 so the weak custom aux endpoint isn't overloaded.

## A/B result (real staged file, live `deepseek-v4-pro`, cache cleared)
Input: the 5 citations in
`test_data/EEG_preprocess_work_dir/middle_process/proposal.staged.json`.

| Metric | Before (`ff0e2c8`) | After (`e461df8`) |
|---|---|---|
| Usable per-citation extractions | 0 / 5 | 4 / 5 |
| `unparseable LLM response` errors | 3 | 0 |
| "Not relevant" refusals | 2 | reduced |
| Final `recommendations` | `[]` | `['time_window_after_cue=0-4s', 'bandpass=4-12Hz', 'high']` |

Two findings:
- Removing the model-facing "distill" wording + lowering concurrency drove live
  `unparseable` errors from 3/5 to **0/5** — validates the "distill triggers
  under-extraction on the weak model" hypothesis.
- Defense-in-depth confirmed: in the live end-to-end run the *aggregate* synthesis LLM
  itself still failed to parse (fell to `confidence: 0.2`), yet `recommendations` was
  non-empty because the salvage logic rescued the per-citation `key_params`. Pre-fix
  code returns `[]` in that scenario.

Tests: `tests/research_extraction/` (4 tests, all pass).
Run a single file with: `venv/bin/python -m pytest <path> -o addopts=""`
(the repo's `addopts = "-n auto"` makes `-p no:xdist` error).

## Precision & breadth follow-up (config-driven)

- `web.research.max_sources` (default 10) — kept+extracted citations (was hard 5).
- `web.research.sources_per_query` (default 8) — candidates per query (was hard 5).
- Tool/library docs (MNE/EEGLAB/FieldTrip/SpikeInterface/sklearn/...) are now
  whitelisted sources — evidence is no longer papers-only.
- Query planner emits 6-8 queries with mandatory ≥1 paper + ≥1 tool-doc query.
- `web_evidence.json` now separates step-format `recommendations` from
  `parameters_extracted` (`param=value` salvage), and never emits a blank
  `rationale` when evidence exists.
