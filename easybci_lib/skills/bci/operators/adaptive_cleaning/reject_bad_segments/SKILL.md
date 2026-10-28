---
name: reject_bad_segments
description: "Bad continuous-segment rejection"
layer: L3
group: adaptive_cleaning
metadata:
  tags: [operator, artifact, reject_bad_segments]
  modalities: [eeg, seeg, ecog, meg]
  step_string: "reject_bad_segments:auto"
  analysis_goal_allowed: [classification, source_localization, feature_extraction, clinical_screening, exploratory, generic, connectivity, phase_amplitude_coupling]
  analysis_goal_forbidden: [online_inference]
---
# Bad continuous-segment rejection

## Function

Remove or exclude contaminated spans of continuous data — either dropped outright, or omitted from a downstream fit (as when annotated spans are withheld from ICA).

## Parameter Format

`reject_bad_segments[:auto|thr=<amp>,pad=<s>,k=<mad>,action=mark|drop]` — runs
in the generated `pipeline.py`. Legacy positional
`amplitude_threshold,<thr>,<window_ms>` is accepted (thr + pad = window/2).

| Parameter | Default | Description |
|-----------|---------|-------------|
| `thr` | — | Absolute deviation from the channel median, in the data's units at this point of the chain. Overrides `k`. |
| `k` | 10 | `auto` mode: deviation > k robust SDs (median / MAD) per channel. |
| `pad` | 0.25 | Seconds added on both sides of every flagged run. |
| `action` | `mark` | `mark`: signal intact; `drop`: excise like `reject_by_labels`. |

`mark` (default): spans go to `meta["bad_spans"]` → trials overlapping a span
are hard-rejected with reason `bad_segment` (independent of `reject_epochs`
thresholds) → spans are written to the NWB `invalid_times` table
(`reason=bad_segment`, `trial_index=-1`) and `easybci_bad_spans` scratch →
excluded from the cross-session PSD; sliding-window AI_ready skips windows
that overlap them. `drop`: spans are excised and events remapped across the
gaps. Flagging >30 % of the recording raises a warning (threshold too strict).
Summary in `pipeline_status.bad_segments`.

## When to Use

Refer to the reference pipelines that use this operator — it fills a gap that no existing registry operator covers.

## Ordering

- Apply AFTER: `detect_artifact_spans`
- Apply BEFORE: `ica`, `epoch`

## Relationship to Existing Operators

**Nearest:** `adaptive_cleaning/asr`

asr is the repair half of clean_rawdata: it reconstructs bursts and returns data of the same length. The removal half (clean_windows / OSL bad-segment detection / ERPLAB pre-ICA pruning) shortens the recording, which no registry operator does.

## Reference Code

```python
def reject_bad_segments(d, method="annotation", threshold=None, action="drop", **_):
    spans = d.get("meta", {}).get("artifact_spans", []); x=np.asarray(d["data"]); sf=_sfreq(d); mask=np.ones(x.shape[-1],bool)
    for a,b in spans: mask[int(a*sf):int(b*sf)] = False
    return _out(d, x[...,mask] if action == "drop" else x, "reject_bad_segments", omitted_samples=int((~mask).sum()))
```
