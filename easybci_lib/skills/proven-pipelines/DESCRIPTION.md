---
name: proven-pipelines
description: "Validated preprocessing pipelines accumulated from successful processing sessions — growing library of proven experience"
---

Proven pipelines saved from successful preprocessing runs. Each skill here represents a pipeline that was:
- Executed without errors
- Passed automated QC checks (quality_check tool)
- Confirmed satisfactory by the user

## How to Use

When planning a new pipeline (Step 2 of pipeline), check here first:
1. Look for a skill matching your modality + paradigm
2. Load it with `skill_view(name="<pipeline-name>")` 
3. Use its steps and parameters as a starting point, adjusting for the new data's specifics

## Naming Convention

`<modality>-<paradigm>-<N>ch-<freq>hz-<YYYYMMDD>`

The date suffix ensures uniqueness when processing multiple datasets of the same type.

Examples:
- `eeg-motor-imagery-64ch-256hz-20260527`
- `seeg-epilepsy-128ch-1000hz-20260415`
- `eeg-p300-32ch-512hz-20260503`
- `meg-ssvep-306ch-1000hz-20260527`

## When to Save

After Final Step passes AND the user confirms satisfaction, save the pipeline here using `skill_manage(action="create", category="bci/proven-pipelines", ...)`.
