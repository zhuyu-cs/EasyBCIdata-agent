---
name: no_op
description: "Declared omission — records a deliberate non-action"
layer: L3
group: misc
metadata:
  tags: [operator, misc, no_op]
  modalities: [eeg, seeg, ecog, meg, spike, fnirs]
  step_string: "no_op"
  analysis_goal_allowed: [classification, source_localization, feature_extraction, clinical_screening, exploratory, generic, connectivity, phase_amplitude_coupling, online_inference]
  analysis_goal_forbidden: []
---
# Declared Omission (no_op)

## Function

An explicit statement that a conventional processing stage was deliberately NOT applied. Carries no computation; it exists so that a reference step which asserts an omission can be represented, and so a scorer can check that the agent's pipeline does not contain the named operators.

## Parameter Format

`no_op:{skipped_operations},{forbidden_operators},{reason}`

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `skipped_operations` | varies | — | what the paper says it did not do, in its own words |
| `forbidden_operators` | varies | — | registry step strings whose presence would contradict the reference |
| `reason` | varies | — | why the omission was deliberate |

## When to Use

When a pipeline explicitly omits a step that would conventionally be applied (e.g. no artifact rejection, no baseline correction, no interpolation) and the omission is a deliberate analytic choice, not an oversight.

## Ordering

- No strict ordering constraints.

## Relationship to Existing Operators

**No near equivalent in the registry.**

Not a capability gap: no operator can represent the absence of an operator. Without it, a paper that explicitly releases minimally processed data with no interpolation, no artefact rejection and no baseline correction has reference steps that vanish, and an agent that adds those stages looks compliant.

## Reference Code

```python
def no_op(d, skipped_operations=None, forbidden_operators=None, reason=None, **_):
    return _out(d, step="no_op", no_op={"skipped_operations": skipped_operations,
        "forbidden_operators": forbidden_operators, "reason": reason})
```
