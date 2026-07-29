# EasyBCI Pipeline Code Standard

> **Version**: 1.1.0 (T7 Sub-phase P-B.1, 2026-06)
>
> Every `pipeline.py`, `qc.py`, `run.py`, and every operator step LLMs
> emit MUST conform to this standard.  `easybci_lib/tools/neural_processing/
> codegen/code_standard_check.py` enforces the rules at codegen time; the
> generator returns violations as structured errors so the agent self-
> repairs in the same loop.

This document is **injected into every `plan_pipeline` / `generate_code`
system prompt** so the model has the contract in-context, not just at
review time.  Keep the rule set short, mechanical, and AST-checkable.

## Why a code standard

Pipelines emitted by different LLMs (and by the same LLM across turns)
read very differently in the absence of a standard — different parameter
names, different exception types, different cache-key conventions.  That
breaks three downstream goals:

1. **Reproducibility.**  A reviewer must be able to read one pipeline,
   learn the conventions, then read N more without re-learning.
2. **Static lint.**  We need a single AST checker, not N checkers, to
   reject violations at codegen time.
3. **Error recovery.**  The 3-tier `error_recovery` chain dispatches on
   `EasyBCIOperatorError(recoverable=…)`.  Generic `Exception` defeats
   the chain.

The 14 rules below are the smallest set that achieves these three goals
while still letting authors use the underlying numerical libraries
freely.

## Rules

### Rule 1 — Function signature

Every operator function MUST be named ``operator_<name>`` and have
``data_dict`` as its first positional parameter.  All other parameters
MUST be keyword-only.

```python
def operator_bandpass_filter(
    data_dict: Dict[str, Any],
    *,
    low: float | None = None,
    high: float | None = None,
    method: str = "fir",
) -> Dict[str, Any]:
    ...
```

### Rule 2 — Type annotations on everything

Every parameter and the return type MUST be annotated.  Use
`Dict[str, Any]` for the data_dict; `Optional[T]` for nullables;
`Literal[...]` for enumerated string options when the choice is small.

### Rule 3 — Docstring contract

The docstring MUST include all five of:

- `Parameters`
- `Returns`
- `Raises`
- `Modality coverage`
- `References`

Operators without all five fail `check_pipeline_code_standard`.

### Rule 4 — `OperatorIO` schema

The `data_dict` MUST be shaped exactly like
:class:`easybci_lib.tools.neural_processing.operator_schema.OperatorIO`:

```python
{
    "data":      np.ndarray (n_channels, n_times)  # float32 or float64
    "channels":  list[str]                          # len == n_channels
    "frequency": float                              # Hz
    "duration":  float                              # seconds
    "meta":      dict[str, Any]                     # per-step state
    "elapsed_s": float | None                       # filled by record_step_elapsed
}
```

`OPERATOR_IO_ALL_KEYS` is the canonical set of allowed keys.  Operators
MUST NOT introduce ad-hoc keys without extending the schema first.

### Rule 5 — Immutability

Operators MUST NOT mutate `data_dict["data"]` in place.

```python
# WRONG
data_dict["data"][:] = filtered

# RIGHT
out = dict(data_dict)
out["data"] = filtered
return out
```

In-place mutation breaks `step_cache` (the cached pre-step copy
silently mutates after the read) and breaks reproducibility tests that
re-run a step with the same input.

### Rule 6 — Random seeds

Any operator that uses RNG MUST seed from the env var `EASYBCI_SEED`.

```python
import numpy as np
rng = np.random.default_rng(int(os.environ.get("EASYBCI_SEED", "0")))
```

Bare `np.random.rand(...)` / `np.random.randn(...)` / `random.random()`
are forbidden — they make pipelines non-reproducible across machines.

### Rule 7 — Deterministic cache keys

`step_cache` keys MUST be a pure function of the step string and the
input fingerprint.  Operators MUST NOT use wall-clock time, PID, or
machine identifiers in cache-key derivation.

### Rule 8 — Errors via `EasyBCIOperatorError`

The only exception class operators MAY raise is
:class:`easybci_lib.tools.neural_processing.operator_errors.EasyBCIOperatorError`.
Wrap library exceptions (numpy `ValueError`, mne `FilterError`, etc.) in
`EasyBCIOperatorError` with `recoverable=True` and a sensible
`fallback_step`.

```python
try:
    filtered = bandpass_filter(data, sfreq, low, high)
except ValueError as exc:
    raise EasyBCIOperatorError(
        operator="bandpass_filter",
        reason=str(exc),
        recoverable=True,
        fallback_step="bandpass:1,40",
    ) from exc
```

### Rule 9 — Elapsed time

Every operator MUST record its wall-clock elapsed time into
``data_dict["elapsed_s"]`` (or via
``preprocess.step_cache.record_step_elapsed``).  The progress tracker
reads this for cross-session ETA.

### Rule 10 — Modality coverage

The docstring `Modality coverage` field MUST enumerate every modality
the operator supports (EEG / MEG / sEEG / ECoG / fNIRS / spike).
Unsupported modalities MUST raise
`EasyBCIOperatorError(recoverable=False)` rather than silently passing.

### Rule 11 — No external network

Operator function bodies MUST NOT import or call `requests`, `urllib`,
`httpx`, `aiohttp`, or `socket`.  Network calls belong in web-research
tools, not in the data pipeline.

### Rule 12 — No file I/O

Operator function bodies MUST NOT call `open`, `Path.write_text`,
`Path.write_bytes`, `np.save`, `np.savez`, `pickle.dump`,
`shutil.copy*`, or `os.remove`.  File I/O belongs in `run.py` /
`qc.py` / `build_ai_ready.py`, not in operator bodies (which are pure
in-memory transforms).

Reading the source file IS permitted in dedicated I/O operators (under
`easybci_lib/tools/neural_processing/io/`); they're flagged via a
`# easybci-allow-file-io` marker on the function definition.

### Rule 13 — Numpy dtype

`data_dict["data"]` dtype MUST be a floating type (`float32` or
`float64`).  `int8` / `uint8` / `float16` are forbidden — they break MNE
filtering and lead to silent precision loss in scipy.

### Rule 14 — Channel name preservation

Unless the operator's documented behaviour is to drop channels (e.g.
`drop_bads`, `pick_channels`), the output `channels` list MUST equal the
input `channels` list, in order.

### Rule 15 — No easybci_* imports in generated scripts

Generated `pipeline.py`, `qc.py`, `run.py`, `build_ai_ready.py` (the
contents of `<work_dir>/code/`) MUST NOT import any module under
`easybci_lib`, `easybci_agent`, `easybci_cli`, `easybci_web`,
`easybcidata_agent`, `services.gateway`, `services.plugins`,
`services.providers`, or `run_agent`.

The mini-repo is meant to be **shareable** — a colleague who pip-installs
`mne numpy scipy scikit-learn matplotlib` should be able to run
`python run.py <raw>` on a machine where easybci is not installed. An
import like `from easybci_lib.tools.neural_processing.io.loader import
load_neural` breaks that promise: it makes the mini-repo a thin wrapper
around the agent's library rather than a standalone reproducible script.

You MAY reference easybci sources as documentation (e.g. "based on
`easybci_lib/.../preprocess/pipeline.py:_step_notch`" in a comment), but
the function body itself must inline the implementation using
`mne.io.read_raw`, `raw.notch_filter`, `mne.preprocessing.ICA`, scipy,
numpy, etc.

Approved replacement patterns for the most common needs:

| Need | Use |
|---|---|
| Read EEG/MEG file | `mne.io.read_raw(path, preload=True, verbose="ERROR")` |
| Notch filter | `raw.notch_filter([50])` |
| Bandpass | `raw.filter(l_freq=lo, h_freq=hi)` |
| Resample | `raw.resample(sfreq=target)` |
| Re-reference | `raw.set_eeg_reference("average")` |
| ICA artifact removal | `mne.preprocessing.ICA(...).fit(raw); ica.exclude=[...]; ica.apply(raw)` |
| PSD | `scipy.signal.welch` or `mne.time_frequency.psd_array_welch` |
| Robust scaling | `(x - median) / iqr` via numpy |
| Plot | `matplotlib.pyplot.savefig(...)` |

The lint rule `no-easybci-import` is **blocking**; codegen returns a
structured error and the agent must fix every match before the mini-repo
is finalised.

There is no `easybci-allow:` escape hatch for this rule — if you find
yourself wanting one, file an issue instead. The contract is binary:
either the mini-repo is standalone, or the rule has failed.

## Banner header — `code/run.py`

Every `code/run.py` produced by codegen MUST begin with:

```python
# EASYBCI_CODE_STANDARD: 1.1.0
# Conformance verified by easybci_lib/tools/neural_processing/codegen/code_standard_check.py
# at codegen time.
```

The version number bumps when the rules change.  The lint refuses to
verify a script that omits the banner.

## Suppressing rules at function level

For genuine exceptions (e.g. an I/O operator that legitimately writes
to disk), authors MAY add a top-of-function comment:

```python
def operator_save_intermediate(data_dict, *, path):
    # easybci-allow: file-io  # write intermediate cache; reviewer-approved
    Path(path).write_bytes(...)
```

The lint recognises `easybci-allow: <rule-name>` and skips that rule
for the function.  Approval comments MUST include a one-line reason.

## Versioning

The standard is versioned (this is 1.0.0).  When rules change:

1. Bump the version in the banner header.
2. Update this document.
3. Update `check_pipeline_code_standard` to enforce the new rule set.
4. Note the bump in CLAUDE.md.

Pipelines written against an older version stay readable but are
flagged at lint time as `code_standard_version_outdated`.
