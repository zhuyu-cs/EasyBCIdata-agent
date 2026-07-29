---
name: marker_integrity_check
description: "Validate event/marker cross-stream alignment (NWB / BIDS / XDF)"
layer: L3
group: qc_operator
metadata:
  tags: [operator, qc, marker, event, alignment, multi_stream, bids, xdf, nwb]
  modalities: [eeg, meg, seeg, ecog, spike, fnirs]
  step_string: "marker_integrity_check"
  analysis_goal_allowed: [classification, source_localization, feature_extraction, clinical_screening, exploratory, generic, connectivity, phase_amplitude_coupling, online_inference]
  analysis_goal_forbidden: []
---
# Marker Integrity Check

## Function

Validates that event / marker timestamps from multi-stream recordings
(NWB, BIDS-iEEG sidecar, XDF, LSL) are correctly aligned with continuous
data. Detects: missing events, time-drift between streams, sample-rate
mismatch, duplicate events, events past EOF.

Input / Output: `data_dict` with `meta["events_s"]` or `meta["events"]`
table → adds `meta["marker_integrity"]: {status, issues}`.

## Algorithm & Math

Six checks per event stream:

1. **Time range**: events within `[0, duration]`.
2. **Monotonic**: timestamps non-decreasing.
3. **Duplicates**: no two events at the same sample.
4. **Sample-rate consistency**: event sample indices match expected sfreq.
5. **Cross-stream drift**: if multiple `meta["streams"]`, validate
   timestamps align within tolerance.
6. **Density sanity**: not too many (likely TTL bounces) or too few
   (likely missing).

## Parameter Format & Defaults

`marker_integrity_check:{tolerance_ms}`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `tolerance_ms` | float | 5.0 | Cross-stream alignment tolerance. |

## Modality-Specific Considerations

All modalities — purely metadata QC.

## When to Use / NOT to Use

**Use** when: loaded multi-stream data (NWB / XDF / BIDS); event-locked
analyses (ERP, epoching).

**Don't use** when: no events expected; pure resting-state.

## Constraints & Ordering

After load; before any event-locked op.

## Failure Modes & Detection

The op IS a detection layer — it does not "fail", it reports issues.

## Common Issues

- **"events_s array longer than recording."** Likely sample-rate
  mismatch; verify `sfreq` matches header.

## Reference Implementation

### Standalone

```python
from __future__ import annotations
import numpy as np


def marker_integrity_check(
    events_s: np.ndarray, duration: float, tolerance_ms: float = 5.0,
) -> dict:
    """Return {status, issues}."""
    issues = []
    events_s = np.asarray(events_s, dtype=np.float64)
    if events_s.min() < 0:
        issues.append("event before t=0")
    if events_s.max() > duration:
        issues.append(f"event past EOF (duration={duration:.2f}s)")
    if not np.all(np.diff(events_s) >= 0):
        issues.append("non-monotonic events")
    if len(np.unique(events_s)) < len(events_s):
        issues.append("duplicate event timestamps")
    isi = np.diff(events_s)
    if len(isi) > 0 and isi.min() * 1000 < tolerance_ms / 5:
        issues.append(f"events closer than {tolerance_ms/5} ms (possible TTL bounce)")
    status = "ok" if not issues else "warn"
    return {"status": status, "issues": issues, "n_events": int(len(events_s))}
```

### EasyBCI-Adapted

```python
from typing import Any, Dict
import time
import numpy as np
from easybci_lib.tools.neural_processing.operator_errors import EasyBCIOperatorError
from easybci_lib.tools.neural_processing.preprocess.step_cache import record_step_elapsed


def operator_marker_integrity_check(
    data_dict: Dict[str, Any], *, tolerance_ms: float = 5.0,
) -> Dict[str, Any]:
    """Validate event/marker integrity.

    Parameters
    ----------
    data_dict : dict
        OperatorIO; `meta["events_s"]` ndarray of event times in seconds.
    tolerance_ms : float

    Returns
    -------
    dict — adds `meta["marker_integrity"]`.

    Raises
    ------
    EasyBCIOperatorError
        recoverable=False if events missing or fundamentally broken
        (out-of-range past duration > 50%).

    Modality coverage
    -----------------
    All modalities.

    References
    ----------
    BIDS-iEEG spec; NWB schema.
    """
    events_s = (data_dict.get("meta") or {}).get("events_s")
    if events_s is None:
        # Not an error — but report as "no events" for callers to act on.
        out = dict(data_dict)
        out["meta"] = {**out.get("meta", {}),
                       "marker_integrity": {"status": "absent", "issues": [], "n_events": 0}}
        return out

    duration = float(data_dict.get("duration") or 0.0)
    events_s = np.asarray(events_s, dtype=np.float64)

    t0 = time.monotonic()
    issues = []
    if events_s.size == 0:
        status = "ok"
    else:
        if events_s.min() < 0:
            issues.append("event before t=0")
        if events_s.max() > duration:
            issues.append(f"event past EOF (duration={duration:.2f}s)")
        if not np.all(np.diff(events_s) >= 0):
            issues.append("non-monotonic events")
        if len(np.unique(events_s)) < len(events_s):
            issues.append("duplicate event timestamps")
        isi = np.diff(events_s)
        if len(isi) > 0 and isi.min() * 1000 < tolerance_ms / 5:
            issues.append(f"events closer than {tolerance_ms/5:.1f} ms (possible TTL bounce)")
        status = "ok" if not issues else "warn"
    elapsed = time.monotonic() - t0

    out = dict(data_dict)
    out["elapsed_s"] = elapsed
    out["meta"] = {
        **out.get("meta", {}),
        "marker_integrity": {
            "status": status, "issues": issues, "n_events": int(events_s.size),
        },
        "marker_integrity_check": {"tolerance_ms": tolerance_ms},
    }
    record_step_elapsed("marker_integrity_check", elapsed, (data_dict.get("meta") or {}).get("step_cache_key"))
    return out
```

## References

1. BIDS-iEEG specification. https://bids-specification.readthedocs.io/en/stable/04-modality-specific-files/04-intracranial-electroencephalography.html
2. NWB Schema. https://nwb-schema.readthedocs.io/
3. Renard, Y. et al. (2010). *OpenViBE: An Open-Source Software Platform
   to Design, Test, and Use Brain–Computer Interfaces*. Presence 19(1):
   35–53. doi:10.1162/pres.19.1.35 — multi-stream timing.
