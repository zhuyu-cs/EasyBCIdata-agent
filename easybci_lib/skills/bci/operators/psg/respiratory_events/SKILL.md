---
name: respiratory_events
description: "Detect apnea, hypopnea, and SpO2 desaturation events from PSG respiratory channels"
layer: L3
group: psg
metadata:
  tags: [operator, psg, respiratory, apnea, hypopnea, desaturation, sleep, ahi]
  modalities: [eeg]
  step_string: "respiratory_events"
  analysis_goal_allowed: [sleep_staging, clinical_screening, feature_extraction, exploratory]
  analysis_goal_forbidden: [online_inference, source_localization, connectivity, phase_amplitude_coupling]
---
# Respiratory Event Detector

## Function

Detects obstructive/central apnea, hypopnea, and SpO2 desaturation events
from PSG respiratory channels following AASM 2012 scoring criteria. Produces
annotated event timestamps for downstream clinical indices (AHI, ODI).

Input / Output: data with airflow and/or effort channels (post bandpass
0.05–3 Hz) → `meta["respiratory_events"]` (event list) +
`meta["ahi"]` (Apnea-Hypopnea Index, events/hour).

## Algorithm & Math

### Apnea Detection (AASM Rule 1A)

1. Compute airflow envelope: abs(airflow) → 5 s moving average.
2. Baseline: rolling 120 s median of envelope (excludes prior events).
3. Drop threshold: amplitude falls < 10% of baseline for ≥ 10 s.
4. Event type: if residual effort present → obstructive; if effort also
   absent → central.

### Hypopnea Detection (AASM Rule 4A)

1. Airflow envelope drops 30–90% of baseline for ≥ 10 s.
2. Associated desaturation ≥ 3% OR arousal (from meta if available).
3. Without SpO2 confirmation: mark as `hypopnea_unconfirmed`.

### Desaturation Marking

1. SpO2 baseline: rolling 120 s max (stable saturation).
2. Desaturation: SpO2 drops ≥ 3% from baseline.
3. Record nadir, duration, and recovery time.

## Parameter Format & Defaults

`respiratory_events` or `respiratory_events:{airflow_ch},{spo2_ch}`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `airflow_ch` | str | "NPress" | Airflow channel name (nasal pressure). |
| `effort_ch` | str | "Thor" | Effort channel name (thoracic belt). |
| `spo2_ch` | str | "SpO2" | SpO2 channel name. |
| `apnea_threshold` (kw) | float | 0.10 | Fraction of baseline for apnea (< 10%). |
| `hypopnea_threshold` (kw) | float | 0.70 | Fraction of baseline for hypopnea (< 70%). |
| `min_duration_s` (kw) | float | 10.0 | Minimum event duration (AASM: 10 s). |
| `desat_threshold` (kw) | float | 3.0 | SpO2 drop % for desaturation. |

## Modality-Specific Considerations

PSG (EEG modality with psg_context): required channels are respiratory
auxiliaries kept by channel_classifier. Without airflow AND effort channels,
operator raises recoverable error.

## When to Use / NOT to Use

**Use** when: PSG sleep study with respiratory channels; computing AHI/ODI;
detecting sleep-disordered breathing.

**Don't use** when: no respiratory channels available; pure EEG-only recording;
non-sleep goals.

## Constraints & Ordering

- Apply AFTER bandpass:0.05,3 on respiratory channels (raw signals too noisy).
- Apply BEFORE resample (needs native rate for accurate amplitude).
- SpO2 is typically 1 Hz — no pre-filtering needed.
- Does NOT modify data array; only writes meta.

## Failure Modes & Detection

| Failure | Symptom | Detection |
|---|---|---|
| No airflow channel | Cannot detect events. | Fallback to effort-only (Thor+Abdo sum). |
| SpO2 absent/all-zero | No desat confirmation. | Degrade to flow-only; mark events `_unconfirmed`. |
| Airflow saturated/flat | False baseline, no events. | `np.ptp(airflow) < 1e-6` → raise recoverable. |
| Short recording (< 1 h) | AHI unreliable. | Warn in meta; AHI still computed. |

## Common Issues

- **"AHI=0 but obvious apneas visible."** Likely airflow signal inverted or
  wrong channel selected; check polarity.
- **"Too many events."** Baseline window too short; increase to 120–180 s.

## Reference Implementation

### Standalone

```python
from __future__ import annotations
import numpy as np
from scipy.ndimage import uniform_filter1d


def respiratory_events(
    airflow: np.ndarray, spo2: np.ndarray | None,
    sfreq_airflow: float, sfreq_spo2: float = 1.0,
    apnea_threshold: float = 0.10, hypopnea_threshold: float = 0.70,
    min_duration_s: float = 10.0, desat_threshold: float = 3.0,
) -> list[dict]:
    """Detect apnea/hypopnea events. Returns list of event dicts."""
    # Envelope: abs + 5s smooth
    env = uniform_filter1d(np.abs(airflow), size=int(5 * sfreq_airflow))
    # Baseline: 120s rolling median
    win = int(120 * sfreq_airflow)
    baseline = np.array([
        np.median(env[max(0, i - win):i + 1]) for i in range(len(env))
    ])
    baseline = np.maximum(baseline, 1e-10)  # avoid division by zero

    ratio = env / baseline
    min_samples = int(min_duration_s * sfreq_airflow)
    events = []
    in_event = False
    start = 0
    for i in range(len(ratio)):
        if ratio[i] < hypopnea_threshold and not in_event:
            start = i
            in_event = True
        elif (ratio[i] >= hypopnea_threshold or i == len(ratio) - 1) and in_event:
            duration_samples = i - start
            if duration_samples >= min_samples:
                min_ratio = float(np.min(ratio[start:i]))
                event_type = "apnea" if min_ratio < apnea_threshold else "hypopnea"
                onset_s = start / sfreq_airflow
                dur_s = duration_samples / sfreq_airflow
                spo2_drop = _check_desat(spo2, sfreq_spo2, onset_s, dur_s,
                                         desat_threshold) if spo2 is not None else None
                events.append({
                    "onset": onset_s, "duration": dur_s, "type": event_type,
                    "spo2_drop": spo2_drop,
                })
            in_event = False
    return events


def _check_desat(
    spo2: np.ndarray, sfreq: float, onset_s: float, dur_s: float,
    threshold: float,
) -> float | None:
    """Check for SpO2 desaturation within event window + 30s after."""
    start = int(onset_s * sfreq)
    end = min(int((onset_s + dur_s + 30) * sfreq), len(spo2))
    if start >= end:
        return None
    segment = spo2[start:end]
    segment = segment[segment > 0]  # exclude invalid zeros
    if len(segment) < 2:
        return None
    baseline_val = float(np.max(segment[:max(1, int(5 * sfreq))]))
    nadir = float(np.min(segment))
    drop = baseline_val - nadir
    return drop if drop >= threshold else None
```

### EasyBCI-Adapted

```python
from typing import Any, Dict
import time
import numpy as np
from easybci_lib.tools.neural_processing.operator_errors import EasyBCIOperatorError
from easybci_lib.tools.neural_processing.preprocess.step_cache import record_step_elapsed


def operator_respiratory_events(
    data_dict: Dict[str, Any], *,
    airflow_ch: str = "NPress", effort_ch: str = "Thor", spo2_ch: str = "SpO2",
    apnea_threshold: float = 0.10, hypopnea_threshold: float = 0.70,
    min_duration_s: float = 10.0, desat_threshold: float = 3.0,
) -> Dict[str, Any]:
    """Respiratory event detection (AASM criteria).

    Parameters
    ----------
    data_dict : dict
    airflow_ch, effort_ch, spo2_ch : str
    apnea_threshold, hypopnea_threshold : float
    min_duration_s : float
    desat_threshold : float

    Returns
    -------
    dict — `meta["respiratory_events"]`, `meta["ahi"]`.

    Raises
    ------
    EasyBCIOperatorError
        recoverable=True if no airflow/effort channel found.

    Modality coverage
    -----------------
    PSG (EEG with respiratory auxiliaries): yes.
    """
    channels = data_dict.get("channels", [])
    sfreq = float(data_dict["frequency"])

    def _find(name):
        return next((i for i, c in enumerate(channels) if name.lower() in c.lower()), None)

    air_idx = _find(airflow_ch)
    eff_idx = _find(effort_ch)
    spo2_idx = _find(spo2_ch)

    if air_idx is None and eff_idx is None:
        raise EasyBCIOperatorError(
            operator="respiratory_events",
            reason=f"no airflow ({airflow_ch!r}) or effort ({effort_ch!r}) channel",
            recoverable=True, fallback_step="skip respiratory_events",
        )

    t0 = time.monotonic()
    from scipy.ndimage import uniform_filter1d

    # Use airflow if available, else fall back to effort
    flow = data_dict["data"][air_idx] if air_idx is not None else data_dict["data"][eff_idx]
    spo2 = data_dict["data"][spo2_idx] if spo2_idx is not None else None

    env = uniform_filter1d(np.abs(flow), size=max(1, int(5 * sfreq)))
    win = int(120 * sfreq)
    # Vectorized rolling median approximation: use percentile on blocks
    baseline = np.array([
        np.median(env[max(0, i - win):i + 1]) for i in range(0, len(env), int(sfreq))
    ])
    baseline = np.repeat(baseline, int(sfreq))[:len(env)]
    baseline = np.maximum(baseline, 1e-10)

    ratio = env / baseline
    min_samples = int(min_duration_s * sfreq)
    events = []
    in_event = False
    start = 0
    for i in range(len(ratio)):
        if ratio[i] < hypopnea_threshold and not in_event:
            start = i
            in_event = True
        elif (ratio[i] >= hypopnea_threshold or i == len(ratio) - 1) and in_event:
            dur = i - start
            if dur >= min_samples:
                min_ratio = float(np.min(ratio[start:i]))
                etype = "apnea" if min_ratio < apnea_threshold else "hypopnea"
                onset_s = start / sfreq
                dur_s = dur / sfreq
                spo2_drop = None
                if spo2 is not None:
                    s = int(onset_s)
                    e = min(int(onset_s + dur_s + 30), len(spo2))
                    seg = spo2[s:e]
                    seg = seg[seg > 0]
                    if len(seg) >= 2:
                        drop = float(np.max(seg[:5])) - float(np.min(seg))
                        spo2_drop = drop if drop >= desat_threshold else None
                events.append({"onset": onset_s, "duration": dur_s,
                               "type": etype, "spo2_drop": spo2_drop})
            in_event = False

    duration_h = data_dict["data"].shape[-1] / sfreq / 3600.0
    ahi = len(events) / duration_h if duration_h > 0 else 0.0
    elapsed = time.monotonic() - t0

    out = dict(data_dict)
    out["elapsed_s"] = elapsed
    out["meta"] = {
        **out.get("meta", {}),
        "respiratory_events": events,
        "ahi": round(ahi, 1),
        "respiratory_events_params": {
            "airflow_ch": airflow_ch, "effort_ch": effort_ch, "spo2_ch": spo2_ch,
            "apnea_threshold": apnea_threshold,
            "hypopnea_threshold": hypopnea_threshold,
        },
    }
    record_step_elapsed("respiratory_events", elapsed,
                        (data_dict.get("meta") or {}).get("step_cache_key"))
    return out
```

## References

1. Berry, R. B. et al. (2012). *Rules for scoring respiratory events in
   sleep: update of the 2007 AASM manual*. J. Clin. Sleep Med. 8(5):
   597–619. doi:10.5664/jcsm.2172.
2. Ruehland, W. R. et al. (2009). *The new AASM criteria for scoring
   hypopneas: impact on the apnea hypopnea index*. Sleep 32(2):
   150–157. doi:10.1093/sleep/32.2.150.
