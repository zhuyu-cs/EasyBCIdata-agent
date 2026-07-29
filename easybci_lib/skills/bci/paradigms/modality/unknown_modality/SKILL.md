---
name: unknown_modality
description: 'Fallback paradigm — LLM cannot match any specific modality; conservative inspect / low-pass / drop / report flow'
layer: L2
group: modality
metadata:
  tags: [fallback, unknown, generic, modality_agnostic]
  modalities: [unknown]
  paradigms: [unknown_modality, generic_fallback]
  analysis_goal_allowed:
    - exploratory
    - generic
  analysis_goal_forbidden:
    - source_localization
    - online_inference
    - phase_amplitude_coupling
    - connectivity
    - classification
    - feature_extraction
    - clinical_screening
---
# Unknown Modality — Generic Fallback

## Neuroscience Background

When the upstream `inspect_neural` cannot confidently identify the
modality (sample rate inconclusive, channel naming non-standard,
metadata absent), the LLM **must not** silently fall back to "generic
EEG" defaults — that produces wrong filtering and silent garbage.
Instead, take this paradigm: inspect → very conservative low-pass +
DC removal → drop_bads → emit a report → tell the user what's needed.

## Inspection Checklist

Before applying any operator, fill this table from `data_dict`:

| Field | Value | What this implies |
|---|---|---|
| `sfreq` | Hz | < 200: EEG/MEG/sleep/clinical; 200–2k: LFP/sEEG; ≥ 2k: spike-band candidate. |
| `n_channels` | int | 1–8: clinical / single-probe; 16–96: standard EEG/UEA; 64–128: high-density EEG; 384: Neuropixels candidate. |
| `duration` | s | Drives any time-frequency analysis. |
| `voltage_range_uV` | µV | < 100: scalp EEG range; 1000+: invasive / saturated. |
| `psd_peak` | Hz | Where most power lives — helps modality inference. |
| `channels[]` naming | str | "Fp1/Cz/Oz" → EEG; "MEG…" → MEG; "imec*" → Neuropixels. |

## Channel Selection

All channels until the user can clarify modality.

## Frequency Bands

Conservative: 1–80 Hz general bandpass; 0.1–80 if ERP / sleep; not
narrower until modality is confirmed.

## Recommended Pipeline

```
load → inspect_neural → hp_dc:0.1 → bandpass:1,80 → drop_bads:auto →
report_to_user (specify ambiguity: "modality unclear; suggested
clarification: [a / b / c]")
```

Do **not** apply:
- Modality-specific operators (`threshold_spike`, `mBLL`, `ssp_eog`).
- Aggressive notch / ICA.
- Source localization / connectivity.

## Common Artifacts

Unknown until modality clarified.

## Pitfalls & Failure Modes

- **"Silent generic default."** The whole point of this paradigm is to
  prevent this. If LLM proposes any modality-specific operator while
  in this paradigm, reject with reason "unknown modality".
- **No modality field in `meta`.** Set `meta["modality"] = "unknown"`
  explicitly so downstream guards trigger.

## Boundary with Related Paradigms

- **`custom_binary.md`**: tiny step further when even the format is
  unknown (raw `.bin / .dat`); this paradigm assumes the format is
  loaded.
- **Any specific paradigm**: should be entered only after the user
  confirms modality.

## Standalone End-to-End

```python
import mne
raw = mne.io.read_raw(path, preload=True, verbose="ERROR")
# Conservative
raw.filter(1, 80)
raw.drop_channels([c for c, t in zip(raw.ch_names, raw.get_data())
                   if np.ptp(t) < 1e-9])
raw.save("inspection.fif")
print(f"sfreq={raw.info['sfreq']}, n_channels={len(raw.ch_names)}, "
      f"duration={raw.times[-1]}s — modality unclear; ask user.")
```

## EasyBCI Pipeline Spec

```yaml
modality: unknown
paradigm: unknown_modality
analysis_goal: exploratory
steps:
  - load:auto
  - inspect_neural
  - hp_dc:0.1
  - bandpass:1,80
  - drop_bads:auto
  - report_to_user:require_modality_clarification
```

## References

1. (none — this is a workflow convention, not a methodology paper.)
