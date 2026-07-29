---
name: custom_binary
description: 'Fallback for raw .bin / .dat / int16 stream — human-in-the-loop format declaration'
layer: L2
group: modality
metadata:
  tags: [custom, raw_binary, dat, int16, fallback, human_loop]
  modalities: [unknown]
  paradigms: [custom_binary, raw_dat]
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
# Custom Binary — Raw `.bin / .dat / int16` Stream

## Neuroscience Background

When a user provides a raw binary file with no header (`.bin`, `.dat`,
arbitrary int16/float32 stream) the LLM must engage in a **human-in-the-loop
interaction** to declare:

1. dtype (`int16`, `int32`, `float32`, `float64`)
2. Sample rate (Hz)
3. Channel count
4. Channel layout (row-major vs column-major / interleaved)
5. Voltage scaling (raw → V)

Without these, no preprocessing is possible. This paradigm formalizes
the dialogue.

## Required Declarations

Before any pipeline can run, `meta` must contain:

```yaml
dtype: "int16"          # or float32 etc.
sfreq: 30000.0
n_channels: 384
layout: "channel_interleaved"  # or "channel_separate"
voltage_per_count: 2.34e-6     # to convert to V
modality: "spike"              # user-asserted
```

## Inspection Checklist

After declaration, validate:
- File size = `n_channels × n_samples × bytes_per_sample`.
- First-channel PSD: does it look like neural data (1/f shape)?
- Voltage range: matches stated modality (µV scale for surface EEG;
  mV for raw extracellular).

## Recommended Pipeline

```
human-declare:dtype,sfreq,n_channels,layout,voltage_per_count,modality
  → load_custom_binary
  → bandpass:1,80 (or modality-specific once declared)
  → drop_bads:auto
  → report_to_user (PSD of first channel; user confirms or revises)
```

## Pitfalls & Failure Modes

- **Wrong dtype.** Reading int16 as float32 gives nonsense; the PSD
  first-channel sanity is the catch.
- **Wrong sample rate.** Frequencies appear shifted; PSD peak at unexpected
  frequency.
- **Interleaved vs separate.** Most acquisition systems are channel-interleaved
  (`[t0 c0, t0 c1, ..., t1 c0, ...]`); SpikeGLX is this. Some labs save
  channel-separate (`[c0 t0 t1 ..., c1 t0 t1 ...]`).
- **Sign / polarity.** If user reports extracellular spikes go positive
  → polarity inversion; flip after load.

## Boundary with Related Paradigms

- **`unknown_modality.md`**: this paradigm is the *next step* after
  `inspect_neural` fails. `unknown_modality` assumes the format is
  loadable; `custom_binary` assumes format requires declaration.

## EasyBCI Pipeline Spec

```yaml
modality: <user-declared>
paradigm: custom_binary
analysis_goal: exploratory
steps:
  - declare_format:dtype,sfreq,n_channels,layout,voltage_per_count
  - load_custom_binary:per_declaration
  - inspect_neural
  - bandpass:1,80
  - report_to_user:psd_first_channel
```

## References

1. (none — this is a workflow convention.)
