---
name: neural-io-index
description: "Index of L0 data-format loader skills by family"
layer: L0
group: ephys_wide_band
metadata:
  tags: [index, neural-io, loader, navigation, l0]
  analysis_goal_allowed: [classification, source_localization, feature_extraction, clinical_screening, exploratory, generic, connectivity, phase_amplitude_coupling, online_inference]
  analysis_goal_forbidden: []
---

# L0 Neural-IO Format Index

> **Navigation.**  
> **L0 (this page)** → [L1 Orchestration](../pipeline/SKILL.md) →
> [L2 Paradigms](../paradigms/SKILL.md) → [L3 Operators](../operators/SKILL.md).

This index lists L0 IO format loader skills, grouped by file-format family.
Each format skill documents the loader library, channel/sample-rate/events
mapping into the `OperatorIO` schema, and known pitfalls. Load one with
`skill_view(name='<format>')` only when `_handle_inspect_data` hits a
non-built-in format.

The dispatcher hits L0 **last**, after L1 → L2 → L3 — L0 is consulted
only when `_handle_inspect_data` hits a format the built-in dispatcher
can't already load; the only legitimate reason to read an L0 skill is
"I need to write a custom loader for this format before any pipeline
step can proceed."

## Format families

> Format-skill stubs live as `io/<family>/<format>/SKILL.md`. The list below
> matches the planned content drop in T9 sibling plan
> `03-preprocessing-skill-expansion/3-3-io-formats.md`. Anything tagged
> *(planned)* below is a slot reserved for that plan; the row should land
> on this page automatically once the format skill is written.

### `ephys_wide_band/` — General wide-band electrophysiology  *(planned: 3)*

| Format | Extension | Loader |
|---|---|---|
| NWB / Neurodata Without Borders | `.nwb` | `pynwb` ≥ 2.5 |
| BIDS-iEEG sidecar | sidecar `.tsv` / `.json` | `mne-bids` |
| MEF3 (Mayo) | `.mefd/` directory | `pymef` |

### `spike_ephys/` — High-rate single-unit ephys  *(planned: 5)*

| Format | Extension | Loader |
|---|---|---|
| SpikeGLX (Neuropixel) | `.bin + .meta` | `SpikeInterface.read_spikeglx` |
| Open Ephys | `continuous.dat + .oebin` | `SpikeInterface.read_openephys` |
| Blackrock | `.ns5 / .ns2 / .nev` | `brpylib` |
| Plexon | `.plx / .pl2` | `neo.io.PlexonIO` |
| Neuralynx | `.ncs / .nse / .nev` | `neo.io.NeuralynxIO` |

### `clinical_ieeg/` — Clinical iEEG / sEEG  *(planned)*

Clinical depth/grid recordings often arrive as proprietary vendor formats
plus DICOM-MR for electrode localization. Format skills here cover the
loader contract, anonymization expectations, and the BIDS-iEEG export path.

### `consumer_eeg/` — Mainstream EEG formats *(in-code loaders → planned stubs)*

| Format | Extension | Loader |
|---|---|---|
| EDF / EDF+ | `.edf` | `mne.io.read_raw_edf` |
| FIF | `.fif` | `mne.io.read_raw_fif` |
| BrainVision | `.vhdr + .vmrk + .eeg` | `mne.io.read_raw_brainvision` |
| EEGLAB SET | `.set` | `mne.io.read_raw_eeglab` |
| LSL XDF | `.xdf` | `pyxdf` |
| CSV / NPY | `.csv` / `.npy` | `numpy` + manual schema |
| MAT | `.mat` | `scipy.io.loadmat` |

### `fnirs/` — fNIRS  *(planned: 1)*

| Format | Extension | Loader |
|---|---|---|
| SNIRF | `.snirf` | `mne.io.read_raw_snirf` |

### `streaming/` — Real-time stream  *(planned: 1)*

| Format | Source | Loader |
|---|---|---|
| LSL stream | Lab Streaming Layer | `pylsl` |

## L1 ↔ L0 integration

When `_handle_inspect_data` encounters a format that is **not** in the
"in-code loaders" list above (EDF / FIF / BrainVision / EEGLAB SET / XDF /
CSV / NPY / MAT), it must:

1. Call `skill_view(name='neural-io-index')` to discover the right format family.
2. Call `skill_view(name='<format>')` to fetch the loader contract before
   writing or generating a custom loader.

See `bci/pipeline/SKILL.md` Step 1 (Format-Agnostic Loading) for
the corresponding orchestration text.

## Layering contract

- **layer**: every format skill carries `layer: L0` in frontmatter.
- **group**: one of `{ephys_wide_band, spike_ephys, clinical_ieeg, consumer_eeg, fnirs, streaming}`,
  matching its parent directory (`bci/neural-io/<group>/<format>/`).
- **analysis_goal_allowed/forbidden**: optional for L0 (the consistency
  checker only requires these on L2 paradigm skills). When present,
  values must come from the 9-enum REGISTRY in
  `easybci_lib/tools/neural_processing/preprocess/analysis_goals.py`.

The layer / group enums are defined in
`easybci_lib/tools/neural_processing/skill_layers.py` (single source of
truth). Run `python -m easybci_lib.tools.neural_processing._check_consistency --strict`
to verify the contract before merging new format skills.
