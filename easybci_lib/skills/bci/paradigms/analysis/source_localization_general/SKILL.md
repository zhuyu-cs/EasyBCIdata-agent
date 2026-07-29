---
name: source_localization_general
description: 'Source localization paradigm — distributed and focal source modeling for EEG/MEG'
layer: L2
group: analysis
metadata:
  tags: [source, inverse, eeg, meg, sloreta, dipole, lcmv]
  modalities: [eeg, meg]
  paradigms: [source_localization, inverse_modelling]
  analysis_goal_allowed:
    - source_localization
    - feature_extraction
    - clinical_screening
    - exploratory
  analysis_goal_forbidden:
    - online_inference
---
# Source Localization (General)

## Neuroscience Background

Source localization recovers the cortical generators of EEG / MEG
signals via inverse modelling. The forward model (G, leadfield) maps
hypothetical cortical sources to sensor signals; the inverse asks: given
the sensor data, which source distribution most likely produced it?
There is no unique solution — assumptions (smoothness, focality, sparsity)
disambiguate. Three families dominate:

1. **Distributed inverse** (sLORETA, MNE, eLORETA): smooth source maps
   over a 1000+ dipole grid; mathematically equivalent to regularized
   minimum-norm.
2. **Adaptive beamformer** (LCMV, DICS): per-source spatial filter
   minimizes total variance; sharper than distributed but covariance-dependent.
3. **Equivalent dipole** (ECD, RAP-MUSIC): one or two free-position
   dipoles; canonical for focal generators (N20, M100).

## Channel Selection

- High-density montage required: ≥ 32 channels for EEG (64+ preferred),
  ≥ 64 for MEG.
- Electrode positions in 3D required (`meta["electrode_positions"]`).
- For EEG, individual head shape via MRI improves accuracy; sphere
  approximation is the floor.

## Frequency Bands

Source-localization is band-agnostic — apply on the bandpassed data of
interest. Common bands:

| Band | Use |
|---|---|
| Broad (1–40 Hz) | General sensor → source mapping. |
| Alpha (8–13) | Resting-state alpha origin. |
| Beta (13–30) | Motor cortex. |
| Gamma (30–80) | High-frequency oscillation localization. |
| Narrowband (per task) | Task-specific source. |

## Recommended Pipeline

```
load → bandpass → notch → drop_bads → average_reference (or REST) →
epoch → compute_noise_cov → compute_data_cov → forward_model_compute →
sloreta (or lcmv / dipole_fit) → source_visualization
```

Each `_compute_*` step is upstream; the inverse op (`sloreta` / `lcmv`
/ `dipole_fit`) consumes them from `meta`.

## Common Artifacts

| Artifact | Effect on inverse | Mitigation |
|---|---|---|
| Wrong noise cov | Source map shows stripes / sLORETA z-scores wrong. | Recompute from baseline. |
| Channel topology errors | Source position shifted. | Validate `meta["electrode_positions"]`. |
| ICA over-removal | Source signal weakened. | Reduce IC removal to preserve task SNR. |
| Forward-model defects | Position bias. | Use BEM > sphere. |

## Quality Metrics

| Metric | Range | Use |
|---|---|---|
| GOF (dipole_fit) | 0–1 | Per-dipole explained variance. |
| Source-resolution | mm | FWHM of the point-spread function. |
| Localization error | mm vs phantom | Validation. |

## Classification / Decoding Baselines

Source-space features (e.g. region-of-interest amplitude, source-space
PSD) feed standard classifiers; no canonical accuracy benchmark across
all tasks.

## Public Datasets

| Dataset | Format | Notes |
|---|---|---|
| MNE sample dataset | `.fif` | Auditory / visual MEG + MRI. |
| OpenfMRI/OpenNeuro M/EEG | BIDS-MEG | Various tasks; includes head models. |

## Pitfalls & Failure Modes

- **Mixing reference choices.** Using CAR for the data while the forward
  model assumes an absolute reference causes systematic bias. Use REST or
  match references explicitly.
- **Channel-position mismatch.** A swapped channel breaks the inverse;
  always validate `len(channels) == leadfield.shape[0]`.
- **Source-grid resolution.** Too few grid points → coarse localization;
  too many → numerical overhead.

## Boundary with Related Paradigms

- **`connectivity.md`**: source-space connectivity uses sources as inputs
  to PLV / PLI / Granger. The pipeline is sequential: localize → then
  connectivity.
- **`phase_amplitude_coupling.md`**: PAC at source level requires this
  paradigm upstream.

## Standalone End-to-End Pipeline

```python
import mne

raw = mne.io.read_raw_fif("sample-raw.fif", preload=True)
raw.filter(1, 40); raw.notch_filter(50)
events = mne.find_events(raw)
epochs = mne.Epochs(raw, events, tmin=-0.2, tmax=0.5,
                    baseline=(None, 0), preload=True)
noise_cov = mne.compute_covariance(epochs, tmax=0)
fwd = mne.make_forward_solution(epochs.info,
                                trans="sample-trans.fif",
                                src="sample-src.fif",
                                bem="sample-bem.fif")
inv = mne.minimum_norm.make_inverse_operator(epochs.info, fwd, noise_cov)
stc = mne.minimum_norm.apply_inverse(epochs.average(), inv,
                                     lambda2=1./9, method="sLORETA")
stc.plot(subject="sample", hemi="both")
```

## EasyBCI Pipeline Spec

```yaml
modality: eeg
paradigm: source_localization_general
analysis_goal: source_localization
steps:
  - load:fif
  - bandpass:1,40
  - notch:50
  - drop_bads:auto
  - rest_reference
  - compute_noise_cov:baseline
  - forward_model_compute:bem
  - sloreta:snr=3.0
```

## References

1. Hämäläinen, M. S., & Ilmoniemi, R. J. (1994). *Interpreting magnetic
   fields of the brain: minimum norm estimates*. Med. Biol. Eng. Comput.
   32(1): 35–42. doi:10.1007/BF02512476.
2. Pascual-Marqui, R. D. (2002). *sLORETA technical details*. Methods
   Find Exp Clin Pharmacol 24(Suppl D): 5–12.
3. Van Veen, B. D. et al. (1997). *Localization of brain electrical
   activity via linearly constrained minimum variance spatial filtering*.
   IEEE Trans. Biomed. Eng. 44(9): 867–880. doi:10.1109/10.623056.
4. Baillet, S. (2017). *Magnetoencephalography for brain electrophysiology
   and imaging*. Nature Neuroscience 20: 327–339. doi:10.1038/nn.4504.
