#!/usr/bin/env python3
"""
End-to-end spike sorting pipeline for raw Neuropixels NWB data.
Usage: python sort_raw_neuropixels.py <input.nwb> <output_dir>

This is the REFERENCE workflow for agent-driven spike sorting.
Copy and adapt for each session's specific data.
"""
import sys
import os
import argparse
import spikeinterface.full as si

def main():
    parser = argparse.ArgumentParser(description="Sort raw Neuropixels NWB data")
    parser.add_argument("input_nwb", help="Path to input NWB file")
    parser.add_argument("output_dir", help="Output directory for sorting results")
    parser.add_argument("--sorter", default=None,
                        help="Sorter name (auto-detected if omitted)")
    parser.add_argument("--freq-min", type=float, default=300,
                        help="High-pass frequency (Hz)")
    parser.add_argument("--freq-max", type=float, default=6000,
                        help="Low-pass frequency (Hz)")
    parser.add_argument("--n-jobs", type=int, default=8,
                        help="Parallel jobs")
    args = parser.parse_args()

    si.set_global_job_kwargs(n_jobs=args.n_jobs)

    # ── Step 0: Check available sorters ──
    installed = si.installed_sorters()
    print(f"Available sorters: {installed}")

    if args.sorter:
        sorter = args.sorter
    else:
        # Preferred order for Neuropixels: kilosort > mountainsort5 > spykingcircus2
        for candidate in ['kilosort4', 'kilosort3', 'kilosort2_5', 'kilosort2',
                          'mountainsort5', 'spykingcircus2', 'tridesclous2']:
            if candidate in installed:
                sorter = candidate
                break
        else:
            raise RuntimeError(f"No known sorter installed. Available: {installed}")
    print(f"Using sorter: {sorter}")

    # ── Step 1: Load AP-band recording ──
    # Neuropixels NWB typically has ElectricalSeriesAP* in /acquisition/
    # SpikeInterface's read_nwb_recording auto-selects the ElectricalSeries
    # with the highest sampling rate.
    recording = si.read_nwb_recording(
        args.input_nwb,
        electrical_series_name=None,  # auto-select (prefers AP band)
    )
    print(f"\nRecording loaded:")
    print(f"  Channels: {recording.get_num_channels()}")
    print(f"  Sampling rate: {recording.get_sampling_frequency()} Hz")
    print(f"  Duration: {recording.get_total_duration() / 60:.1f} min")
    print(f"  Dtype: {recording.get_dtype()}")

    # ── Step 2: Preprocessing ──
    print("\nPreprocessing...")
    # Bandpass filter (isolate spike frequency band)
    recording_f = si.bandpass_filter(
        recording, freq_min=args.freq_min, freq_max=args.freq_max
    )
    # Common median reference (remove shared noise)
    recording_cmr = si.common_reference(
        recording_f, reference='global', operator='median'
    )
    print("Preprocessing complete.")

    # ── Step 3: Run sorter ──
    sorter_dir = os.path.join(args.output_dir, f"{sorter}_output")
    print(f"\nRunning {sorter}... (output: {sorter_dir})")
    sorting = si.run_sorter(
        sorter,
        recording_cmr,
        output_folder=sorter_dir,
        remove_existing_folder=True,
        verbose=True,
    )
    print(f"Sorting complete. Found {len(sorting.get_unit_ids())} units.")

    # ── Step 4: Post-processing ──
    waveform_dir = os.path.join(args.output_dir, "waveforms")
    print(f"\nExtracting waveforms to {waveform_dir}...")
    we = si.extract_waveforms(
        recording_cmr, sorting, folder=waveform_dir,
        ms_before=1.0, ms_after=2.0,
        max_spikes_per_unit=500,
        overwrite=True,
    )

    print("Computing quality metrics...")
    metrics = si.compute_quality_metrics(
        we, metric_names=['snr', 'isi_violations_ratio',
                          'presence_ratio', 'amplitude_cutoff',
                          'firing_rate', 'num_spikes']
    )
    print(f"Metrics for {len(metrics)} units:")
    print(metrics.to_string())

    # ── Step 5: Auto-curation ──
    print("\nAuto-curating...")
    sorting_curated = si.auto_curation(
        sorting, metrics,
        isi_violations_ratio_threshold=0.5,
        snr_threshold=5.0,
        presence_ratio_threshold=0.9,
    )
    n_total = len(sorting.get_unit_ids())
    n_kept = len(sorting_curated.get_unit_ids())
    print(f"Retained {n_kept}/{n_total} units after curation.")

    # ── Step 6: Save ──
    curated_dir = os.path.join(args.output_dir, "curated_sorting")
    sorting_curated = sorting_curated.save(folder=curated_dir)
    metrics_path = os.path.join(args.output_dir, "quality_metrics.csv")
    metrics.to_csv(metrics_path)

    print(f"\nDone. Results in {args.output_dir}/")
    print(f"  Sorting:    {sorter_dir}")
    print(f"  Waveforms:  {waveform_dir}")
    print(f"  Curated:    {curated_dir}")
    print(f"  Metrics:    {metrics_path}")

    # ── Summary ──
    good_units = (metrics['snr'] > 5) & (metrics['isi_violations_ratio'] < 0.005) & (metrics['presence_ratio'] > 0.9)
    n_good = good_units.sum()
    print(f"\nSummary: {n_good} good single units (SNR>5, ISI<0.5%, presence>0.9)")

if __name__ == "__main__":
    main()
