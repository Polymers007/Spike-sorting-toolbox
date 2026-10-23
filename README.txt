SPIKE SORTING V8 - QUICK USER GUIDE

Before starting
- Use a configured Kilosort Python environment and a separate Phy environment.
- Batch sorting requires an NVIDIA GPU and CUDA-enabled PyTorch.
- Keep the numbered project folders together and allow enough free disk space.

1. ADD BIN RECORDINGS
Place headerless BIN files directly inside a device folder:
  001_Bin/Data/<device>/<recording>.bin
Create the device folder if needed. Each BIN is sorted separately; files are
not combined automatically.

The included configuration expects:
  data type:       little-endian signed int16 (2 bytes per value)
  sample layout:   all channels for sample 0, then all channels for sample 1, etc.
  channel count:   16
  sampling rate:   30000 Hz
  channel order:   the chanMap in 002_Kilosort/probes/16chs_AAB.json

Use BIN files exported before running this program. Their channel count,
sampling rate, channel order and voltage calibration must be known.
For the bundled format, file size must be a multiple of 32 bytes.
Processing files with different acquisition settings requires separate batches
and corresponding parameter/probe changes.

2. CHECK THE PROBE AND PARAMETERS
Probe layout:
  002_Kilosort/probes/16chs_AAB.json
Batch configuration:
  002_Kilosort/kilosort_batch_configuration.json

Open:
  002_Kilosort/Kilosort4.bat
Select a BIN file, select 16chs_AAB.json, and check Preview Probe.
Confirm channel mapping, channel count, sampling rate and int16 data type.
Close the Kilosort GUI.

Open:
  002_Kilosort/Kilosort4_Parameter_Setting.bat
Check Current probe, n_chan_bin, fs, data_dtype and bad_channels.
Set waveform_scale_uV_per_bit to the number of microvolts represented by one
stored BIN count. The bundled value is 0.195; change it for other calibrations.
This setting controls waveform voltage display; it does not alter sorting.
Click Save and Close.

Optional per-recording calibration:
  001_Bin/Data/<device>/<recording>_metadata.json
with content such as:
  {"scale_uV_per_bit": 0.195}
This value overrides waveform_scale_uV_per_bit for that recording.
Calibration values must be finite and positive.

The saved configuration is applied to every BIN in the batch.
If changing the probe, first rename the existing kilosort_batch_configuration.json,
select the new probe in the GUI, and save a new batch configuration.

3. RUN KILOSORT
With the Kilosort GUI closed, double-click:
  002_Kilosort/Batch_Kilosort4_All.bat
Sorting results are saved to:
  004_Result/Kilosort4/<device>/<recording>/
Check each recording in:
  004_Result/Kilosort4/batch_sort_status.csv
Failed recordings have a batch_sort_error.txt file in their result folder.
Rerunning the batch skips result folders containing the required output files.
Changing parameters alone does not rerun existing results.

4. REVIEW IN PHY
Double-click:
  003_Phy/01_Curation Manager.bat
Click Refresh, select one recording, and click Open in Phy.
The Manager creates a working copy for an Unprocessed recording.
Review waveforms, amplitudes, correlograms and similar clusters.
Label clusters as Good, MUA or Noise, and merge or split them as needed.
All clusters can be reviewed regardless of spike count or current label.
Press Ctrl+S before closing Phy, then wait for validation to finish.
If no files changed, confirm completion only after reviewing the recording.
Click Select Next Unfinished to continue.

Optional: Cached Refinement
After a working copy exists, click Open Cached Refinement to preview edits.
Click Finish Editing when done, then Apply All Edited in the Manager.
Finish Editing alone does not change the final results.
Apply All Edited applies every Edited recording, including filtered-out rows.
Apply pending edits before editing the same recording directly in Phy.

5. USE THE CURATED RESULTS
Final working results:
  004_Result/Phy/Working_Curated/<device>/<recording>/
Main files:
  spike_times.npy       - Spike times in samples; divide by fs for seconds.
  spike_clusters.npy    - Curated cluster ID for each spike.
  cluster_group.tsv     - Good, MUA and Noise labels.
  params.py             - BIN reference and recording settings.

Keep the corresponding BIN files available for Phy and waveform review.
BIN references are stored relative to the result folder.
Recovery backups are stored in 004_Result/Phy/Snapshots/.
Completed indicates workflow completion; verify unit quality before analysis.

IF A LAUNCHER CANNOT FIND PYTHON
Set these Windows user environment variables to the environment executable:
  AP_KILOSORT_PYTHON = Kilosort environment's python.exe
  AP_PHY_PYTHON      = Phy environment's python.exe
Executable paths may be absolute or relative to the project folder.
The Phy environment needs Phy, phylib, NumPy, PyQt5, Matplotlib and Tkinter.
