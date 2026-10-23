BIN input

Place each headerless .bin recording in Data/<device>/<recording>.bin.
Each file is sorted separately.

The included configuration expects little-endian int16 data, with channels
interleaved for each sample, 16 channels, and a 30 kHz sampling rate.
Channel order must match 002_Kilosort/probes/16chs_AAB.json.

Set waveform_scale_uV_per_bit in the batch parameter window to match the BIN
calibration. The bundled value is 0.195 uV per stored count. A per-recording
<recording>_metadata.json containing scale_uV_per_bit overrides that value.
