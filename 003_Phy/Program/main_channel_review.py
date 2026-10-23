from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from curation_common import portable_paths
from phy.apps.template.gui import TemplateController
from phylib.io.model import load_model

from curation_common import OUTPUT_ROOT, PROJECT_ROOT, now_text, parse_params

ALGORITHM_VERSION = "phy-main-channel-filtered-v3"
CACHE_ROOT = OUTPUT_ROOT / "Main_Channel_Review_Cache"
CACHE_NAMES = (
    "_phy_spikes_subset.waveforms.npy",
    "_phy_spikes_subset.spikes.npy",
    "_phy_spikes_subset.channels.npy",
)

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def _waveform_scale(folder: Path) -> tuple[float, str]:
    config_path = PROJECT_ROOT / "002_Kilosort" / "kilosort_batch_configuration.json"
    scale = 1.0
    source = "BIN data units"
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if "waveform_scale_uV_per_bit" in config:
            scale = float(config["waveform_scale_uV_per_bit"])
            source = str(config_path)
    params = parse_params(folder / "params.py")
    data_path = Path(str(params.get("dat_path", "")))
    metadata_path = data_path.with_name(data_path.stem + "_metadata.json")
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if "scale_uV_per_bit" in metadata:
            scale = float(metadata["scale_uV_per_bit"])
            source = str(metadata_path)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("The BIN waveform scale must be finite and greater than zero")
    return scale, source

def _signature(folder: Path) -> dict:
    signature = {
        name: _sha256(folder / name)
        for name in ("spike_clusters.npy", "spike_times.npy", "params.py")
    }
    signature["waveform_scale_uV_per_bit"] = _waveform_scale(folder)[0]
    return signature

def cache_path(folder: Path) -> Path:
    return CACHE_ROOT / folder.parent.name / folder.name / "main_channel_waveforms.npz"

def metadata_path(folder: Path) -> Path:
    return cache_path(folder).with_name("metadata.json")

def _shift_traces(values: np.ndarray, amount: int) -> np.ndarray:
    if amount == 0:
        return values.copy()
    result = np.zeros_like(values)
    if amount > 0:
        result[:, amount:] = values[:, :-amount]
    else:
        width = -amount
        result[:, :-width] = values[:, width:]
    return result

def _shift_waveform(values: np.ndarray, amount: int) -> np.ndarray:
    if amount == 0:
        return values.copy()
    result = np.full_like(values, np.nan)
    if amount > 0:
        result[amount:, :] = values[:-amount, :]
    else:
        width = -amount
        result[:-width, :] = values[width:, :]
    return result

def _baseline_traces(values: np.ndarray) -> np.ndarray:
    edge = min(10, max(2, values.shape[1] // 8))
    baseline = np.median(np.concatenate((values[:, :edge], values[:, -edge:]), axis=1), axis=1)
    return values - baseline[:, None]

def _ensure_fresh_phy_cache(folder: Path) -> None:
    if all((folder / name).is_file() for name in CACHE_NAMES):
        return
    model = load_model(folder / "params.py")
    try:
        model.save_spikes_subset_waveforms(max_n_spikes_per_template=200, max_n_channels=16)
    finally:
        model.close()
    missing = [name for name in CACHE_NAMES if not (folder / name).is_file()]
    if missing:
        raise RuntimeError("Phy failed to extract filtered waveform cache: " + ", ".join(missing))

def _main_channels(folder: Path, cluster_ids: np.ndarray) -> dict[int, int]:
    model = load_model(folder / "params.py")
    controller = TemplateController(model=model, dir_path=folder, enable_threading=False)
    try:
        available = {int(value) for value in controller.supervisor.clustering.cluster_ids}
        missing = [int(value) for value in cluster_ids if int(value) not in available]
        if missing:
            raise RuntimeError(f"Phy does not recognize clusters: {missing[:10]}")
        return {int(cluster_id): int(controller.get_best_channel(int(cluster_id)))
                for cluster_id in cluster_ids}
    finally:
        controller.close()

def build_main_channel_cache(folder: Path, force: bool = False) -> dict:
    folder = Path(folder)
    _ensure_fresh_phy_cache(folder)
    destination = cache_path(folder)
    meta_file = metadata_path(folder)
    signature = _signature(folder)
    if not force and destination.is_file() and meta_file.is_file():
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            if meta.get("algorithm_version") == ALGORITHM_VERSION and meta.get("signature") == signature:
                return {"status": "skipped", "cache": str(destination), **meta}
        except (OSError, ValueError):
            pass

    assignments = np.load(folder / "spike_clusters.npy", mmap_mode="r").ravel()
    cluster_ids = np.unique(assignments).astype(np.int64)
    main_by_cluster = _main_channels(folder, cluster_ids)
    sample_ids = np.load(folder / CACHE_NAMES[1]).astype(np.int64)
    sample_waveforms = np.load(folder / CACHE_NAMES[0], mmap_mode="r")
    sample_channels = np.load(folder / CACHE_NAMES[2], mmap_mode="r").astype(np.int32)
    if sample_waveforms.ndim != 3 or sample_channels.shape[:1] != sample_waveforms.shape[:1]:
        raise RuntimeError("Phy filtered waveform cache shapes are inconsistent")
    sample_clusters = assignments[sample_ids]
    channel_positions = np.asarray(np.load(folder / "channel_positions.npy", mmap_mode="r"), dtype=np.float32)
    n_channels = int(channel_positions.shape[0])
    n_samples = int(sample_waveforms.shape[1])
    center = n_samples // 2
    params = parse_params(folder / "params.py")
    sample_rate = float(params.get("sample_rate", 30000.0))
    scale_uv, scale_source = _waveform_scale(folder)

    medians = np.full((len(cluster_ids), n_samples, n_channels), np.nan, dtype=np.float32)
    main_channels = np.full(len(cluster_ids), -1, dtype=np.int32)
    shifts = np.zeros(len(cluster_ids), dtype=np.int16)
    sample_counts = np.zeros(len(cluster_ids), dtype=np.int32)
    all_trace_clusters: list[np.ndarray] = []
    all_trace_spikes: list[np.ndarray] = []
    all_traces: list[np.ndarray] = []
    metrics: list[dict] = []

    for index, cluster_id_value in enumerate(cluster_ids):
        cluster_id = int(cluster_id_value)
        main_channel = main_by_cluster[cluster_id]
        main_channels[index] = main_channel
        rows = np.flatnonzero(sample_clusters == cluster_id)
        if not 0 <= main_channel < n_channels:
            metrics.append({"cluster_id": cluster_id, "status": "invalid_main_channel"})
            continue
        main_traces = []
        main_rows = []
        for row in rows:
            slots = np.flatnonzero(np.asarray(sample_channels[row]) == main_channel)
            if slots.size:
                main_traces.append(np.asarray(sample_waveforms[row, :, int(slots[0])], dtype=np.float32))
                main_rows.append(int(row))
        if len(main_traces) < 3:
            metrics.append({
                "cluster_id": cluster_id, "status": "too_few_main_channel_waveforms",
                "phy_main_channel": main_channel, "display_channel": main_channel,
                "decision_channel": main_channel, "sample_count": len(main_traces),
            })
            continue
        main_raw = _baseline_traces(np.stack(main_traces))
        median_main_raw = np.median(main_raw, axis=0)
        safe_start, safe_stop = 3, n_samples - 3
        peak_before = int(np.argmin(median_main_raw[safe_start:safe_stop]) + safe_start)
        edge = min(10, max(2, n_samples // 8))
        edge_values = np.concatenate((main_raw[:, :edge], main_raw[:, -edge:]), axis=1)
        noise = float(1.4826 * np.median(np.abs(edge_values - np.median(edge_values))))
        trough_before = max(0.0, -float(median_main_raw[peak_before]))
        preliminary_snr = trough_before / max(noise, 1e-6)
        proposed_shift = int(center - peak_before)
        reliable_alignment = preliminary_snr >= 2.5 and abs(proposed_shift) <= center - 3
        applied_shift = proposed_shift if reliable_alignment else 0
        aligned_main = _shift_traces(main_raw, applied_shift)

        physical_median = np.full((n_samples, n_channels), np.nan, dtype=np.float32)
        for physical in range(n_channels):
            traces = []
            for row in rows:
                slots = np.flatnonzero(np.asarray(sample_channels[row]) == physical)
                if slots.size:
                    traces.append(np.asarray(sample_waveforms[row, :, int(slots[0])], dtype=np.float32))
            if traces:
                clean = _baseline_traces(np.stack(traces))
                physical_median[:, physical] = np.median(clean, axis=0)
        physical_median = _shift_waveform(physical_median, applied_shift)
        trace = physical_median[:, main_channel]

        search_start, search_stop = max(1, center - 18), min(n_samples - 1, center + 19)
        individual_locations = np.argmin(aligned_main[:, search_start:search_stop], axis=1) + search_start
        location_median = float(np.median(individual_locations))
        jitter_mad = float(1.4826 * np.median(np.abs(individual_locations - location_median)))
        center_fraction = float(np.mean(np.abs(individual_locations - center) <= 6))
        individual_troughs = -aligned_main[np.arange(aligned_main.shape[0]), individual_locations]
        individual_noise = 1.4826 * np.median(
            np.abs(edge_values - np.median(edge_values, axis=1, keepdims=True)), axis=1
        )
        amplitude_fraction = float(np.mean(individual_troughs >= 2.5 * np.maximum(individual_noise, 1e-6)))
        reference = trace[search_start:search_stop] - np.nanmean(trace[search_start:search_stop])
        ref_norm = float(np.linalg.norm(reference))
        correlations = []
        for item in aligned_main:
            part = item[search_start:search_stop] - np.mean(item[search_start:search_stop])
            denominator = float(np.linalg.norm(part) * ref_norm)
            correlations.append(float(np.dot(part, reference) / denominator) if denominator else 0.0)
        median_correlation = float(np.median(correlations))
        trough_after = max(0.0, -float(np.nanmin(trace[search_start:search_stop])))
        positive = max(0.0, float(np.nanmax(trace)))
        negative_dominance = trough_after / max(trough_after, positive, 1e-6)
        peak_snr = trough_after / max(noise, 1e-6)
        half_width = int(np.sum(trace < -0.5 * max(trough_after, 1e-6)))
        status = "completed" if reliable_alignment else "needs_review"
        medians[index] = physical_median
        shifts[index] = applied_shift
        sample_counts[index] = len(main_rows)
        all_trace_clusters.append(np.full(len(main_rows), cluster_id, dtype=np.int64))
        all_trace_spikes.append(sample_ids[np.asarray(main_rows, dtype=np.int64)])
        all_traces.append(aligned_main.astype(np.float32))
        metrics.append({
            "cluster_id": cluster_id, "status": status,
            "phy_main_channel": main_channel, "display_channel": main_channel,
            "decision_channel": main_channel, "sample_count": len(main_rows),
            "peak_before": peak_before, "peak_target": center, "applied_shift": applied_shift,
            "preliminary_snr": preliminary_snr, "peak_snr": peak_snr,
            "negative_dominance": negative_dominance, "center_fraction": center_fraction,
            "jitter_mad": jitter_mad, "amplitude_fraction": amplitude_fraction,
            "median_correlation": median_correlation, "half_width": half_width,
            "channel_invariant": True,
        })

    trace_clusters = np.concatenate(all_trace_clusters) if all_trace_clusters else np.empty(0, dtype=np.int64)
    trace_spikes = np.concatenate(all_trace_spikes) if all_trace_spikes else np.empty(0, dtype=np.int64)
    traces = np.concatenate(all_traces) if all_traces else np.empty((0, n_samples), dtype=np.float32)
    if any(item.get("phy_main_channel") != item.get("display_channel") or
           item.get("phy_main_channel") != item.get("decision_channel")
           for item in metrics if "phy_main_channel" in item):
        raise RuntimeError("main-channel invariant failed")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".tmp.{os.getpid()}")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream, cluster_ids=cluster_ids, main_channels=main_channels, medians=medians,
            shifts=shifts, sample_counts=sample_counts, trace_clusters=trace_clusters,
            trace_spikes=trace_spikes, main_traces=traces,
            channel_positions=channel_positions,
        )
    os.replace(temporary, destination)
    meta = {
        "schema_version": 2, "algorithm_version": ALGORITHM_VERSION,
        "created": now_text(), "working_folder": str(folder),
        "source": "fresh Phy high-pass waveform extraction",
        "signature": signature, "cluster_count": int(len(cluster_ids)),
        "waveform_samples": n_samples, "sample_rate_hz": sample_rate,
        "scale_uV_per_bit": scale_uv, "scale_source": scale_source,
        "main_channel_invariant": True,
        "completed_clusters": sum(item.get("status") == "completed" for item in metrics),
        "needs_review_clusters": [item["cluster_id"] for item in metrics if item.get("status") != "completed"],
        "metrics": metrics,
    }
    meta_file.write_text(json.dumps(portable_paths(meta), ensure_ascii=False, indent=2), encoding="utf-8")
    return {"status": "completed", "cache": str(destination), **meta}

def load_main_channel_cache(folder: Path, force: bool = False) -> tuple[dict, dict[str, np.ndarray]]:
    build_main_channel_cache(folder, force=force)
    meta = json.loads(metadata_path(folder).read_text(encoding="utf-8"))
    arrays_file = np.load(cache_path(folder))
    arrays = {name: arrays_file[name] for name in arrays_file.files}
    arrays_file.close()
    return meta, arrays
