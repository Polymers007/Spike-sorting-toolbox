from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path

import numpy as np

from curation_common import portable_paths
from phylib.stats import correlograms

from cached_refinement_common import CACHE_ROOT, base_signature, cache_dir, read_groups
from curation_common import WORKING_ROOT, now_text, parse_params
from main_channel_review import load_main_channel_cache, _waveform_scale

MAX_CORRELOGRAM_SPIKES = 50_000
BIN_SIZE = 0.001
WINDOW_SIZE = 0.05

def _cosine_shift(a: np.ndarray, b: np.ndarray, max_shift: int = 3) -> float:
    a = np.nan_to_num(np.asarray(a, dtype=np.float64))
    b = np.nan_to_num(np.asarray(b, dtype=np.float64))
    best = -1.0
    for shift in range(-max_shift, max_shift + 1):
        if shift < 0: left, right = a[-shift:], b[:shift]
        elif shift > 0: left, right = a[:-shift], b[shift:]
        else: left, right = a, b
        x, y = left.ravel(), right.ravel()
        denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
        score = float(np.dot(x, y) / denominator) if denominator else -1.0
        best = max(best, score)
    return best

def build_one(folder: Path, force: bool = False) -> dict:
    folder = Path(folder)
    destination = cache_dir(folder)
    arrays_path = destination / "cached_refinement.npz"
    meta_path = destination / "metadata.json"
    signature = base_signature(folder)
    if not force and arrays_path.is_file() and meta_path.is_file():
        try:
            old = json.loads(meta_path.read_text(encoding="utf-8"))
            if (old.get("schema_version") == 2 and old.get("base_signature") == signature
                    and old.get("scale_uV_per_bit") == _waveform_scale(folder)[0]):
                return {"status": "skipped", "folder": str(folder), "cache": str(arrays_path)}
        except (OSError, ValueError):
            pass

    main_meta, main = load_main_channel_cache(folder, force=False)
    assignments = np.load(folder / "spike_clusters.npy", mmap_mode="r").ravel()
    spike_samples = np.load(folder / "spike_times.npy", mmap_mode="r").ravel()
    rate = float(parse_params(folder / "params.py").get("sample_rate", 30_000.0))
    cluster_ids = np.asarray(main["cluster_ids"], dtype=np.int64)
    medians = np.asarray(main["medians"], dtype=np.float32)
    n_clusters = len(cluster_ids)

    flattened = np.nan_to_num(medians.reshape(n_clusters, -1).astype(np.float32))
    norms = np.linalg.norm(flattened, axis=1, keepdims=True)
    normalized = np.divide(flattened, norms, out=np.zeros_like(flattened), where=norms > 0)
    similarity = np.clip(normalized @ normalized.T, -1.0, 1.0).astype(np.float32)
    np.fill_diagonal(similarity, 1.0)

    n_bins = 2 * int(0.5 * WINDOW_SIZE / BIN_SIZE) + 1
    autocorrelograms = np.zeros((n_clusters, n_bins), dtype=np.int32)
    full_counts = np.zeros(n_clusters, dtype=np.int64)
    for index, cluster_id in enumerate(cluster_ids):
        spike_ids = np.flatnonzero(assignments == int(cluster_id))
        full_counts[index] = len(spike_ids)
        if len(spike_ids) < 2:
            continue
        if len(spike_ids) > MAX_CORRELOGRAM_SPIKES:
            spike_ids = spike_ids[np.linspace(0, len(spike_ids) - 1, MAX_CORRELOGRAM_SPIKES).astype(np.int64)]
        times = np.asarray(spike_samples[spike_ids], dtype=np.float64) / rate
        labels = np.zeros(len(times), dtype=np.int64)
        autocorrelograms[index] = correlograms(
            times, labels, cluster_ids=[0], sample_rate=rate,
            bin_size=BIN_SIZE, window_size=WINDOW_SIZE,
        )[0, 0]

    destination.mkdir(parents=True, exist_ok=True)
    temporary = arrays_path.with_name(arrays_path.name + f".tmp.{os.getpid()}")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream, cluster_ids=cluster_ids, medians=medians,
            main_channels=np.asarray(main["main_channels"], dtype=np.int32),
            channel_positions=np.asarray(main["channel_positions"], dtype=np.float32),
            sample_counts=np.asarray(main["sample_counts"], dtype=np.int32),
            full_counts=full_counts, similarity=similarity,
            autocorrelograms=autocorrelograms,
            correlogram_lags_ms=np.linspace(-WINDOW_SIZE * 500, WINDOW_SIZE * 500, n_bins).astype(np.float32),
        )
    os.replace(temporary, arrays_path)
    groups = read_groups(folder)
    meta = {
        "schema_version": 2, "created": now_text(), "working_folder": str(folder),
        "base_signature": signature, "cluster_count": n_clusters,
        "labels": {str(key): value for key, value in groups.items()},
        "main_cache": main_meta.get("cache"),
        "sample_rate_hz": float(main_meta.get("sample_rate_hz", rate)),
        "scale_uV_per_bit": float(main_meta.get("scale_uV_per_bit", 1.0)),
        "correlogram": {"max_spikes": MAX_CORRELOGRAM_SPIKES, "bin_size_s": BIN_SIZE, "window_size_s": WINDOW_SIZE},
    }
    temporary_meta = meta_path.with_name(meta_path.name + f".tmp.{os.getpid()}")
    temporary_meta.write_text(json.dumps(portable_paths(meta), ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary_meta, meta_path)
    return {"status": "completed", "folder": str(folder), "clusters": n_clusters, "cache": str(arrays_path)}

def folders() -> list[Path]:
    return sorted(path.parent for path in WORKING_ROOT.rglob("params.py")
                  if len(path.parent.relative_to(WORKING_ROOT).parts) == 2)

def main() -> int:
    parser = argparse.ArgumentParser(description="Build cache packages for Cached Refinement.")
    parser.add_argument("--recording", action="append", default=[])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(); selected = folders()
    if args.recording:
        wanted = {value.replace("\\", "/").lower() for value in args.recording}
        selected = [folder for folder in selected if f"{folder.parent.name}/{folder.name}".lower() in wanted]
    failures = 0
    for index, folder in enumerate(selected, 1):
        try:
            result = build_one(folder, force=args.force)
            print(f"[{index}/{len(selected)}] {folder.parent.name}/{folder.name}: {result['status']}", flush=True)
        except Exception as error:
            failures += 1
            print(f"[{index}/{len(selected)}] {folder.parent.name}/{folder.name}: FAILED {error}", flush=True)
            traceback.print_exc()
    return 1 if failures else 0

if __name__ == "__main__":
    raise SystemExit(main())
