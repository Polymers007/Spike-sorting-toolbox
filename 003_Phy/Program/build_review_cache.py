from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path

import numpy as np

from curation_common import portable_paths

from curation_common import WORKING_ROOT, now_text
from fast_review_common import CACHE_ROOT, base_signature, cache_dir, read_groups

def build_one(folder: Path, force: bool = False) -> dict:
    destination = cache_dir(folder)
    meta_path = destination / "metadata.json"
    signature = base_signature(folder)
    if not force and meta_path.is_file():
        try:
            old = json.loads(meta_path.read_text(encoding="utf-8"))
            if old.get("base_signature") == signature and (destination / "cluster_waveforms.npz").is_file():
                return {"status": "skipped", "folder": str(folder)}
        except (OSError, ValueError):
            pass
    required = [
        folder / "spike_clusters.npy", folder / "spike_times.npy", folder / "amplitudes.npy",
        folder / "_phy_spikes_subset.waveforms.npy", folder / "_phy_spikes_subset.spikes.npy",
        folder / "_phy_spikes_subset.channels.npy",
    ]
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing cache inputs: " + ", ".join(missing))
    assignments = np.load(required[0], mmap_mode="r").ravel()
    sample_waveforms = np.load(required[3], mmap_mode="r")
    sample_ids = np.load(required[4]).astype(np.int64)
    sample_channels = np.load(required[5]).astype(np.int32)
    sample_clusters = assignments[sample_ids]
    cluster_ids = np.unique(assignments).astype(np.int64)
    shape = (len(cluster_ids), sample_waveforms.shape[1], sample_waveforms.shape[2])
    medians = np.full(shape, np.nan, dtype=np.float32)
    channels = np.full((len(cluster_ids), sample_channels.shape[1]), -1, dtype=np.int32)
    sample_counts = np.zeros(len(cluster_ids), dtype=np.int32)
    for index, cluster_id in enumerate(cluster_ids):
        rows = np.flatnonzero(sample_clusters == cluster_id)
        sample_counts[index] = len(rows)
        if rows.size:
            medians[index] = np.median(sample_waveforms[rows], axis=0).astype(np.float32)
            channels[index] = sample_channels[rows[0]]
    labels = read_groups(folder)
    destination.mkdir(parents=True, exist_ok=True)
    tmp = destination / f"cluster_waveforms.npz.tmp.{os.getpid()}"
    with tmp.open("wb") as stream:
        np.savez_compressed(stream, cluster_ids=cluster_ids, waveforms=medians, channels=channels,
                            sample_counts=sample_counts)
    os.replace(tmp, destination / "cluster_waveforms.npz")
    metadata = {
        "schema_version": 1, "created": now_text(), "working_folder": str(folder),
        "base_signature": signature, "cluster_count": int(len(cluster_ids)),
        "spike_count": int(assignments.size), "labels": {str(k): v for k, v in labels.items()},
    }
    meta_tmp = meta_path.with_name(meta_path.name + f".tmp.{os.getpid()}")
    meta_tmp.write_text(json.dumps(portable_paths(metadata), ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(meta_tmp, meta_path)
    return {"status": "completed", "folder": str(folder), "clusters": len(cluster_ids)}

def folders() -> list[Path]:
    return sorted(path.parent for path in WORKING_ROOT.rglob("params.py") if len(path.parent.relative_to(WORKING_ROOT).parts) == 2)

def main() -> int:
    parser = argparse.ArgumentParser(description="Build lightweight waveform indexes for Fast Review.")
    parser.add_argument("--recording", help="DEVICE/RECORDING")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    selected = folders()
    if args.recording:
        key = args.recording.replace("\\", "/").lower()
        selected = [p for p in selected if f"{p.parent.name}/{p.name}".lower() == key]
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    failures = 0
    for index, folder in enumerate(selected, 1):
        try:
            result = build_one(folder, args.force)
            print(f"[{index}/{len(selected)}] {folder.parent.name}/{folder.name}: {result['status']}", flush=True)
        except Exception as error:
            failures += 1
            print(f"[{index}/{len(selected)}] {folder}: FAILED {error}", flush=True)
            traceback.print_exc()
    print(f"Finished: {len(selected)-failures} successful, {failures} failed")
    return 1 if failures else 0

if __name__ == "__main__":
    raise SystemExit(main())
