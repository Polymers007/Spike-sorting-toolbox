from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

from curation_common import portable_paths
from phy.apps.template.gui import TemplateController
from phylib.io.model import load_model

PROGRAM_DIR = Path(__file__).resolve().parent
TOOL_DIR = PROGRAM_DIR.parent
PROJECT_ROOT = TOOL_DIR.parent
DEFAULT_SORTING_ROOT = PROJECT_ROOT / "004_Result" / "Phy" / "Working_Curated"
LOG_DIR = PROJECT_ROOT / "004_Result" / "Phy" / "Batch_Logs" / "Phy_Cache_Prepare"
LOCK_FILE = PROJECT_ROOT / "004_Result" / "Phy" / "batch_phy_prepare.lock"
MARKER_NAME = "PHY_BATCH_PREPARED.json"
CACHE_WAVEFORMS = "_phy_spikes_subset.waveforms.npy"
CACHE_SPIKES = "_phy_spikes_subset.spikes.npy"
CACHE_CHANNELS = "_phy_spikes_subset.channels.npy"
BACKUP_WAVEFORMS = "_phy_spikes_subset.waveforms.unaligned.npy"
PROTECTED_FILES = (
    "spike_times.npy",
    "spike_clusters.npy",
    "spike_templates.npy",
    "templates.npy",
    "params.py",
)

def natural_key(path: Path):
    return [int(piece) if piece.isdigit() else piece.casefold()
            for piece in re.split(r"(\d+)", str(path))]

def discover_results(root: Path, device: str | None = None) -> list[Path]:
    results = []
    for params in root.rglob("params.py"):
        folder = params.parent
        required = ("spike_times.npy", "spike_clusters.npy", "spike_templates.npy", "templates.npy")
        if not all((folder / name).is_file() for name in required):
            continue
        relative = folder.relative_to(root)
        if device and (not relative.parts or relative.parts[0].casefold() != device.casefold()):
            continue
        results.append(folder)
    return sorted(results, key=lambda path: natural_key(path.relative_to(root)))

def file_state(folder: Path) -> dict[str, tuple[int, int]]:
    state = {}
    for name in PROTECTED_FILES:
        path = folder / name
        stat = path.stat()
        state[name] = (stat.st_size, stat.st_mtime_ns)
    return state

def expected_cache_bytes(folder: Path, max_spikes: int, max_channels: int) -> tuple[int, int]:
    spike_templates = np.load(folder / "spike_templates.npy", mmap_mode="r")
    templates = np.load(folder / "templates.npy", mmap_mode="r")
    counts = np.bincount(np.asarray(spike_templates, dtype=np.int64))
    selected_upper = int(np.minimum(counts, max_spikes).sum())
    n_samples = int(templates.shape[1])
    n_channels = min(max_channels, int(templates.shape[2]))
    
    one_copy = selected_upper * (n_samples * n_channels * 8 + 8 + n_channels * 4)
    return one_copy, selected_upper

def correlation_score(trace: np.ndarray, reference: np.ndarray, shift: int) -> float:
    if shift > 0:
        a, b = trace[:-shift], reference[shift:]
    elif shift < 0:
        amount = -shift
        a, b = trace[amount:], reference[:-amount]
    else:
        a, b = trace, reference
    a = a.astype(np.float64, copy=False)
    b = b.astype(np.float64, copy=False)
    a = a - np.mean(a)
    b = b - np.mean(b)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator > 0 else -1.0

def shift_waveform(waveform: np.ndarray, shift: int) -> np.ndarray:
    if shift == 0:
        return waveform
    shifted = np.empty_like(waveform)
    edge = max(1, min(5, waveform.shape[0] // 8))
    baseline = np.median(
        np.concatenate((waveform[:edge], waveform[-edge:]), axis=0), axis=0
    )
    if shift > 0:
        shifted[:shift] = baseline
        shifted[shift:] = waveform[:-shift]
    else:
        amount = -shift
        shifted[:-amount] = waveform[amount:]
        shifted[-amount:] = baseline
    return shifted

def align_waveform_cache(
    folder: Path,
    max_shift: int,
    minimum_correlation: float,
    keep_backup: bool = True,
) -> dict:
    waveform_path = folder / CACHE_WAVEFORMS
    spike_path = folder / CACHE_SPIKES
    channel_path = folder / CACHE_CHANNELS
    backup_path = folder / BACKUP_WAVEFORMS
    partial_path = folder / "_phy_spikes_subset.waveforms.aligned.partial.npy"

    if not backup_path.exists():
        shutil.copy2(waveform_path, backup_path)
    source_path = backup_path if keep_backup else waveform_path
    waveforms = np.load(source_path, mmap_mode="r")
    spike_ids = np.load(spike_path, mmap_mode="r")
    spike_channels = np.load(channel_path, mmap_mode="r")
    all_clusters = np.load(folder / "spike_clusters.npy", mmap_mode="r")
    selected_clusters = np.asarray(all_clusters[np.asarray(spike_ids, dtype=np.int64)])

    if waveforms.ndim != 3 or waveforms.shape[0] != spike_ids.shape[0]:
        raise RuntimeError("Phy waveform cache shapes are inconsistent.")
    if spike_channels.shape[0] != waveforms.shape[0]:
        raise RuntimeError("Phy waveform channel cache is inconsistent.")

    if partial_path.exists():
        partial_path.unlink()
    aligned = np.lib.format.open_memmap(
        partial_path, mode="w+", dtype=waveforms.dtype, shape=waveforms.shape
    )
    aligned[:] = waveforms[:]

    histogram: Counter[int] = Counter()
    changed_channels: Counter[int] = Counter()
    changed_clusters: list[dict] = []
    rejected = 0
    missing_main_channel = 0
    shifted_count = 0
    for cluster_id in np.unique(selected_clusters):
        indices = np.flatnonzero(selected_clusters == cluster_id)
        if indices.size < 2:
            histogram[0] += int(indices.size)
            continue
        first_channels = np.asarray(spike_channels[indices, 0], dtype=np.int64)
        main_channel = Counter(first_channels.tolist()).most_common(1)[0][0]
        traces = []
        valid_indices = []
        channel_columns = []
        for index in indices:
            columns = np.flatnonzero(np.asarray(spike_channels[index]) == main_channel)
            if columns.size == 0:
                missing_main_channel += 1
                histogram[0] += 1
                continue
            column = int(columns[0])
            traces.append(np.asarray(waveforms[index, :, column]))
            valid_indices.append(int(index))
            channel_columns.append(column)
        if len(traces) < 2:
            continue
        trace_matrix = np.asarray(traces)
        reference = np.median(trace_matrix, axis=0)
        cluster_shifted = 0
        for index, trace in zip(valid_indices, trace_matrix):
            scores = [(correlation_score(trace, reference, shift), shift)
                      for shift in range(-max_shift, max_shift + 1)]
            score, best_shift = max(scores, key=lambda pair: (pair[0], -abs(pair[1])))
            if score < minimum_correlation:
                best_shift = 0
                rejected += 1
            if best_shift:
                aligned[index] = shift_waveform(np.asarray(waveforms[index]), best_shift)
                shifted_count += 1
                cluster_shifted += 1
                changed_channels[int(main_channel)] += 1
            histogram[int(best_shift)] += 1
        if cluster_shifted:
            changed_clusters.append({
                "cluster_id": int(cluster_id),
                "main_channel": int(main_channel),
                "waveforms_shifted": int(cluster_shifted),
            })

    aligned.flush()
    del aligned
    del waveforms
    os.replace(partial_path, waveform_path)
    return {
        "algorithm": "cluster_median_primary_channel_integer_xcorr_v1",
        "max_shift_samples": max_shift,
        "minimum_correlation": minimum_correlation,
        "waveforms_total": int(spike_ids.shape[0]),
        "waveforms_shifted": shifted_count,
        "waveforms_rejected_low_correlation": rejected,
        "waveforms_missing_cluster_main_channel": missing_main_channel,
        "shift_histogram": {str(key): histogram[key] for key in sorted(histogram)},
        "changed_main_channels": {str(key): int(changed_channels[key]) for key in sorted(changed_channels)},
        "changed_clusters": changed_clusters,
        "unaligned_backup": str(backup_path) if keep_backup else None,
    }

def warm_phy_controller(folder: Path) -> dict:
    pass
    model = load_model(folder / "params.py")
    controller = TemplateController(
        model=model,
        dir_path=folder,
        enable_threading=False,
    )
    try:
        cluster_ids = list(controller.supervisor.clustering.cluster_ids)
        for cluster_id in cluster_ids:
            controller.get_best_channel(cluster_id)
            controller.get_best_channels(cluster_id)
            controller.get_mean_firing_rate(cluster_id)
            controller.get_probe_depth(cluster_id)
            controller.get_template_for_cluster(cluster_id)
        controller.context.save_memcache()
        cache_dir = folder / ".phy"
        cache_bytes = sum(path.stat().st_size for path in cache_dir.rglob("*") if path.is_file())
        return {
            "cluster_metrics_warmed": len(cluster_ids),
            "phy_context_directory": str(cache_dir),
            "phy_context_bytes": cache_bytes,
        }
    finally:
        controller.close()

def prepare_one(
    folder: Path,
    max_spikes: int,
    max_channels: int,
    max_shift: int,
    minimum_correlation: float,
    align: bool,
    force: bool,
) -> dict:
    marker_path = folder / MARKER_NAME
    cache_paths = [folder / CACHE_WAVEFORMS, folder / CACHE_SPIKES, folder / CACHE_CHANNELS]
    protected_before = file_state(folder)
    input_state = {name: list(state) for name, state in protected_before.items()}
    if marker_path.is_file() and all(path.is_file() for path in cache_paths) and not force:
        saved = json.loads(marker_path.read_text(encoding="utf-8"))
        if (
            saved.get("format_version") == 2
            and saved.get("controller_warmup")
            and saved.get("max_spikes_per_template") == max_spikes
            and saved.get("max_channels") == max_channels
            and bool(saved.get("alignment")) == align
            and saved.get("input_state") == input_state
        ):
            return {"status": "skipped", "message": "already prepared"}

    if force:
        for path in cache_paths + [folder / BACKUP_WAVEFORMS, marker_path]:
            if path.exists():
                path.unlink()

    if not all(path.is_file() for path in cache_paths):
        model = load_model(folder / "params.py")
        try:
            model.save_spikes_subset_waveforms(
                max_n_spikes_per_template=max_spikes,
                max_n_channels=max_channels,
            )
        finally:
            model.close()
    if not all(path.is_file() for path in cache_paths):
        raise RuntimeError("Phy did not create all three waveform cache files.")

    alignment = None
    if align:
        alignment = align_waveform_cache(
            folder,
            max_shift=max_shift,
            minimum_correlation=minimum_correlation,
            keep_backup=True,
        )

    controller_warmup = warm_phy_controller(folder)

    if file_state(folder) != protected_before:
        raise RuntimeError("A protected Kilosort result file changed during preparation.")

    marker = {
        "format_version": 2,
        "completed": datetime.now().isoformat(timespec="seconds"),
        "result_folder": str(folder),
        "max_spikes_per_template": max_spikes,
        "max_channels": max_channels,
        "alignment": alignment,
        "input_state": input_state,
        "controller_warmup": controller_warmup,
        "protected_kilosort_files_unchanged": True,
        "cache_files": {
            path.name: path.stat().st_size for path in cache_paths
        },
    }
    marker_path.write_text(json.dumps(portable_paths(marker), ensure_ascii=False, indent=2), encoding="utf-8")
    return {"status": "completed", "message": json.dumps(alignment or {})}

def write_status(path: Path, row: dict) -> None:
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("time", "status", "result_folder", "elapsed_seconds", "message")
        )
        if new_file:
            writer.writeheader()
        writer.writerow(row)

def acquire_lock() -> int:
    try:
        return os.open(LOCK_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError(f"Another Phy preparation batch may already be running: {LOCK_FILE}") from error

def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare and align Phy waveform caches in batch.")
    parser.add_argument("--sorting-root", type=Path, default=DEFAULT_SORTING_ROOT)
    parser.add_argument("--device")
    parser.add_argument("--max-spikes", type=int, default=200)
    parser.add_argument("--max-channels", type=int, default=16)
    parser.add_argument("--max-shift", type=int, default=6)
    parser.add_argument("--minimum-correlation", type=float, default=0.30)
    parser.add_argument("--no-align", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--pilot", action="store_true", help="Prepare Au1/DCH1/ICH1 w02 only.")
    args = parser.parse_args()

    root = args.sorting_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Sorting root not found: {root}")
    results = discover_results(root, args.device)
    if args.pilot:
        pilot_names = {"Au1/Au1_w02", "DCH1/DCH1_w02", "ICH1/ICH1_w02"}
        results = [folder for folder in results
                   if folder.relative_to(root).as_posix() in pilot_names]
    print(f"Found {len(results)} complete Phy result folders under {root}")
    estimates = [expected_cache_bytes(folder, args.max_spikes, args.max_channels) for folder in results]
    one_copy = sum(item[0] for item in estimates)
    multiplier = 2 if not args.no_align else 1
    print(f"Estimated cache upper bound: {one_copy * multiplier / 2**30:.2f} GiB")
    print(f"Settings: max_spikes={args.max_spikes}, max_channels={args.max_channels}, "
          f"align={not args.no_align}, max_shift={args.max_shift}")
    if args.dry_run:
        for folder, (_, selected) in zip(results, estimates):
            print(f"PLAN {folder.relative_to(root)} (<= {selected} waveforms)")
        return 0

    free = shutil.disk_usage(root).free
    required = one_copy * multiplier
    if required + 5 * 2**30 > free:
        raise RuntimeError(
            f"Insufficient disk space: need about {required / 2**30:.2f} GiB plus 5 GiB reserve, "
            f"available {free / 2**30:.2f} GiB."
        )

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    status_path = LOG_DIR / f"phy_prepare_{datetime.now():%Y%m%d_%H%M%S}.csv"
    lock_fd = acquire_lock()
    failures = 0
    try:
        for index, folder in enumerate(results, start=1):
            started = time.time()
            print(f"[{index}/{len(results)}] {folder.relative_to(root)}", flush=True)
            try:
                outcome = prepare_one(
                    folder,
                    args.max_spikes,
                    args.max_channels,
                    args.max_shift,
                    args.minimum_correlation,
                    not args.no_align,
                    args.force,
                )
                print(f"  {outcome['status'].upper()}", flush=True)
            except KeyboardInterrupt:
                print("Interrupted; completed caches remain usable.")
                return 130
            except Exception as error:
                failures += 1
                outcome = {"status": "failed", "message": f"{type(error).__name__}: {error}"}
                print(f"  FAILED: {outcome['message']}", flush=True)
            write_status(status_path, {
                "time": datetime.now().isoformat(timespec="seconds"),
                "status": outcome["status"],
                "result_folder": str(folder),
                "elapsed_seconds": round(time.time() - started, 3),
                "message": outcome["message"],
            })
    finally:
        os.close(lock_fd)
        if LOCK_FILE.exists():
            LOCK_FILE.unlink()
    print(f"Finished with {failures} failures. Status: {status_path}")
    return 1 if failures else 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
