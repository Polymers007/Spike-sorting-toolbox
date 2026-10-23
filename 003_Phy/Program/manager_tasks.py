from __future__ import annotations

import argparse
import json
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

from curation_common import portable_paths

from build_review_cache import build_one as build_review_cache
from build_cached_refinement_cache import build_one as build_cached_refinement_cache
from cached_refinement_common import cache_dir as cached_cache_dir
from curation_common import OUTPUT_ROOT, WORKING_ROOT, now_text
from fast_review_common import base_signature, cache_dir
from main_channel_review import build_main_channel_cache, metadata_path
from batch_phy_prepare import prepare_one
from completed_reference import build_reference_model

def infer_phy_changes(folder: Path) -> dict:
    pass
    aligned_path = folder / "_phy_spikes_subset.waveforms.npy"
    original_path = folder / "_phy_spikes_subset.waveforms.unaligned.npy"
    spike_path = folder / "_phy_spikes_subset.spikes.npy"
    channel_path = folder / "_phy_spikes_subset.channels.npy"
    if not all(path.is_file() for path in (aligned_path, original_path, spike_path, channel_path)):
        return {}
    aligned = np.load(aligned_path, mmap_mode="r")
    original = np.load(original_path, mmap_mode="r")
    if aligned.shape != original.shape:
        return {}
    changed_rows = []
    for start in range(0, aligned.shape[0], 256):
        stop = min(start + 256, aligned.shape[0])
        changed = np.any(np.asarray(aligned[start:stop]) != np.asarray(original[start:stop]), axis=(1, 2))
        changed_rows.extend((np.flatnonzero(changed) + start).tolist())
    spike_ids = np.load(spike_path, mmap_mode="r").astype(np.int64)
    channels = np.load(channel_path, mmap_mode="r")
    assignments = np.load(folder / "spike_clusters.npy", mmap_mode="r").ravel()
    channel_counts: dict[int, int] = {}
    cluster_counts: dict[tuple[int, int], int] = {}
    for row in changed_rows:
        channel = int(channels[row, 0]); cluster = int(assignments[int(spike_ids[row])])
        channel_counts[channel] = channel_counts.get(channel, 0) + 1
        cluster_counts[(cluster, channel)] = cluster_counts.get((cluster, channel), 0) + 1
    return {
        "waveforms_shifted": len(changed_rows),
        "changed_main_channels": {str(key): value for key, value in sorted(channel_counts.items())},
        "changed_clusters": [{"cluster_id": cluster, "main_channel": channel, "waveforms_shifted": count}
                             for (cluster, channel), count in sorted(cluster_counts.items())],
        "change_summary_source": "comparison with preserved unaligned waveform cache",
    }

def folders(recordings: list[str] | None = None) -> list[Path]:
    found = sorted(
        (path.parent for path in WORKING_ROOT.rglob("params.py")
         if len(path.parent.relative_to(WORKING_ROOT).parts) == 2),
        key=lambda path: (path.parent.name.lower(), path.name.lower()),
    )
    if not recordings:
        return found
    wanted = {value.replace("\\", "/").lower() for value in recordings}
    return [path for path in found if f"{path.parent.name}/{path.name}".lower() in wanted]

def _main_cache_is_current(folder: Path) -> bool:
    path = metadata_path(folder)
    if not path.is_file():
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        
        return value.get("signature") is not None
    except (OSError, ValueError):
        return False

def _review_cache_is_current(folder: Path) -> bool:
    path = cache_dir(folder) / "metadata.json"
    waveform = cache_dir(folder) / "cluster_waveforms.npz"
    if not path.is_file() or not waveform.is_file():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("base_signature") == base_signature(folder)
    except (OSError, ValueError):
        return False

def _cached_refinement_cache_is_current(folder: Path) -> bool:
    root = cached_cache_dir(folder)
    path, arrays = root / "metadata.json", root / "cached_refinement.npz"
    if not path.is_file() or not arrays.is_file(): return False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value.get("schema_version") == 2 and value.get("base_signature") == base_signature(folder)
    except (OSError, ValueError):
        return False

def cache_one(folder: Path, force: bool) -> dict:
    main = build_main_channel_cache(folder, force=force)
    review = build_review_cache(folder, force=force)
    cached = build_cached_refinement_cache(folder, force=force)
    return {"main": main.get("status", "completed"), "review": review.get("status", "completed"),
            "cached_refinement": cached.get("status", "completed")}

def run_cache(selected: list[Path], force: bool, update_only: bool) -> int:
    failures = 0
    for index, folder in enumerate(selected, 1):
        key = f"{folder.parent.name}/{folder.name}"
        try:
            if update_only and _main_cache_is_current(folder) and _review_cache_is_current(folder) and _cached_refinement_cache_is_current(folder):
                
                outcome = cache_one(folder, force=False)
            else:
                outcome = cache_one(folder, force=force)
            print(f"[{index}/{len(selected)}] {key}: {outcome}", flush=True)
        except Exception as error:
            failures += 1
            print(f"[{index}/{len(selected)}] {key}: FAILED {error}", flush=True)
            traceback.print_exc()
    if not failures and not selected:
        return 0
    if not failures:
        try:
            summary = build_reference_model()
            print(f"Completed-reference model: {summary}", flush=True)
        except Exception as error:
            failures += 1
            print(f"Completed-reference model: FAILED {error}", flush=True)
            traceback.print_exc()
    return 1 if failures else 0

def run_align(selected: list[Path]) -> int:

    failures = 0
    results = []
    for index, folder in enumerate(selected, 1):
        key = f"{folder.parent.name}/{folder.name}"
        try:
            phy = prepare_one(folder, max_spikes=200, max_channels=16, max_shift=6,
                              minimum_correlation=0.30, align=True, force=False)
            main = build_main_channel_cache(folder, force=True)
            review = build_review_cache(folder, force=True)
            cached = build_cached_refinement_cache(folder, force=True)
            phy_alignment = {}
            marker = folder / "PHY_BATCH_PREPARED.json"
            if marker.is_file():
                try:
                    phy_alignment = json.loads(marker.read_text(encoding="utf-8")).get("alignment") or {}
                except (OSError, ValueError):
                    phy_alignment = {}
            if not phy_alignment.get("changed_main_channels"):
                recovered = infer_phy_changes(folder)
                if recovered:
                    phy_alignment = {**phy_alignment, **recovered}
            main_changes = [
                {"cluster_id": int(metric["cluster_id"]),
                 "main_channel": int(metric["phy_main_channel"]),
                 "shift_samples": int(metric.get("applied_shift", 0))}
                for metric in main.get("metrics", [])
                if int(metric.get("applied_shift", 0)) != 0 and "phy_main_channel" in metric
            ]
            phy_channels = {int(channel): int(count) for channel, count in phy_alignment.get("changed_main_channels", {}).items()}
            channels = sorted(set(phy_channels) | {item["main_channel"] for item in main_changes})
            results.append({
                "device": folder.parent.name, "recording": folder.name,
                "changed_channels": channels,
                "phy_waveforms_shifted": int(phy_alignment.get("waveforms_shifted", 0)),
                "phy_changed_channel_counts": phy_channels,
                "phy_changed_clusters": phy_alignment.get("changed_clusters", []),
                "main_channel_centering": main_changes,
            })
            print(f"[{index}/{len(selected)}] {key}: aligned; phy={phy.get('status')}; review={review.get('status')}; "
                  f"main={main.get('status')}; cached_refinement={cached.get('status')}", flush=True)
        except Exception as error:
            failures += 1
            results.append({"device": folder.parent.name, "recording": folder.name,
                            "error": f"{type(error).__name__}: {error}"})
            print(f"[{index}/{len(selected)}] {key}: FAILED {error}", flush=True)
            traceback.print_exc()
    report_root = OUTPUT_ROOT / "Batch_Logs" / "Unified_Manager" / "Align_Reports"
    report_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    changed = [item for item in results if item.get("changed_channels")]
    report = {
        "created": now_text(), "recordings_total": len(selected),
        "recordings_succeeded": len(selected) - failures, "recordings_failed": failures,
        "recordings_with_channel_changes": len(changed), "changed_recordings": changed,
        "failures": [item for item in results if item.get("error")],
        "definition": {
            "phy_waveforms_shifted": "individual cached waveforms shifted by cluster-median cross-correlation",
            "main_channel_centering": "derived main-channel median shifted to the display center",
        },
    }
    report_path = report_root / f"align_report_{stamp}.json"
    report_path.write_text(json.dumps(portable_paths(report), ensure_ascii=False, indent=2), encoding="utf-8")
    text_path = report_path.with_suffix(".txt")
    lines = [f"Align report: {report['created']}",
             f"Succeeded: {report['recordings_succeeded']}/{report['recordings_total']}; failed: {failures}",
             f"Recordings with changed channels: {len(changed)}", ""]
    for item in changed:
        lines.append(f"{item['device']}/{item['recording']}: channels {', '.join(map(str,item['changed_channels']))}; "
                     f"Phy waveforms shifted={item['phy_waveforms_shifted']}; centered clusters={len(item['main_channel_centering'])}")
    for item in report["failures"]:
        lines.append(f"FAILED {item['device']}/{item['recording']}: {item['error']}")
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"ALIGN_REPORT={report_path}", flush=True)
    return 1 if failures else 0

def run_apply(selected: list[Path]) -> int:
    from apply_review_actions import apply_one

    failures = 0
    for index, folder in enumerate(selected, 1):
        key = f"{folder.parent.name}/{folder.name}"
        try:
            checked = apply_one(folder, dry_run=True)
            if checked.get("status") == "skipped":
                print(f"[{index}/{len(selected)}] {key}: skipped (not Ready to Apply)", flush=True)
                continue
            result = apply_one(folder, dry_run=False)
            print(f"[{index}/{len(selected)}] {key}: {result.get('status')}", flush=True)
        except Exception as error:
            failures += 1
            print(f"[{index}/{len(selected)}] {key}: FAILED {error}", flush=True)
            traceback.print_exc()
    return 1 if failures else 0

def main() -> int:
    parser = argparse.ArgumentParser(description="Background jobs for the unified Curation Manager.")
    parser.add_argument("action", choices=("cache", "align", "apply"))
    parser.add_argument("--recording", action="append", default=[])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--update-only", action="store_true")
    args = parser.parse_args()
    selected = folders(args.recording)
    if args.recording and len(selected) != len(set(value.lower() for value in args.recording)):
        found = {f"{path.parent.name}/{path.name}".lower() for path in selected}
        missing = [value for value in args.recording if value.replace("\\", "/").lower() not in found]
        raise FileNotFoundError("Working_Curated result not found: " + ", ".join(missing))
    print(f"Action={args.action}; recordings={len(selected)}", flush=True)
    if args.action == "cache":
        return run_cache(selected, force=args.force, update_only=args.update_only)
    if args.action == "align":
        return run_align(selected)
    return run_apply(selected)

if __name__ == "__main__":
    raise SystemExit(main())
