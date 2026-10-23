from __future__ import annotations

import argparse
import json
import os
import shutil
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

from curation_common import resolve_project_path

from build_cached_refinement_cache import build_one as build_cached_cache
from build_review_cache import build_one as build_review_cache
from cached_refinement_common import (
    BACKUP_ROOT, QUEUE_ROOT, base_signature, journal_path, load_journal, replay,
    save_journal, write_groups,
)
from curation_common import (
    AUTHORITATIVE_FILES, AUTO_MARKER, MANUAL_CONFIRMED_MARKER, MANUAL_SAVED_MARKER,
    now_text, read_json, result_signature, validate_result, write_json,
)
from main_channel_review import ALGORITHM_VERSION, build_main_channel_cache
from refinement_common import REFINEMENT_DONE_MARKER, REFINEMENT_SESSION_MARKER

MUTABLE_FILES = AUTHORITATIVE_FILES + (
    AUTO_MARKER, MANUAL_SAVED_MARKER, MANUAL_CONFIRMED_MARKER,
    REFINEMENT_SESSION_MARKER, REFINEMENT_DONE_MARKER,
)

def edited_folders() -> list[Path]:
    answer: list[Path] = []
    if not QUEUE_ROOT.is_dir(): return answer
    for path in QUEUE_ROOT.rglob("cached_refinement_actions.json"):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if value.get("review_status") == "edited": answer.append(resolve_project_path(value["working_folder"]))
        except (OSError, ValueError, KeyError):
            continue
    return sorted(set(answer), key=lambda folder: (folder.parent.name.lower(), folder.name.lower()))

def plan(folder: Path) -> dict:
    journal = load_journal(folder)
    if journal.get("review_status") != "edited":
        raise RuntimeError("journal is not marked Edited")
    if journal.get("base_signature") != base_signature(folder):
        raise RuntimeError("result changed after cached review began; reopen this recording")
    assignments, labels = replay(folder, journal.get("operations", []))
    if assignments.size != np.load(folder / "spike_times.npy", mmap_mode="r").size:
        raise RuntimeError("planned spike assignment length does not match spike_times.npy")
    return {"folder": folder, "journal": journal, "assignments": assignments,
            "labels": labels}

def backup_one(folder: Path, batch_root: Path) -> Path:
    destination = batch_root / folder.parent.name / folder.name
    destination.mkdir(parents=True, exist_ok=False)
    for name in MUTABLE_FILES:
        source = folder / name
        if source.is_file(): shutil.copy2(source, destination / name)
    queue = journal_path(folder)
    if queue.is_file(): shutil.copy2(queue, destination / "cached_refinement_actions.before_apply.json")
    write_json(destination / "signature.before.json", result_signature(folder))
    return destination

def restore_one(folder: Path, backup: Path) -> None:
    for name in MUTABLE_FILES:
        current, saved = folder / name, backup / name
        if current.exists(): current.unlink()
        if saved.is_file(): shutil.copy2(saved, current)
    saved_journal = backup / "cached_refinement_actions.before_apply.json"
    if saved_journal.is_file():
        target = journal_path(folder); target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(saved_journal, target)

def apply_plan(item: dict, backup: Path) -> dict:
    folder: Path = item["folder"]
    assignments: np.ndarray = item["assignments"]
    labels: dict[int, str] = item["labels"]
    journal: dict = item["journal"]
    cluster_tmp = folder / f"spike_clusters.npy.cached_refinement.{os.getpid()}.tmp"
    group_tmp = folder / f"cluster_group.tsv.cached_refinement.{os.getpid()}.tmp"
    try:
        with cluster_tmp.open("wb") as stream: np.save(stream, assignments)
        write_groups(group_tmp, assignments, labels)
        
        staged = np.load(cluster_tmp)
        if staged.size != assignments.size or np.any(np.asarray(staged) != assignments):
            raise RuntimeError("staged spike cluster verification failed")
        os.replace(cluster_tmp, folder / "spike_clusters.npy")
        os.replace(group_tmp, folder / "cluster_group.tsv")
        cluster_info = folder / "cluster_info.tsv"
        if cluster_info.is_file(): cluster_info.unlink()

        main = build_main_channel_cache(folder, force=True)
        marker = read_json(folder / AUTO_MARKER)
        marker["stage"] = "cached_refinement_applied"
        marker["peak_alignment"] = {
            "status": "completed", "algorithm": ALGORITHM_VERSION,
            "completed_clusters": main.get("completed_clusters", 0),
            "needs_review_clusters": main.get("needs_review_clusters", []),
            "cache": main.get("cache"),
        }
        marker["cached_refinement_apply"] = {
            "applied": now_text(), "operation_count": len(journal.get("operations", [])),
            "backup": str(backup),
        }
        write_json(folder / AUTO_MARKER, marker)
        build_review_cache(folder, force=True)
        build_cached_cache(folder, force=True)
        validation = validate_result(folder)
        done = {
            "stage": "completed", "completed": now_text(), "source": "Cached Refinement",
            "operation_count": len(journal.get("operations", [])),
            "backup": str(backup),
            "signature": result_signature(folder),
        }
        write_json(folder / REFINEMENT_DONE_MARKER, done)
        session = folder / REFINEMENT_SESSION_MARKER
        if session.is_file(): session.unlink()
        journal["review_status"] = "applied"; journal["applied"] = now_text()
        journal["applied_operation_count"] = len(journal.get("operations", []))
        journal["applied_signature"] = base_signature(folder); journal["base_signature"] = base_signature(folder)
        journal["backup"] = str(backup); journal["operations"] = []; journal["redo_operations"] = []
        save_journal(folder, journal)
        return {"validation": validation}
    finally:
        for temporary in (cluster_tmp, group_tmp):
            if temporary.exists(): temporary.unlink()

def main() -> int:
    parser = argparse.ArgumentParser(description="Validate or atomically apply all Edited Cached Refinement journals.")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--recording", action="append", default=[])
    args = parser.parse_args()
    selected = edited_folders()
    if args.recording:
        wanted = {value.replace("\\", "/").lower() for value in args.recording}
        selected = [folder for folder in selected if f"{folder.parent.name}/{folder.name}".lower() in wanted]
    if not selected:
        print("No Edited Cached Refinement journals found.", flush=True); return 0

    plans = []
    try:
        for index, folder in enumerate(selected, 1):
            item = plan(folder); plans.append(item)
            print(f"PREFLIGHT [{index}/{len(selected)}] {folder.parent.name}/{folder.name}: valid; "
                  f"operations={len(item['journal'].get('operations', []))}", flush=True)
    except Exception as error:
        print(f"Preflight failed. No result modified: {error}", flush=True); traceback.print_exc(); return 1
    if not args.apply:
        print(f"Validated {len(plans)} Edited recording(s). No result modified.", flush=True); return 0

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    batch_root = BACKUP_ROOT / stamp; batch_root.mkdir(parents=True, exist_ok=False)
    write_json(batch_root / "batch.json", {"created": now_text(), "recordings": [str(item["folder"]) for item in plans]})
    backups: dict[Path, Path] = {}; applied: list[Path] = []
    try:
        for item in plans: backups[item["folder"]] = backup_one(item["folder"], batch_root)
        for index, item in enumerate(plans, 1):
            folder = item["folder"]; applied.append(folder)
            result = apply_plan(item, backups[folder])
            print(f"[{index}/{len(plans)}] {folder.parent.name}/{folder.name}: completed", flush=True)
        print(f"Applied {len(plans)} recording(s). Backup: {batch_root}", flush=True); return 0
    except Exception as error:
        print(f"Apply failed: {error}. Restoring all touched results...", flush=True); traceback.print_exc()
        for folder in reversed(applied):
            try:
                restore_one(folder, backups[folder])
                build_main_channel_cache(folder, force=True)
                build_review_cache(folder, force=True)
                build_cached_cache(folder, force=True)
            except Exception: traceback.print_exc()
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
