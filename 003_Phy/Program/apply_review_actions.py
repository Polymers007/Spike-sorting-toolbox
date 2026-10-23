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

from build_review_cache import build_one
from curation_common import (
    AUTO_MARKER, MANUAL_CONFIRMED_MARKER, MANUAL_SAVED_MARKER,
    WORKING_ROOT, now_text, read_json, validate_result, write_json,
)
from fast_review_common import (
    APPLY_BACKUP_ROOT, QUEUE_ROOT, base_signature, load_journal, replay,
    journal_path, read_groups, save_journal, write_groups,
)
from main_channel_review import ALGORITHM_VERSION, build_main_channel_cache

def ready_folders() -> list[Path]:
    answer=[]
    for path in QUEUE_ROOT.rglob("review_actions.json") if QUEUE_ROOT.is_dir() else []:
        try:
            value=json.loads(path.read_text(encoding="utf-8"))
            if value.get("review_status")=="ready_to_apply": answer.append(resolve_project_path(value["working_folder"]))
        except (OSError,ValueError,KeyError):
            continue
    return sorted(answer)

def apply_one(folder: Path, dry_run: bool) -> dict:
    journal=load_journal(folder)
    if journal.get("review_status")!="ready_to_apply": return {"status":"skipped","folder":str(folder)}
    if journal.get("base_signature")!=base_signature(folder):
        raise RuntimeError("working result changed after review began; reload and review again")
    assignments,labels=replay(folder,journal["operations"])
    if dry_run:
        return {"status":"validated","folder":str(folder),"operations":len(journal["operations"]),
                "clusters":int(np.unique(assignments).size)}
    device,recording=folder.parent.name,folder.name
    stamp=datetime.now().strftime("%Y%m%d_%H%M%S")
    backup=APPLY_BACKUP_ROOT/device/recording/stamp; backup.mkdir(parents=True,exist_ok=False)
    mutable=["spike_clusters.npy","cluster_group.tsv","cluster_info.tsv",AUTO_MARKER,MANUAL_SAVED_MARKER,MANUAL_CONFIRMED_MARKER]
    for name in mutable:
        source=folder/name
        if source.is_file(): shutil.copy2(source,backup/name)
    review_log=journal_path(folder)
    if review_log.is_file(): shutil.copy2(review_log,backup/"review_actions_applied.json")
    try:
        cluster_tmp=folder/"spike_clusters.npy.fast_review.tmp"
        with cluster_tmp.open("wb") as stream: np.save(stream,assignments)
        group_tmp=folder/"cluster_group.tsv.fast_review.tmp"; write_groups(group_tmp,assignments,labels)
        os.replace(cluster_tmp,folder/"spike_clusters.npy"); os.replace(group_tmp,folder/"cluster_group.tsv")
        info=folder/"cluster_info.tsv"
        if info.is_file(): info.unlink()
        cache_result=build_main_channel_cache(folder,force=True)
        alignment={
            "status":"completed", "algorithm":ALGORITHM_VERSION,
            "completed_clusters":cache_result.get("completed_clusters",0),
            "needs_review_clusters":cache_result.get("needs_review_clusters",[]),
            "cache":cache_result.get("cache"),
        }
        marker=read_json(folder/AUTO_MARKER); marker["peak_alignment"]=alignment; marker["peak_alignment_updated"]=now_text(); write_json(folder/AUTO_MARKER,marker)
        marker["stage"]="fast_review_applied"
        marker["fast_review_apply"]={"applied":now_text(),"operation_count":len(journal["operations"]),
                                      "backup":str(backup)}
        write_json(folder/AUTO_MARKER,marker)
        build_one(folder,force=True)
        validation=validate_result(folder)
        operation_count=len(journal["operations"])
        journal["review_status"]="applied"; journal["applied"]=now_text(); journal["applied_signature"]=base_signature(folder)
        journal["applied_operation_count"]=operation_count; journal["backup"]=str(backup); journal["operations"]=[]
        journal["base_signature"]=base_signature(folder); save_journal(folder,journal)
        return {"status":"completed","folder":str(folder),"alignment":alignment["status"],"backup":str(backup),
                "validation":validation}
    except Exception:
        for name in mutable:
            current=folder/name; saved=backup/name
            if current.exists(): current.unlink()
            if saved.is_file(): shutil.copy2(saved,current)
        raise

def main() -> int:
    parser=argparse.ArgumentParser(description="Validate or apply non-destructive Fast Review journals.")
    parser.add_argument("--apply",action="store_true")
    parser.add_argument("--recording",help="DEVICE/RECORDING")
    args=parser.parse_args(); selected=ready_folders()
    if args.recording:
        key=args.recording.replace("\\","/").lower(); selected=[p for p in selected if f"{p.parent.name}/{p.name}".lower()==key]
    print(f"Ready journals: {len(selected)}; mode: {'APPLY' if args.apply else 'VALIDATE ONLY'}")
    failures=0
    for index,folder in enumerate(selected,1):
        try:
            result=apply_one(folder,not args.apply); print(f"[{index}/{len(selected)}] {folder.parent.name}/{folder.name}: {result['status']}",flush=True)
        except Exception as error:
            failures+=1; print(f"[{index}/{len(selected)}] {folder}: FAILED {error}",flush=True); traceback.print_exc()
    return 1 if failures else 0

if __name__=="__main__": raise SystemExit(main())
