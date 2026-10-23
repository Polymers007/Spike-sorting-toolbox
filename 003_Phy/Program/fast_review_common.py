from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np

from curation_common import portable_paths

from curation_common import OUTPUT_ROOT, WORKING_ROOT, now_text, result_signature

CACHE_ROOT = OUTPUT_ROOT / "Review_Cache"
QUEUE_ROOT = OUTPUT_ROOT / "Review_Queue"
APPLY_BACKUP_ROOT = OUTPUT_ROOT / "Snapshots" / "Fast_Review_Apply"
JOURNAL_NAME = "review_actions.json"

def recording_key(folder: Path) -> tuple[str, str]:
    return folder.parent.name, folder.name

def cache_dir(folder: Path) -> Path:
    device, recording = recording_key(folder)
    return CACHE_ROOT / device / recording

def journal_path(folder: Path) -> Path:
    device, recording = recording_key(folder)
    return QUEUE_ROOT / device / recording / JOURNAL_NAME

def read_groups(folder: Path) -> dict[int, str]:
    path = folder / "cluster_group.tsv"
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return {int(row["cluster_id"]): row.get("group", "mua") for row in csv.DictReader(stream, delimiter="\t")}

def base_signature(folder: Path) -> dict:
    signature = result_signature(folder)
    return {name: signature[name] for name in ("spike_clusters.npy", "cluster_group.tsv")}

def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(portable_paths(value), ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)

def load_journal(folder: Path) -> dict:
    path = journal_path(folder)
    if path.is_file():
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                current = base_signature(folder)

                if value.get("review_status") == "applied":
                    return {
                        "schema_version": 2,
                        "device": folder.parent.name,
                        "recording": folder.name,
                        "working_folder": str(folder),
                        "base_signature": current,
                        "created": now_text(),
                        "updated": now_text(),
                        "review_status": "applied",
                        "operations": [],
                        "previous_application": {
                            "applied": value.get("applied"),
                            "backup": value.get("backup"),
                            "operation_count": value.get("applied_operation_count", len(value.get("operations", []))),
                        },
                    }
                if value.get("base_signature") != current:
                    raise RuntimeError(
                        "This Fast Review draft was created from an older result. "
                        "Archive or discard the stale draft before continuing."
                    )
                return value
        except (OSError, ValueError):
            pass
    device, recording = recording_key(folder)
    return {
        "schema_version": 2,
        "device": device,
        "recording": recording,
        "working_folder": str(folder),
        "base_signature": base_signature(folder),
        "created": now_text(),
        "updated": now_text(),
        "review_status": "draft",
        "operations": [],
    }

def save_journal(folder: Path, journal: dict) -> None:
    journal["updated"] = now_text()
    atomic_json(journal_path(folder), journal)

def replay(folder: Path, operations: list[dict]) -> tuple[np.ndarray, dict[int, str]]:
    assignments = np.load(folder / "spike_clusters.npy").ravel().astype(np.int64, copy=True)
    labels = read_groups(folder)
    for op in operations:
        kind = op.get("type")
        if kind == "label":
            labels[int(op["cluster_id"])] = op["group"]
        elif kind == "merge":
            sources = [int(value) for value in op["sources"]]
            target = int(op["target"])
            assignments[np.isin(assignments, sources)] = target
            for source in sources:
                if source != target:
                    labels.pop(source, None)
            labels[target] = op.get("group", "good")
        elif kind == "split":
            source = int(op["source"])
            target = int(op["new_cluster_id"])
            spike_ids = np.asarray(op["selected_spike_ids"], dtype=np.int64)
            if spike_ids.size and (spike_ids.min() < 0 or spike_ids.max() >= assignments.size):
                raise ValueError("split contains an out-of-range spike id")
            if spike_ids.size and np.any(assignments[spike_ids] != source):
                raise ValueError(f"split source changed before replay: {source}")
            assignments[spike_ids] = target
            labels[source] = op.get("remainder_group", labels.get(source, "mua"))
            labels[target] = op.get("selected_group", "good")
        else:
            raise ValueError(f"unknown review operation: {kind}")
    return assignments, labels

def write_groups(path: Path, assignments: np.ndarray, labels: dict[int, str]) -> None:
    lines = ["cluster_id\tgroup"]
    for cluster_id in sorted(int(value) for value in np.unique(assignments)):
        lines.append(f"{cluster_id}\t{labels.get(cluster_id, 'mua')}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
