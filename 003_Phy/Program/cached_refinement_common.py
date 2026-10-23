from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from curation_common import portable_paths

from curation_common import OUTPUT_ROOT, now_text
from fast_review_common import base_signature, read_groups, replay, write_groups

CACHE_ROOT = OUTPUT_ROOT / "Cached_Refinement_Cache"
QUEUE_ROOT = OUTPUT_ROOT / "Cached_Refinement_Queue"
BACKUP_ROOT = OUTPUT_ROOT / "Snapshots" / "Cached_Refinement_Apply"
JOURNAL_NAME = "cached_refinement_actions.json"
SCHEMA_VERSION = 1

def cache_dir(folder: Path) -> Path:
    return CACHE_ROOT / folder.parent.name / folder.name

def journal_path(folder: Path) -> Path:
    return QUEUE_ROOT / folder.parent.name / folder.name / JOURNAL_NAME

def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(portable_paths(value), ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)

def new_journal(folder: Path, previous: dict | None = None) -> dict:
    value = {
        "schema_version": SCHEMA_VERSION,
        "device": folder.parent.name,
        "recording": folder.name,
        "working_folder": str(folder),
        "base_signature": base_signature(folder),
        "created": now_text(),
        "updated": now_text(),
        "review_status": "draft",
        "operations": [],
        "redo_operations": [],
    }
    if previous:
        value["previous_application"] = {
            key: previous.get(key) for key in ("applied", "backup", "applied_operation_count")
        }
    return value

def load_journal(folder: Path) -> dict:
    path = journal_path(folder)
    if not path.is_file():
        return new_journal(folder)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return new_journal(folder)
    current = base_signature(folder)
    if value.get("review_status") == "applied":
        return new_journal(folder, previous=value)
    if value.get("base_signature") != current:
        raise RuntimeError("The result changed after this cached review began. Archive the stale draft and reopen the recording.")
    value.setdefault("operations", [])
    value.setdefault("redo_operations", [])
    return value

def save_journal(folder: Path, value: dict) -> None:
    value["updated"] = now_text()
    atomic_json(journal_path(folder), value)

def journal_status(folder: Path) -> str:
    path = journal_path(folder)
    if not path.is_file():
        return ""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("base_signature") != base_signature(folder):
            return "stale"
        return str(value.get("review_status", ""))
    except (OSError, ValueError):
        return "error"

__all__ = [
    "BACKUP_ROOT", "CACHE_ROOT", "QUEUE_ROOT", "atomic_json", "base_signature",
    "cache_dir", "journal_path", "journal_status", "load_journal", "read_groups",
    "replay", "save_journal", "write_groups",
]
