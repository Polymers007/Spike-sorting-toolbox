from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np

PROGRAM_DIR = Path(__file__).resolve().parent
TOOL_DIR = PROGRAM_DIR.parent
PROJECT_ROOT = TOOL_DIR.parent
SORTING_ROOT = PROJECT_ROOT / "004_Result" / "Kilosort4"
OUTPUT_ROOT = PROJECT_ROOT / "004_Result" / "Phy"
WORKING_ROOT = OUTPUT_ROOT / "Working_Curated"

AUTO_ROOT = WORKING_ROOT
MANUAL_ROOT = WORKING_ROOT
HISTORY_ROOT = OUTPUT_ROOT / "Split_History"
SNAPSHOT_ROOT = OUTPUT_ROOT / "Snapshots"
AUTO_BASELINE_ROOT = SNAPSHOT_ROOT / "Auto_Baseline"
BACKUP_ROOT = SNAPSHOT_ROOT / "Manual_Backups"
LOG_ROOT = OUTPUT_ROOT / "Batch_Logs"

AUTO_MARKER = "AUTO_CURATION_DONE.json"
MANUAL_SAVED_MARKER = "MANUAL_CURATION_SAVED.json"
MANUAL_CONFIRMED_MARKER = "MANUAL_CURATION_CONFIRMED.json"
SESSION_MARKER = "CURATION_SESSION.json"

AUTHORITATIVE_FILES = (
    "spike_clusters.npy",
    "cluster_group.tsv",
    "cluster_info.tsv",
)

def project_relative_path(value: str | Path) -> str:
    return Path(os.path.relpath(Path(value).resolve(), PROJECT_ROOT)).as_posix()

def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()

def portable_paths(value):
    if isinstance(value, Path):
        return project_relative_path(value)
    if isinstance(value, str) and Path(value).is_absolute():
        return project_relative_path(value)
    if isinstance(value, dict):
        return {key: portable_paths(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [portable_paths(item) for item in value]
    return value

def now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")

def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)

def write_json(path: Path, value: dict) -> None:
    def json_default(item):

        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(f"Object of type {type(item).__name__} is not JSON serializable")

    atomic_write_text(path, json.dumps(portable_paths(value), ensure_ascii=False, indent=2, default=json_default))

def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}

def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def result_signature(folder: Path) -> dict:
    return {
        name: {
            "sha256": sha256(folder / name),
            "size": (folder / name).stat().st_size if (folder / name).is_file() else None,
        }
        for name in AUTHORITATIVE_FILES
    }

def recording_key(folder: Path, root: Path = SORTING_ROOT) -> tuple[str, str]:
    relative = folder.resolve().relative_to(root.resolve())
    if len(relative.parts) < 2:
        raise ValueError(f"Unexpected result path: {folder}")
    return relative.parts[-2], relative.parts[-1]

def result_at(root: Path, device: str, recording: str) -> Path:
    return root / device / recording

def discover_results(root: Path = SORTING_ROOT) -> list[Path]:
    if not root.is_dir():
        return []
    results = []
    for params in root.rglob("params.py"):
        folder = params.parent
        if all((folder / name).is_file() for name in ("spike_times.npy", "spike_clusters.npy", "templates.npy")):
            results.append(folder)
    return sorted(results, key=lambda item: str(item).lower())

def parse_params(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    output = {}
    for key in ("n_channels_dat", "sample_rate", "offset"):
        match = re.search(rf"(?m)^\s*{key}\s*=\s*([0-9.]+)", text)
        if match:
            output[key] = float(match.group(1))
    dtype_match = re.search(r"(?m)^\s*dtype\s*=\s*['\"]([^'\"]+)", text)
    if dtype_match:
        output["dtype"] = dtype_match.group(1)
    path_match = re.search(r"(?m)^\s*dat_path\s*=\s*\[?\s*['\"]([^'\"]+)", text)
    if path_match:
        configured = Path(path_match.group(1))
        output["dat_path"] = str(configured if configured.is_absolute() else (path.parent / configured).resolve())
    return output

def normalize_cluster_group(folder: Path, backup: bool = True) -> dict:
    pass
    path = folder / "cluster_group.tsv"
    if not path.is_file():
        labels = folder / "cluster_KSLabel.tsv"
        if not labels.is_file():
            return {"changed": False, "reason": "no metadata file"}
        path = labels

    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
        fields = list(rows[0].keys()) if rows else []
    if "cluster_id" not in fields:
        raise ValueError(f"Invalid cluster metadata: {path}")
    label_field = "group" if "group" in fields else "KSLabel" if "KSLabel" in fields else None
    if label_field is None:
        raise ValueError(f"No group/KSLabel column in {path}")
    target = folder / "cluster_group.tsv"
    if path == target and label_field == "group":
        return {"changed": False, "reason": "already normalized"}

    if backup and target.is_file():
        device, recording = infer_device_recording(folder)
        backup_dir = BACKUP_ROOT / device / recording / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_metadata_fix")
        backup_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target, backup_dir / target.name)

    lines = ["cluster_id\tgroup"]
    for row in rows:
        lines.append(f"{int(row['cluster_id'])}\t{row.get(label_field, 'mua') or 'mua'}")
    atomic_write_text(target, "\n".join(lines) + "\n")
    return {"changed": True, "source_field": label_field, "rows": len(rows)}

def infer_device_recording(folder: Path) -> tuple[str, str]:
    return folder.parent.name, folder.name

def validate_result(folder: Path) -> dict:
    spike_times = np.load(folder / "spike_times.npy", mmap_mode="r")
    spike_clusters = np.load(folder / "spike_clusters.npy", mmap_mode="r")
    if spike_times.size != spike_clusters.size:
        raise ValueError("spike_times.npy and spike_clusters.npy have different lengths")
    if spike_clusters.size and int(np.min(spike_clusters)) < 0:
        raise ValueError("negative cluster id detected")
    group_path = folder / "cluster_group.tsv"
    groups = {}
    if group_path.is_file():
        with group_path.open("r", encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                if "cluster_id" in row and "group" in row:
                    groups[int(row["cluster_id"])] = row["group"]
    return {
        "spike_count": int(spike_times.size),
        "cluster_count": int(np.unique(spike_clusters).size),
        "group_count": len(groups),
        "good_count": sum(value == "good" for value in groups.values()),
        "noise_count": sum(value == "noise" for value in groups.values()),
    }

def backup_authoritative(folder: Path, suffix: str = "session") -> Path:
    device, recording = infer_device_recording(folder)
    destination = BACKUP_ROOT / device / recording / (datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{suffix}")
    destination.mkdir(parents=True, exist_ok=True)
    for name in AUTHORITATIVE_FILES:
        source = folder / name
        if source.is_file():
            shutil.copy2(source, destination / name)
    write_json(destination / "signature.json", result_signature(folder))
    return destination

def snapshot_auto_baseline(folder: Path) -> Path:
    device, recording = infer_device_recording(folder)
    destination = AUTO_BASELINE_ROOT / device / recording
    destination.mkdir(parents=True, exist_ok=True)
    for name in AUTHORITATIVE_FILES:
        source = folder / name
        if source.is_file():
            shutil.copy2(source, destination / name)
    marker = folder / AUTO_MARKER
    if marker.is_file():
        shutil.copy2(marker, destination / AUTO_MARKER)
    write_json(destination / "signature.json", result_signature(folder))
    return destination

def copy_result(source: Path, destination: Path) -> None:
    def ignore(_directory: str, names: list[str]) -> set[str]:
        ignored = {name for name in names if name == ".phy" or name.startswith("_phy_spikes_subset")}
        ignored.update({AUTO_MARKER, MANUAL_SAVED_MARKER, MANUAL_CONFIRMED_MARKER, SESSION_MARKER})
        ignored.add("cluster_info.tsv")
        return ignored

    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".copying.{os.getpid()}")
    shutil.copytree(source, temporary, ignore=ignore)
    os.replace(temporary, destination)

def get_status(device: str, recording: str) -> tuple[str, Path]:
    manual = result_at(MANUAL_ROOT, device, recording)
    auto = result_at(AUTO_ROOT, device, recording)
    if (manual / MANUAL_CONFIRMED_MARKER).is_file():
        return "Manually confirmed", manual
    if (manual / MANUAL_SAVED_MARKER).is_file() or (manual / SESSION_MARKER).is_file():
        return "Manual review in progress", manual
    if (auto / AUTO_MARKER).is_file():
        return "Auto-curated", auto
    return "Unprocessed", result_at(SORTING_ROOT, device, recording)

def ensure_manual_copy(device: str, recording: str) -> Path:
    destination = result_at(MANUAL_ROOT, device, recording)
    if destination.exists():
        repair_binary_path(destination, device, recording)
        if not any((destination / marker).is_file() for marker in (SESSION_MARKER, MANUAL_SAVED_MARKER, MANUAL_CONFIRMED_MARKER)):
            write_json(destination / SESSION_MARKER, {
                "stage": "manual_in_progress",
                "created": now_text(),
                "source": "existing manual working copy",
            })
        return destination
    source = result_at(SORTING_ROOT, device, recording)
    copy_result(source, destination)
    repair_binary_path(destination, device, recording)
    normalize_cluster_group(destination, backup=False)
    write_json(destination / SESSION_MARKER, {
        "stage": "manual_in_progress",
        "created": now_text(),
        "source": str(source),
    })
    return destination

def repair_binary_path(folder: Path, device: str, recording: str) -> None:
    params = folder / "params.py"
    values = parse_params(params)
    configured = Path(values.get("dat_path", ""))
    local_root = PROJECT_ROOT / "001_Bin" / "Data"
    candidates = [local_root / device / f"{recording}.bin", local_root / f"{recording}.bin"]
    if configured.name:
        candidates.extend([local_root / device / configured.name, local_root / configured.name])
    binary = next((item for item in candidates if item.is_file()), None)
    if binary is None and configured.is_file():
        binary = configured
    if binary is None:
        raise FileNotFoundError(f"Binary data unavailable for {device}/{recording}: {configured}")
    text = params.read_text(encoding="utf-8")
    assignment = f"dat_path = {Path(os.path.relpath(binary.resolve(), folder.resolve())).as_posix()!r}"
    updated, count = re.subn(r"(?m)^\s*dat_path\s*=.*$", lambda _match: assignment, text)
    if not count:
        updated = text.rstrip() + "\n" + assignment + "\n"
    atomic_write_text(params, updated)
