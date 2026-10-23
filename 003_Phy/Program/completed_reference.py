from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from curation_common import OUTPUT_ROOT, WORKING_ROOT, now_text
from fast_review_common import read_groups
from main_channel_review import _signature, metadata_path
from refinement_common import refinement_status

MODEL_ROOT = OUTPUT_ROOT / "Manager_State"
MODEL_ARRAYS = MODEL_ROOT / "completed_reference_model.npz"
MODEL_META = MODEL_ROOT / "completed_reference_model.json"
FEATURES = (
    "peak_snr", "negative_dominance", "center_fraction", "jitter_mad",
    "amplitude_fraction", "median_correlation", "half_width", "sample_count",
)

def vector(metric: dict) -> np.ndarray | None:
    try:
        row = np.asarray([float(metric[name]) for name in FEATURES], dtype=np.float64)
    except (KeyError, TypeError, ValueError):
        return None
    return row if np.all(np.isfinite(row)) else None

def build_reference_model() -> dict:
    rows: list[np.ndarray] = []
    labels: list[int] = []
    recordings = 0
    for params in sorted(WORKING_ROOT.rglob("params.py")):
        folder = params.parent
        if len(folder.relative_to(WORKING_ROOT).parts) != 2 or refinement_status(folder)[0] != "Completed":
            continue
        meta_path = metadata_path(folder)
        if not meta_path.is_file():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if meta.get("signature") != _signature(folder):
            continue
        groups = read_groups(folder); used = 0
        for metric in meta.get("metrics", []):
            group = groups.get(int(metric.get("cluster_id", -1)))
            value = vector(metric)
            if group not in ("good", "noise") or value is None:
                continue
            rows.append(value); labels.append(1 if group == "good" else 0); used += 1
        recordings += bool(used)
    if not rows:
        raise RuntimeError("No current Completed reference clusters are available. Build caches first.")
    values = np.stack(rows); targets = np.asarray(labels, dtype=np.uint8)
    median = np.median(values, axis=0)
    q1, q3 = np.percentile(values, [25, 75], axis=0)
    scale = np.where(q3 - q1 > 1e-9, q3 - q1, np.std(values, axis=0) + 1e-6)
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    temporary = MODEL_ARRAYS.with_name(MODEL_ARRAYS.name + f".tmp.{os.getpid()}")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, values=values, targets=targets, median=median, scale=scale)
    os.replace(temporary, MODEL_ARRAYS)
    summary = {"schema_version": 1, "created": now_text(), "features": FEATURES,
               "completed_recordings": recordings, "training_clusters": len(rows),
               "training_good": int(targets.sum()), "training_noise": int((targets == 0).sum()),
               "policy": "advisory only; never changes labels or creates splits"}
    temporary_meta = MODEL_META.with_name(MODEL_META.name + f".tmp.{os.getpid()}")
    temporary_meta.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary_meta, MODEL_META)
    return summary

def load_reference_model() -> dict | None:
    if not MODEL_ARRAYS.is_file() or not MODEL_META.is_file():
        return None
    try:
        arrays = np.load(MODEL_ARRAYS)
        result = {name: arrays[name] for name in arrays.files}
        arrays.close()
        result["metadata"] = json.loads(MODEL_META.read_text(encoding="utf-8"))
        return result
    except (OSError, ValueError, KeyError):
        return None

def predict_good(model: dict | None, metric: dict, k: int = 31) -> float | None:
    row = vector(metric)
    if model is None or row is None:
        return None
    values = np.asarray(model["values"], dtype=np.float64)
    targets = np.asarray(model["targets"], dtype=np.float64)
    if not len(values):
        return None
    train = (values - model["median"]) / model["scale"]
    test = (row - model["median"]) / model["scale"]
    distance = np.sqrt(np.sum((train - test) ** 2, axis=1))
    nearest = np.argsort(distance)[:min(k, len(distance))]
    weights = 1.0 / (distance[nearest] + 0.15)
    return float(np.sum(weights * targets[nearest]) / np.sum(weights))
