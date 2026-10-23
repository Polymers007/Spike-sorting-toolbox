from __future__ import annotations

import argparse
import ast
import csv
import gc
import json
import math
import os
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from qtpy import QtCore

from kilosort import run_kilosort
from kilosort.parameters import DEFAULT_SETTINGS
from kilosort.run_kilosort import close_logger

PROGRAM_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PROGRAM_DIR.parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "001_Bin" / "Data"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "004_Result" / "Kilosort4"
PARAMETER_FILE = PROGRAM_DIR / "kilosort_batch_configuration.json"
LOCK_FILE = DEFAULT_OUTPUT_ROOT / "batch_sort_running.lock"
SUCCESS_MARKER = "BATCH_SORT_SUCCESS.txt"
REQUIRED_OUTPUTS = ("ops.npy", "spike_times.npy", "spike_clusters.npy", "params.py")
BATCH_OPTION_DEFAULTS = {
    "do_CAR": True,
    "invert_sign": False,
    "clear_cache": False,
    "save_preprocessed_copy": False,
    "verbose_log": False,
}

def parse_scalar(text: str):
    value = text.strip()
    lowered = value.lower()
    if lowered == "none":
        return None
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"inf", "+inf", "infinity", "+infinity"}:
        return math.inf
    if lowered in {"-inf", "-infinity"}:
        return -math.inf
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value

def coerce_setting(raw, default):
    pass
    if raw is None:
        return None
    if isinstance(raw, list):
        return [parse_scalar(str(item)) if isinstance(item, str) else item for item in raw]
    if isinstance(raw, str):
        parsed = parse_scalar(raw)
    else:
        parsed = raw

    if default is None:
        return parsed
    if isinstance(default, bool):
        return bool(parsed)
    if isinstance(default, int) and not isinstance(default, bool):
        return int(parsed)
    if isinstance(default, float):
        return float(parsed)
    return parsed

def load_gui_configuration():
    cache = QtCore.QSettings("Janelia", "Kilosort4")
    settings = {}
    for key, default in DEFAULT_SETTINGS.items():
        raw = cache.value(key) if cache.contains(key) else default
        settings[key] = coerce_setting(raw, default)

    probe = cache.value("probe_layout")
    probe_name = cache.value("probe_name")
    data_dtype = cache.value("data_dtype", "int16")
    bad_channels = cache.value("bad_channels", [])
    shank_idx = coerce_setting(cache.value("shank_idx", None), None)

    if not isinstance(probe, dict):
        raise RuntimeError("No valid probe layout was found. Select a probe in the Kilosort4 GUI first.")
    if settings.get("n_chan_bin") is None:
        raise RuntimeError("n_chan_bin is not configured in the Kilosort4 GUI.")

    return settings, probe, str(probe_name), str(data_dtype), bad_channels, shank_idx

def load_batch_configuration():
    batch_options = BATCH_OPTION_DEFAULTS.copy()
    if PARAMETER_FILE.is_file():
        saved = json.loads(PARAMETER_FILE.read_text(encoding="utf-8"))
        if not isinstance(saved.get("probe"), dict):
            raise RuntimeError(f"Saved electrode geometry is missing from {PARAMETER_FILE}")
        settings = DEFAULT_SETTINGS.copy()
        for key, default in DEFAULT_SETTINGS.items():
            if key in saved.get("settings", {}):
                settings[key] = coerce_setting(saved["settings"][key], default)
        probe = saved["probe"]
        for key, dtype in (
            ("chanMap", np.int32),
            ("chanMapBackup", np.float32),
            ("xc", np.float32),
            ("yc", np.float32),
            ("kcoords", np.float32),
        ):
            if key in probe:
                probe[key] = np.asarray(probe[key], dtype=dtype)
        probe_name = str(saved.get("probe_name", "saved_probe"))
        data_dtype = str(saved.get("data_dtype", "int16"))
        bad_channels = saved.get("bad_channels", [])
        shank_idx = saved.get("shank_idx")
        for key, default in BATCH_OPTION_DEFAULTS.items():
            if key in saved.get("batch_options", {}):
                batch_options[key] = bool(saved["batch_options"][key])
        source = str(PARAMETER_FILE)
    else:
        settings, probe, probe_name, data_dtype, bad_channels, shank_idx = load_gui_configuration()
        source = "Kilosort GUI last-used parameters"
    return (
        settings,
        probe,
        probe_name,
        data_dtype,
        bad_channels,
        shank_idx,
        batch_options,
        source,
    )

def is_complete(result_dir: Path) -> bool:
    return all((result_dir / filename).is_file() for filename in REQUIRED_OUTPUTS)

def natural_parts(text: str):
    return [int(piece) if piece.isdigit() else piece.lower() for piece in re.split(r"(\d+)", text)]

def binary_sort_key(path: Path):
    return natural_parts(path.name)

def find_binaries(data_root: Path):
    return sorted(
        (path for path in data_root.rglob("*.bin") if path.is_file()),
        key=lambda path: [natural_parts(part) for part in path.relative_to(data_root).parts],
    )

def write_status(status_csv: Path, status: str, binary: Path, result_dir: Path, message: str = ""):
    new_file = not status_csv.exists()
    with status_csv.open("a", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        if new_file:
            writer.writerow(["time", "status", "bin_file", "result_folder", "message"])
        writer.writerow([
            datetime.now().isoformat(timespec="seconds"), status, Path(os.path.relpath(binary, PROJECT_ROOT)).as_posix(), Path(os.path.relpath(result_dir, PROJECT_ROOT)).as_posix(), message
        ])

def safe_json_value(value):
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)

def show_configuration(data_root, settings, probe_name, data_dtype, bad_channels, total, pending, source):
    print("\nKilosort4 batch sorting")
    print(f"Data root : {data_root}")
    print(f"Probe     : {probe_name}")
    print(f"Data type : {data_dtype}")
    print(f"GPU       : {torch.cuda.get_device_name(0)}")
    print(f"nt        : {settings['nt']}")
    print(f"Channels  : {settings['n_chan_bin']}")
    print(f"fs        : {settings['fs']}")
    print(f"Bad chans : {bad_channels}")
    print(f"Parameters: {source}")
    print(f"Found     : {total} bin files; {pending} pending\n")

def main() -> int:
    parser = argparse.ArgumentParser(description="Batch-run Kilosort4 using the current GUI settings.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dry-run", action="store_true", help="List work without running Kilosort.")
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is unavailable. Kilosort batch sorting was not started.")

    (
        settings,
        probe,
        probe_name,
        data_dtype,
        bad_channels,
        shank_idx,
        batch_options,
        parameter_source,
    ) = load_batch_configuration()
    output_root.mkdir(parents=True, exist_ok=True)
    status_csv = output_root / "batch_sort_status.csv"
    binaries = find_binaries(data_root)
    def result_folder(path: Path) -> Path:
        relative = path.relative_to(data_root)
        return output_root / (relative.parent if relative.parent != Path(".") else Path("Default")) / path.stem
    pending = [path for path in binaries if not is_complete(result_folder(path))]

    show_configuration(
        data_root,
        settings,
        probe_name,
        data_dtype,
        bad_channels,
        len(binaries),
        len(pending),
        parameter_source,
    )

    if not pending:
        print("All bin files already have complete Kilosort results.")
        return 0

    for index, binary in enumerate(pending, start=1):
        result_dir = result_folder(binary)
        print(f"[{index}/{len(pending)}] {binary}")
        print(f"           -> {result_dir}")
        if args.dry_run:
            continue

        result_dir.mkdir(parents=True, exist_ok=True)
        write_status(status_csv, "STARTED", binary, result_dir)
        try:
            run_kilosort(
                settings=settings.copy(),
                probe=probe,
                probe_name=probe_name,
                filename=binary,
                results_dir=result_dir,
                data_dtype=data_dtype,
                device=torch.device("cuda:0"),
                do_CAR=batch_options["do_CAR"],
                invert_sign=batch_options["invert_sign"],
                clear_cache=batch_options["clear_cache"],
                save_preprocessed_copy=batch_options["save_preprocessed_copy"],
                bad_channels=bad_channels,
                shank_idx=shank_idx,
                verbose_log=batch_options["verbose_log"],
            )
            if not is_complete(result_dir):
                raise RuntimeError("Kilosort returned, but one or more required output files are missing.")

            marker = {
                "completed": datetime.now().isoformat(timespec="seconds"),
                "bin_file": Path(os.path.relpath(binary, PROJECT_ROOT)).as_posix(),
                "probe_name": probe_name,
                "data_dtype": data_dtype,
                "settings": settings,
            }
            (result_dir / SUCCESS_MARKER).write_text(
                json.dumps(marker, ensure_ascii=False, indent=2, default=safe_json_value),
                encoding="utf-8",
            )
            write_status(status_csv, "COMPLETED", binary, result_dir)
            print("           COMPLETED\n")
        except KeyboardInterrupt:
            write_status(status_csv, "INTERRUPTED", binary, result_dir, "Stopped by user")
            print("\nBatch sorting interrupted. Run the BAT file again to continue.")
            return 130
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            write_status(status_csv, "FAILED", binary, result_dir, message)
            (result_dir / "batch_sort_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            print(f"           FAILED: {message}")
            print("           The error was recorded; continuing with the next file.\n")
        finally:
            close_logger()
            gc.collect()
            torch.cuda.empty_cache()

    print(f"Batch pass finished. Status log: {status_csv}")
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
