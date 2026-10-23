from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from tkinter import Tk, messagebox

from curation_common import (
    AUTO_ROOT,
    MANUAL_CONFIRMED_MARKER,
    MANUAL_SAVED_MARKER,
    SESSION_MARKER,
    SORTING_ROOT,
    backup_authoritative,
    ensure_manual_copy,
    normalize_cluster_group,
    now_text,
    result_signature,
    validate_result,
    write_json,
)

def dialog(kind: str, title: str, message: str) -> bool | None:
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        if kind == "yesno":
            return messagebox.askyesno(title, message, parent=root)
        if kind == "error":
            messagebox.showerror(title, message, parent=root)
        elif kind == "warning":
            messagebox.showwarning(title, message, parent=root)
        else:
            messagebox.showinfo(title, message, parent=root)
    finally:
        root.destroy()
    return None

def main() -> int:
    parser = argparse.ArgumentParser(description="Launch Phy with save verification and backups.")
    parser.add_argument("--result-folder", required=True, type=Path)
    parser.add_argument("--params-file", type=Path)
    parser.add_argument("--phy-exe", type=Path, default=Path.home() / ".conda" / "envs" / "phy" / "Scripts" / "phy.exe")
    args = parser.parse_args()

    folder = args.result_folder.resolve()
    params = (args.params_file or (folder / "params.py")).resolve()
    for protected_root in (SORTING_ROOT.resolve(), AUTO_ROOT.resolve()):
        try:
            relative = folder.relative_to(protected_root)
        except ValueError:
            continue
        if len(relative.parts) >= 2:
            folder = ensure_manual_copy(relative.parts[-2], relative.parts[-1]).resolve()
            params = folder / "params.py"
        break
    if not folder.is_dir() or not params.is_file() or not args.phy_exe.is_file():
        dialog("error", "Phy could not start", f"The result directory, parameter file, or Phy executable is unavailable:\n{folder}")
        return 2

    try:
        normalization = normalize_cluster_group(folder, backup=True)
        validation_before = validate_result(folder)
        backup = backup_authoritative(folder, "before_phy")
        before = result_signature(folder)
    except Exception as error:
        dialog("error", "Phy preflight failed", str(error))
        return 3

    write_json(folder / SESSION_MARKER, {
        "stage": "phy_open",
        "started": now_text(),
        "backup": str(backup),
        "metadata_normalization": normalization,
        "before": before,
        "validation_before": validation_before,
    })

    command = [str(args.phy_exe), "template-gui", "--clear-state", str(params)]
    exit_code = subprocess.call(command, cwd=folder)

    try:
        after = result_signature(folder)
        validation_after = validate_result(folder)
    except Exception as error:
        write_json(folder / SESSION_MARKER, {
            "stage": "save_validation_failed",
            "finished": now_text(),
            "backup": str(backup),
            "error": str(error),
        })
        dialog(
            "error",
            "Phy result validation failed",
            f"The saved result failed integrity validation and was not marked complete.\n\nBackup: {backup}\n\n{error}",
        )
        return 4

    changed = [name for name in before if before[name] != after[name]]
    session = {
        "stage": "phy_closed",
        "finished": now_text(),
        "exit_code": exit_code,
        "backup": str(backup),
        "changed_authoritative_files": changed,
        "before": before,
        "after": after,
        "validation_after": validation_after,
    }
    write_json(folder / SESSION_MARKER, session)

    if not changed:
        dialog(
            "warning",
            "No Phy save detected",
            "No saved change was detected in spike_clusters.npy, cluster_group.tsv, or cluster_info.tsv.\n\n"
            "This recording will not be marked complete. After future edits, press Ctrl+S and wait for saving to finish before closing Phy.",
        )
        return exit_code

    saved = {
        "stage": "manual_saved",
        "saved": now_text(),
        "changed_authoritative_files": changed,
        "backup": str(backup),
        "signature": after,
        "validation": validation_after,
    }
    write_json(folder / MANUAL_SAVED_MARKER, saved)
    confirmed = dialog(
        "yesno",
        "Saved manual edits detected",
        "Phy edits were saved and passed integrity validation.\n\n"
        f"Changed files: {', '.join(changed)}\n"
        f"Cluster:{validation_after['cluster_count']},good:{validation_after['good_count']},noise:{validation_after['noise_count']}\n\n"
        "Mark this recording as manually reviewed and complete?",
    )
    if confirmed:
        write_json(folder / MANUAL_CONFIRMED_MARKER, {
            **saved,
            "stage": "manual_confirmed",
            "confirmed": now_text(),
        })
        dialog("info", "Manual review completed", "The result was saved, validated, and marked as manually reviewed and complete.")
    return exit_code

if __name__ == "__main__":
    raise SystemExit(main())
