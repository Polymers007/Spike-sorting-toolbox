from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, ttk

import numpy as np

from curation_common import OUTPUT_ROOT, SORTING_ROOT, WORKING_ROOT, validate_result, ensure_manual_copy
from fast_review_common import journal_path, read_groups
from cached_refinement_common import journal_status as cached_journal_status
from refinement_common import REFINEMENT_DONE_MARKER, refinement_status

STATUS_TAGS = {"Unprocessed": "unprocessed", "Auto-curated": "automatic", "Edited": "edited", "Completed": "completed", "Error": "error"}

class UnifiedCurationManager(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Phy Curation Manager")
        self.geometry("1320x790")
        self.minsize(1080, 650)
        self.rows: list[dict] = []
        self.worker: subprocess.Popen | None = None
        self.worker_log = None
        self.worker_log_path: Path | None = None
        self.worker_log_offset = 0
        self.worker_report_path: Path | None = None
        self.worker_label = ""
        self.last_selection: list[tuple[str, str]] = []
        self._build()
        self.refresh()

    def _build(self) -> None:
        header = ttk.Frame(self, padding=10)
        header.pack(fill="x")
        ttk.Label(header, text="Phy Curation Manager", font=("Segoe UI", 18, "bold")).pack(side="left")
        self.summary = ttk.Label(header)
        self.summary.pack(side="right")

        filters = ttk.Frame(self, padding=(10, 0, 10, 8))
        filters.pack(fill="x")
        ttk.Label(filters, text="Device").pack(side="left")
        self.device_filter = ttk.Combobox(filters, width=13, state="readonly")
        self.device_filter.pack(side="left", padx=(5, 18))
        ttk.Label(filters, text="Status").pack(side="left")
        self.status_filter = ttk.Combobox(filters, values=["All", "Unprocessed", "Auto-curated", "Edited", "Completed", "Error"], width=14, state="readonly")
        self.status_filter.set("All")
        self.status_filter.pack(side="left", padx=(5, 18))
        ttk.Button(filters, text="Refresh", command=self.refresh).pack(side="left")
        self.device_filter.bind("<<ComboboxSelected>>", lambda _event: self.render())
        self.status_filter.bind("<<ComboboxSelected>>", lambda _event: self.render())

        columns = ("device", "recording", "status", "clusters", "good", "noise", "completed")
        self.tree = ttk.Treeview(self, columns=columns, show="headings", selectmode="extended")
        headings = {"device": "Device", "recording": "Recording", "status": "Status", "clusters": "Clusters", "good": "Good", "noise": "Noise", "completed": "Completed At"}
        widths = {"device": 100, "recording": 150, "status": 145, "clusters": 100, "good": 90, "noise": 90, "completed": 220}
        for column in columns:
            self.tree.heading(column, text=headings[column])
            self.tree.column(column, width=widths[column], anchor="center")
        for tag, color in (("unprocessed", "#f2f2f2"), ("automatic", "#dbeeff"), ("edited", "#fff0bd"), ("completed", "#d9f3df"), ("error", "#ffdada")):
            self.tree.tag_configure(tag, background=color)
        self.tree.pack(fill="both", expand=True, padx=10)
        self.tree.bind("<Double-1>", lambda _event: self.open_phy())

        row1 = ttk.Frame(self, padding=(10, 10, 10, 3)); row1.pack(fill="x")
        for text, command in (
            ("Open Cached Refinement", self.open_cached_refinement), ("Apply All Edited", self.apply_all_edited),
            ("Open in Phy", self.open_phy), ("Open Fast Review", self.open_fast_review),
            ("Apply Fast Review", self.apply_fast_review), ("Select Next Unfinished", self.select_next),
            ("Open Result Folder", self.open_folder),
        ):
            ttk.Button(row1, text=text, command=command).pack(side="left", padx=3)

        row2 = ttk.Frame(self, padding=(10, 3, 10, 10)); row2.pack(fill="x")
        for text, command in (
            ("Align Selected", self.align_selected), ("Align All", self.align_all),
            ("Build Cache for All", self.build_cache_all), ("Update Cache", self.update_cache),
        ):
            ttk.Button(row2, text=text, command=command).pack(side="left", padx=3)
        footer = ttk.Frame(self, padding=(13, 0, 13, 5))
        footer.pack(side="bottom", fill="x")
        ttk.Label(footer, text="Developed by Liang Quanduo, 2026", font=("Segoe UI", 8), foreground="#777777", anchor="e").pack(side="right")
        statusbar = ttk.Frame(self, padding=(13, 0, 13, 9)); statusbar.pack(fill="x")
        self.message = ttk.Label(statusbar, text="All clusters are available for review in Phy.", foreground="#245b85")
        self.message.pack(side="left")
        self.progress_text = ttk.Label(statusbar, text="", width=26, anchor="e")
        self.progress_text.pack(side="right", padx=(8, 0))
        self.progress = ttk.Progressbar(statusbar, orient="horizontal", mode="determinate", length=300, maximum=1)
        self.progress.pack(side="right")

    @staticmethod
    def _folder_map(root: Path) -> dict[tuple[str, str], Path]:
        if not root.is_dir():
            return {}
        return {(path.parent.parent.name, path.parent.name): path.parent for path in root.rglob("params.py") if len(path.parent.relative_to(root).parts) == 2}

    def refresh(self) -> None:
        selected_keys = self.selected_keys(silent=True) or self.last_selection
        working = self._folder_map(WORKING_ROOT)
        source = self._folder_map(SORTING_ROOT)
        self.rows = []
        for device, recording in sorted(set(source) | set(working), key=lambda value: (value[0].lower(), value[1].lower())):
            folder = working.get((device, recording)) or source[(device, recording)]
            status = "Auto-curated" if (device, recording) in working else "Unprocessed"
            completed = ""
            try:
                validation = validate_result(folder)
                if (device, recording) in working:
                    refinement, marker = refinement_status(folder)
                    if refinement == "Completed":
                        status, completed = "Completed", marker.get("completed", "")
                    cached_status = cached_journal_status(folder)
                    if cached_status == "edited":
                        status, completed = "Edited", "Pending batch apply"
                queue = journal_path(folder)
                queue_status = ""
                if queue.is_file():
                    queue_status = json.loads(queue.read_text(encoding="utf-8")).get("review_status", "")
                self.rows.append({"device": device, "recording": recording, "status": status, "path": folder,
                                  "clusters": validation.get("cluster_count", ""), "good": validation.get("good_count", ""),
                                  "noise": validation.get("noise_count", ""), "completed": completed, "review_status": queue_status})
            except Exception as error:
                self.rows.append({"device": device, "recording": recording, "status": "Error", "path": folder,
                                  "clusters": "", "good": "", "noise": "", "completed": str(error), "review_status": ""})
        devices = sorted({row["device"] for row in self.rows})
        self.device_filter.configure(values=["All"] + devices)
        if self.device_filter.get() not in ["All"] + devices:
            self.device_filter.set("All")
        if not self.device_filter.get():
            self.device_filter.set("All")
        self.render(selected_keys)

    def render(self, selected_keys: list[tuple[str, str]] | None = None) -> None:
        for item in self.tree.get_children(): self.tree.delete(item)
        wanted = set(selected_keys or [])
        device_filter = self.device_filter.get() or "All"
        status_filter = self.status_filter.get() or "All"
        counts: dict[str, int] = {}; shown = 0
        for index, row in enumerate(self.rows):
            counts[row["status"]] = counts.get(row["status"], 0) + 1
            if device_filter != "All" and row["device"] != device_filter: continue
            if status_filter != "All" and row["status"] != status_filter: continue
            values = tuple(row[key] for key in ("device", "recording", "status", "clusters", "good", "noise", "completed"))
            iid = str(index); self.tree.insert("", "end", iid=iid, values=values, tags=(STATUS_TAGS[row["status"]],)); shown += 1
            if (row["device"], row["recording"]) in wanted: self.tree.selection_add(iid); self.tree.see(iid)
        self.summary.configure(text=f"Total {len(self.rows)}  |  Unprocessed {counts.get('Unprocessed', 0)}  |  Auto-curated {counts.get('Auto-curated', 0)}  |  Edited {counts.get('Edited', 0)}  |  Completed {counts.get('Completed', 0)}  |  Showing {shown}")

    def selected_rows(self, silent: bool = False) -> list[dict]:
        result = [self.rows[int(iid)] for iid in self.tree.selection()]
        if not result and not silent: messagebox.showinfo("Select recording", "Select at least one recording first.", parent=self)
        return result

    def selected_keys(self, silent: bool = False) -> list[tuple[str, str]]:
        return [(row["device"], row["recording"]) for row in self.selected_rows(silent=silent)]

    def _one(self) -> dict | None:
        rows = self.selected_rows()
        if not rows: return None
        if len(rows) != 1:
            messagebox.showinfo("Select one recording", "This action opens one recording at a time.", parent=self); return None
        return rows[0]

    def _start(self, command: list[str], label: str, keys: list[tuple[str, str]] | None = None,
               total: int = 0) -> None:
        if self.worker and self.worker.poll() is None:
            messagebox.showwarning("Task running", "Another Manager task is still running.", parent=self); return
        self.last_selection = keys or self.selected_keys(silent=True)
        self.worker_label = label
        self.worker_log_offset = 0
        self.worker_report_path = None
        self.progress.configure(maximum=max(total, 1), value=0)
        self.progress_text.configure(text=f"0 / {total}" if total else "")
        log_dir = OUTPUT_ROOT / "Batch_Logs" / "Unified_Manager"
        log_dir.mkdir(parents=True, exist_ok=True)
        safe = "".join(character if character.isalnum() else "_" for character in label)[:55]
        self.worker_log_path = log_dir / f"{datetime.now():%Y%m%d_%H%M%S_%f}_{safe}.log"
        self.worker_log = self.worker_log_path.open("w", encoding="utf-8")
        self.worker = subprocess.Popen(command, cwd=OUTPUT_ROOT,
                                       stdout=self.worker_log, stderr=subprocess.STDOUT)
        self.message.configure(text=label)
        self.after(800, self._poll)

    def _read_worker_progress(self) -> None:
        if not self.worker_log_path or not self.worker_log_path.is_file(): return
        try:
            with self.worker_log_path.open("rb") as stream:
                stream.seek(self.worker_log_offset)
                block = stream.read(); self.worker_log_offset = stream.tell()
            text = block.decode("utf-8", errors="replace")
        except OSError:
            return
        for line in text.splitlines():
            match = re.match(r"\[(\d+)/(\d+)\]\s+([^:]+):", line)
            if match:
                current, total, recording = int(match.group(1)), int(match.group(2)), match.group(3)
                self.progress.configure(maximum=max(total, 1), value=current)
                self.progress_text.configure(text=f"{current} / {total}   {recording}")
            if line.startswith("ALIGN_REPORT="):
                self.worker_report_path = Path(line.split("=", 1)[1].strip())

    def _show_align_report(self) -> None:
        path = self.worker_report_path
        if not path or not path.is_file():
            messagebox.showinfo("Alignment finished", "Alignment finished, but no structured channel-change report was produced. See the task log for details.", parent=self)
            return
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
            changed = report.get("changed_recordings", [])
            lines = [f"Completed: {report.get('recordings_succeeded', 0)} / {report.get('recordings_total', 0)}",
                     f"Failed: {report.get('recordings_failed', 0)}",
                     f"Recordings with changed channels: {len(changed)}", ""]
            for item in changed[:35]:
                channels = ", ".join(str(value) for value in item.get("changed_channels", [])) or "none"
                lines.append(f"{item['device']}/{item['recording']}: Ch {channels}")
            if len(changed) > 35:
                lines.append(f"... and {len(changed)-35} more recordings")
            lines += ["", f"Full report: {path}"]
            messagebox.showinfo("Alignment report", "\n".join(lines), parent=self)
        except (OSError, ValueError, KeyError) as error:
            messagebox.showerror("Alignment report error", f"Could not read the report.\n\n{path}\n\n{error}", parent=self)

    def _poll(self) -> None:
        self._read_worker_progress()
        if self.worker and self.worker.poll() is None:
            self.after(800, self._poll); return
        code = self.worker.returncode if self.worker else 0
        if self.worker_log:
            self.worker_log.close(); self.worker_log = None
        self._read_worker_progress()
        label = self.worker_label; self.worker = None
        self.refresh(); self.message.configure(text=f"{label} finished (exit code {code}).")
        if code:
            messagebox.showerror("Task failed", f"{label} ended with exit code {code}.\n\nLog: {self.worker_log_path}", parent=self)
        if label.startswith("Aligning"):
            self._show_align_report()

    def open_phy(self) -> None:
        row = self._one()
        if not row: return
        if row["status"] == "Unprocessed":
            try:
                row["path"] = ensure_manual_copy(row["device"], row["recording"])
            except Exception as error:
                messagebox.showerror("Could not prepare Phy result", str(error), parent=self); return
        command = [sys.executable, str(Path(__file__).with_name("refinement_session_guard.py")), "--result-folder", str(row["path"])]
        self._start(command, f"Phy refinement: {row['device']}/{row['recording']}", [(row["device"], row["recording"])])

    def open_cached_refinement(self) -> None:
        row = self._one()
        if not row: return
        if row["status"] == "Unprocessed":
            messagebox.showinfo("No curated result", "Cached Refinement requires a Working_Curated result.", parent=self); return
        command = [sys.executable, str(Path(__file__).with_name("cached_refinement.py")), "--result-folder", str(row["path"])]
        self._start(command, f"Cached Refinement: {row['device']}/{row['recording']}", [(row["device"], row["recording"])])

    def apply_all_edited(self) -> None:
        rows = [row for row in self.rows if row["status"] == "Edited"]
        if not rows:
            messagebox.showinfo("No Edited recordings", "There are no Cached Refinement journals marked Edited.", parent=self); return
        if not messagebox.askyesno(
            "Apply All Edited",
            f"Apply all {len(rows)} Edited recordings?\n\nAll journals will be validated before any result is changed. "
            "A recovery backup will be created.",
            parent=self,
        ): return
        command = [sys.executable, str(Path(__file__).with_name("apply_cached_refinement.py")), "--apply"]
        keys = [(row["device"], row["recording"]) for row in rows]
        self._start(command, "Applying all Edited Cached Refinement journals", keys, len(rows))

    def open_fast_review(self) -> None:
        row = self._one()
        if not row: return
        if row["status"] == "Unprocessed":
            messagebox.showinfo("No curated result", "Fast Review requires a Working_Curated result.", parent=self); return
        command = [sys.executable, str(Path(__file__).with_name("fast_review.py")), "--recording", f"{row['device']}/{row['recording']}"]
        self._start(command, f"Fast Review: {row['device']}/{row['recording']}", [(row["device"], row["recording"])])

    def apply_fast_review(self) -> None:
        rows = self.selected_rows()
        if not rows: return
        not_ready = [f"{row['device']}/{row['recording']}" for row in rows if row.get("review_status") != "ready_to_apply"]
        if not_ready:
            messagebox.showinfo("Not ready", "These recordings are not marked Ready to Apply:\n\n" + "\n".join(not_ready[:20]), parent=self); return
        if not messagebox.askyesno("Apply Fast Review", "Apply the selected review journals? A recovery backup will be created first.", parent=self): return
        command = [sys.executable, str(Path(__file__).with_name("manager_tasks.py")), "apply"]
        for row in rows: command += ["--recording", f"{row['device']}/{row['recording']}"]
        self._start(command, "Applying Fast Review", [(row["device"], row["recording"]) for row in rows], len(rows))

    def align_selected(self) -> None:
        rows = [row for row in self.selected_rows() if row["status"] != "Unprocessed"]
        if not rows: return
        command = [sys.executable, str(Path(__file__).with_name("manager_tasks.py")), "align"]
        for row in rows: command += ["--recording", f"{row['device']}/{row['recording']}"]
        self._start(command, "Aligning selected recordings", [(row["device"], row["recording"]) for row in rows], len(rows))

    def align_all(self) -> None:
        if not messagebox.askyesno("Align all", "Rebuild the derived main-channel aligned waveform cache for every Working_Curated recording? Labels and spike assignments will not change.", parent=self): return
        total=sum(row["status"] != "Unprocessed" for row in self.rows)
        self._start([sys.executable, str(Path(__file__).with_name("manager_tasks.py")), "align"], "Aligning all recordings", total=total)

    def build_cache_all(self) -> None:
        if not messagebox.askyesno("Build cache for all", "Build/rebuild Fast Review and main-channel waveform caches for all Working_Curated recordings? This can take a long time but does not copy binary data.", parent=self): return
        total=sum(row["status"] != "Unprocessed" for row in self.rows)
        self._start([sys.executable, str(Path(__file__).with_name("manager_tasks.py")), "cache", "--force"], "Building cache for all recordings", total=total)

    def update_cache(self) -> None:
        total=sum(row["status"] != "Unprocessed" for row in self.rows)
        self._start([sys.executable, str(Path(__file__).with_name("manager_tasks.py")), "cache", "--update-only"], "Updating changed or missing caches", total=total)

    def select_next(self) -> None:
        current = self.selected_rows(silent=True)
        start = self.rows.index(current[-1]) if current else -1
        order = list(range(start + 1, len(self.rows))) + list(range(0, start + 1))
        for index in order:
            if self.rows[index]["status"] != "Completed":
                self.device_filter.set("All"); self.status_filter.set("All")
                key = [(self.rows[index]["device"], self.rows[index]["recording"])]
                self.render(key); return
        messagebox.showinfo("All completed", "Every recording is marked Completed.", parent=self)

    def open_folder(self) -> None:
        row = self._one()
        if row: os.startfile(row["path"])

if __name__ == "__main__":
    UnifiedCurationManager().mainloop()
