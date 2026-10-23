from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PyQt5 import QtCore, QtWidgets
from matplotlib import rcParams
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector
from phylib.stats import correlograms

from build_cached_refinement_cache import BIN_SIZE, MAX_CORRELOGRAM_SPIKES, WINDOW_SIZE, build_one
from cached_refinement_common import cache_dir, journal_status, load_journal, replay, save_journal
from curation_common import WORKING_ROOT, parse_params, validate_result
from refinement_common import refinement_status

rcParams["font.sans-serif"] = ["Segoe UI", "Microsoft YaHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

class Canvas(FigureCanvasQTAgg):
    def __init__(self, rows: int = 1):
        self.figure = Figure(figsize=(7, 5), tight_layout=True)
        if rows == 1:
            self.axes = [self.figure.add_subplot(111)]
        else:
            grid = self.figure.add_gridspec(rows, 1)
            self.axes = [self.figure.add_subplot(grid[index]) for index in range(rows)]
        super().__init__(self.figure)

class CachedRefinement(QtWidgets.QMainWindow):
    def __init__(self, folder: Path):
        super().__init__()
        self.folder = Path(folder)
        self.setWindowTitle(f"Cached Refinement - {self.folder.parent.name}/{self.folder.name}")
        self.resize(1700, 960)
        self.journal: dict = {}
        self.base_assignments = self.assignments = None
        self.labels: dict[int, str] = {}
        self.times = self.amplitudes = None
        self.cache = self.meta = None
        self.sample_ids = self.sample_waveforms = self.sample_channels = None
        self.cluster_rows: dict[int, int] = {}
        self.preferred_channels: dict[int, int] = {}
        self.waveform_memory: dict[tuple[int, int], tuple[np.ndarray, int]] = {}
        self.selection_bounds = None
        self.selector = None
        self.current_cluster: int | None = None
        self._updating_similarity = False
        self._build_ui()
        QtCore.QTimer.singleShot(0, self.load)

    def _build_ui(self) -> None:
        root = QtWidgets.QWidget(); self.setCentralWidget(root)
        outer = QtWidgets.QVBoxLayout(root)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal); outer.addWidget(splitter, 1)

        left = QtWidgets.QWidget(); left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.addWidget(QtWidgets.QLabel("ClusterView  (channel order)"))
        self.clusters = QtWidgets.QListWidget()
        self.clusters.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.clusters.currentItemChanged.connect(self.draw_current)
        self.clusters.itemSelectionChanged.connect(self.draw_selected_overlays)
        left_layout.addWidget(self.clusters, 3)
        left_layout.addWidget(QtWidgets.QLabel("SimilarityView  (double-click to select pair)"))
        self.similarity = QtWidgets.QListWidget()
        self.similarity.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.similarity.itemSelectionChanged.connect(self.draw_selected_overlays)
        self.similarity.itemDoubleClicked.connect(self.select_similarity_pair)
        left_layout.addWidget(self.similarity, 2)
        buttons = QtWidgets.QGridLayout(); left_layout.addLayout(buttons)
        specs = [
            ("Good [G]", lambda: self.set_label("good"), 0, 0),
            ("Noise [N]", lambda: self.set_label("noise"), 0, 1),
            ("MUA [U]", lambda: self.set_label("mua"), 0, 2),
            ("Merge -> Good", self.merge_selected, 1, 0),
            ("Selection -> Good\nremainder Noise", lambda: self.split_selected("good", "noise"), 1, 1),
            ("Selection -> Noise\nremainder unchanged", lambda: self.split_selected("noise", None), 1, 2),
            ("Undo [Ctrl+Z]", self.undo, 2, 0),
            ("Redo [Ctrl+Y]", self.redo, 2, 1),
            ("Finish Editing", self.finish, 2, 2),
            ("Finish Editing and Open Next Auto-curated", self.finish_and_next, 3, 0),
        ]
        for text, callback, row, column in specs:
            button = QtWidgets.QPushButton(text); button.clicked.connect(callback)
            if row == 3: buttons.addWidget(button, row, column, 1, 3)
            else: buttons.addWidget(button, row, column)
        splitter.addWidget(left); left.setMinimumWidth(475)

        plots = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        upper = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.wave = Canvas(); self.correlogram = Canvas(); self.amplitude = Canvas()
        upper.addWidget(self.wave); upper.addWidget(self.correlogram); upper.setSizes([760, 460])
        plots.addWidget(upper); plots.addWidget(self.amplitude)
        
        plots.setSizes([650, 300]); splitter.addWidget(plots); splitter.setSizes([480, 1220])
        self.message = QtWidgets.QLabel("Loading cached preview...")
        self.message.setStyleSheet("padding: 5px; color: #245b85"); outer.addWidget(self.message)
        QtWidgets.QShortcut("G", self, lambda: self.set_label("good"))
        QtWidgets.QShortcut("N", self, lambda: self.set_label("noise"))
        QtWidgets.QShortcut("U", self, lambda: self.set_label("mua"))
        QtWidgets.QShortcut("Ctrl+Z", self, self.undo); QtWidgets.QShortcut("Ctrl+Y", self, self.redo)

    def load(self) -> None:
        try:
            self.setWindowTitle(f"Cached Refinement - {self.folder.parent.name}/{self.folder.name}")
            if self.selector is not None: self.selector.set_active(False)
            self.selector = None; self.selection_bounds = None; self.current_cluster = None
            self.clusters.blockSignals(True); self.clusters.clear(); self.clusters.blockSignals(False)
            self.similarity.blockSignals(True); self.similarity.clear(); self.similarity.blockSignals(False)
            self.waveform_memory.clear(); self.cluster_rows = {}; self.preferred_channels = {}
            self.message.setText("Preparing cache for this recording..."); QtWidgets.QApplication.processEvents()
            build_one(self.folder, force=False)
            destination = cache_dir(self.folder)
            self.meta = json.loads((destination / "metadata.json").read_text(encoding="utf-8"))
            self.cache = dict(np.load(destination / "cached_refinement.npz", allow_pickle=False))
            self.journal = load_journal(self.folder)
            self.base_assignments = np.load(self.folder / "spike_clusters.npy", mmap_mode="r").ravel()
            self.assignments, self.labels = replay(self.folder, self.journal.get("operations", []))
            rate = float(parse_params(self.folder / "params.py").get("sample_rate", 30000.0))
            self.times = np.load(self.folder / "spike_times.npy", mmap_mode="r").ravel() / rate
            self.amplitudes = np.load(self.folder / "amplitudes.npy", mmap_mode="r").ravel()
            subset = self.folder / "_phy_spikes_subset.spikes.npy"
            if subset.is_file():
                self.sample_ids = np.load(subset, mmap_mode="r").astype(np.int64)
                self.sample_waveforms = np.load(self.folder / "_phy_spikes_subset.waveforms.npy", mmap_mode="r")
                self.sample_channels = np.load(self.folder / "_phy_spikes_subset.channels.npy", mmap_mode="r")
            ids = np.asarray(self.cache["cluster_ids"], dtype=np.int64)
            self.cluster_rows = {int(cluster_id): index for index, cluster_id in enumerate(ids)}
            self.preferred_channels = {int(cluster_id): int(self.cache["main_channels"][index]) for index, cluster_id in enumerate(ids)}
            for operation in self.journal.get("operations", []): self._inherit_channel(operation)
            self.refresh_clusters()
            self.message.setText(f"Cached preview loaded. {len(self.journal.get('operations', []))} pending operation(s); results are unchanged until Apply All Edited.")
        except Exception as error:
            QtWidgets.QMessageBox.critical(self, "Cached Refinement", str(error)); self.close()

    def _inherit_channel(self, operation: dict) -> None:
        if operation.get("type") == "split":
            source, target = int(operation["source"]), int(operation["new_cluster_id"])
            if source in self.preferred_channels: self.preferred_channels[target] = self.preferred_channels[source]
        elif operation.get("type") == "merge":
            target = int(operation["target"])
            if target not in self.preferred_channels:
                for source in operation.get("sources", []):
                    if int(source) in self.preferred_channels:
                        self.preferred_channels[target] = self.preferred_channels[int(source)]; break

    def cluster_count(self, cluster_id: int) -> int:
        return int(np.count_nonzero(self.assignments == cluster_id))

    def main_channel(self, cluster_id: int) -> int:
        return int(self.preferred_channels.get(cluster_id, -1))

    def refresh_clusters(self, select: int | None = None) -> None:
        previous = select if select is not None else self.current_cluster
        self.clusters.blockSignals(True); self.clusters.clear()
        ids = [int(value) for value in np.unique(self.assignments)]
        ids.sort(key=lambda cid: (self.main_channel(cid), cid))
        for cluster_id in ids:
            count = self.cluster_count(cluster_id); group = self.labels.get(cluster_id, "mua")
            item = QtWidgets.QListWidgetItem(f"{cluster_id:5d}   Ch {self.main_channel(cluster_id):2d}   {group:5s}   n={count}")
            item.setData(QtCore.Qt.UserRole, cluster_id)
            item.setForeground({"good": QtCore.Qt.darkGreen, "noise": QtCore.Qt.darkRed}.get(group, QtCore.Qt.black))
            self.clusters.addItem(item)
            if cluster_id == previous: self.clusters.setCurrentItem(item)
        self.clusters.blockSignals(False)
        if self.clusters.currentItem() is None and self.clusters.count(): self.clusters.setCurrentRow(0)
        self.draw_current()

    def selected_ids(self) -> list[int]:
        return [int(item.data(QtCore.Qt.UserRole)) for item in self.clusters.selectedItems()]

    def _membership_unchanged(self, cluster_id: int) -> bool:
        if cluster_id not in self.cluster_rows: return False
        return np.array_equal(np.asarray(self.assignments == cluster_id), np.asarray(self.base_assignments == cluster_id))

    def waveform(self, cluster_id: int) -> tuple[np.ndarray, int]:
        revision = len(self.journal.get("operations", [])); key = (cluster_id, revision)
        if key in self.waveform_memory: return self.waveform_memory[key]
        preferred = self.main_channel(cluster_id)
        if self._membership_unchanged(cluster_id):
            waveform = np.asarray(self.cache["medians"][self.cluster_rows[cluster_id]], dtype=np.float32)
            result = waveform, preferred; self.waveform_memory[key] = result; return result
        n_channels = int(np.asarray(self.cache["channel_positions"]).shape[0])
        n_samples = int(self.sample_waveforms.shape[1]) if self.sample_waveforms is not None else 0
        physical = np.full((n_samples, n_channels), np.nan, dtype=np.float32)
        if self.sample_ids is not None:
            rows = np.flatnonzero(self.assignments[np.asarray(self.sample_ids)] == cluster_id)
            for channel in range(n_channels):
                pieces = []
                for row in rows:
                    columns = np.flatnonzero(np.asarray(self.sample_channels[row]) == channel)
                    if columns.size: pieces.append(np.asarray(self.sample_waveforms[row, :, int(columns[0])], dtype=np.float32))
                if pieces: physical[:, channel] = np.nanmedian(np.asarray(pieces), axis=0)
        result = physical, preferred; self.waveform_memory[key] = result; return result

    def draw_current(self, *_args) -> None:
        item = self.clusters.currentItem()
        if item is None or self.assignments is None: return
        cluster_id = int(item.data(QtCore.Qt.UserRole)); self.current_cluster = cluster_id
        self.populate_similarity(cluster_id)
        self.draw_selected_overlays()

    def display_ids(self) -> list[int]:
        if self.current_cluster is None: return []
        selected = self.selected_ids()
        ordered = [self.current_cluster] + [cluster_id for cluster_id in selected if cluster_id != self.current_cluster]
        if len(ordered) >= 2: return ordered[:2]
        similarity_items = self.similarity.selectedItems()
        if similarity_items:
            other = int(similarity_items[0].data(QtCore.Qt.UserRole))
            if other != self.current_cluster: return [self.current_cluster, other]
        return [self.current_cluster]

    def draw_selected_overlays(self) -> None:
        if self.assignments is None or self.current_cluster is None or self._updating_similarity: return
        cluster_ids = self.display_ids()
        self.draw_waveforms(cluster_ids)
        self.draw_correlogram(self.current_cluster)
        self.draw_amplitude(cluster_ids)

    def draw_waveforms(self, cluster_ids: list[int]) -> None:
        ax = self.wave.axes[0]; ax.clear()
        scale = float(self.meta.get("scale_uV_per_bit", 1.0)); rate = float(self.meta.get("sample_rate_hz", 30000.0))
        positions = np.asarray(self.cache["channel_positions"], dtype=float)
        colors = ["#087fdb", "#ff7f0e"]
        entries = []
        channel_candidates: list[int] = []
        for cluster_id in cluster_ids:
            waveform, channel = self.waveform(cluster_id)
            entries.append((cluster_id, waveform, channel))
            if 0 <= channel < waveform.shape[1]:
                distances = np.linalg.norm(positions - positions[channel], axis=1) if positions.ndim == 2 else np.abs(np.arange(waveform.shape[1]) - channel)
                channel_candidates.extend(int(value) for value in np.argsort(distances)[:6])
        nearby = []
        for channel in channel_candidates:
            if channel not in nearby and any(channel < waveform.shape[1] and np.any(np.isfinite(waveform[:, channel])) for _, waveform, _ in entries):
                nearby.append(channel)
            if len(nearby) >= 8: break
        available = [waveform[:, nearby] * scale for _, waveform, _ in entries if nearby and waveform.size]
        span = max([float(np.nanmax(value) - np.nanmin(value)) for value in available if np.any(np.isfinite(value))] + [2.0])
        gap = span * 1.35
        for index, (cluster_id, waveform, main_channel) in enumerate(entries):
            if not waveform.size: continue
            time_ms = (np.arange(waveform.shape[0]) - waveform.shape[0] // 2) * 1000.0 / rate
            color = colors[index % len(colors)]
            label_added = False
            for row, channel in enumerate(nearby):
                if channel >= waveform.shape[1] or not np.any(np.isfinite(waveform[:, channel])): continue
                trace = waveform[:, channel] * scale
                trace -= np.nanmedian(np.r_[trace[:8], trace[-8:]])
                offset = (len(nearby) - 1 - row) * gap
                is_main = channel == main_channel
                label = f"Cluster {cluster_id} (main Ch {main_channel})" if is_main and not label_added else None
                ax.plot(time_ms, trace + offset, color=color, lw=3.0 if is_main else 1.25,
                        alpha=1.0 if is_main else .55, label=label)
                label_added = label_added or is_main
        if nearby and entries:
            sample_waveform = entries[0][1]
            time_ms = (np.arange(sample_waveform.shape[0]) - sample_waveform.shape[0] // 2) * 1000.0 / rate
            for row, channel in enumerate(nearby):
                offset = (len(nearby) - 1 - row) * gap
                ax.text(time_ms[0], offset + .05 * gap, f"Ch {channel}", color="#555555", fontsize=8)
            ax.axvline(0, color="gray", alpha=.35); ax.set_yticks([])
            ax.set(title="Main and nearest physical channels", xlabel="Time from aligned center (ms)",
                   ylabel="Channels (same uV scale, offset)")
            if len(cluster_ids) > 1: ax.legend(loc="best")
        else:
            ax.text(.5, .5, "No cached waveform available", ha="center", va="center", transform=ax.transAxes)
        self.wave.draw_idle()

    def correlogram_values(self, cluster_id: int) -> tuple[np.ndarray, np.ndarray]:
        if self._membership_unchanged(cluster_id):
            row = self.cluster_rows[cluster_id]
            return np.asarray(self.cache["correlogram_lags_ms"]), np.asarray(self.cache["autocorrelograms"][row])
        spike_ids = np.flatnonzero(self.assignments == cluster_id)
        if len(spike_ids) > MAX_CORRELOGRAM_SPIKES:
            spike_ids = spike_ids[np.linspace(0, len(spike_ids) - 1, MAX_CORRELOGRAM_SPIKES).astype(np.int64)]
        n_bins = 2 * int(.5 * WINDOW_SIZE / BIN_SIZE) + 1
        lags = np.linspace(-WINDOW_SIZE * 500, WINDOW_SIZE * 500, n_bins)
        if len(spike_ids) < 2: return lags, np.zeros(n_bins)
        values = correlograms(np.asarray(self.times[spike_ids]), np.zeros(len(spike_ids), dtype=np.int64),
                              cluster_ids=[0], sample_rate=float(self.meta.get("sample_rate_hz", 30000)),
                              bin_size=BIN_SIZE, window_size=WINDOW_SIZE)[0, 0]
        return lags, values

    def draw_correlogram(self, cluster_id: int) -> None:
        ax = self.correlogram.axes[0]; ax.clear(); lags, values = self.correlogram_values(cluster_id)
        width = float(np.diff(lags).mean()) * .9 if len(lags) > 1 else 1.0
        ax.bar(lags, values, width=width, color="#159cff"); ax.axvspan(-1, 1, color="#ffb000", alpha=.15)
        ax.set(title=f"CorrelogramView - Cluster {cluster_id} (max 50,000 spikes)", xlabel="Lag (ms)", ylabel="Count")
        self.correlogram.draw_idle()

    def draw_amplitude(self, cluster_ids: list[int]) -> None:
        ax = self.amplitude.axes[0]; ax.clear(); colors = ["#159cff", "#ff7f0e"]
        summaries = []
        for index, cluster_id in enumerate(cluster_ids):
            ids = np.flatnonzero(self.assignments == cluster_id)
            display = ids if len(ids) <= 25000 else ids[np.linspace(0, len(ids)-1, 25000).astype(np.int64)]
            ax.scatter(self.times[display], self.amplitudes[display], s=3, color=colors[index % len(colors)],
                       alpha=.55, linewidths=0, label=f"Cluster {cluster_id}")
            summaries.append(f"{cluster_id}: n={len(ids)}")
        ax.set(title="AmplitudeView - " + " | ".join(summaries), xlabel="Time (s)", ylabel="Amplitude")
        if len(cluster_ids) > 1: ax.legend(loc="best")
        if self.selector is not None: self.selector.set_active(False)
        self.selector = RectangleSelector(ax, self.rectangle_selected, useblit=True, button=[1], interactive=True,
                                          props=dict(facecolor="#20a5ff", edgecolor="white", alpha=.18))
        self.selection_bounds = None; self.amplitude.draw_idle()

    @staticmethod
    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        a, b = np.nan_to_num(a.astype(float)), np.nan_to_num(b.astype(float))
        best = -1.0
        for shift in range(-3, 4):
            x, y = (a[-shift:], b[:shift]) if shift < 0 else ((a[:-shift], b[shift:]) if shift > 0 else (a, b))
            denominator = np.linalg.norm(x) * np.linalg.norm(y)
            if denominator: best = max(best, float(np.dot(x.ravel(), y.ravel()) / denominator))
        return best

    def populate_similarity(self, cluster_id: int) -> None:
        self._updating_similarity = True
        self.similarity.blockSignals(True); self.similarity.clear(); source, _ = self.waveform(cluster_id); scores = []
        unchanged_source = self._membership_unchanged(cluster_id)
        for other in np.unique(self.assignments):
            other = int(other)
            if other == cluster_id: continue
            if unchanged_source and self._membership_unchanged(other):
                score = float(self.cache["similarity"][self.cluster_rows[cluster_id], self.cluster_rows[other]])
            else:
                target, _ = self.waveform(other); score = self.cosine(source, target)
            scores.append((score, other))
        for score, other in sorted(scores, reverse=True)[:80]:
            item = QtWidgets.QListWidgetItem(f"{other:5d}   sim={score:.4f}   {self.labels.get(other, 'mua')}   Ch {self.main_channel(other)}")
            item.setData(QtCore.Qt.UserRole, other); self.similarity.addItem(item)
        self.similarity.blockSignals(False); self._updating_similarity = False

    def select_similarity_pair(self, item) -> None:
        other = int(item.data(QtCore.Qt.UserRole)); wanted = {self.current_cluster, other}
        self.clusters.clearSelection()
        for index in range(self.clusters.count()):
            entry = self.clusters.item(index)
            if int(entry.data(QtCore.Qt.UserRole)) in wanted: entry.setSelected(True)
        self.draw_selected_overlays()

    def commit(self, operation: dict) -> None:
        self.journal.setdefault("operations", []).append(operation); self.journal["redo_operations"] = []
        self.journal["review_status"] = "draft"; save_journal(self.folder, self.journal)
        self._inherit_channel(operation); self.assignments, self.labels = replay(self.folder, self.journal["operations"])
        self.waveform_memory.clear(); self.refresh_clusters(int(operation.get("new_cluster_id", operation.get("target", operation.get("cluster_id", -1)))))

    def set_label(self, group: str) -> None:
        ids = self.selected_ids() or ([self.current_cluster] if self.current_cluster is not None else [])
        if not ids: return
        for cluster_id in ids: self.journal.setdefault("operations", []).append({"type": "label", "cluster_id": cluster_id, "group": group})
        self.journal["redo_operations"] = []; self.journal["review_status"] = "draft"; save_journal(self.folder, self.journal)
        self.assignments, self.labels = replay(self.folder, self.journal["operations"]); self.refresh_clusters(ids[0])

    def merge_selected(self) -> None:
        ids = self.selected_ids()
        if len(ids) < 2: self.message.setText("Select at least two clusters before Merge."); return
        target = self.current_cluster if self.current_cluster in ids else min(ids)
        self.commit({"type": "merge", "sources": ids, "target": int(target), "group": "good"})

    def rectangle_selected(self, start, end) -> None:
        if None in (start.xdata, start.ydata, end.xdata, end.ydata): return
        self.selection_bounds = [min(start.xdata, end.xdata), max(start.xdata, end.xdata), min(start.ydata, end.ydata), max(start.ydata, end.ydata)]
        self.message.setText("Selection recorded. Choose one of the two Split actions.")

    def split_selected(self, selected_group: str, remainder_group: str | None) -> None:
        if self.current_cluster is None or not self.selection_bounds:
            self.message.setText("Drag a rectangle in AmplitudeView first."); return
        x0, x1, y0, y1 = self.selection_bounds; source = int(self.current_cluster)
        mask = ((self.assignments == source) & (self.times >= x0) & (self.times <= x1) &
                (self.amplitudes >= y0) & (self.amplitudes <= y1))
        selected = np.flatnonzero(mask); total = self.cluster_count(source)
        if not len(selected) or len(selected) == total:
            self.message.setText("A Split selection must contain only part of the current cluster."); return
        target = int(np.max(self.assignments)) + 1
        self.commit({"type": "split", "source": source, "new_cluster_id": target,
                     "selected_spike_ids": selected.tolist(), "selected_group": selected_group,
                     "remainder_group": remainder_group or self.labels.get(source, "mua"),
                     "bounds": {"time": [x0, x1], "amplitude": [y0, y1]}})

    def undo(self) -> None:
        operations = self.journal.get("operations", [])
        if not operations: return
        self.journal.setdefault("redo_operations", []).append(operations.pop()); self.journal["review_status"] = "draft"
        save_journal(self.folder, self.journal); self.assignments, self.labels = replay(self.folder, operations)
        self.waveform_memory.clear(); self.refresh_clusters()

    def redo(self) -> None:
        redo = self.journal.get("redo_operations", [])
        if not redo: return
        operation = redo.pop(); self.journal.setdefault("operations", []).append(operation); self._inherit_channel(operation)
        self.journal["review_status"] = "draft"; save_journal(self.folder, self.journal)
        self.assignments, self.labels = replay(self.folder, self.journal["operations"]); self.waveform_memory.clear(); self.refresh_clusters()

    def finish(self) -> None:
        if not self.mark_edited(): return
        QtWidgets.QMessageBox.information(self, "Cached Refinement", "This recording is now Edited.\nNo Phy result has been changed yet.\nUse Apply All Edited in the Manager when ready.")
        self.close()

    def mark_edited(self) -> bool:
        if not self.journal: return False
        self.journal["review_status"] = "edited"; save_journal(self.folder, self.journal)
        return True

    @staticmethod
    def working_folders() -> list[Path]:
        return sorted(
            (path.parent for path in WORKING_ROOT.rglob("params.py")
             if len(path.parent.relative_to(WORKING_ROOT).parts) == 2),
            key=lambda path: (path.parent.name.lower(), path.name.lower()),
        )

    @staticmethod
    def is_auto_curated(folder: Path) -> bool:
        try:
            validate_result(folder)
            if journal_status(folder) == "edited": return False
            status, _marker = refinement_status(folder)
            return status != "Completed"
        except Exception:
            return False

    def next_auto_curated(self) -> Path | None:
        folders = self.working_folders()
        if not folders: return None
        try: start = folders.index(self.folder)
        except ValueError: start = -1
        order = folders[start + 1:] + folders[:start + 1]
        for folder in order:
            if folder != self.folder and self.is_auto_curated(folder): return folder
        return None

    def finish_and_next(self) -> None:
        if not self.mark_edited(): return
        current = f"{self.folder.parent.name}/{self.folder.name}"
        self.message.setText(f"{current} marked Edited. Finding the next Auto-curated recording...")
        QtWidgets.QApplication.processEvents()
        try:
            next_folder = self.next_auto_curated()
        except Exception as error:
            QtWidgets.QMessageBox.critical(self, "Find next recording", f"The current recording was saved as Edited, but the next recording could not be found.\n\n{error}")
            return
        if next_folder is None:
            QtWidgets.QMessageBox.information(self, "Cached Refinement", "The current recording is now Edited.\nThere are no remaining Auto-curated recordings.")
            self.close(); return
        self.folder = next_folder
        self.load()

def main() -> int:
    parser = argparse.ArgumentParser(description="Cached, non-destructive Phy-like refinement preview.")
    parser.add_argument("--result-folder", required=True)
    args = parser.parse_args()
    app = QtWidgets.QApplication([sys.argv[0]]); window = CachedRefinement(Path(args.result_folder)); window.show()
    return app.exec_()

if __name__ == "__main__":
    raise SystemExit(main())
