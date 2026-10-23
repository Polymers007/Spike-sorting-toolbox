from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PyQt5 import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector
from matplotlib import rcParams

from curation_common import WORKING_ROOT, parse_params
from fast_review_common import journal_path, load_journal, replay, save_journal
from main_channel_review import load_main_channel_cache
from completed_reference import load_reference_model, predict_good

rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

class PlotCanvas(FigureCanvasQTAgg):
    def __init__(self):
        self.figure = Figure(figsize=(6, 4), tight_layout=True)
        self.ax = self.figure.add_subplot(111)
        super().__init__(self.figure)

class WaveformCanvas(FigureCanvasQTAgg):
    def __init__(self):
        self.figure = Figure(figsize=(8, 6), tight_layout=True)
        grid = self.figure.add_gridspec(2, 1, height_ratios=(1.15, 1.0))
        self.main_ax = self.figure.add_subplot(grid[0])
        self.neighbor_ax = self.figure.add_subplot(grid[1])
        super().__init__(self.figure)

class FastReview(QtWidgets.QMainWindow):
    def __init__(self, initial_recording: str | None = None):
        super().__init__()
        self.setWindowTitle("Fast manual review (non-destructive annotations)")
        self.resize(1550, 900)
        self.folder: Path | None = None
        self.journal = None
        self.assignments = self.base_assignments = None
        self.labels = {}
        self.times = self.amplitudes = None
        self.sample_ids = self.sample_waveforms = self.sample_channels = None
        self.main_cache_meta = self.main_cache = None
        self.reference_model = load_reference_model()
        self.reference_probabilities = {}
        self.current_cluster = None
        self.selection_bounds = None
        self._build()
        self._load_records(initial_recording)

    def _build(self):
        root = QtWidgets.QWidget(); self.setCentralWidget(root)
        layout = QtWidgets.QHBoxLayout(root)
        self.records = QtWidgets.QTreeWidget(); self.records.setHeaderLabels(["Device", "Recording", "Processing status", "Review queue", "Peak alignment"])
        self.records.setColumnWidth(0, 70); self.records.setColumnWidth(1, 115); self.records.setMinimumWidth(470)
        self.records.itemSelectionChanged.connect(self.load_record)
        layout.addWidget(self.records, 0)
        middle = QtWidgets.QVBoxLayout(); layout.addLayout(middle, 0)
        self.clusters = QtWidgets.QListWidget(); self.clusters.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.clusters.currentItemChanged.connect(self.draw_cluster); self.clusters.setMinimumWidth(260)
        middle.addWidget(QtWidgets.QLabel("Clusters (select multiple clusters to merge)")); middle.addWidget(self.clusters, 1)
        buttons = QtWidgets.QGridLayout(); middle.addLayout(buttons)
        for text, slot, row, col in [
            ("Good [G]", lambda:self.set_label("good"),0,0), ("Noise [N]",lambda:self.set_label("noise"),0,1),
            ("MUA [U]",lambda:self.set_label("mua"),1,0), ("Merge→Good",self.merge_selected,1,1),
            ("Selection -> Good\nRemainder -> Noise",lambda:self.split_selected("good","noise"),2,0),
            ("Selection -> Noise\nKeep remainder",lambda:self.split_selected("noise",None),2,1),
            ("Undo [Ctrl+Z]",self.undo,3,0), ("Confirm and mark ready to apply",self.confirm,3,1),
        ]:
            button=QtWidgets.QPushButton(text); button.clicked.connect(slot); buttons.addWidget(button,row,col)
        self.message=QtWidgets.QLabel("Actions are written only to Review_Queue; Phy results are not modified.")
        self.message.setWordWrap(True); middle.addWidget(self.message)
        plots=QtWidgets.QSplitter(QtCore.Qt.Vertical); layout.addWidget(plots,1)
        self.wave=WaveformCanvas(); self.amp=PlotCanvas(); plots.addWidget(self.wave); plots.addWidget(self.amp)
        self.selector=None
        QtWidgets.QShortcut("G",self,self._good); QtWidgets.QShortcut("N",self,self._noise)
        QtWidgets.QShortcut("U",self,self._mua); QtWidgets.QShortcut("Ctrl+Z",self,self.undo)

    def _good(self): self.set_label("good")
    def _noise(self): self.set_label("noise")
    def _mua(self): self.set_label("mua")

    def _load_records(self, initial_recording: str | None = None):
        self.records.clear()
        wanted=(initial_recording or "").replace("\\","/").lower()
        selected=None
        for params in sorted(WORKING_ROOT.rglob("params.py")):
            folder=params.parent
            if len(folder.relative_to(WORKING_ROOT).parts)!=2: continue
            if (folder/"FINAL_REFINEMENT_DONE.json").is_file(): status="Completed"
            else: status="Available"
            queue_path=journal_path(folder)
            if queue_path.is_file():
                try:
                    import json
                    review_status=json.loads(queue_path.read_text(encoding="utf-8")).get("review_status","")
                except (OSError,ValueError):
                    review_status="Invalid journal"
            else:
                review_status=""
            q={"draft":"Draft","ready_to_apply":"Ready to apply","applied":"Applied"}.get(review_status,review_status)
            alignment="Regenerate on demand"
            item=QtWidgets.QTreeWidgetItem([folder.parent.name,folder.name,status,q,alignment]); item.setData(0,QtCore.Qt.UserRole,str(folder))
            self.records.addTopLevelItem(item)
            if wanted==f"{folder.parent.name}/{folder.name}".lower(): selected=item
        if selected is not None:
            self.records.setCurrentItem(selected); selected.setSelected(True); self.records.scrollToItem(selected)

    def load_record(self):
        items=self.records.selectedItems()
        if not items:return
        try:
            folder=Path(items[0].data(0,QtCore.Qt.UserRole)); self.message.setText("Loading the validated Phy main-channel cache..."); QtWidgets.QApplication.processEvents()
            self.folder=folder; self.journal=load_journal(folder)
            self.base_assignments=np.load(folder/"spike_clusters.npy",mmap_mode="r").ravel()
            self.assignments,self.labels=replay(folder,self.journal["operations"])
            params=parse_params(folder/"params.py"); rate=float(params.get("sample_rate",30000))
            self.times=np.load(folder/"spike_times.npy",mmap_mode="r").ravel()/rate
            self.amplitudes=np.load(folder/"amplitudes.npy",mmap_mode="r").ravel()
            self.main_cache_meta,self.main_cache=load_main_channel_cache(folder)
            metrics={int(value.get("cluster_id",-1)):value for value in self.main_cache_meta.get("metrics",[])}
            self.reference_probabilities={cluster_id:predict_good(self.reference_model,metric)
                                          for cluster_id,metric in metrics.items()}
            self.refresh_clusters(); self.message.setText(f"Loaded {folder.parent.name}/{folder.name}; {len(self.journal['operations'])} pending operation(s)")
        except Exception as error:
            QtWidgets.QMessageBox.critical(self,"Load failed",str(error))

    def refresh_clusters(self, select=None):
        self.clusters.blockSignals(True); self.clusters.clear()
        for cid in np.unique(self.assignments):
            cid=int(cid); n=int(np.sum(self.assignments==cid)); group=self.labels.get(cid,"mua")
            probability=self.reference_probabilities.get(cid)
            reference="" if probability is None else f"  refGood={probability:.0%}"
            item=QtWidgets.QListWidgetItem(f"{cid:4d}  {group:5s}  n={n}{reference}"); item.setData(QtCore.Qt.UserRole,cid)
            item.setForeground({"good":QtCore.Qt.darkGreen,"noise":QtCore.Qt.darkRed}.get(group,QtCore.Qt.black))
            self.clusters.addItem(item)
            if select==cid:self.clusters.setCurrentItem(item)
        self.clusters.blockSignals(False)
        if self.clusters.currentItem() is None and self.clusters.count():self.clusters.setCurrentRow(0)
        self.draw_cluster()

    def selected_ids(self): return [int(x.data(QtCore.Qt.UserRole)) for x in self.clusters.selectedItems()]

    def draw_cluster(self,*_):
        item=self.clusters.currentItem()
        if item is None or self.assignments is None:return
        cid=int(item.data(QtCore.Qt.UserRole)); self.current_cluster=cid; mask=self.assignments==cid
        self.wave.main_ax.clear(); self.wave.neighbor_ax.clear()
        base_membership=np.asarray(self.base_assignments==cid)
        membership_unchanged=np.array_equal(np.asarray(mask),base_membership)
        ids_in_cache=np.asarray(self.main_cache["cluster_ids"],dtype=np.int64)
        matches=np.flatnonzero(ids_in_cache==cid)
        if matches.size and membership_unchanged:
            index=int(matches[0]); med=np.asarray(self.main_cache["medians"][index]); physical=int(self.main_cache["main_channels"][index])
            metric=next((value for value in self.main_cache_meta.get("metrics",[]) if int(value.get("cluster_id",-1))==cid),{})
            if not (physical==metric.get("phy_main_channel")==metric.get("display_channel")==metric.get("decision_channel")):
                raise RuntimeError(f"Main-channel consistency check failed for cluster {cid}")
            trace_rows=np.flatnonzero(np.asarray(self.main_cache["trace_clusters"])==cid)
            scale=float(self.main_cache_meta.get("scale_uV_per_bit",.195))
            rate=float(self.main_cache_meta.get("sample_rate_hz",30000.0))
            traces=np.asarray(self.main_cache["main_traces"][trace_rows],dtype=float)*scale
            if traces.shape[0]>100:traces=traces[np.linspace(0,traces.shape[0]-1,100).astype(int)]
            main=np.asarray(med[:,physical],dtype=float)*scale
            center=med.shape[0]//2
            time_ms=(np.arange(med.shape[0])-center)*1000.0/rate
            for trace in traces:
                self.wave.main_ax.plot(time_ms,trace,color="#8ec9ee",alpha=.10,lw=.55,zorder=1)
            self.wave.main_ax.plot(time_ms,main,alpha=1.0,lw=3.4,color="#087fdb",zorder=6)
            finite=main[np.isfinite(main)]
            if finite.size:
                lo=float(np.nanmin(finite)); hi=float(np.nanmax(finite)); span=max(hi-lo,2.0)
                self.wave.main_ax.set_ylim(lo-.28*span,hi+.28*span)
            self.wave.main_ax.axvline(0,color="gray",alpha=.4)
            self.wave.main_ax.axhline(0,color="gray",alpha=.18,lw=.7)
            self.wave.main_ax.set_xlabel("Time from aligned trough (ms)")
            self.wave.main_ax.set_ylabel("Amplitude (uV)")
            self.wave.main_ax.set_title(
                f"Cluster {cid} | Phy main channel {physical} | display/decision channel {physical} | "
                f"spikes {int(self.main_cache['sample_counts'][index])} | alignment {metric.get('status','')} shift {metric.get('applied_shift',0)}"
            )
            positions=np.asarray(self.main_cache.get("channel_positions"),dtype=float)
            if positions.ndim==2 and physical < len(positions):
                distances=np.linalg.norm(positions-positions[physical],axis=1)
                valid=[int(ch) for ch in np.argsort(distances) if ch<med.shape[1] and np.any(np.isfinite(med[:,ch]))][:5]
            else:
                distances=np.abs(np.arange(med.shape[1])-physical).astype(float); valid=[physical]
            neighbor_medians=np.asarray(med[:,valid],dtype=float)*scale
            ranges=np.nanmax(neighbor_medians,axis=0)-np.nanmin(neighbor_medians,axis=0)
            shared=max(float(np.nanmax(ranges)),2.0)
            row_gap=1.45*shared
            for row,channel in enumerate(valid):
                waveform=neighbor_medians[:,row]
                baseline=float(np.nanmedian(np.r_[waveform[:8],waveform[-8:]]))
                offset=(len(valid)-1-row)*row_gap
                color="#087fdb" if channel==physical else "#666666"
                width=2.8 if channel==physical else 1.25
                self.wave.neighbor_ax.plot(time_ms,waveform-baseline+offset,color=color,lw=width)
                self.wave.neighbor_ax.text(time_ms[0],offset+.12*shared,
                    f"Ch {channel}  ({distances[channel]:.0f} um)",color=color,fontsize=8,va="bottom")
            self.wave.neighbor_ax.set_yticks([])
            self.wave.neighbor_ax.axvline(0,color="gray",alpha=.35)
            self.wave.neighbor_ax.set_xlabel("Time from aligned trough (ms)")
            self.wave.neighbor_ax.set_ylabel("Adjacent channels (same uV scale, vertically offset)")
            self.wave.neighbor_ax.set_title(f"Main channel and {max(0,len(valid)-1)} nearest physical channel(s); blue denotes the main channel")
        else:
            reason="Cluster membership changed after merge/split; rebuild the main-channel cache after applying changes" if not membership_unchanged else "Validated Phy main-channel cache is unavailable"
            self.wave.main_ax.text(.5,.5,reason,ha="center",va="center",color="darkred",transform=self.wave.main_ax.transAxes)
            self.wave.neighbor_ax.text(.5,.5,"No inferred channel is displayed",ha="center",va="center",color="darkred",transform=self.wave.neighbor_ax.transAxes)
        self.wave.draw_idle()
        self.amp.ax.clear(); ids=np.flatnonzero(mask)
        if ids.size>25000: ids=ids[np.linspace(0,ids.size-1,25000).astype(int)]
        self.amp.ax.scatter(self.times[ids],self.amplitudes[ids],s=3,c="#159cff",alpha=.55,linewidths=0)
        self.amp.ax.set(title=f"Cluster {cid} | {int(mask.sum())} spikes",xlabel="Time (s)",ylabel="Amplitude")
        if self.selector is not None:self.selector.set_active(False)
        self.selector=RectangleSelector(self.amp.ax,self.rectangle_selected,useblit=True,button=[1],interactive=True,
                                        props=dict(facecolor="#20a5ff",edgecolor="white",alpha=.18))
        self.selection_bounds=None; self.amp.draw_idle()

    def add_operation(self,op):
        self.journal["operations"].append(op); self.journal["review_status"]="draft"; save_journal(self.folder,self.journal)
        self.assignments,self.labels=replay(self.folder,self.journal["operations"]); self.refresh_clusters(op.get("new_cluster_id",op.get("target",op.get("cluster_id"))))

    def set_label(self,group):
        ids=self.selected_ids() or ([self.current_cluster] if self.current_cluster is not None else [])
        for cid in ids:self.journal["operations"].append({"type":"label","cluster_id":cid,"group":group})
        if ids:self.journal["review_status"]="draft"; save_journal(self.folder,self.journal); self.assignments,self.labels=replay(self.folder,self.journal["operations"]); self.refresh_clusters(ids[0])

    def merge_selected(self):
        ids=self.selected_ids()
        if len(ids)<2: self.message.setText("Select at least two clusters before merging.") ; return
        self.add_operation({"type":"merge","sources":ids,"target":min(ids),"group":"good"})

    def rectangle_selected(self,start,end):
        if None in (start.xdata,start.ydata,end.xdata,end.ydata):return
        self.selection_bounds=[min(start.xdata,end.xdata),max(start.xdata,end.xdata),min(start.ydata,end.ydata),max(start.ydata,end.ydata)]
        self.message.setText("Selection complete; choose Selection -> Good or Selection -> Noise.")

    def split_selected(self,selected_group,remainder_group):
        if self.current_cluster is None or not self.selection_bounds:self.message.setText("Draw a selection box in the amplitude plot first.") ; return
        x0,x1,y0,y1=self.selection_bounds; source=int(self.current_cluster)
        mask=(self.assignments==source)&(self.times>=x0)&(self.times<=x1)&(self.amplitudes>=y0)&(self.amplitudes<=y1)
        ids=np.flatnonzero(mask)
        if ids.size==0 or ids.size==np.sum(self.assignments==source):self.message.setText("The selection must contain a proper subset of spikes from the current cluster.") ; return
        new_id=int(self.assignments.max())+1
        op={"type":"split","source":source,"new_cluster_id":new_id,"selected_spike_ids":ids.tolist(),"selected_group":selected_group,
            "remainder_group":remainder_group or self.labels.get(source,"mua"),"bounds":{"time":[x0,x1],"amplitude":[y0,y1]}}
        self.add_operation(op)

    def undo(self):
        if self.journal and self.journal["operations"]:
            self.journal["operations"].pop(); self.journal["review_status"]="draft"; save_journal(self.folder,self.journal)
            self.assignments,self.labels=replay(self.folder,self.journal["operations"]); self.refresh_clusters()

    def confirm(self):
        if not self.journal:return
        self.journal["review_status"]="ready_to_apply"; save_journal(self.folder,self.journal)
        self.message.setText("Marked ready to apply; the current Phy result remains unchanged.")
        item=self.records.currentItem(); item.setText(3,"Ready to apply")

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description="Fast non-destructive cluster review")
    parser.add_argument("--recording",help="DEVICE/RECORDING to select on launch")
    args=parser.parse_args()
    app=QtWidgets.QApplication([sys.argv[0]]); window=FastReview(args.recording); window.show(); raise SystemExit(app.exec_())
