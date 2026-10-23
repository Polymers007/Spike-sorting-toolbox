from pathlib import Path

import pyqtgraph as pg
import pyqtgraph.exporters
import numpy as np
import matplotlib
import torch
from qtpy import QtWidgets

from kilosort.plots import COLOR_CODES, PROBE_PLOT_COLORS

class PlotWindow(QtWidgets.QWidget):
    def __init__(self, *args, title=None, width=500, height=400,
                 background=None, **kwargs):
        super().__init__()
        if title is not None:
            self.setWindowTitle(title)
        self.resize(width, height)

        layout = QtWidgets.QVBoxLayout()
        self.plot_widget = pg.GraphicsLayoutWidget(parent=self)
        if background is not None:
            self.plot_widget.setBackground(background)
        layout.addWidget(self.plot_widget)
        layout.setContentsMargins(0,0,0,0)
        self.setLayout(layout)

        self.hide()

def plot_drift_amount(plot_window, dshift, settings):
    
    p1 = plot_window.plot_widget.addPlot(
        row=0, col=0, labels={'left': 'Depth shift (um)', 'bottom': 'Time (s)'}
        )
    p1.setTitle('Drift amount per probe section, across batches')
    fs = settings['fs']
    NT = settings['batch_size']
    t = np.arange(dshift.shape[0])*(NT/fs) + settings['tmin']

    for i in range(dshift.shape[1]):
        color = COLOR_CODES[i % len(COLOR_CODES)]
        p1.plot(t, dshift[:,i], pen=color)

    plot_window.show()
    save_path = str(Path(settings['results_dir']) / 'drift_amount.png')
    pg.exporters.ImageExporter(plot_window.plot_widget.scene()).export(save_path)

def plot_drift_scatter(plot_window, st0, settings):
    
    p1 = plot_window.plot_widget.addPlot(
        row=0, col=0, labels={'left': 'Depth (um)', 'bottom': 'Time (s)'}
    )
    p1.setTitle('Spike amplitude across time and depth', color='black')

    x = st0[:,0] + settings['tmin']  
    y = st0[:,1]                     
    z = st0[:,2]                     
    z[z < 10] = 10
    z[z > 100] = 100

    bin_idx = np.digitize(z, np.logspace(1, 2, 90))
    cm = matplotlib.colormaps['binary']
    brushes = np.empty_like(z, dtype=object)
    pens = np.empty_like(z, dtype=object)
    for i in np.unique(bin_idx):
        
        subset = (bin_idx == i)
        a = z[subset].mean()
        rgba = cm(((a-10)/90))
        
        rgba = tuple([c*255 for c in rgba])
        brush = pg.mkBrush(rgba)
        brushes[subset] = brush
        pen = pg.mkPen(rgba)
        pens[subset] = pen

    scatter = pg.ScatterPlotItem(x, y, symbol='o', size=3, pen=None,
                                 brush=brushes)
    p1.addItem(scatter)
    
    p1.getViewBox().setBackgroundColor('w')
    p1.getViewBox().invertY(True)
    bottom_ax = p1.getAxis('bottom')
    bottom_ax.setPen('k')
    bottom_ax.setTextPen('k')
    bottom_ax.setTickPen('k')
    left_ax = p1.getAxis('left')
    left_ax.setPen('k')
    left_ax.setTextPen('k')
    left_ax.setTickPen('k')

    plot_window.show()
    save_path = str(Path(settings['results_dir']) / 'drift_scatter.png')
    pg.exporters.ImageExporter(plot_window.plot_widget.scene()).export(save_path)

def plot_diagnostics(plot_window, wPCA, Wall0, clu0, settings):
    
    p1 = plot_window.plot_widget.addPlot(
        row=0, col=0, labels={'bottom': 'Time (s)'}
        )
    p1.setTitle('Temporal Features')
    t = np.arange(wPCA.shape[1])/(settings['fs']/1000)
    for i in range(wPCA.shape[0]):
        color = COLOR_CODES[i % len(COLOR_CODES)]
        p1.plot(t, wPCA[i,:], pen=color)

    p2 = plot_window.plot_widget.addPlot(
        row=0, col=1, labels={'bottom': 'Channel Number', 'left': 'Unit Number'}
        )
    p2.setTitle('Spatial Features')
    features = torch.linalg.norm(Wall0, dim=2).cpu().numpy()
    img = pg.ImageItem(image=features.T)
    img.setLevels([0, 25])
    p2.addItem(img)

    n_units = int(clu0.max()) + 1
    spike_counts = np.zeros(n_units)
    for i in range(n_units):
        spike_counts[i] = (clu0[clu0 == i]).size
    mean_amp = torch.linalg.norm(Wall0, dim=(1,2)).cpu().numpy()

    p3 = plot_window.plot_widget.addPlot(
        row=1, col=0, labels={'bottom': 'Unit Number', 'left': 'Amplitude (a.u.)'}
        )
    p3.setTitle('Unit Amplitudes')
    p3.plot(mean_amp)

    p4 = plot_window.plot_widget.addPlot(
        row=1, col=1,
        labels={'bottom': 'Log(1 + Spike Count)', 'left': 'Amplitude (a.u.)'}
        )
    p4.setTitle('Amplitude vs Spike Count')
    scatter = pg.ScatterPlotItem(np.log(1 + spike_counts), mean_amp,
                                 symbol='o', size=3)
    p4.addItem(scatter)

    plot_window.show()
    save_path = str(Path(settings['results_dir']) / 'diagnostics.png')
    pg.exporters.ImageExporter(plot_window.plot_widget.scene()).export(save_path)

def plot_spike_positions(plot_window, clu, is_refractory, settings):

    p1 = plot_window.plot_widget.addPlot(
        row=0, col=0, labels={'bottom': 'Depth (um)', 'left': 'Lateral (um)'}
    )
    p1.setTitle('Spike position across probe, colored by cluster')

    clu = clu.copy()
    bad_units = np.unique(clu)[is_refractory == 0]
    bad_idx = np.isin(clu, bad_units)
    clu = np.mod(clu, 9)
    clu[bad_idx] = 9
    cm = (PROBE_PLOT_COLORS*255).astype('int')  

    brushes = np.empty_like(clu, dtype=object)
    pens = np.empty_like(clu, dtype=object)
    for i in range(10):
        subset = (clu == i)
        rgba = cm[i]
        brush = pg.mkBrush(rgba)
        brushes[subset] = brush
        pen = pg.mkPen(rgba)
        pens[subset] = pen

    results_dir = Path(settings['results_dir'])
    positions = np.load(results_dir / 'spike_positions.npy')
    xs, ys = positions[:,0], positions[:,1]
    scatter = pg.ScatterPlotItem(ys, xs, symbol='o', size=3, pen=None,
                                 brush=brushes)
    p1.addItem(scatter)
    plot_window.show()
    save_path = str(results_dir / 'spike_positions.png')
    pg.exporters.ImageExporter(plot_window.plot_widget.scene()).export(save_path)
