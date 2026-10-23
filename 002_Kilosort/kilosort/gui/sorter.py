import logging
logger = logging.getLogger(__name__)

import torch
from qtpy import QtCore

from kilosort.run_kilosort import setup_logger, _sort

class KiloSortWorker(QtCore.QThread):
    finishedPreprocess = QtCore.Signal(object)
    finishedSpikesort = QtCore.Signal(object)
    finishedAll = QtCore.Signal(object)
    progress_bar = QtCore.Signal(int)
    plotDataReady = QtCore.Signal(str)

    def __init__(self, context, results_directory, steps,
                 device=torch.device('cuda'), file_object=None, *args, **kwargs):
        super(KiloSortWorker, self).__init__(*args, **kwargs)
        self.context = context
        self.data_path = context.data_path
        self.results_directory = results_directory
        assert isinstance(steps, list) or isinstance(steps, str)
        self.steps = steps if isinstance(steps, list) else [steps]
        self.device = device
        self.file_object = file_object

    def run(self):
        if "spikesort" in self.steps:
            settings = self.context.params
            probe = self.context.probe
            settings["data_dir"] = self.data_path[0].parent
            settings["filename"] = self.data_path
            results_dir = self.results_directory
            results_dir.mkdir(parents=True, exist_ok=True)
            setup_logger(results_dir)

            _ = _sort(
                settings['filename'], results_dir, probe, settings,
                settings['data_dtype'], self.device, settings['do_CAR'],
                settings['clear_cache'], settings['invert_sign'],
                settings['save_preprocessed_copy'], settings['verbose_log'],
                False, self.file_object, self.progress_bar, gui_sorter=self
                )

            self.finishedSpikesort.emit(self.context)
