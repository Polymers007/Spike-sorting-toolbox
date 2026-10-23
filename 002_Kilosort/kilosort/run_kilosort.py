import time
from pathlib import Path
import logging
import warnings
import platform
logger = logging.getLogger(__name__)

import numpy as np
import torch

import kilosort
from kilosort import (preprocessing, datashift, template_matching, clustering_qr, 
                      clustering_qr, io, spikedetect, CCG, PROBE_DIR)
from kilosort.parameters import DEFAULT_SETTINGS
from kilosort.utils import (
    log_performance, log_cuda_details, probe_as_string, ops_as_string,
    get_performance, log_sorting_summary, log_thread_count
    )
import kilosort.plots as kplots

RECOGNIZED_SETTINGS = list(DEFAULT_SETTINGS.keys())
RECOGNIZED_SETTINGS.extend([
    'filename', 'data_dir', 'results_dir', 'probe_name', 'probe_path',
])

GUI_SETTINGS = [
    'data_file_path', 'probe', 'data_dtype', 'save_preprocessed_copy',
    'clear_cache', 'do_CAR', 'invert_sign', 'verbose_log'
]

def run_kilosort(settings, probe=None, probe_name=None, filename=None,
                 data_dir=None, file_object=None, results_dir=None,
                 data_dtype=None, do_CAR=True, invert_sign=False, device=None,
                 progress_bar=None, save_extra_vars=False, clear_cache=False,
                 save_preprocessed_copy=False, bad_channels=None, shank_idx=None,
                 verbose_console=False, verbose_log=False, torch_thread_lim=None):
    pass
    if torch_thread_lim is not None:
        torch.set_num_threads(torch_thread_lim)

    if settings is None or settings.get('n_chan_bin', None) is None:
        raise ValueError(
            '`n_chan_bin` is a required setting. This is the total number of '
            'channels in the binary file, which may or may not be equal to the '
            'number of channels specified by the probe.'
            )
    settings = {**DEFAULT_SETTINGS, **settings}
    
    if not isinstance(shank_idx, list): shank_idx = [shank_idx]
    for idx in shank_idx:
        _filename, _data_dir, _results_dir, _probe = \
            set_files(settings, filename, probe, probe_name, data_dir,
                      results_dir, bad_channels, idx)
        setup_logger(_results_dir, verbose_console=verbose_console)

        ops, st, clu, tF, Wall, similar_templates, \
            is_ref, est_contam_rate, kept_spikes = _sort(
                _filename, _results_dir, _probe, settings, data_dtype, device,
                do_CAR, clear_cache, invert_sign, save_preprocessed_copy,
                verbose_log, save_extra_vars, file_object, progress_bar,
            )

    return ops, st, clu, tF, Wall, similar_templates, \
           is_ref, est_contam_rate, kept_spikes

def _sort(filename, results_dir, probe, settings, data_dtype, device, do_CAR,
          clear_cache, invert_sign, save_preprocessed_copy, verbose_log,
          save_extra_vars, file_object, progress_bar, gui_sorter=None):
    pass

    try:
        logger.info(f"Kilosort version {kilosort.__version__}")
        logger.info(f"Python version {platform.python_version()}")
        logger.info('-'*40)

        logger.info('System information:')
        logger.info(f'{platform.platform()} {platform.machine()}')
        logger.info(platform.processor())
        if device is None:
            if torch.cuda.is_available():
                logger.info('Using GPU for PyTorch computations. '
                            'Specify `device` to change this.')
                device = torch.device('cuda')
            else:
                logger.info('Using CPU for PyTorch computations. '
                            'Specify `device` to change this.')
                device = torch.device('cpu')

        if device != torch.device('cpu'):
            memory = torch.cuda.get_device_properties(device).total_memory/1024**3
            logger.info(f'Using CUDA device: {torch.cuda.get_device_name()} {memory:.2f}GB')

        logger.info('-'*40)
        if len(filename) == 1:
            logger.info(f"Sorting {filename}")
        else:
            logger.info(f"Sorting {filename[0].parent}/... (multiple files)")

        if data_dtype is None:
            logger.info(
                "Interpreting binary file as default dtype='int16'. If data was "
                "saved in a different format, specify `data_dtype`."
                )
            data_dtype = 'int16'

        if not do_CAR:
            logger.info("Skipping common average reference.")
        if clear_cache:
            logger.info('clear_cache=True')

        if probe['chanMap'].max() >= settings['n_chan_bin']:
            raise ValueError(
                f'Largest value of chanMap exceeds channel count of data, '
                'make sure chanMap is 0-indexed.'
            )

        tic0 = time.time()
        ops, settings = initialize_ops(
            settings, probe, data_dtype, do_CAR, invert_sign,
            device, save_preprocessed_copy, gui_mode=(gui_sorter is not None)
            )

        logger.debug(f"Initial ops:\n\n{ops_as_string(ops)}\n")
        logger.debug(f"Probe dictionary:\n\n{probe_as_string(ops['probe'])}\n")

        log_performance(logger, 'info', 'Resource usage before sorting')
        log_thread_count(logger)

        ops = compute_preprocessing(ops, device, tic0=tic0, file_object=file_object)
        np.random.seed(1)
        torch.cuda.manual_seed_all(1)
        torch.random.manual_seed(1) 
        ops, bfile, st0 = compute_drift_correction(
            ops, device, tic0=tic0, progress_bar=progress_bar,
            file_object=file_object, clear_cache=clear_cache,
            verbose=verbose_log
            )

        log_thread_count(logger)

        if save_preprocessed_copy:
            io.save_preprocessing(results_dir / 'temp_wh.dat', ops, bfile)
            log_performance(logger, 'info', 'Resource usage after saving preprocessing.',
                            reset=True)

        logger.info('Generating drift plots ...')
        
        if st0 is not None:
            if gui_sorter is not None:
                gui_sorter.dshift = ops['dshift']
                gui_sorter.st0 = st0
                gui_sorter.plotDataReady.emit('drift')
            else:
                kplots.plot_drift_amount(ops, results_dir, tmin=settings['tmin'])
                kplots.plot_drift_scatter(st0, results_dir, tmin=settings['tmin'])

        st,tF, Wall0, clu0 = detect_spikes(
            ops, device, bfile, tic0=tic0, progress_bar=progress_bar,
            clear_cache=clear_cache, verbose=verbose_log
            )

        log_thread_count(logger)

        logger.info('Generating diagnostic plots ...')
        if gui_sorter is not None:
            gui_sorter.Wall0 = Wall0
            gui_sorter.wPCA = torch.clone(ops['wPCA'].cpu()).numpy()
            gui_sorter.clu0 = clu0
            gui_sorter.plotDataReady.emit('diagnostics')
        else:
            kplots.plot_diagnostics(Wall0, clu0, ops, results_dir)

        clu, Wall, st, tF = cluster_spikes(
            st, tF, ops, device, bfile, tic0=tic0, progress_bar=progress_bar,
            clear_cache=clear_cache, verbose=verbose_log,
            )

        log_thread_count(logger)

        ops, similar_templates, is_ref, est_contam_rate, kept_spikes = \
            save_sorting(
                ops, results_dir, st, clu, tF, Wall, bfile.imin, tic0,
                save_extra_vars=save_extra_vars,
                save_preprocessed_copy=save_preprocessed_copy,
                skip_dat_path=(file_object is not None)
                )
        if torch.cuda.is_available():
            ops['cuda_postproc'] = torch.cuda.memory_stats(device)

        log_thread_count(logger)

        logger.info('Generating spike position plot ...')
        if gui_sorter is not None:
            gui_sorter.clu = clu[kept_spikes]
            gui_sorter.is_refractory = is_ref
            gui_sorter.plotDataReady.emit('probe')
        else:
            kplots.plot_spike_positions(clu[kept_spikes], is_ref, results_dir)
        logger.info('Sorting finished.')
        log_sorting_summary(ops, log=logger, level='info')
        
    except Exception as e:
        if isinstance(e, torch.cuda.OutOfMemoryError):
            logger.exception('Out of memory error, printing performance...')
            log_cuda_details(logger)
            log_performance(logger, level='info')

        logger.exception('Encountered error in `run_kilosort`:')

        raise
    
    finally:
        close_logger()

    return ops, st, clu, tF, Wall, similar_templates, \
           is_ref, est_contam_rate, kept_spikes

def set_files(settings, filename, probe, probe_name, data_dir, results_dir,
              bad_channels, shank_idx):
    pass

    filename = settings.get('filename', None) if filename is None else filename 

    if filename is None:
        data_dir = settings.get('data_dir', None) if data_dir is None else data_dir
        if data_dir is None:
            raise ValueError('no path to data provided, set "data_dir=" or "filename="')
        data_dir = Path(data_dir).resolve()
        if not data_dir.exists():
            raise FileExistsError(f"data_dir '{data_dir}' does not exist")

        filename  = io.find_binary(data_dir=data_dir)
        filename = [filename]
    else:
        if not isinstance(filename, list):
            filename = [filename]
        filename = [Path(f) for f in filename]
        for f in filename:
            if not f.exists():
                raise FileExistsError(f"filename '{filename}' does not exist")
        data_dir = filename[0].parent

    settings['filename'] = filename
    settings['data_dir'] = data_dir

    results_dir = settings.get('results_dir', None) if results_dir is None else results_dir
    results_dir = Path(results_dir).resolve() if results_dir is not None else None
    if results_dir is None:
        from .result_paths import default_result_directory
        results_dir = default_result_directory(filename)
    if shank_idx is not None:
        results_dir = results_dir / f'shank_{shank_idx}'
    
    results_dir.mkdir(exist_ok=True, parents=True)

    if probe is None:
        if probe_name is not None:     probe_path = PROBE_DIR / probe_name
        elif 'probe_name' in settings: probe_path = PROBE_DIR / settings['probe_name']
        elif 'probe_path' in settings: probe_path = Path(settings['probe_path']).resolve()
        else: raise ValueError('no probe_name or probe_path provided, set probe_name=')
        if not probe_path.exists():
            raise FileExistsError(f"probe_path '{probe_path}' does not exist")
        
        probe  = io.load_probe(probe_path)
    else:

        probe['xc'] = probe['xc'].astype(np.float32)
        probe['yc'] = probe['yc'].astype(np.float32)

    for k in ['xc', 'yc', 'kcoords', 'chanMap']:
        if probe[k].ndim > 1:
            raise ValueError(f"Array-valued probe entries should have 1 dim, "
                             f"but key: {k} has ndim == {probe[k].ndim}.")

    if bad_channels is not None:
        probe = io.remove_bad_channels(probe, bad_channels)
    if shank_idx is not None:
        probe = io.select_shank(probe, shank_idx)

    return filename, data_dir, results_dir, probe

def setup_logger(results_dir, verbose_console=False):
    results_dir = Path(results_dir)

    ks_log = logging.getLogger('kilosort')
    ks_log.setLevel(logging.DEBUG)

    file = logging.FileHandler(results_dir / 'kilosort4.log', mode='w')
    file.setLevel(logging.DEBUG)
    text_format = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    file_formatter = logging.Formatter(text_format)
    file.setFormatter(file_formatter)

    if not ks_log.handlers:

        console = logging.StreamHandler()
        if verbose_console:
            console.setLevel(logging.DEBUG)
            console.setFormatter(file_formatter)
        else:
            console.setLevel(logging.INFO)
            console_formatter = logging.Formatter('%(name)-12s: %(message)s')
            console.setFormatter(console_formatter)
        ks_log.addHandler(console)

    ks_log.addHandler(file)

def close_logger():
    ks_log = logging.getLogger('kilosort')
    for handler in ks_log.handlers.copy():
        ks_log.removeHandler(handler)
        handler.close()

def initialize_ops(settings, probe, data_dtype, do_CAR, invert_sign,
                   device, save_preprocessed_copy, gui_mode=False) -> dict:
    pass

    settings = settings.copy()
    if settings['nt'] % 2 == 0:
        raise ValueError(f'`nt` must be odd, but got nt={settings["nt"]}')
    if settings['nt0min'] is None:
        settings['nt0min'] = int(20 * settings['nt']/61)
    if settings['max_channel_distance'] is None:

        settings['max_channel_distance'] = DEFAULT_SETTINGS['max_channel_distance']

    if settings['nearest_chans'] > len(probe['chanMap']):
        msg = f"""
            Parameter `nearest_chans` must be less than or equal to the number 
            of data channels being sorted.\n
            Changing from {settings['nearest_chans']} to {len(probe['chanMap'])}.
            """
        warnings.warn(msg, UserWarning)
        settings['nearest_chans'] = len(probe['chanMap'])

    if 'duplicate_spike_bins' in settings:
        msg = """
            The `duplicate_spike_bins` parameter has been replaced with 
            `duplicate_spike_ms`. Specifying the former will have no effect, 
            since it gets overwritten based on sampling rate.
            """
        warnings.warn(msg, DeprecationWarning)
    dup_bins = int(settings['duplicate_spike_ms'] * (settings['fs']/1000))

    recognized = RECOGNIZED_SETTINGS.copy()
    if gui_mode:
        recognized.extend(GUI_SETTINGS.copy())

    unrecognized = []
    for k, _ in settings.items():
        if k not in recognized:
            unrecognized.append(k)
    if len(unrecognized) > 0:
        logger.info('Unrecognized keys found in `settings`')
        logger.info('See `kilosort.run_kilosort.RECOGNIZED_SETTINGS`')
        raise ValueError(f'Unrecognized settings: {unrecognized}')

    ops = settings.copy()
    ops['settings'] = settings
    ops['probe'] = probe
    ops['data_dtype'] = data_dtype
    ops['do_CAR'] = do_CAR
    ops['invert_sign'] = invert_sign
    ops['NTbuff'] = ops['batch_size'] + 2 * ops['nt']
    ops['Nchan'] = len(probe['chanMap'])
    ops['n_chan_bin'] = settings['n_chan_bin']
    ops['duplicate_spike_bins'] = dup_bins
    ops['torch_device'] = str(device)
    ops['save_preprocessed_copy'] = save_preprocessed_copy

    if not settings['templates_from_data'] and settings['nt'] != 61:
        raise ValueError('If using pre-computed universal templates '
                         '(templates_from_data=False), nt must be 61')

    ops = {**ops, **probe}

    return ops, settings

def get_run_parameters(ops) -> list:
    pass

    parameters = [
        ops['settings']['n_chan_bin'],
        ops['settings']['fs'],
        ops['settings']['batch_size'],
        ops['settings']['nt'],
        ops['settings']['nt0min'],  
        ops['probe']['chanMap'],
        ops['data_dtype'],
        ops['do_CAR'],
        ops['invert_sign'],
        ops['probe']['xc'],
        ops['probe']['yc'],
        ops['settings']['tmin'],
        ops['settings']['tmax'],
        ops['settings']['artifact_threshold'],
        ops['settings']['shift'],
        ops['settings']['scale'],
        ops['settings']['batch_downsampling']
    ]

    return parameters

def compute_preprocessing(ops, device, tic0=np.nan, file_object=None):
    pass

    tic = time.time()
    logger.info(' ')
    logger.info('Computing preprocessing variables.')
    logger.info('-'*40)

    n_chan_bin, fs, NT, nt, twav_min, chan_map, dtype, do_CAR, invert, \
        xc, yc, tmin, tmax, artifact, shift, scale, batch_downsampling = \
            get_run_parameters(ops)
    nskip = ops['settings']['nskip']
    whitening_range = ops['settings']['whitening_range']

    cutoff = ops['settings']['highpass_cutoff']
    hp_filter = preprocessing.get_highpass_filter(fs, cutoff, device=device)
    
    bfile = io.BinaryFiltered(ops['filename'], n_chan_bin, fs, NT, nt, twav_min,
                              chan_map, hp_filter, device=device, do_CAR=do_CAR,
                              invert_sign=invert, dtype=dtype, tmin=tmin,
                              tmax=tmax, artifact_threshold=artifact,
                              shift=shift, scale=scale, file_object=file_object,
                              batch_downsampling=batch_downsampling)

    logger.info(f'N samples: {bfile.n_samples}')
    logger.info(f'N seconds: {bfile.n_samples/fs}')
    logger.info(f'N batches: {bfile.n_batches}')

    whiten_mat = preprocessing.get_whitening_matrix(bfile, xc, yc, nskip=nskip,
                                                    nrange=whitening_range)

    ops['Nbatches'] = bfile.n_batches
    ops['preprocessing'] = {}
    ops['preprocessing']['whiten_mat'] = whiten_mat
    ops['preprocessing']['hp_filter'] = hp_filter
    ops['Wrot'] = whiten_mat
    ops['fwav'] = hp_filter

    elapsed = time.time() - tic
    total = time.time() - tic0
    ops['runtime_preproc'] = elapsed
    ops['usage_preproc'] = get_performance()
    logger.info(f'Preprocessing filters computed in {elapsed:.2f}s; ' +
                f'total {total:.2f}s')
    logger.debug(f'hp_filter shape: {hp_filter.shape}')
    logger.debug(f'whiten_mat shape: {whiten_mat.shape}')
    
    b1 = bfile.padded_batch_to_torch(0).cpu().numpy()
    logger.debug(f"First batch min, max: {b1.min(), b1.max()}")

    log_performance(logger, 'info', 'Resource usage after preprocessing',
                    reset=True)

    return ops

def compute_drift_correction(ops, device, tic0=np.nan, progress_bar=None,
                             file_object=None, clear_cache=False, verbose=False):
    pass

    tic = time.time()
    logger.info(' ')
    logger.info('Computing drift correction.')
    logger.info('-'*40)

    n_chan_bin, fs, NT, nt, twav_min, chan_map, dtype, do_CAR, invert, \
        _, _, tmin, tmax, artifact, shift, scale, batch_downsampling = \
            get_run_parameters(ops)
    hp_filter = ops['preprocessing']['hp_filter']
    whiten_mat = ops['preprocessing']['whiten_mat']
    bfile = io.BinaryFiltered(
        ops['filename'], n_chan_bin, fs, NT, nt, twav_min, chan_map, 
        hp_filter=hp_filter, whiten_mat=whiten_mat, device=device, do_CAR=do_CAR,
        invert_sign=invert, dtype=dtype, tmin=tmin, tmax=tmax,
        artifact_threshold=artifact, shift=shift, scale=scale,
        file_object=file_object, batch_downsampling=batch_downsampling
        )

    ops, st = datashift.run(ops, bfile, device=device, progress_bar=progress_bar,
                            clear_cache=clear_cache, verbose=verbose)
    
    elapsed = time.time() - tic
    total = time.time() - tic0
    ops['runtime_drift'] = elapsed
    ops['usage_drift'] = get_performance()
    if torch.cuda.is_available():
        ops['cuda_drift'] = torch.cuda.memory_stats()
    logger.info(f'drift computed in {elapsed:.2f}s; total {total:.2f}s')

    if st is not None:
        logger.debug(f'st shape: {st.shape}')
        logger.debug(f'yblk shape: {ops["yblk"].shape}')
        logger.debug(f'dshift shape: {ops["dshift"].shape}')
        logger.debug(f'iKxx shape: {ops["iKxx"].shape}')

    bfile = io.BinaryFiltered(
        ops['filename'], n_chan_bin, fs, NT, nt, twav_min, chan_map, 
        hp_filter=hp_filter, whiten_mat=whiten_mat, device=device,
        dshift=ops['dshift'], do_CAR=do_CAR, dtype=dtype, tmin=tmin, tmax=tmax,
        artifact_threshold=artifact, shift=shift, scale=scale,
        file_object=file_object, batch_downsampling=batch_downsampling
        )

    log_cuda_details(logger)
    log_performance(logger, 'info', 'Resource usage after drift correction',
                    reset=True)

    return ops, bfile, st

def detect_spikes(ops, device, bfile, tic0=np.nan, progress_bar=None,
                  clear_cache=False, verbose=False):
    pass

    tic = time.time()
    logger.info(' ')
    logger.info(f'Extracting spikes using templates')
    logger.info('-'*40)
    st0, tF, ops = spikedetect.run(
        ops, bfile, device=device, progress_bar=progress_bar,
        clear_cache=clear_cache, verbose=verbose
        )
    tF = torch.from_numpy(tF)

    elapsed = time.time() - tic
    total = time.time() - tic0
    ops['runtime_st0'] = elapsed
    ops['usage_st0'] = get_performance()
    if torch.cuda.is_available():
        ops['cuda_st0'] = torch.cuda.memory_stats(device)
    logger.info(f'{len(st0)} spikes extracted in {elapsed:.2f}s; ' + 
                f'total {total:.2f}s')
    logger.debug(f'st0 shape: {st0.shape}')
    logger.debug(f'tF shape: {tF.shape}')
    if len(st0) == 0:
        raise ValueError('No spikes detected, cannot continue sorting.')
    log_performance(logger, 'info', 'Resource usage after spike detect (univ)',
                    reset=True)
    log_thread_count(logger)

    tic = time.time()
    logger.info(' ')
    logger.info('First clustering')
    logger.info('-'*40)
    clu, Wall = clustering_qr.run(
        ops, st0, tF, mode='spikes', device=device, progress_bar=progress_bar,
        clear_cache=clear_cache, verbose=verbose
        )
    Wall3 = template_matching.postprocess_templates(
        Wall, ops, clu, st0, tF, device=device
        )

    elapsed = time.time() - tic
    total = time.time() - tic0
    ops['runtime_clu0'] = elapsed
    ops['usage_clu0'] = get_performance()
    if torch.cuda.is_available():
        ops['cuda_clu0'] = torch.cuda.memory_stats(device)
    logger.info(f'{clu.max()+1} clusters found, in {elapsed:.2f}s; ' +
                f'total {total:.2f}s')
    logger.debug(f'clu shape: {clu.shape}')
    logger.debug(f'Wall shape: {Wall.shape}')
    log_performance(logger, 'info', 'Resource usage after first clustering',
                    reset=True)
   
    log_thread_count(logger)

    tic = time.time()
    logger.info(' ')
    logger.info('Extracting spikes using cluster waveforms')
    logger.info('-'*40)
    st, tF, ops = template_matching.extract(
        ops, bfile, Wall3, device=device, progress_bar=progress_bar
        )
   
    log_thread_count(logger)

    elapsed = time.time() - tic
    total = time.time() - tic0
    ops['runtime_st'] = elapsed
    ops['usage_st'] = get_performance()
    if torch.cuda.is_available():
        ops['cuda_st'] = torch.cuda.memory_stats(device)
    logger.info(f'{len(st)} spikes extracted in {elapsed:.2f}s; ' +
                f'total {total:.2f}s')
    logger.debug(f'st shape: {st.shape}')
    logger.debug(f'tF shape: {tF.shape}')
    logger.debug(f'iCC shape: {ops["iCC"].shape}')
    logger.debug(f'iU shape: {ops["iU"].shape}')

    log_cuda_details(logger)
    log_performance(logger, 'info', 'Resource usage after spike detect (learned)',
                    reset=True)

    return st, tF, Wall, clu

def cluster_spikes(st, tF, ops, device, bfile, tic0=np.nan, progress_bar=None,
                   clear_cache=False, verbose=False):
    pass
    tic = time.time()
    logger.info(' ')
    logger.info('Final clustering')
    logger.info('-'*40)
    clu, Wall = clustering_qr.run(
        ops, st, tF,  mode = 'template', device=device, progress_bar=progress_bar,
        clear_cache=clear_cache, verbose=verbose
        )
    
    elapsed = time.time() - tic
    total = time.time() - tic0
    ops['runtime_clu'] = elapsed
    ops['usage_clu'] = get_performance()
    if torch.cuda.is_available():
        ops['cuda_clu'] = torch.cuda.memory_stats(device)
    logger.info(f'{clu.max()+1} clusters found, in {elapsed:.2f}s; ' + 
                f'total {total:.2f}s')
    logger.debug(f'clu shape: {clu.shape}')
    logger.debug(f'Wall shape: {Wall.shape}')
    log_thread_count(logger)

    tic = time.time()
    logger.info(' ')
    logger.info('Merging clusters')
    logger.info('-'*40)
    Wall, clu, is_ref, st, tF = template_matching.merging_function(
        ops, Wall, clu, st, tF, device=device, check_dt=True
        )
    clu = clu.astype('int32')

    elapsed = time.time() - tic
    total = time.time() - tic0
    ops['runtime_merge'] = elapsed
    ops['usage_merge'] = get_performance()
    if torch.cuda.is_available():
        ops['cuda_merge'] = torch.cuda.memory_stats(device)
    logger.info(f'{clu.max()+1} units found, in {elapsed:.2f}s; ' + 
                f'total {total:.2f}s')
    logger.debug(f'clu shape: {clu.shape}')
    logger.debug(f'Wall shape: {Wall.shape}')

    log_cuda_details(logger)
    log_performance(logger, 'info', 'Resource usage after clustering',
                    reset=True)

    return clu, Wall, st, tF

def save_sorting(ops, results_dir, st, clu, tF, Wall, imin, tic0=np.nan,
                 save_extra_vars=False, save_preprocessed_copy=False,
                 skip_dat_path=False):  
    pass

    tic = time.time()
    logger.info(' ')
    logger.info('Saving to phy and computing refractory periods')
    logger.info('-'*40)
    results_dir, similar_templates, is_ref, est_contam_rate, kept_spikes = \
        io.save_to_phy(
            st, clu, tF, Wall, ops['probe'], ops, imin, results_dir=results_dir,
            data_dtype=ops['data_dtype'], save_extra_vars=save_extra_vars,
            save_preprocessed_copy=save_preprocessed_copy,
            skip_dat_path=skip_dat_path
            )
    logger.info(f'{int(is_ref.sum())} units found with good refractory periods')
    
    ops['n_units_total'] = np.unique(clu).size
    ops['n_units_good'] = int(is_ref.sum())
    ops['n_spikes'] = st[kept_spikes].shape[0]
    if ops.get('dshift', None) is not None:
        ops['mean_drift'] = np.abs(ops['dshift']).mean(axis=0)[0]
    else:
        ops['mean_drift'] = np.nan

    elapsed = elapsed = time.time() - tic
    ops['runtime_postproc'] = elapsed
    ops['usage_postproc'] = get_performance()
    logger.info(f'Exporting to Phy took: {elapsed:.2f}s')

    runtime = time.time()-tic0
    seconds = runtime % 60
    mins = runtime // 60
    hrs = mins // 60
    mins = mins % 60

    logger.info(f'Total runtime: {runtime:.2f}s = {int(hrs):02d}:' +
                f'{int(mins):02d}:{round(seconds)} h:m:s')
    ops['runtime'] = runtime 
    io.save_ops(ops, results_dir)
    logger.info(f'Sorting output saved in: {results_dir}.')

    log_cuda_details(logger)
    log_performance(logger, 'info', 'Resource usage after saving',
                    reset=True)

    return ops, similar_templates, is_ref, est_contam_rate, kept_spikes

def load_sorting(results_dir, device=None, load_extra_vars=False):
    pass
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

    results_dir = Path(results_dir)
    ops = io.load_ops(results_dir / 'ops.npy', device=device)
    similar_templates = np.load(results_dir / 'similar_templates.npy')

    clu = np.load(results_dir / 'spike_clusters.npy')
    st = np.load(results_dir / 'spike_times.npy')
    kept_spikes = np.load(results_dir / 'kept_spikes.npy')
    acg_threshold = ops['settings']['acg_threshold']
    ccg_threshold = ops['settings']['ccg_threshold']
    is_ref, est_contam_rate = CCG.refract(clu, st / ops['fs'],
                                          acg_threshold=acg_threshold,
                                          ccg_threshold=ccg_threshold)

    results = [ops, st, clu, similar_templates, is_ref,
               est_contam_rate, kept_spikes]

    if load_extra_vars:
        
        tF = np.load(results_dir / 'tF.npy')
        tF = torch.from_numpy(tF)
        Wall = np.load(results_dir / 'Wall.npy')
        Wall = torch.from_numpy(Wall)
        full_st = np.load(results_dir / 'full_st.npy')
        full_clu = np.load(results_dir / 'full_clu.npy')
        full_amp = np.load(results_dir / 'full_amp.npy')
        results.extend([tF, Wall, full_st, full_clu, full_amp])

    return results
