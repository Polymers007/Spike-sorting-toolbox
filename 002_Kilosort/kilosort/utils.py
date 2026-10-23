import os, sys, tempfile, shutil, pathlib, psutil
import subprocess
import time
import urllib
import importlib.util
import logging
import pprint
logger = logging.getLogger(__name__)

import numpy as np
from tqdm import tqdm

import torch

if importlib.util.find_spec('pynvml') is not None:
    _NVML_EXISTS = True
else:
    _NVML_EXISTS = False

_DOWNLOADS_DIR_ENV = os.environ.get("KILOSORT_LOCAL_DOWNLOADS_PATH")
_DOWNLOADS_DIR_DEFAULT = pathlib.Path(__file__).resolve().parents[1] / '.kilosort'
DOWNLOADS_DIR = pathlib.Path(_DOWNLOADS_DIR_ENV) if _DOWNLOADS_DIR_ENV else _DOWNLOADS_DIR_DEFAULT
PROBE_DIR = pathlib.Path(__file__).resolve().parents[1] / 'probes'
PROBE_URLS = {
    
    'Linear16x1_test.mat': 'https://osf.io/download/67f012cbc56bef203cb25416/',
    
    'NeuroPix1_default.mat': 'https://osf.io/download/67f012cc7e1fd38cad82980a/',
    
    'NeuroPix2_default.mat': 'https://osf.io/download/67f012ce033d25194f829812/'
}

def template_path(basename='wTEMP.npz'):
    pass
    return cache_template_path(basename)

def cache_template_path(basename):
    cached_file = os.fspath(DOWNLOADS_DIR.joinpath(basename)) 
    if not os.path.exists(cached_file):
        url = 'https://osf.io/download/6807fb5958b763aae139aa60/'
        logger.info('Downloading: "{}" to {}\n'.format(url, cached_file))
        download_url_to_file(url, cached_file, progress=True)
    return cached_file

def download_probes(probe_dir=None):
    if probe_dir is None:
        probe_dir = PROBE_DIR
    probe_dir.mkdir(parents=True, exist_ok=True)
    for probe_name, url in PROBE_URLS.items():
        cached_file = os.fspath(probe_dir.joinpath(probe_name)) 
        if not os.path.exists(cached_file):
            logger.info('Downloading: "{}" to {}\n'.format(url, cached_file))
            try:
                download_url_to_file(url, cached_file, progress=True)
            except urllib.error.HTTPError as e:
                logger.info(f'Unable to download probe {probe_name}, error:')
                logger.info(e)

def retry_download(func, n_tries=5):
    def retry(*args, **kwargs):
        i = 0
        while True:
            try:
                func(*args, **kwargs)
                break
            except urllib.error.HTTPError as e:
                
                if i < n_tries:
                    print(f'Download failed, retrying... {i}/{n_tries-1}')
                    i += 1
                    time.sleep(1)
                else:
                    raise e
    
    return retry

@retry_download
def download_url_to_file(url, dst, progress=True):
    pass
    file_size = None
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    u = urllib.request.urlopen(url)
    meta = u.info()
    if hasattr(meta, 'getheaders'):
        content_length = meta.getheaders("Content-Length")
    else:
        content_length = meta.get_all("Content-Length")
    if content_length is not None and len(content_length) > 0:
        file_size = int(content_length[0])
    
    dst = os.path.expanduser(dst)
    dst_dir = os.path.dirname(dst)
    f = tempfile.NamedTemporaryFile(delete=False, dir=dst_dir)
    try:
        with tqdm(total=file_size, disable=not progress,
                  unit='B', unit_scale=True, unit_divisor=1024) as pbar:
            while True:
                buffer = u.read(8192)
                if len(buffer) == 0:
                    break
                f.write(buffer)
                pbar.update(len(buffer))
        f.close()
        shutil.move(f.name, dst)
    finally:
        f.close()
        if os.path.exists(f.name):
            os.remove(f.name)

def get_performance():
    pass

    cpu = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    avail = memory.available / 2**30
    total = memory.total / 2**30
    used = total - avail
    pct = memory.percent

    cpu_stats = {
        'util': cpu, 'mem_avail': avail, 'mem_total': total,
        'mem_used': used, 'mem_pct': pct
    }

    if torch.cuda.is_available():
        if _NVML_EXISTS:
            gpu = torch.cuda.utilization()
        else:
            gpu = None
        gpu_avail, gpu_total = torch.cuda.mem_get_info()
        gpu_avail /= (2**30)  
        gpu_total /= (2**30)
        gpu_used = gpu_total - gpu_avail
        gpu_pct = (gpu_used / gpu_total) * 100
        allocated = torch.cuda.memory_allocated() / 2**30
        alloc_pct = (allocated / gpu_total) * 100
        max_alloc = torch.cuda.max_memory_allocated() / 2**30
        max_pct = (max_alloc / gpu_total) * 100

        gpu_stats = {
            'util': gpu, 'mem_avail': gpu_avail, 'mem_total': gpu_total,
            'mem_used': gpu_used, 'mem_pct': gpu_pct, 'alloc': allocated,
            'alloc_pct': alloc_pct, 'max_alloc': max_alloc,
            'max_alloc_pct': max_pct
        }
    else:
        gpu_stats = None

    perf = {'cpu': cpu_stats, 'gpu': gpu_stats}
    return perf

def log_performance(log=None, level=None, header=None, reset=False):
    pass

    perf = get_performance()

    if log is None:
        log = logger
    if level is None:
        level = 'debug'
    if header is not None:
        getattr(log, level)(' ')
        getattr(log, level)(f'{header}')

    util = perf['cpu']['util']
    avail = perf['cpu']['mem_avail']
    total = perf['cpu']['mem_total']
    used = perf['cpu']['mem_used']
    pct = perf['cpu']['mem_pct']

    getattr(log, level)('*'*56)
    getattr(log, level)(f'CPU usage:    {util:5.2f} %')
    getattr(log, level)(f'Mem used:     {pct:5.2f} %     | {used:10.2f} GB')
    getattr(log, level)(f'Mem avail:    {avail:5.2f} / {total:5.2f} GB')
    getattr(log, level)('-'*54)

    if perf['gpu'] is not None:
        gpu_util = perf['gpu']['util']
        if gpu_util is not None:
            getattr(log, level)(f'GPU usage:    {gpu_util:5.2f} %')
        else:
            getattr(log, level)(f'GPU usage:    `conda install pynvml` for GPU usage')

        gpu_total = perf['gpu']['mem_total']
        gpu_used = perf['gpu']['mem_used']
        gpu_pct = perf['gpu']['mem_pct']
        allocated = perf['gpu']['alloc']
        alloc_pct = perf['gpu']['alloc_pct']
        max_alloc = perf['gpu']['max_alloc']
        max_pct = perf['gpu']['max_alloc_pct']

        getattr(log, level)(f'GPU memory:   {gpu_pct:5.2f} %     |{gpu_used:10.2f}   / {gpu_total:8.2f} GB')
        getattr(log, level)(f'Allocated:    {alloc_pct:5.2f} %     |{allocated:10.2f}   / {gpu_total:8.2f} GB')
        getattr(log, level)(f'Max alloc:    {max_pct:5.2f} %     |{max_alloc:10.2f}   / {gpu_total:8.2f} GB')
    else:
        getattr(log, level)('GPU usage:    N/A')
        getattr(log, level)('GPU memory:   N/A')

    getattr(log, level)('*'*56)

    if reset and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

def log_cuda_details(log=None):
    pass
    if log is None: log = logger
    if torch.cuda.is_available():
        log.debug(f'\n\n{torch.cuda.memory_summary(abbreviated=True)}\n')

def log_sorting_summary(ops, log=None, level=None):
    pass

    if ops.get('runtime_postproc', None) is None:
        raise ValueError('Ops must contain runtime stats (v4.0.38 or later).')

    if log is None:
        log = logger
    if level is None:
        level = 'debug'
    log_fn = getattr(log, level)

    step_keys = {
        'preprocessing': 'preproc', 'drift corr': 'drift',
        'spike det. (univ)': 'st0', 'cluster (temp)': 'clu0',
        'spike det. (learn)': 'st', 'cluster (final)': 'clu',
        'cluster merge': 'merge', 'postprocessing': 'postproc'
    }

    log_fn(' ')
    log_fn('*'*56)
    log_fn('Sorting summary')
    log_fn('-'*56)
    log_fn(f"{'Total number of units:':<30}{ops['n_units_total']:>25}")
    log_fn(f"{'Number of good units:':<30}{ops['n_units_good']:>25}")
    log_fn(f"{'Number of spikes':<30}{ops['n_spikes']:>25}")
    log_fn(f"{'Mean abs. drift':<30}{ops['mean_drift']:>22.1f} um")
    log_fn(f"{'Total runtime':<30}{ops['runtime']:>24.2f}s")

    log_fn(' ')
    log_fn('Runtime by step')
    log_fn('-'*56)
    for name, key in step_keys.items():
        log_fn(_format_runtime(ops, name, key))

    gpu = True if ops['usage_postproc']['gpu'] is not None else False
    log_fn(' ')
    log_fn('Memory usage by step')
    log_fn('-'*56)
    for name, key in step_keys.items():
        log_fn(_format_usage(ops, name, key, gpu))
    log_fn('*'*56)

def _format_runtime(ops, name, key):
    total = ops['runtime']
    step = ops['runtime_' + key]
    pct = step / total * 100
    return f"{(name + ':'):<27} {step:>15.1f}s {('(' + f'{pct:.2f}' + ') %'):>10}"

def _format_usage(ops, name, key, gpu):
    s = (f"{(name + ':'):<23}"
         + f"sys  {ops['usage_' + key]['cpu']['mem_used']:>6.1f} GB  |  ")
    if gpu:
        s += f"gpu  {ops['usage_' + key]['gpu']['max_alloc']:>5.1f} GB"
    else:
        s += f"gpu  {'N/A':>7}   "
    return s

def log_thread_count(log=None):
    if log is None:
        log = logger

    if sys.platform == 'linux':
        result = subprocess.run(['ps', '-o', 'thcount', f'{os.getpid()}'], stdout=subprocess.PIPE)
        thread_count = result.stdout.decode('utf-8').replace('\n','').replace(' ','').strip('THCNT')
    else:
        thread_count = 'N/A (linux only)'

    log.debug('')
    log.debug('--------------------')
    log.debug(f'Process thread count: {thread_count}')
    log.debug(f'num torch threads: {torch.get_num_threads()}')
    log.debug('--------------------\n')

def probe_as_string(probe):
    pass

    opt = np.get_printoptions()
    np.set_printoptions(threshold=np.inf)
    
    p = pprint.pformat(probe, indent=4, sort_dicts=False)
    
    p = 'np.array'.join(p.split('array'))
    p = 'dtype=np.'.join(p.split('dtype='))
    probe_text = "probe = "
    
    probe_text += p[0] + '\n ' + p[1:-1] + '\n' + p[-1]

    np.set_printoptions(**opt)

    return probe_text

def ops_as_string(ops):
    pass

    ops_copy = ops.copy()
    probe_keys = list(ops['probe'].keys())
    for k in ['settings', 'probe'] + probe_keys:
        _ = ops_copy.pop(k)
    n_files = len(ops_copy['filename'])
    if n_files > 5:
        ops_copy['filename'] = ops_copy['filename'][:5] + [f'... ({n_files} total files)']

    ops_text = "ops = "
    p = pprint.pformat(ops_copy, indent=4, sort_dicts=False)
    
    ops_text += p[0] + '\n ' + p[1:-1] + '\n' + p[-1]

    return ops_text
