import json
from pathlib import Path
from typing import Tuple, Union
import os, shutil
import warnings
from concurrent.futures import ThreadPoolExecutor
import time
import logging
logger = logging.getLogger(__name__)

from scipy.io import loadmat
import numpy as np
import torch
from torch.fft import fft, ifft, fftshift

from kilosort import CCG
from kilosort.preprocessing import get_drift_matrix, fft_highpass
from kilosort.postprocessing import (
    remove_duplicates, compute_spike_positions, make_pc_features
    )
from kilosort.utils import log_performance

_torch_warning = ".*PyTorch does not support non-writable tensors"

def find_binary(data_dir: Union[str, os.PathLike]) -> Path:
    pass

    data_dir = Path(data_dir)
    filenames = list(data_dir.glob('*.bin')) + list(data_dir.glob('*.bat')) \
                + list(data_dir.glob('*.dat')) + list(data_dir.glob('*.raw'))
    if len(filenames) == 0:
        raise FileNotFoundError(
            f'No binary file found in {data_dir}. Expected extensions are:\n'
            '*.bin, *.bat, *.dat, or *.raw.'
            )

    if len(filenames) > 1:
        filenames = [f for f in filenames if 'ap.bin' in f.as_posix()]

    if len(filenames) > 1:
        raise ValueError('Multiple binary files in folder with "ap" tag, '
                         'please specify filename')

    return filenames[0]

def load_probe(probe_path):
    pass
    probe = {}
    probe_path = Path(probe_path).resolve()
    required_keys = ['chanMap', 'yc', 'xc', 'n_chan']

    if probe_path.suffix == '.prb':

        contents = probe_path.read_text()
        metadata = {'np': np}
        exec(contents, {}, metadata)

        probe['chanMap'] = []
        probe['xc'] = []
        probe['yc'] = []
        probe['kcoords'] = []
        probe['n_chan'] = 0 
        for cg in sorted(metadata['channel_groups']):
            d = metadata['channel_groups'][cg]
            ch = d['channels']
            pos = d.get('geometry', {})
            probe['chanMap'].append(ch)
            probe['n_chan'] += len(ch)
            probe['xc'].append([pos[c][0] for c in ch])
            probe['yc'].append([pos[c][1] for c in ch])
            probe['kcoords'].append([cg for c in ch])
        probe['chanMap'] = np.concatenate(probe['chanMap']).ravel().astype(np.int32)
        probe['xc'] = np.concatenate(probe['xc']).astype('float32')
        probe['yc'] = np.concatenate(probe['yc']).astype('float32')
        probe['kcoords'] = np.concatenate(probe['kcoords']).astype('float32')

    elif probe_path.suffix == '.mat':
        mat = loadmat(probe_path)
        connected = mat['connected'].ravel().astype('bool')
        probe['xc'] = mat['xcoords'].ravel().astype(np.float32)[connected]
        nc = len(probe['xc'])
        probe['yc'] = mat['ycoords'].ravel().astype(np.float32)[connected]
        kc = mat.get('kcoords', None)
        if kc is None:
            kc = np.zeros(nc).ravel().astype(np.float32)
        else:
            kc = kc.ravel().astype(np.float32)[connected]
        probe['kcoords'] = kc
        probe['chanMap'] = (mat['chanMap'] - 1).ravel().astype(np.int32)[connected]  
        probe['n_chan'] = (mat['chanMap'] - 1).ravel().astype(np.int32).shape[0]  

    elif probe_path.suffix == '.json':
        with open(probe_path, 'r') as f:
            probe = json.load(f)
        for k in list(probe.keys()):
            
            v = probe[k]
            if isinstance(v, list):
                dtype = np.int32 if k == 'chanMap' else np.float32
                probe[k] = np.array(v, dtype=dtype)

    else:
        raise ValueError(
            f"Unrecognized extension for probe: {probe_path.name}\n"
            "Recognized extensions are .mat, .prb, or .json."
            )

    for n in required_keys:
        if n not in probe.keys():
            raise ValueError(f'Missing required key: {n} after loading probe.')

    size = None
    for k, v in probe.items():
        if isinstance(v, np.ndarray):
            if size is None:
                size = v.size
            elif size != v.size:
                raise ValueError(
                    f"All probe variables must have the same length."
                )

    return probe

def save_probe(probe_dict, filepath):
    pass

    if Path(filepath).suffix != '.json':
        raise RuntimeWarning(
            'Saving json probe to a file whose suffix is not .json. '
            'kilosort.io.load_probe will not recognize this file.' 
        )

    d = probe_dict.copy()
    
    for k in list(d.keys()):
        v = d[k]
        if isinstance(v, np.ndarray):
            d[k] = v.tolist()

    length = None
    for k, v in d.items():
        if isinstance(v, list):
            if length is None:
                length = len(v)
            elif length != len(v):
                raise ValueError(
                    f"All probe variables must have the same length."
                )

    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        f.write(json.dumps(d))

def remove_bad_channels(probe, bad_channels):
    pass
    probe = probe.copy()

    bad_idx = np.empty_like(bad_channels, dtype=int)
    for i, k in enumerate(bad_channels):
        try:
            idx = np.where(probe['chanMap'] == k)[0][0]
        except IndexError:
            raise IndexError(f"Channel '{k}' was not in probe['chanMap']")
        bad_idx[i] = idx
    
    for k in ['xc', 'yc', 'chanMap', 'kcoords']:
        probe[k] = np.delete(probe[k], bad_idx)
    probe['n_chan'] = probe['n_chan'] - bad_idx.size

    return probe

def select_shank(probe, shank_idx):
    pass
    probe = probe.copy()
    keep = (probe['kcoords'] == shank_idx)
    n = keep.sum()
    if n == 0:
        raise ValueError(f"Selected empty shank index: {shank_idx}, can't sort.")
    else:
        probe['n_chan'] = n
        for k in ['xc', 'yc', 'chanMap', 'kcoords']:
            probe[k] = probe[k][keep]

    return probe

def save_to_phy(st, clu, tF, Wall, probe, ops, imin, results_dir=None,
                data_dtype=None, save_extra_vars=False,
                save_preprocessed_copy=False, skip_dat_path=False):
    pass

    if results_dir is None:
        results_dir = ops['data_dir'].joinpath('kilosort4')
    results_dir = Path(results_dir)
    results_dir.mkdir(exist_ok=True)

    chan_map = probe['chanMap']
    channel_positions = np.stack((probe['xc'], probe['yc']), axis=-1)
    np.save((results_dir / 'channel_map.npy'), chan_map)
    np.save((results_dir / 'channel_positions.npy'), channel_positions)
    np.save((results_dir / 'channel_shanks.npy'), probe['kcoords'])

    whitening_mat = ops['Wrot']
    np.save((results_dir / 'whitening_mat_dat.npy'), whitening_mat.cpu())

    whitening_mat_inv = torch.inverse(
        whitening_mat
        + 1e-5 * torch.eye(whitening_mat.shape[0]).to(whitening_mat.device)
        )
    np.save((results_dir / 'whitening_mat.npy'), whitening_mat.cpu())
    np.save((results_dir / 'whitening_mat_inv.npy'), whitening_mat_inv.cpu())
    log_performance(logger, level='debug', header='save_to_phy, whitening',
                    reset=True)

    spike_times = st[:,0].astype('int64') + imin  
    spike_templates = st[:,1].astype('int32')
    spike_clusters = clu
    xs, ys = compute_spike_positions(st, tF, ops)
    spike_positions = np.vstack([xs, ys]).T
    amplitudes = torch.norm(tF, dim=[-2,-1]).cpu().numpy()
    log_performance(logger, level='debug', header='save_to_phy, spike positions',
                    reset=True)

    spike_times, spike_clusters, kept_spikes = remove_duplicates(
        spike_times, spike_clusters, dt=ops['duplicate_spike_bins']
    )
    amp = amplitudes[kept_spikes]
    spike_templates = spike_templates[kept_spikes]
    spike_positions = spike_positions[kept_spikes]
    np.save((results_dir / 'spike_times.npy'), spike_times)
    np.save((results_dir / 'spike_templates.npy'), spike_clusters)
    np.save((results_dir / 'spike_clusters.npy'), spike_clusters)
    np.save((results_dir / 'spike_positions.npy'), spike_positions)
    np.save((results_dir / 'spike_detection_templates.npy'), spike_templates)
    np.save((results_dir / 'amplitudes.npy'), amp)

    np.save((results_dir / 'kept_spikes.npy'), kept_spikes)
    log_performance(logger, level='debug', header='save_to_phy, remove duplicates',
                    reset=True)

    similar_templates = CCG.similarity(Wall, ops['wPCA'].contiguous(), nt=ops['nt'])
    template_amplitudes = torch.norm(Wall, dim=[-2,-1]).cpu().numpy()
    templates = (Wall.unsqueeze(-1).cpu() * ops['wPCA'].cpu()).sum(axis=-2).numpy()
    templates = templates.transpose(0,2,1)
    templates_ind = np.tile(np.arange(Wall.shape[1])[np.newaxis, :], (templates.shape[0],1))
    np.save((results_dir / 'similar_templates.npy'), similar_templates)
    np.save((results_dir / 'templates.npy'), templates)
    np.save((results_dir / 'templates_ind.npy'), templates_ind)
    log_performance(logger, level='debug', header='save_to_phy, template props',
                    reset=True)

    if save_extra_vars:
        
        np.save(results_dir / 'tF.npy', tF.cpu().numpy())

    tF = tF[kept_spikes]
    pc_features, pc_feature_ind = make_pc_features(
        ops, spike_templates, spike_clusters, tF
        )
    np.save(results_dir / 'pc_features.npy', pc_features)
    np.save(results_dir / 'pc_feature_ind.npy', pc_feature_ind)
    log_performance(logger, level='debug', header='save_to_phy, pc features',
                    reset=True)

    acg_threshold = ops['settings']['acg_threshold']
    ccg_threshold = ops['settings']['ccg_threshold']
    is_ref, est_contam_rate = CCG.refract(spike_clusters, spike_times / ops['fs'],
                                          acg_threshold=acg_threshold,
                                          ccg_threshold=ccg_threshold)
    log_performance(logger, level='debug', header='save_to_phy, est contam',
                    reset=True)

    stypes = ['ContamPct', 'Amplitude', 'KSLabel']
    ks_labels = [['mua', 'good'][int(r)] for r in is_ref]
    props = [est_contam_rate*100, template_amplitudes, ks_labels]
    for stype, prop in zip(stypes, props):
        with open((results_dir / f'cluster_{stype}.tsv'), 'w') as f:
            f.write(f'cluster_id\t{stype}\n')
            for i,p in enumerate(prop):
                if stype != 'KSLabel':
                    f.write(f'{i}\t{p:.1f}\n')
                else:
                    f.write(f'{i}\t{p}\n')
        if stype == 'KSLabel':
            shutil.copyfile((results_dir / f'cluster_{stype}.tsv'), 
                            (results_dir / f'cluster_group.tsv'))

    dtype = "'int16'" if data_dtype is None else f"'{data_dtype}'"
    params = {
        'n_channels_dat': ops['settings']['n_chan_bin'],
        'offset': 0,
        'sample_rate': ops['settings']['fs']
        }
    if save_preprocessed_copy:
        dat_path = results_dir / 'temp_wh.dat'
        params['dtype'] = "'int16'"
        params['hp_filtered'] = True
        params['dat_path'] = repr(dat_path.name)
    else:
        f = ops['settings']['filename']
        if not isinstance(f, list): f = [f]
        dat_path = [Path(os.path.relpath(Path(p).resolve(), results_dir.resolve())).as_posix() for p in f]
        params['dtype'] = dtype
        params['hp_filtered'] = False
        params['dat_path'] = dat_path

    with open((results_dir / 'params.py'), 'w') as f: 
        for key in params.keys():
            if (key == 'dat_path') and skip_dat_path:
                f.write(f'dat_path = "no_path.bin"\n')
                f.write('# ')
            f.write(f'{key} = {params[key]}\n')

    if save_extra_vars:
        
        np.save(results_dir / 'Wall.npy', Wall.cpu().numpy())
        
        np.save(results_dir / 'full_st.npy', st)
        np.save(results_dir / 'full_clu.npy', clu)
        np.save(results_dir / 'full_amp.npy', amplitudes)

    phy_cache_path = Path(results_dir / '.phy')
    if phy_cache_path.is_dir():
        shutil.rmtree(phy_cache_path)

    return results_dir, similar_templates, is_ref, est_contam_rate, kept_spikes

def save_ops(ops, results_dir=None):
    pass

    if results_dir is None:
        results_dir = Path(ops['data_dir']) / 'kilosort4'
    else:
        results_dir = Path(results_dir)
    results_dir.mkdir(exist_ok=True)

    ops = ops.copy()

    ops['settings']['results_dir'] = str(results_dir)
    if isinstance(ops['filename'], list):
        ops['filename'] = [str(f) for f in ops['filename']]
    else:
        ops['filename'] = str(ops['filename'])
    ops['data_dir'] = str(ops['data_dir'])
    if isinstance(ops['settings']['filename'], list):
        ops['settings']['filename'] = [str(f) for f in ops['settings']['filename']]
    else:
        ops['settings']['filename'] = str(ops['settings']['filename'])
    ops['settings']['data_dir'] = str(ops['settings']['data_dir'])

    ops['is_tensor'] = []
    for k, v in ops.items():
        if isinstance(v, torch.Tensor):
            ops[k] = v.cpu().numpy()
            ops['is_tensor'].append(k)
    ops['preprocessing'] = {k: v.cpu().numpy()
                            for k, v in ops['preprocessing'].items()}

    np.save(results_dir / 'ops.npy', np.array(ops))

def load_ops(ops_path, device=None):
    pass
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

    ops = np.load(ops_path, allow_pickle=True).item()
    for k, v in ops.items():
        if k in ops['is_tensor']:
            ops[k] = torch.from_numpy(v).to(device)

    ops['preprocessing'] = {k: torch.from_numpy(v).to(device)
                            for k,v in ops['preprocessing'].items()}

    return ops

def bfile_from_ops(ops=None, ops_path=None, filename=None, device=None):
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

    if ops is None:
        if ops_path is not None:
            ops = load_ops(ops_path, device=device)
        else:
            raise ValueError('Must specify either `ops` or `ops_path`.')
    
    if filename is None:

        filename = ops['filename']
    
    bfile = BinaryFiltered(
        filename, ops['n_chan_bin'], fs=ops['fs'], NT=ops['batch_size'],
        nt=ops['nt'], nt0min=ops['nt0min'], chan_map=ops['probe']['chanMap'],
        hp_filter=ops['fwav'], whiten_mat=ops['Wrot'], dshift=ops['dshift'],
        device=device, do_CAR=ops['do_CAR'], artifact_threshold=ops['artifact_threshold'],
        invert_sign=ops['invert_sign'], dtype=ops['data_dtype'], tmin=ops['tmin'],
        tmax=ops['tmax'], shift=ops['shift'], scale=ops['scale']
        )

    return bfile

class BinaryRWFile:

    supported_dtypes = ['int16', 'uint16', 'int32', 'float32']

    def __init__(self, filename, n_chan_bin: int, fs: int = 30000, 
                 NT: int = 60000, nt: int = 61, nt0min: int = 20,
                 device: torch.device = None, write: bool = False,
                 dtype: str = None, tmin: float = 0.0, tmax: float = np.inf,
                 shift=None, scale=None, file_object=None, batch_downsampling=1):
        pass

        self.fs = np.float64(fs)
        self.n_chan_bin = n_chan_bin
        self.filename = filename
        self.NT = np.int64(NT) 
        self.nt = np.int64(nt) 
        self.nt0min = np.int64(nt0min)
        self.shift = shift
        self.scale = scale
        tmin = np.float64(tmin)
        tmax = np.float64(tmax)

        if device is None:
            if torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')
        self.device = device
        self.uint_set_warning = True
        self.writable = write
        self.mode = 'w+' if write else 'r'

        if file_object is not None:
            dtype = file_object.dtype
        if dtype is None:
            dtype = 'int16'
        self.dtype = dtype

        if isinstance(filename, list) and len(filename) > 1:
            if file_object is None:
                file_object = BinaryFileGroup(
                    filenames=filename, n_channels=n_chan_bin, dtype=dtype
                    )
            else:
                logger.info(
                    "`file_object is not None`, "
                    "it will be used instead of loading from filename."
                    )
        self.file_object = file_object

        if str(self.dtype) not in self.supported_dtypes:
            message = f"""
                {self.dtype} is not supported and may result in unexpected
                behavior or errors. Supported types are:\n
                {self.supported_dtypes}
                """
            warnings.warn(message, RuntimeWarning)

        if file_object is None:
            self.total_samples = get_total_samples(filename, n_chan_bin, dtype)
        else:
            n, c = file_object.shape
            assert c == n_chan_bin
            self.total_samples = n

        self.imin = max(np.int64(tmin*self.fs), 0)
        self.imax = self.total_samples if tmax==np.inf \
                    else min(np.int64(tmax*self.fs), self.total_samples)
        self.n_batches_raw = np.int64(np.ceil(self.n_samples / self.NT))
        
        a, b = self._get_batch_edges(self.n_batches_raw-1)
        batch_size = b - a - self.nt
        if batch_size < self.nt:
            self.n_batches_raw -= 1
            self.imax -= batch_size

        self.set_downsampling(batch_downsampling)

    @property
    def n_samples(self) -> int:
        pass
        return self.imax - self.imin

    @property
    def shape(self) -> Tuple[int, int]:
        pass
        return self.n_samples, self.n_chan_bin

    @property
    def size(self) -> int:
        pass
        return np.prod(np.array(self.shape).astype(np.int64))
    
    @property
    def file(self):
        pass
        if self.file_object is not None:
            file = self.file_object
        else:
            f = self.filename
            if isinstance(f, list):
                f = f[0]
            file = np.memmap(f, mode=self.mode, dtype=self.dtype,
                             shape=(self.total_samples, self.n_chan_bin))
        return file
        
    def __enter__(self):
        return self

    def __setitem__(self, *items):
        if not self.writable:
            raise ValueError('Binary file was loaded as read-only.')

        idx, data = items
        
        sample_indices = self._get_shifted_indices(idx)
        
        if self.dtype == 'uint16':
            data = data + 2**15
            if self.uint_set_warning:

                logger.info(
                    "NOTE: When setting new values for uint16 data, 2**15 will "
                    "be added to the given values before writing to file."
                    )
                self.uint_set_warning = False
        
        data = data.astype(self.dtype)

        self.file[sample_indices] = data
        if self.file_object is None:
            self.file.flush()

    def __getitem__(self, *items):
        idx, *crop = items
        
        sample_indices = self._get_shifted_indices(idx)
        samples = self.file[sample_indices]
        
        if self.dtype == 'uint16':
            
            samples = samples.astype('float32')
            samples = samples - 2**15

        if self.scale is not None:
            samples = samples * self.scale
        if self.shift is not None:
            samples = samples + self.shift

        return samples

    def _get_shifted_indices(self, idx):
        if not isinstance(idx, tuple): idx = tuple([idx])
        new_idx = []

        i = idx[0]
        if isinstance(i, slice):
            
            start = self.imin if i.start is None else i.start + self.imin
            stop = self.imax if i.stop is None else min(i.stop + self.imin, self.imax)
            new_idx.append(slice(start, stop, i.step))
        else:
            new_idx.append(i)

        if len(idx) == 2:

            new_idx.append(idx[1])

        return tuple(new_idx)

    def _get_batch_edges(self, ibatch):
        if ibatch==0:
            bstart = self.imin
            bend = self.imin + self.NT + self.nt
        else:

            ibatch = np.int64(ibatch)
            bstart = np.int64(self.imin + (ibatch * self.NT) - self.nt)
            bend = min(self.imax, np.int64(bstart + self.NT + 2*self.nt))

        return bstart, bend
    
    def set_downsampling(self, downsampling):
        self.batch_downsampling = downsampling
        self.n_batches = np.int64(self.n_batches_raw / self.batch_downsampling)

    def padded_batch_to_torch(self, ibatch, return_inds=False):
        pass
        ibatch *= self.batch_downsampling
        bstart, bend = self._get_batch_edges(ibatch)
        data = self.file[bstart : bend]
        data = data.T

        if self.dtype == 'uint16':
            
            data = data.astype('float32')
            data = data - 2**15

        if self.scale is not None:
            data = data * self.scale
        if self.shift is not None:
            data = data + self.shift

        nsamp = data.shape[-1]
        X = torch.zeros((self.n_chan_bin, self.NT + 2*self.nt), device=self.device)

        with warnings.catch_warnings():

            warnings.filterwarnings("ignore", message=_torch_warning)

            if ibatch == 0:
                X[:, self.nt : self.nt+nsamp] = torch.from_numpy(data).to(self.device).float()
                X[:, :self.nt] = X[:, self.nt : self.nt+1]
                bstart = self.imin - self.nt
            elif ibatch == self.n_batches-1:
                X[:, :nsamp] = torch.from_numpy(data).to(self.device).float()
                X[:, nsamp:] = X[:, nsamp-1:nsamp]
                bend += self.nt
            else:
                X[:] = torch.from_numpy(data).to(self.device).float()

        inds = [bstart, bend]
        if return_inds:
            return X, inds
        else:
            return X

def get_total_samples(filename, n_channels, dtype=np.int16):
    pass
    if isinstance(filename, list):
        filename = filename[0]
    bytes_per_value = np.dtype(dtype).itemsize
    bytes_per_sample = np.int64(bytes_per_value * n_channels)
    total_bytes = os.path.getsize(filename)
    samples = np.float64(total_bytes / bytes_per_sample)

    if samples%1 != 0:
        raise ValueError(
            "Bytes in binary file did not divide evenly, "
            "incorrect n_chan_bin ('number of channels' in GUI)."
        )
    else:
        return np.int64(samples)

class BinaryFileGroup:
    def __init__(self, file_objects=None, filenames=None,
                 n_channels=None, dtype=None):

        self._file_objects = None
        self._filenames = None
        self.n_files = 0

        if file_objects is not None:
            self._file_objects = file_objects
            self.dtype = file_objects[0].dtype
            self.n_chans = file_objects[0].shape[1]
            self.n_files = len(file_objects)
            for f in file_objects[1:]:
                assert f.dtype == self.dtype, 'All files must have the same dtype'
                assert f.shape[1] == self.n_chans, \
                    'All files must have the same number of channels'
        elif filenames is not None:
            
            if n_channels is None or dtype is None:
                raise ValueError("BinaryFileGroup requires n_channels and dtype "
                                 "when using filenames option.")
            self._filenames = filenames
            self.dtype = dtype
            self.n_chans = n_channels
            self.n_files = len(filenames)
        else:
            raise ValueError("Must specify either file_objects or filenames")

        i = 0
        self.split_indices = []
        for j in range(self.n_files):
            i += self.get_file_size(j)
            self.split_indices.append(i)

    def get_file(self, i):
        if self._file_objects is not None:
            file = self._file_objects[i]
        else:
            name = self._filenames[i]
            n_samples = get_total_samples(name, self.n_chans, self.dtype)
            file = np.memmap(name, mode='r', dtype=self.dtype,
                                shape=(n_samples, self.n_chans))

        return file
    
    def get_file_size(self, i):
        if self._file_objects is not None:
            size = self._file_objects[i].shape[0]
        else:
            size = get_total_samples(self._filenames[i], self.n_chans, self.dtype)

        return size

    def __getitem__(self, *items):

        idx, *crop = items
        if not isinstance(idx, tuple): idx = tuple([idx])
        channel_idx = idx[1] if len(idx) > 1 else slice(None)
        time_idx = idx[0]

        if not isinstance(time_idx, slice):
            time_idx = slice(time_idx, time_idx+1)
        i = time_idx.start
        j = time_idx.stop

        if i is None: i = 0
        if i < 0: i = self.shape[0] + i
        if j is None: j = self.shape[0]
        if j < 0: j = self.shape[0] + j
        time_idx = slice(i, j)

        data = []
        shift = 0
        for idx, k in enumerate(self.split_indices):
            f = self.get_file(idx)
            n_samples = f.shape[0]
            if time_idx.start < k:
                
                ii = max(int(time_idx.start - shift), 0)
                jj = max(int(time_idx.stop - shift), 0)
                t = slice(ii, jj)
                data.append(f[t, channel_idx])
                if time_idx.stop <= k:
                    
                    break
            shift += n_samples

        if len(data) == 0:
            d = None
        elif len(data) == 1:
            d = data[0]
        else:
            d = np.concatenate(data, axis=0)
    
        return d

    @property
    def shape(self):
        return self.split_indices[-1], self.n_chans
    
    @staticmethod
    def from_filenames(filenames, n_channels, dtype=np.int16):

        return BinaryFileGroup(filenames=filenames, n_channels=n_channels,
                               dtype=dtype)

class BinaryFiltered(BinaryRWFile):
    def __init__(self, filename: str, n_chan_bin: int, fs: int = 30000, 
                 NT: int = 60000, nt: int = 61, nt0min: int = 20,
                 chan_map: np.ndarray = None, hp_filter: torch.Tensor = None,
                 whiten_mat: torch.Tensor = None, dshift: torch.Tensor = None,
                 device: torch.device = None, do_CAR: bool = True,
                 artifact_threshold: float = np.inf, invert_sign: bool = False,
                 dtype=None, tmin: float = 0.0, tmax: float = np.inf,
                 shift=None, scale=None, file_object=None, batch_downsampling=1):

        super().__init__(filename, n_chan_bin, fs, NT, nt, nt0min, device,
                         dtype=dtype, tmin=tmin, tmax=tmax, shift=shift,
                         scale=scale, file_object=file_object,
                         batch_downsampling=batch_downsampling) 
        self.chan_map = chan_map
        self.whiten_mat = whiten_mat
        self.hp_filter = hp_filter
        self.dshift = dshift
        self.do_CAR = do_CAR
        self.invert_sign=invert_sign
        self.artifact_threshold = artifact_threshold

    def filter(self, X, ops=None, ibatch=None, skip_preproc=False):
        
        if self.chan_map is not None:
            X = X[self.chan_map]

        if self.invert_sign:
            X = X * -1

        X = X - X.mean(1).unsqueeze(1)
        if self.do_CAR:
            
            X = X - torch.median(X, 0)[0]
    
        if skip_preproc:
            return X

        if self.hp_filter is not None:
            fwav = fft_highpass(self.hp_filter, NT=X.shape[1])
            X = torch.real(ifft(fft(X) * torch.conj(fwav)))
            X = fftshift(X, dim = -1)

        if self.artifact_threshold < np.inf:
            if torch.any(torch.abs(X) >= self.artifact_threshold):

                return torch.zeros_like(X)

        if self.whiten_mat is not None:
            if self.dshift is not None and ops is not None and ibatch is not None:
                M = get_drift_matrix(ops, self.dshift[ibatch], device=self.device)
                
                X = (M @ self.whiten_mat) @ X
            else:
                X = self.whiten_mat @ X
        return X

    def __getitem__(self, *items):
        samples = super().__getitem__(*items)
        with warnings.catch_warnings():

            warnings.filterwarnings("ignore", message=_torch_warning)
            X = torch.from_numpy(samples.T).to(self.device).float()
        return self.filter(X)
        
    def padded_batch_to_torch(self, ibatch, ops=None, return_inds=False,
                              skip_preproc=False):
        if return_inds:
            X, inds = super().padded_batch_to_torch(ibatch, return_inds=return_inds)
            return self.filter(X, ops, ibatch, skip_preproc=skip_preproc), inds
        else:
            X = super().padded_batch_to_torch(ibatch)
            return self.filter(X, ops, ibatch, skip_preproc=skip_preproc)

def save_preprocessing(filename, ops, bfile=None, bfile_path=None, verbose=False):
    pass

    n_batches = ops['Nbatches']
    nt = ops['nt']
    NT = ops['batch_size']
    whiten_mat = ops['preprocessing']['whiten_mat']
    hp_filter = ops['preprocessing']['hp_filter']
    dshift = ops['dshift']
    chan_map = ops['chanMap']
    dtype = ops['data_dtype']
    n_chans = ops['n_chan_bin']

    if bfile is None:
        if bfile_path is None:
            raise ValueError("Must specify either `bfile` or `bfile_path`.")
        bfile = BinaryFiltered(
            filename=bfile_path, n_chan_bin=n_chans, chan_map=chan_map, nt=nt,
            NT=NT, hp_filter=hp_filter, whiten_mat=whiten_mat, dshift=dshift,
            dtype=dtype
            )

    n_chans_used = len(chan_map)
    weights = np.linspace(1, 0, 2*nt+2)[1:-1]
    W = np.vstack([weights, np.flip(weights)])
    W = np.tile(W, n_chans_used).reshape(2, n_chans_used, 2*nt)

    logger.info(' ')
    logger.info('='*40)
    logger.info(f'Saving drift-corrected copy of data to: {filename}...')
    for i in range(n_batches):
        if i % 100 == 0:
            logger.info(f'Writing batch {i}/{n_batches}...')
        if i > 0 and i % 500 == 0:
            log_performance(logger, level='debug', header=None)

        if i == 0:
            
            batch1 = bfile.padded_batch_to_torch(i, ops=ops)
        else:
            
            batch1 = batch2

        mode = 'w+' if i == 0 else 'r+'
        z = np.memmap(filename, dtype='int16', mode=mode,
                      shape=(NT*n_batches, n_chans))

        if i == n_batches-1:

            y = batch1[:, 2*nt:-nt].cpu().numpy().T
            z[(i*NT)+nt:, chan_map] = (y*200).astype('int16')
        else:
            batch2 = bfile.padded_batch_to_torch(i+1, ops=ops)

            x1 = batch1[:, (NT-1) + 1:].cpu().numpy()
            x2 = batch2[:, :2*nt].cpu().numpy()
            X = np.vstack([x1[np.newaxis,...], x2[np.newaxis,...]])
            y2 = (X*W).sum(axis=0).T

            y1 = batch1[:, nt*2:-nt].cpu().numpy().T
            z[(i*NT)+(nt) : ((i+1)*NT), chan_map] = (y1*200).astype('int16')
            if i == 0:
                
                y0 = batch1[:, nt:2*nt].cpu().numpy().T
                z[:nt, chan_map] = (y0*200).astype('int16')

            z[((i+1)*NT)-nt : ((i+1)*NT)+nt, chan_map] = (y2*200).astype('int16')

        z.flush()
        del(z)

    logger.info('='*40)
    logger.info('Copying finished.')
    logger.info(' ')

def spikeinterface_to_binary(recording, filepath, data_name='data.bin',
                             dtype=np.int16, chunksize=300000, export_probe=True,
                             probe_name='probe.prb', max_workers=None):
    pass

    filepath = Path(filepath)
    filepath.mkdir(exist_ok=True, parents=True)
    binary_filename = filepath / f'{data_name}'
    probe_filename = filepath / f'{probe_name}'

    logger.info('='*40)
    logger.info('Loading recording with SpikeInterface...')

    N = recording.get_total_samples()
    logger.info(f'number of samples: {N}')
    c = recording.get_traces(start_frame=0, end_frame=1, segment_index=0).shape[1]
    logger.info(f'number of channels: {c}')
    s = recording.get_num_segments()
    logger.info(f'numbef of segments: {s}')
    fs = recording.get_sampling_frequency()
    logger.info(f'sampling rate: {fs}')
    dtype = recording.get_dtype()
    logger.info(f'dtype: {dtype}')

    indices = []
    for k in range(s):
        n = recording.get_num_samples(segment_index=k)
        i = 0 + k*chunksize
        while i < n:
            j = i + chunksize if (i + chunksize) < n else n
            indices.append((i, j, k))
            i += chunksize

    def copy_chunk(memmap, i, j, k):
        t = recording.get_traces(start_frame=i, end_frame=j, segment_index=k)
        memmap[i:j,:] = t
        memmap.flush()
        del(t)

    y = np.memmap(binary_filename, dtype=dtype, mode='w+', shape=(N,c))
    total_chunks = len(indices)
    logger.info('='*40)
    logger.info(
        f'Converting {total_chunks} data chunks '
        f'with a chunksize of {chunksize} samples...'
        )
    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        futures = [exe.submit(copy_chunk, y, i, j, k) for i, j, k in indices]

        while exe._work_queue.qsize() > 0:
            time.sleep(10)
            logger.info(
                f'{total_chunks - exe._work_queue.qsize()} of {total_chunks}'
                ' chunks converted...'
                )
    logger.info(f'Data conversion finished.')
    logger.info('='*40)

    del(y)  

    if export_probe:
        try:
            from probeinterface import write_prb
            try:
                pg = recording.get_probegroup()
                write_prb(probe_filename, pg)
            except ValueError:
                logger.info(
                    'SpikeInterface recording contains no probe information,\n'
                    'could not write .prb file.'
                )
                probe_filename = None
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                'Could not import `write_prb` from probeinterface when exporting '
                'probe, please run `pip install spikeinterface`.'
            )
    else:
        probe_filename = None

    return binary_filename, N, c, s, fs, probe_filename

class RecordingExtractorAsArray:

    def __init__(self, recording_extractor):
        pass

        if recording_extractor.get_num_segments() > 1:
            try:
                import spikeinterface as si
                self.recording = si.concatenate_recordings([recording_extractor])
                logger.info(
                    'SpikeInterface recording contains more than one segment, '
                    'segments will be concatenated as if contiguous.'
                    )
            except ModuleNotFoundError:
                raise ModuleNotFoundError(
                    'SpikeInterface could not be imported, but is needed for '
                    'loading multi-segment SpikeInterface recordings. Please '
                    'run `pip install spikeinterface`.'
                )

        logger.info('='*40)
        logger.info('Loading recording with SpikeInterface...')
        self.recording = recording_extractor
        self.N = self.recording.get_total_samples()
        logger.info(f'number of samples: {self.N}')
        self.c = self.recording.get_traces(start_frame=0, end_frame=1, segment_index=0).shape[1]
        logger.info(f'number of channels: {self.c}')
        self.s = self.recording.get_num_segments()
        logger.info(f'numbef of segments: {self.s}')
        self.fs = self.recording.get_sampling_frequency()
        logger.info(f'sampling rate: {self.fs}')
        self.dtype = self.recording.get_dtype()
        logger.info(f'dtype: {self.dtype}')
        self.shape = (self.N, self.c)
        logger.info('='*40)

    def __getitem__(self, *items):
        idx, *crop = items
        if not isinstance(idx, tuple): idx = tuple([idx])
        sample_idx = idx[0]
        channel_ids = None if len(idx) == 1 else idx[1]

        if not isinstance(sample_idx, slice):
            sample_idx = slice(sample_idx, sample_idx+1)
        i = sample_idx.start
        j = sample_idx.stop

        if i is None: i = 0
        if i < 0: i = self.N + i
        if j is None: j = self.N
        if j < 0: j = self.N + j

        if isinstance(channel_ids, slice):
            c = channel_ids.start
            d = channel_ids.stop
            if c is None: c = 0
            if d is None: d = self.shape[1]
            channel_ids = list(range(c, d))
        elif channel_ids is not None:
            assert isinstance(channel_ids, int)
            channel_ids = [channel_ids]
        else:
            channel_ids = np.arange(self.shape[1])

        channel_ids = self.recording.channel_ids[channel_ids]

        samples = self.recording.get_traces(start_frame=i, end_frame=j,
                                            channel_ids=channel_ids)
        
        return samples

    def __setitem__(self):
        raise ValueError('RecordingExtractorAsBinary is read-only.')
