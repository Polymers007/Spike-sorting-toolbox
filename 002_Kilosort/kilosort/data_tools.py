from pathlib import Path
import numpy as np
from kilosort import io, CCG

def mean_waveform(cluster_id, results_dir, n_spikes=np.inf, bfile=None, best=True):
    pass
    results_dir = Path(results_dir)
    if best:
        chan = get_best_channels(results_dir)[cluster_id]
    else:
        chan = None

    spike_times, spike_subset = get_cluster_spikes(cluster_id, results_dir,
                                                   n_spikes=n_spikes)
    waves = get_spike_waveforms(spike_times, results_dir, bfile=bfile, chan=chan)
    mean_wave = waves.mean(axis=-1)

    return mean_wave, spike_subset

def get_best_channels(results_dir):
    pass
    templates = np.load(results_dir / 'templates.npy')
    best_chans = (templates**2).sum(axis=1).argmax(axis=-1)
    return best_chans

def get_best_channel(results_dir, cluster_id):
    return get_best_channels(results_dir)[cluster_id]

def get_cluster_spikes(cluster_id, results_dir, n_spikes=np.inf):
    pass
    spike_times = np.load(results_dir / 'spike_times.npy')
    spike_clusters = np.load(results_dir / 'spike_clusters.npy')
    spike_idx = (spike_clusters == cluster_id).nonzero()[0]
    if n_spikes != np.inf:
        spike_subset = np.random.choice(
            np.arange(spike_idx.size), min(spike_idx.size, n_spikes), replace=False
            )
    else:
        spike_subset = np.arange(spike_idx.size)
    spike_idx = spike_idx[spike_subset]
    spike_idx.sort()
    spikes = spike_times[spike_idx]

    return spikes, spike_subset

def get_spike_waveforms(spikes, results_dir, bfile=None, chan=None):
    pass
    if isinstance(spikes, int):
        spikes = [spikes]

    if bfile is None:
        ops = io.load_ops(results_dir / 'ops.npy')
        bfile = io.bfile_from_ops(ops)
    whitening_mat_inv = np.load(results_dir / 'whitening_mat_inv.npy')

    waves = []
    for t in spikes:
        tmin = max(t - bfile.nt0min, 0)
        tmax = t + (bfile.nt - bfile.nt0min)
        w = bfile[tmin:tmax].cpu().numpy()
        if whitening_mat_inv is not None:
            w = whitening_mat_inv @ w
        if w.shape[1] == bfile.nt:

            waves.append(w)
    waves = np.stack(waves, axis=-1)

    if chan is not None:
        waves = waves[chan,:]

    return waves

def cluster_templates(cluster_id, results_dir, mean=False, best=False, spike_subset=None):
    pass
    results_dir = Path(results_dir)
    spike_clusters = np.load(results_dir / 'spike_clusters.npy')
    spike_idx = (spike_clusters == cluster_id).nonzero()[0]
    if spike_subset is not None:
        spike_idx = spike_idx[spike_subset]
    temps = get_templates(spike_idx, results_dir)
    if best:
        chan = get_best_channel(results_dir, cluster_id)
        temps = temps[:,:,chan]
    if mean:
        return temps.mean(axis=0)

    return temps

def get_templates(spike_idx, results_dir):
    pass
    results_dir = Path(results_dir)
    if isinstance(spike_idx, int):
        spike_idx = [spike_idx]
    templates = np.load(results_dir / 'templates.npy')
    
    spike_templates = np.load(results_dir / 'spike_clusters.npy')
    amplitudes = np.load(results_dir / 'amplitudes.npy')
    template_idx = spike_templates[spike_idx]
    temps = templates[template_idx, :, :]
    scaled = amplitudes[spike_idx, np.newaxis, np.newaxis] * temps

    return scaled

def get_good_cluster(results_dir, n=1):
    pass
    labels = get_labels(results_dir)
    labels = [x for x in labels if x[1] == 'good']
    rows = np.random.choice(
        np.arange(len(labels)), size=min(n, len(labels)), replace=False
        )
    cluster_ids = [int(labels[r][0]) for r in rows]
    if n == 1:
        cluster_ids = cluster_ids[0]

    return cluster_ids

def get_labels(results_dir):
    pass
    results_dir = Path(results_dir)
    filename = results_dir / 'cluster_KSLabel.tsv'
    with open(filename) as f:
        text = f.read()
    rows = text.split('\n')
    labels = [r.split('\t') for r in rows[1:]][:-1]

    return labels

def get_refractory(results_dir):
    pass

    clu = np.load(results_dir / 'spike_clusters.npy')
    st = np.load(results_dir / 'spike_times.npy')
    ops = io.load_ops(results_dir / 'ops.npy')

    is_ref, est_contam_rate = CCG.refract(
        clu, st/ops['fs'],
        acg_threshold=ops['acg_threshold'],
        ccg_threshold=ops['ccg_threshold']
        )
    
    return is_ref.astype(bool), est_contam_rate
