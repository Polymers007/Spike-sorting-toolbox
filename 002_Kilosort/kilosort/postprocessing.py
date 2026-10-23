from numba import njit
from numba.types import bool_
import numpy as np
import torch

from kilosort.clustering_qr import xy_templates, get_data_cpu

@njit("(int64[:], int32[:], int32)")
def remove_duplicates(spike_times, spike_clusters, dt=15):
    pass
    keep = np.zeros_like(spike_times, bool_)
    cluster_t0 = {}
    for i in range(spike_times.size):
        t = spike_times[i]
        c = spike_clusters[i]
        if c in cluster_t0:
            t0 = cluster_t0[c]
        else:
            t0 = t - dt

        if t >= (t0 + dt):
            
            cluster_t0[c] = t
            keep[i] = True
        else:
            
            continue
    
    return spike_times[keep], spike_clusters[keep], keep

def compute_spike_positions(st, tF, ops):
    pass

    cpu = torch.device('cpu')
    icc_mask = ops['iCC_mask'].to(cpu)
    iU = ops['iU'].to(cpu)
    iCC = ops['iCC'].to(cpu)

    tmass = torch.norm(tF, 2, dim=-1)
    tmask = icc_mask[:, iU[st[:,1]]].T
    tmass = tmass * tmask
    tmass = tmass / tmass.sum(1, keepdim=True)

    xc = torch.from_numpy(ops['xc'])
    yc = torch.from_numpy(ops['yc'])
    chs = iCC[:, iU[st[:,1]]]
    xc0 = xc[chs.T]
    yc0 = yc[chs.T]

    xs = (xc0 * tmass).sum(1).numpy()
    ys = (yc0 * tmass).sum(1).numpy()

    return xs, ys

def make_pc_features(ops, spike_templates, spike_clusters, tF):
    pass

    xy, iC = xy_templates(ops)
    n_templates = iC.shape[1]
    n_clusters = np.unique(spike_clusters).size
    n_chans = ops['nearest_chans']
    feature_ind = np.zeros((n_clusters, n_chans), dtype=np.uint32)

    for i in np.unique(spike_clusters):
        
        iunq = np.unique(spike_templates[spike_clusters==i]).astype(int)
        
        ix = torch.from_numpy(np.zeros(n_templates, bool))
        ix[iunq] = True

        Xd, igood, ichan = get_data_cpu(
            ops, xy, iC, spike_templates, tF, None, None,
            dmin=ops['dmin'], dminx=ops['dminx'], ix=ix, merge_dim=False
            )

        spike_mean = Xd.mean(0)
        chan_norm = torch.linalg.norm(spike_mean, dim=1)
        sorted_chans, ind = torch.sort(chan_norm, descending=True)
        
        tF[igood,:] = Xd[:, ind[:n_chans], :]
        
        feature_ind[i,:] = ichan[ind[:n_chans]].cpu().numpy()

    tF = torch.permute(tF, (0, 2, 1))

    return tF, feature_ind
