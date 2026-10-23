import gc
import logging

import numpy as np
import torch
from torch import sparse_coo_tensor as coo
from scipy.sparse import csr_matrix
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks
from scipy.cluster.vq import kmeans
import faiss
from tqdm import tqdm 

from kilosort import hierarchical, swarmsplitter
from kilosort.utils import log_performance

logger = logging.getLogger(__name__)

def neigh_mat(Xd, nskip=1, n_neigh=10, max_sub=25000):

    n_samples, dim = Xd.shape

    Xsub = Xd[::nskip]
    n1 = Xsub.shape[0]

    if (max_sub is not None) and (n1 > max_sub):
        n2 = n1 - max_sub
        idx, rev_idx = subsample_idx(n1, n2)
        Xsub = Xsub[idx]
    else:
        rev_idx = None

    n_nodes = Xsub.shape[0]

    Xd = np.ascontiguousarray(Xd)
    Xsub = np.ascontiguousarray(Xsub)

    index = faiss.IndexFlatL2(dim)   
    index.add(Xsub)    
    _, kn = index.search(Xd, n_neigh)     

    dexp = np.ones(kn.shape, np.float32)    
    rows = np.tile(np.arange(n_samples)[:, np.newaxis], (1, n_neigh)).flatten()
    M   = csr_matrix((dexp.flatten(), (rows, kn.flatten())),  
                     (kn.shape[0], n_nodes))                  

    skip_idx = np.arange(0, n_samples, nskip)
    if rev_idx is not None:
        skip_idx = skip_idx[rev_idx]
    M[skip_idx, np.arange(n_nodes)] = 0

    return kn, M

def assign_iclust(rows_neigh, isub, kn, tones2, nclust, lam, m, ki, kj, device=torch.device('cuda')):
    n_spikes = kn.shape[0]

    ij = torch.vstack((rows_neigh.flatten(), isub[kn].flatten()))
    xN = coo(ij, tones2.flatten(), (n_spikes, nclust))
    xN = xN.to_dense()

    if lam > 0:
        tones = torch.ones(len(kj), device = device)
        tzeros = torch.zeros(len(kj), device = device)
        ij = torch.vstack((tzeros, isub))    
        kN = coo(ij, tones, (1, nclust))
    
        xN = xN - lam/m * (ki.unsqueeze(-1) * kN.to_dense()) 
    
    iclust = torch.argmax(xN, 1)

    return iclust

def assign_isub(iclust, kn, tones2, nclust, nsub, lam, m,ki,kj, device=torch.device('cuda')):
    n_neigh = kn.shape[1]
    cols = iclust.unsqueeze(-1).tile((1, n_neigh))
    iis = torch.vstack((kn.flatten(), cols.flatten()))

    xS = coo(iis, tones2.flatten(), (nsub, nclust))
    xS = xS.to_dense()

    if lam > 0:
        tones = torch.ones(len(ki), device = device)
        tzeros = torch.zeros(len(ki), device = device)
        ij = torch.vstack((tzeros, iclust))    
        kN = coo(ij, tones, (1, nclust))
        xS = xS - lam / m * (kj.unsqueeze(-1) * kN.to_dense())

    isub = torch.argmax(xS, 1)
    return isub

def Mstats(M, device=torch.device('cuda')):
    m = M.sum()
    ki = np.array(M.sum(1)).flatten()
    kj = np.array(M.sum(0)).flatten()
    ki = m * ki/ki.sum()
    kj = m * kj/kj.sum()

    ki = torch.from_numpy(ki).to(device)
    kj = torch.from_numpy(kj).to(device)
    
    return m, ki, kj

def cluster(Xd, iclust=None, kn=None, nskip=1, n_neigh=10, max_sub=25000,
            nclust=200, seed=1, niter=200, lam=0, device=torch.device('cuda'),
            verbose=False):    

    if kn is None:

        kn, M = neigh_mat(Xd, nskip=nskip, n_neigh=n_neigh, max_sub=max_sub)
    m, ki, kj = Mstats(M, device=device)

    if verbose:
        logger.debug(f'ki: {ki.nbytes / (2**20):.2f} MB, shape: {ki.shape}')
        logger.debug(f'kj: {kj.nbytes / (2**20):.2f} MB, shape: {kj.shape}')
        log_performance(logger, header='clustering_qr.cluster, after Mstats')

    Xg = Xd.to(device)
    kn = torch.from_numpy(kn).to(device)
    n_spikes, n_neigh = kn.shape
    nsub = M.shape[1]  

    rows_neigh = torch.arange(n_spikes, device=device).unsqueeze(-1).tile((1,n_neigh))
    tones2 = torch.ones((n_spikes, n_neigh), device=device)

    if verbose:
        logger.debug(f'Xg: {Xg.nbytes / (2**20):.2f} MB, shape: {Xg.shape}')
        logger.debug(f'kn: {kn.nbytes / (2**20):.2f} MB, shape: {kn.shape}')
        logger.debug(f'rows_neigh: {rows_neigh.nbytes / (2**20):.2f} MB')
        logger.debug(f'tones2: {tones2.nbytes / (2**20):.2f} MB')
        log_performance(logger, header='clustering_qr.cluster, after var init')

    if iclust is None:
        iclust_init =  kmeans_plusplus(Xg, niter=nclust, seed=seed, 
                                       device=device, verbose=verbose)
        iclust = iclust_init.clone()
    else:
        iclust_init = iclust.clone()
        
    for t in range(niter):
        
        isub = assign_isub(iclust, kn, tones2, nclust, nsub, lam, m,
                           ki, kj,device=device)
        
        iclust = assign_iclust(rows_neigh, isub, kn, tones2, nclust, lam, m,
                               ki, kj, device=device)
        
    if verbose:
        logger.debug(f'isub: {isub.nbytes / (2**20):.2f} MB, shape: {isub.shape}')
        log_performance(logger, header='clustering_qr.cluster, after isub loop')
    
    _, iclust = torch.unique(iclust, return_inverse=True)    
    nclust = iclust.max() + 1
    isub = assign_isub(iclust, kn, tones2, nclust , nsub, lam, m,ki,kj, device=device)

    iclust = iclust.cpu().numpy()
    isub = isub.cpu().numpy()

    return iclust, isub, M, iclust_init

def kmeans_plusplus(Xg, niter=200, seed=1, device=torch.device('cuda'), verbose=False):

    vtot = torch.norm(Xg, 2, dim=1)**2

    n1 = vtot.shape[0]
    if n1 > 2**24:

        n2 = n1 - 2**24   
        idx, rev_idx = subsample_idx(n1, n2)
        rev_idx = torch.from_numpy(rev_idx).to(device)
        subsample = True
    else:
        subsample = False

    torch.manual_seed(seed)
    np.random.seed(seed)

    ntry = 100  
    n_spikes, n_features = Xg.shape

    mu = torch.zeros((niter, n_features), device = device)
    vexp0 = torch.zeros(n_spikes, device = device)
    iclust = torch.zeros((n_spikes,), dtype = torch.int, device = device)

    if verbose:
        log_performance(logger, header='clustering_qr.kpp, after var init')

    for j in range(niter):
        
        v2 = torch.relu(vtot - vexp0)

        if subsample:
            isamp = rev_idx[torch.multinomial(v2[idx], ntry)]
        else:
            isamp = torch.multinomial(v2, ntry)

        try:
            
            Xc = Xg[isamp]
            
            vexp = 2 * Xg @ Xc.T - (Xc**2).sum(1)

            dexp = torch.relu(vexp - vexp0.unsqueeze(1))

            vsum = dexp.sum(0)
            
            imax = torch.argmax(vsum)

            ix = dexp[:, imax] > 0

            iclust[ix] = j    
            mu[j] = Xc[imax]  
            
            vexp0[ix] = vexp[ix, imax]

            del(vexp)
            del(dexp)

        except torch.cuda.OutOfMemoryError:
            logger.debug(f"OOM in kmeans_plus_plus iter {j}, nsp: {Xg.shape[0]}, "
                         f"Xg size: {Xg.nbytes / (2**20):.2f} MB.")
            raise

    if verbose:
        log_performance(logger, header='clustering_qr.kpp, after loop')

    return iclust

def subsample_idx(n1, n2):
    pass

    remove = np.round(np.linspace(0, n1-1, n2)).astype(int)
    idx = np.ones(n1, dtype=bool)
    idx[remove] = False

    rev_idx = idx.nonzero()[0]

    return idx, rev_idx

def xy_templates(ops):
    iU = ops['iU'].cpu().numpy()
    iC = ops['iCC'][:, ops['iU']]
    
    xcup, ycup = ops['xc'][iU], ops['yc'][iU]
    xy = np.vstack((xcup, ycup))
    xy = torch.from_numpy(xy)

    return xy, iC

def xy_up(ops):
    xcup, ycup = ops['xcup'], ops['ycup']
    xy = np.vstack((xcup, ycup))
    xy = torch.from_numpy(xy)
    iC = ops['iC'] 

    return xy, iC

def x_centers(ops):
    k = ops.get('x_centers', None)
    if k is not None:

        approx_centers = k
    else:

        bin_width = 50
        min_x = ops['xc'].min()
        max_x = ops['xc'].max()

        num_bins = int((max_x-min_x)/(bin_width)) + 4
        bins = np.linspace(min_x - bin_width*2, max_x + bin_width*2, num_bins)
        hist, edges = np.histogram(ops['xc'], bins=bins)
        
        smoothed = gaussian_filter(hist, sigma=0.5)
        peaks, _ = find_peaks(smoothed)
        
        approx_centers = [edges[p] for p in peaks]

        if len(approx_centers) <= 1: approx_centers = 1

    centers, distortion = kmeans(ops['xc'], approx_centers, seed=5330)

    return centers

def y_centers(ops):
    ycup = ops['ycup']
    dmin = ops['dmin']

    centers = np.arange(ycup.min()+dmin-1, ycup.max()+dmin+1, 2*dmin)

    return centers

def get_nearest_centers(xy, xcent, ycent):
    
    ycent_pos, xcent_pos = np.meshgrid(ycent, xcent)
    ycent_pos = torch.from_numpy(ycent_pos.flatten())
    xcent_pos = torch.from_numpy(xcent_pos.flatten())
    
    center_distance = (
        (xy[0,:] - xcent_pos.unsqueeze(-1))**2
        + (xy[1,:] - ycent_pos.unsqueeze(-1))**2
        )
    
    center_distance += 1e-20*torch.rand(center_distance.shape)
    
    minimum_distance = torch.min(center_distance, 0).indices

    return minimum_distance, xcent_pos, ycent_pos

def run(ops, st, tF, mode='template', device=torch.device('cuda'),
        progress_bar=None, clear_cache=False, verbose=False):

    if mode == 'template':
        xy, iC = xy_templates(ops)
        iclust_template = st[:,1].astype('int32')
        xcup, ycup = ops['xcup'], ops['ycup']
    else:
        xy, iC = xy_up(ops)
        iclust_template = st[:,5].astype('int32')
        xcup, ycup = ops['xcup'], ops['ycup']

    dmin = ops['dmin']
    dminx = ops['dminx']
    nskip = ops['settings']['cluster_downsampling']
    n_neigh = ops['settings']['cluster_neighbors']
    max_sub = ops['settings']['max_cluster_subset']
    seed = ops['settings']['cluster_init_seed']
    ycent = y_centers(ops)
    xcent = x_centers(ops)
    nsp = st.shape[0]
    nearest_center, _, _ = get_nearest_centers(xy, xcent, ycent)
    total_centers = np.unique(nearest_center).size
    
    clu = np.zeros(nsp, 'int32')
    Wall = torch.zeros((0, ops['Nchan'], ops['settings']['n_pcs']))
    Nfilt = None
    nearby_chans_empty = 0
    nmax = 0
    prog = tqdm(np.arange(len(xcent)), miniters=20 if progress_bar else None,
                mininterval=10 if progress_bar else None)
    t = 0
    v = False
    
    try:
        for jj in prog:
            for kk in np.arange(len(ycent)):
                
                ii = kk + jj*ycent.size
                if ii not in nearest_center:
                    
                    continue
                else:
                    t += 1
                ix = (nearest_center == ii)
                ntemp = ix.sum()

                v = False
                if t % 10 == 0:
                    log_performance(
                        logger,
                        header=f'Cluster center: {ii} ({t}/{total_centers})'
                        )
                    if verbose:
                        v = True

                Xd, igood, ichan = get_data_cpu(
                    ops, xy, iC, iclust_template, tF, ycent[kk], xcent[jj],
                    dmin=dmin, dminx=dminx, ix=ix,
                    )
                if Xd is None:
                    nearby_chans_empty += 1
                    continue

                logger.debug(f'Center {ii} | Xd shape: {Xd.shape} | ntemp: {ntemp}')
                if verbose and Xd.nelement() > 10**8:
                    logger.info(f'Resetting cuda memory stats for Center {ii}')
                    if device == torch.device('cuda'):
                        torch.cuda.reset_peak_memory_stats(device)
                    v = True
                if Xd.shape[0] < 1000:
                    iclust = torch.zeros((Xd.shape[0],))
                else:
                    if mode == 'template':
                        st0 = st[igood,0]/ops['fs']
                    else:
                        st0 = None

                    iclust, iclust0, M, _ = cluster(
                        Xd, nskip=nskip, n_neigh=n_neigh, max_sub=max_sub,
                        lam=1, seed=seed, device=device, verbose=v
                        )

                    if clear_cache:
                        if v:
                            log_performance(logger, header='clustering_qr before gc')
                        gc.collect()
                        torch.cuda.empty_cache()
                        if v:
                            log_performance(logger, header='clustering_qr after gc')

                    xtree, tstat, my_clus = hierarchical.maketree(M, iclust, iclust0)

                    xtree, tstat = swarmsplitter.split(
                        Xd.numpy(), xtree, tstat,iclust, my_clus, meta=st0
                        )

                    iclust = swarmsplitter.new_clusters(iclust, my_clus, xtree, tstat)

                if v:
                    log_performance(logger, header='clustering_qr.run, after iclust')

                clu[igood] = iclust + nmax
                Nfilt = int(iclust.max() + 1)
                nmax += Nfilt

                W = torch.zeros((Nfilt, ops['Nchan'], ops['settings']['n_pcs']))
                for j in range(Nfilt):
                    w = Xd[iclust==j].mean(0)
                    W[j, ichan, :] = torch.reshape(w, (-1, ops['settings']['n_pcs'])).cpu()
                
                Wall = torch.cat((Wall, W), 0)

                if progress_bar is not None:
                    progress_bar.emit(int((kk+1) / len(ycent) * 100))
    except:
        logger.exception(f'Error in clustering_qr.run on center {ii}')
        logger.debug(f'Xd shape: {Xd.shape}')
        logger.debug(f'Nfilt: {Nfilt}')
        logger.debug(f'num spikes: {nsp}')
        try:
            logger.debug(f'iclust shape: {iclust.shape}')
        except UnboundLocalError:
            logger.debug('iclust not yet assigned')
            pass
        raise

    if nearby_chans_empty == total_centers:
        raise ValueError(
            f'`get_data_cpu` never found suitable channels in `clustering_qr.run`.'
            f'\ndmin, dminx, and xcenter are: {dmin, dminx, xcup.mean()}'
        )

    if Wall.sum() == 0:
        
        raise ValueError(
            'Wall is empty after `clustering_qr.run`, cannot continue clustering.'
        )

    return clu, Wall

def get_data_cpu(ops, xy, iC, PID, tF, ycenter, xcenter, dmin=20, dminx=32,
                 ix=None, merge_dim=True):
    PID =  torch.from_numpy(PID).long()

    y0 = ycenter 
    x0 = xcenter 

    if ix is None:
        ix = torch.logical_and(
            torch.abs(xy[1] - y0) < dmin,
            torch.abs(xy[0] - x0) < dminx
            )
    igood = ix[PID].nonzero()[:,0]

    if len(igood) == 0:
        return None, None, None

    pid = PID[igood]
    data = tF[igood]
    nspikes, nchanraw, nfeatures = data.shape
    ichan, imap = torch.unique(iC[:, ix], return_inverse=True)
    nchan = ichan.nelement()

    dd = torch.zeros((nspikes, nchan, nfeatures))
    for k,j in enumerate(ix.nonzero()[:,0]):
        ij = torch.nonzero(pid==j)[:, 0]
        dd[ij.unsqueeze(-1), imap[:,k]] = data[ij]

    if merge_dim:
        Xd = torch.reshape(dd, (nspikes, -1))
    else:
        
        Xd = dd

    return Xd, igood, ichan

def assign_clust(rows_neigh, iclust, kn, tones2, nclust):    
    n_spikes = len(iclust)

    ij = torch.vstack((rows_neigh.flatten(), iclust[kn].flatten()))
    xN = coo(ij, tones2.flatten(), (n_spikes, nclust))
    
    xN = xN.to_dense() 
    iclust = torch.argmax(xN, 1)

    return iclust

def assign_iclust0(Xg, mu):
    vv = Xg @ mu.T
    nm = (mu**2).sum(1)
    iclust = torch.argmax(2*vv-nm, 1)
    return iclust
