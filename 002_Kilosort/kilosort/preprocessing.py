import torch, os, scipy
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
from glob import glob
from torch.fft import fft, ifft, fftshift

def whitening_from_covariance(CC):
    pass
    E,D,V =  torch.linalg.svd(CC)
    eps = 1e-6
    Wrot =(E / (D+eps)**.5) @ E.T
    return Wrot

def whitening_local(CC, xc, yc, nrange=32, device=torch.device('cuda')):
    pass
    Nchan = CC.shape[0]
    Wrot = torch.zeros((Nchan,Nchan), device = device)

    for j in range(CC.shape[0]):
        ds = (xc[j] - xc)**2 + (yc[j] - yc)**2
        isort = np.argsort(ds)
        ix = isort[:nrange]

        wrot = whitening_from_covariance(CC[np.ix_(ix, ix)])

        Wrot[j, ix] = wrot[0]
    return Wrot

def kernel2D_torch(x, y, sig = 1):
    pass
    ds = ((x.unsqueeze(1) - y)**2).sum(-1)
    Kn = torch.exp(-ds / (2*sig**2))
    return Kn

def get_drift_matrix(ops, dshift, device=torch.device('cuda')):
    pass

    yblk = ops['yblk']
    if ops['nblocks'] == 1:
        shifts = dshift
    else:
        finterp = interp1d(yblk, dshift, fill_value="extrapolate", kind = 'linear')
        shifts = finterp(ops['probe']['yc'])

    xp = np.vstack((ops['probe']['xc'],ops['probe']['yc'])).T
    yp = xp.copy()
    yp[:,1] -= shifts

    xp = torch.from_numpy(xp).to(device)
    yp = torch.from_numpy(yp).to(device)

    Kyx = kernel2D_torch(yp, xp, ops['settings']['sig_interp'])

    M = Kyx @ ops['iKxx']

    return M

def get_fwav(NT = 30122, fs = 30000, device=torch.device('cuda')):
    pass

    b,a = butter(3, 300, fs = fs, btype = 'high')

    x = np.zeros(NT)
    x[NT//2] = 1

    wav = filtfilt(b,a , x).copy()
    wav = torch.from_numpy(wav).to(device).float()

    fwav = fft(wav)

    return fwav

def get_whitening_matrix(f, xc, yc, nskip=25, nrange=32):
    pass
    n_chan = len(f.chan_map)
    
    CC = torch.zeros((n_chan, n_chan), device=f.device)
    k = 0
    for j in range(0, f.n_batches-1, nskip):
        
        X = f.padded_batch_to_torch(j)        

        X = X[:, f.nt : -f.nt]

        CC = CC + (X @ X.T)/X.shape[1]
        
        k+=1
        
    CC = CC / k

    Wrot = whitening_local(CC, xc, yc, nrange=nrange, device=f.device)

    return Wrot

def get_highpass_filter(fs=30000, cutoff=300, device=torch.device('cuda')):
    pass
    NT = 30122

    b,a = butter(3, cutoff, fs=fs, btype='high')

    x = np.zeros(NT)
    x[NT//2] = 1

    hp_filter = filtfilt(b, a , x).copy()
    
    hp_filter = torch.from_numpy(hp_filter).to(device).float()
    return hp_filter

def fft_highpass(hp_filter, NT=30122):
    pass
    device = hp_filter.device
    ft = hp_filter.shape[0]

    if ft < NT:
        pad = (NT - ft) // 2
        fhp = fft(torch.cat((torch.zeros(pad, device=device), 
                             hp_filter,
                             torch.zeros(pad + (NT-pad*2-ft), device=device))))
    elif ft > NT:
        crop = (ft - NT) // 2 
        fhp = fft(hp_filter[crop : crop + NT])
    else:
        fhp = fft(hp_filter)
    return fhp
