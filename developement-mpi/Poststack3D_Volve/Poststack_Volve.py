r"""
Poststack inversion of 3D Volve dataset distributed over ilines. 

NOTE: currently works only with same number of inlines per rank

Run as: export OMP_NUM_THREADS=4; export MKL_NUM_THREADS=4; export NUMBA_NUM_THREADS=4; mpiexec -n 4 python Poststack_Volve.py
"""

import os
import sys
import numpy as np
import pylops_mpi
import time
import segyio
import warnings
warnings.filterwarnings('ignore')

from matplotlib import pyplot as plt
from mpi4py import MPI

from scipy.interpolate import RegularGridInterpolator
from pylops_mpi.DistributedArray import local_split, Partition
from pylops.basicoperators import Transpose
from pylops.avo.poststack import PoststackLinearModelling
from visual import explode_volume

plt.close("all")

def run():
    comm = MPI.COMM_WORLD
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()

    tic = time.perf_counter()  
    
    if rank == 0:
        print(f'Distributed Poststack inversion of 3D Volve data ({size} ranks)')

    # Create folder to save figures
    figdir = 'Figs'
    if rank == 0:
        if not os.path.exists(figdir):
            os.makedirs(figdir)
    
    ############
    # Parameters
    ############

    # Time axis
    itmin = 100 # index of first time/depth sample in data used in inversion
    itmax = 1200 # index of last time/depth sample in data used in inversion

    # Wavelet estimation
    nt_wav = 21 # number of samples of statistical wavelet
    nfft = 512 # number of samples of FFT

    # Inversion parameters
    niter_sr = 20 # number of iterations of lsqr
    epsR_sr = 1e-1 # spatial regularization

    ##################
    # Data preparation
    ##################

    # Load data 
    segyfile = '../../data/seismicinversion/ST10010ZC11_PZ_PSDM_KIRCH_FULL_D.MIG_FIN.POST_STACK.3D.JS-017536.segy'
    f = segyio.open(segyfile, ignore_geometry=True)

    traces = segyio.collect(f.trace)[:]
    traces = traces[:, itmin:itmax]
    ntraces, nt = traces.shape

    t = f.samples[itmin:itmax]
    il = f.attributes(segyio.TraceField.INLINE_3D)[:]
    xl = f.attributes(segyio.TraceField.CROSSLINE_3D)[:]

    # Define regular IL and XL axes
    il_unique = np.unique(il)
    xl_unique = np.unique(xl)

    il_min, il_max = min(il_unique), max(il_unique)
    xl_min, xl_max = min(xl_unique), max(xl_unique)

    dt = t[1] - t[0]
    dil = min(np.unique(np.diff(il_unique)))
    dxl = min(np.unique(np.diff(xl_unique)))

    ilines = np.arange(il_min, il_max + dil, dil)
    xlines = np.arange(xl_min, xl_max + dxl, dxl)
    nil, nxl = ilines.size, xlines.size

    ilgrid, xlgrid = np.meshgrid(np.arange(nil),
                                np.arange(nxl),
                                indexing='ij')

    # Look-up table
    traces_indeces = np.full((nil, nxl), np.nan)
    iils = (il - il_min) // dil
    ixls = (xl - xl_min) // dxl
    traces_indeces[iils, ixls] = np.arange(ntraces)
    traces_available = np.logical_not(np.isnan(traces_indeces))

    # Reorganize traces in regular grid
    d = np.zeros((nil, nxl, nt))
    d[ilgrid.ravel()[traces_available.ravel()], 
      xlgrid.ravel()[traces_available.ravel()]] = traces
    nil, nxl, nt = len(ilines), len(xlines), len(t)

    # Take away an inline to have same nil per rank (temporary!)
    d = d[:-1]
    ilines = ilines[:-1]
    nil, nxl, nt = len(ilines), len(xlines), len(t)

    # Display data
    if rank == 0:
        explode_volume(d.transpose(2, 1, 0),
                       cmap='RdYlBu', clipval=(-5, 5),
                       figsize=(15, 10))
        plt.savefig(os.path.join(figdir, 'Data.png'), dpi=300)

    # Choose how to split ilines to ranks
    nil_rank = local_split((nil, ), comm, Partition.SCATTER, 0)
    nil_ranks = np.concatenate(comm.allgather(nil_rank))
    ilin_rank = np.insert(np.cumsum(nil_ranks)[:-1] , 0, 0)[rank]
    ilend_rank = np.cumsum(nil_ranks)[rank]
    ilines_rank = ilines[ilin_rank:ilend_rank]
    print(f'Rank: {rank}, nil: {nil_rank}, isin: {ilin_rank}, isend: {ilend_rank}')
    
    # Extract part of data of interest
    d = d[ilin_rank:ilend_rank]

    # Load velocity model
    segyfilev = '../../data/seismicinversion/ST10010ZC11-MIG-VEL.MIG_VEL.VELOCITY.3D.JS-017527.segy'
    fv = segyio.open(segyfilev)
    v = segyio.cube(fv)

    # Regrid velocity model to portion of data used in this rank
    IL, XL, T = np.meshgrid(ilines_rank, xlines, t, indexing='ij')

    vinterp = RegularGridInterpolator((fv.ilines, fv.xlines, fv.samples), v, 
                                       bounds_error=False, fill_value=0)
    vinterp = vinterp(np.vstack((IL.ravel(), XL.ravel(), T.ravel())).T)
    vinterp = vinterp.reshape(nil_rank[0], nxl, nt)

    # Display velocity model
    if rank == 0:
        explode_volume(d.transpose(2, 1, 0),
                       cmap='RdYlBu', clipval=(-5, 5),
                       figsize=(15, 10))
        plt.savefig(os.path.join(figdir, 'Data_rank0.png'), dpi=300)

        explode_volume(vinterp.transpose(2, 1, 0),
                       cmap='gist_rainbow',
                       figsize=(15, 10))
        plt.savefig(os.path.join(figdir, 'Vel_rank0.png'), dpi=300)

    # Compute background AI (from well log analysis done externally)
    intercept = -3218.0003362662665
    gradient = 3.2468122679241023
    aiinterp = intercept + gradient*vinterp
    m0 = np.log(aiinterp)
    m0[np.isnan(m0)] = 0

    # Wavelet estimation (so far done only in rank0 and then broadcasted)
    if rank == 0:
        t_wav = np.arange(nt_wav) * (dt/1000)
        t_wav = np.concatenate((np.flipud(-t_wav[1:]), t_wav), axis=0)

        # Estimate wavelet spectrum
        wav_est_fft = np.mean(np.abs(np.fft.fft(d[::2, ::2], nfft, axis=-1)), axis=(0, 1))
        fwest = np.fft.fftfreq(nfft, d=dt/1000)

        # Create wavelet in time
        wav_est = np.real(np.fft.ifft(wav_est_fft)[:nt_wav])
        wav_est = np.concatenate((np.flipud(wav_est[1:]), wav_est), axis=0)
        wav_est = wav_est / wav_est.max()

        # Display wavelet
        fig, axs = plt.subplots(1, 2, figsize=(20, 5))
        fig.suptitle('Statistical wavelet estimate')
        axs[0].plot(fwest[:nfft//2], wav_est_fft[:nfft//2], 'k')
        axs[0].set_title('Frequency')
        axs[1].plot(t_wav, wav_est, 'k')
        axs[1].set_title('Time')
        plt.savefig(os.path.join(figdir, 'Wav.png'), dpi=300)
    else:
        wav_est = np.empty(nt_wav * 2 - 1)
    comm.Bcast(wav_est, root=0)

    # Create distributed data
    d_dist = pylops_mpi.DistributedArray(global_shape=nil * nxl * nt,
                                         local_shapes=[(nil_r * nxl * nt,) for nil_r in nil_ranks], 
                                         dtype=np.float32)
    d_dist[:] = d.flatten().astype(np.float32)
    
    # Create distributed background model
    m0_dist = pylops_mpi.DistributedArray(global_shape=nil * nxl * nt,
                                         local_shapes=[(nil_r * nxl * nt,) for nil_r in nil_ranks], 
                                         dtype=np.float32)
    m0_dist[:] = m0.flatten().astype(np.float32)
    ai0 = m0_dist.asarray().reshape((nil, nxl, nt))

    # Created PostStackLinearModelling
    PPop = PoststackLinearModelling(1e1*wav_est, nt0=nt, spatdims=(nil_rank[0], nxl))
    Top = Transpose((nil_rank[0], nxl, nt), (2, 0, 1))
    BDiag = pylops_mpi.basicoperators.MPIBlockDiag(ops=[Top.H @ PPop @ Top, ])

    # Regularized inversion with regularized equations
    LapOp = pylops_mpi.MPILaplacian(dims=(nil, nxl, nt), axes=(0, 1, 2), weights=(1, 1, 1),
                                    sampling=(1, 1, 1), dtype=BDiag.dtype)

    StackOp = pylops_mpi.MPIStackedVStack([BDiag, np.sqrt(epsR_sr) * LapOp])
    d0_dist = pylops_mpi.DistributedArray(global_shape=nil * nxl * nt)
    d0_dist[:] = 0.
    dstack_dist = pylops_mpi.StackedDistributedArray([d_dist, d0_dist])

    dnorm_dist = BDiag.H @ d_dist
    aiinv_dist = pylops_mpi.optimization.basic.cgls(StackOp, dstack_dist, 
                                                    x0=m0_dist, 
                                                    niter=niter_sr, 
                                                    show=True)[0]
    aiinv = aiinv_dist.asarray().reshape((nil, nxl, nt))

    # Display background and inverted model
    if rank == 0:
        explode_volume(np.exp(ai0).transpose(2, 1, 0),
                       cmap='gist_rainbow', clipval=(3000, 18000),
                       figsize=(15, 10))
        plt.savefig(os.path.join(figdir, 'BackAI.png'), dpi=300)
        explode_volume(np.exp(aiinv).transpose(2, 1, 0),
                       cmap='gist_rainbow', clipval=(3000, 18000),
                       figsize=(15, 10))
        plt.savefig(os.path.join(figdir, 'InvAI.png'), dpi=300)
    

if __name__ == '__main__':
    run()