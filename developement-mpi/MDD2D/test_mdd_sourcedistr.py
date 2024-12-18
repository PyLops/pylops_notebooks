r"""
Test MDD with synthetic seismic data distributing data over sources

Run as: export OMP_NUM_THREADS=4; export MKL_NUM_THREADS=4; export NUMBA_NUM_THREADS=4; mpiexec -n 2 python test_mdd_sourcedistr.py
"""

import numpy as np
import matplotlib.pyplot as plt
import pylops_mpi

from pylops.utils.seismicevents import hyperbolic2d, makeaxis
from pylops.utils.tapers import taper3d
from pylops.utils.wavelets import ricker
from pylops.basicoperators import Transpose
from pylops.waveeqprocessing import MDC
from pylops.optimization.basic import cgls

from mpi4py import MPI
from pylops_mpi.DistributedArray import local_split, Partition

def run():
    comm = MPI.COMM_WORLD
    size = comm.Get_size() # number of nodes
    rank = comm.Get_rank() # rank of current node
    dtype = np.float32
    cdtype = np.complex64

    # Create data
    par = {
        "ox": -300,
        "dx": 10,
        "nx": 61,
        "oy": -500,
        "dy": 10,
        "ny": 101,
        "ot": 0,
        "dt": 0.004,
        "nt": 400,
        "f0": 20,
        "nfmax": 200,
    }

    t0_m = 0.2
    vrms_m = 1100.0
    amp_m = 1.0

    t0_G = (0.2, 0.5, 0.7)
    vrms_G = (1200.0, 1500.0, 2000.0)
    amp_G = (1.0, 0.6, 0.5)

    # Taper
    tap = taper3d(par["nt"], (par["ny"], par["nx"]), (5, 5), tapertype="hanning")

    # Create axis
    t, t2, x, y = makeaxis(par)

    # Create wavelet
    wav = ricker(t[:41], f0=par["f0"])[0]

    # Generate model
    m, mwav = hyperbolic2d(x, t, t0_m, vrms_m, amp_m, wav)

    # Generate operator
    G, Gwav = np.zeros((par["ny"], par["nx"], par["nt"])), np.zeros(
        (par["ny"], par["nx"], par["nt"])
    )
    for iy, y0 in enumerate(y):
        G[iy], Gwav[iy] = hyperbolic2d(x - y0, t, t0_G, vrms_G, amp_G, wav)
    G, Gwav = G * tap, Gwav * tap

    # Add negative part to data and model
    m = np.concatenate((np.zeros((par["nx"], par["nt"] - 1)), m), axis=-1)
    mwav = np.concatenate((np.zeros((par["nx"], par["nt"] - 1)), mwav), axis=-1)
    Gwav2 = np.concatenate((np.zeros((par["ny"], par["nx"], par["nt"] - 1)), Gwav), axis=-1)

    # Move to frequency
    Gwav_fft = np.fft.rfft(Gwav2, 2 * par["nt"] - 1, axis=-1)
    Gwav_fft = Gwav_fft[..., : par["nfmax"]]

    # Move frequency/time to first axis
    m, mwav = m.T, mwav.T
    Gwav_fft = Gwav_fft.transpose(2, 0, 1)

    # Choose how to split sources to ranks
    ns = par["ny"]
    ns_rank = local_split((ns, ), MPI.COMM_WORLD, Partition.SCATTER, 0)
    ns_ranks = np.concatenate(MPI.COMM_WORLD.allgather(ns_rank))
    isin_rank = np.insert(np.cumsum(ns_ranks)[:-1] , 0, 0)[rank]
    isend_rank = np.cumsum(ns_ranks)[rank]
    print(f'Rank: {rank}, ns: {ns_rank}, ifin: {isin_rank}, ifend: {isend_rank}')

    # Extract batch of frequency slices (in practice, this will be directly read from input file)
    G = Gwav_fft[:,isin_rank:isend_rank].astype(cdtype)
    print(f'Rank: {rank}, G: {G.shape}')
     
    # Define operator
    Fop = MDC((1.0 * 0.004 * np.sqrt(par["nt"])) * G, nt=2 * par["nt"] - 1, nv=1, 
              dt=0.004, dr=1.0, twosided=True, prescaled=True)
    Top = Transpose(dims=(2 * par["nt"] - 1, ns_rank[0]), axes=(1, 0))
    Foptot = pylops_mpi.MPIVStack(ops=[Top * Fop , ])

    # Apply forward
    x = pylops_mpi.DistributedArray(global_shape=(2 * par["nt"] - 1) * par["nx"] * 1, 
                                    partition=Partition.BROADCAST,
                                    dtype=dtype)
    x[:] = m.astype(dtype).ravel()
    xloc = x.asarray()

    y = Foptot @ x
    yloc = y.asarray().real
    print(f'Rank: {rank}, y.localarray: {y.local_array.shape}, ns_rank: {ns_rank[0]}')
    
    if rank == 0:
        plt.figure()
        plt.imshow(yloc.reshape(par["ny"], 2 * par["nt"] - 1).T, aspect="auto", interpolation="nearest",
                   cmap="gray", vmin=-yloc.max(), vmax=yloc.max())
        plt.savefig('data_mpi.png')

    # Compare with serial computation
    if rank == 0:
        Fop_ = MDC((1.0 * 0.004 * np.sqrt(par["nt"])) * Gwav_fft, nt=2 * par["nt"] - 1, nv=1,
                   dt=0.004, dr=1.0, twosided=True, prescaled=True)
        y_ = Fop_ @ m.ravel()

        print('Forward check', np.allclose(yloc, y_, atol=1e-6))
        print(yloc[:10], y_[:10])
        
        plt.figure()
        plt.imshow(y_.reshape(2 * par["nt"] - 1, par["ny"]), aspect="auto", interpolation="nearest",
                   cmap="gray", vmin=-y_.max(), vmax=y_.max())
        plt.savefig('data.png')

        plt.figure()
        plt.imshow(yloc.reshape(par["ny"], 2 * par["nt"] - 1).T - y_.reshape(2 * par["nt"] - 1, par["ny"]), 
                   aspect="auto", interpolation="nearest",
                   cmap="gray", vmin=-0.1*y_.max(), vmax=0.1*y_.max())
        plt.savefig('data_err.png')
    
    # Apply adjoint
    xadj = Foptot.H @ y
    xadjloc = xadj.asarray().real
    
    if rank == 0:
        plt.figure()
        plt.imshow(xadjloc.reshape(2 * par["nt"] - 1, par["nx"]), aspect="auto", interpolation="nearest",
                   cmap="gray", vmin=-xadjloc.max(), vmax=xadjloc.max())
        plt.savefig('adj_mpi.png')

    # Compare with serial computation
    if rank == 0:
        xadj_ = Fop_.H @ y_

        print('Adjoint check', np.allclose(xadjloc, xadj_, atol=1e-6))
        print(xadjloc[:10], xadj_[:10])
        
        plt.figure()
        plt.imshow(xadj_.reshape(2 * par["nt"] - 1, par["nx"]), aspect="auto", interpolation="nearest",
                   cmap="gray", vmin=-xadj_.max(), vmax=xadj_.max())
        plt.savefig('adj.png')

        plt.figure()
        plt.imshow((xadjloc - xadj_).reshape(2 * par["nt"] - 1, par["nx"]), aspect="auto", interpolation="nearest",
                   cmap="gray", vmin=-0.1*xadj_.max(), vmax=0.1*xadj_.max())
        plt.savefig('adj_err.png')

    # Inverse
    x0 = pylops_mpi.DistributedArray(global_shape=(2 * par["nt"] - 1) * par["nx"] * 1, 
                                     partition=Partition.BROADCAST,
                                     dtype=cdtype)
    x0[:] = 0

    xinv = pylops_mpi.cgls(Foptot, y, x0=x0, niter=50, show=True if rank == 0 else False)[0]
    xinvloc = xinv.asarray().real

    if rank == 0:
        plt.figure()
        plt.imshow(xinvloc.reshape(2 * par["nt"] - 1, par["nx"]), aspect="auto", interpolation="nearest",
                   cmap="gray", vmin=-xinvloc.max(), vmax=xinvloc.max())
        plt.savefig('inv_mpi.png')

    # Compare with serial computation
    if rank == 0:
        xinv_ = cgls(Fop_, y_, niter=50, show=True)[0].ravel()

        print('Inv check', np.allclose(xinvloc, xinv_, atol=1e-6))
        print(xinvloc[:10], xinv_[:10])
        
        plt.figure()
        plt.imshow(xinv_.reshape(2 * par["nt"] - 1, par["nx"]), aspect="auto", interpolation="nearest",
                   cmap="gray", vmin=-xinv_.max(), vmax=xinv_.max())
        plt.savefig('inv.png')

        print(xinvloc.shape, xinv_.shape)
        plt.figure()
        plt.imshow((xinvloc - xinv_).reshape(2 * par["nt"] - 1, par["nx"]), aspect="auto", interpolation="nearest",
                   cmap="gray", vmin=-0.1*xinv_.max(), vmax=0.1*xinv_.max())
        plt.savefig('inv_err.png')



if __name__ == '__main__':
    run()