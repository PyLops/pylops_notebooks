r"""
Test MDD with synthetic seismic data distributing data over frequencies and virtual sources with multiple virtual sources

Run as: export OMP_NUM_THREADS=4; export MKL_NUM_THREADS=4; export NUMBA_NUM_THREADS=4; mpiexec -n 4 python test_mdd_freqdist_multivs.py
"""
import numpy as np
import matplotlib.pyplot as plt
import pylops_mpi

from pylops.utils.seismicevents import hyperbolic2d, makeaxis
from pylops.utils.tapers import taper3d
from pylops.utils.wavelets import ricker
from pylops.waveeqprocessing import MDC
from pylops.optimization.basic import cgls

from mpi4py import MPI
from pylops_mpi.DistributedArray import local_split, Partition
from pylops_mpi.waveeqprocessing.MDC import MPIMDC


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
    nvs = 20

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
    m, mwav = np.zeros((nvs, par['nx'], par['nt'])), np.zeros((nvs, par['nx'], par['nt']))
    for ix, x0 in enumerate(x[par["nx"]//2 - nvs//2: par["nx"]//2 + nvs//2 + (0 if nvs % 2 == 0 else 1)]):
        m[ix], mwav[ix] = hyperbolic2d(x - x0, t, t0_m, vrms_m, amp_m, wav)
    print('m.shape', m.shape)

    # Generate operator
    G, Gwav = np.zeros((par["ny"], par["nx"], par["nt"])), np.zeros(
        (par["ny"], par["nx"], par["nt"])
    )
    for iy, y0 in enumerate(y):
        G[iy], Gwav[iy] = hyperbolic2d(x - y0, t, t0_G, vrms_G, amp_G, wav)
    G, Gwav = G * tap, Gwav * tap

    # Add negative part to data and model
    m = np.concatenate((np.zeros((nvs, par['nx'], par["nt"] - 1)), m), axis=-1)
    mwav = np.concatenate((np.zeros((nvs, par['nx'], par["nt"] - 1)), mwav), axis=-1)
    Gwav2 = np.concatenate((np.zeros((par["ny"], par["nx"], par["nt"] - 1)), Gwav), axis=-1)

    # Move to frequency
    Gwav_fft = np.fft.rfft(Gwav2, 2 * par["nt"] - 1, axis=-1)
    Gwav_fft = Gwav_fft[..., : par["nfmax"]]

    # Move frequency/time to first axis
    m, mwav = m.transpose(2, 1, 0), mwav.transpose(2, 1, 0)
    Gwav_fft = Gwav_fft.transpose(2, 0, 1)

    if rank == 0:
        plt.figure(figsize=(20, 6))
        plt.imshow(mwav.reshape(2 * par["nt"] - 1, par["nx"], nvs).transpose(0, 2, 1).reshape(2 * par["nt"] - 1, -1),
                   aspect="auto", interpolation="nearest",
                   cmap="gray", vmin=-mwav.max(), vmax=mwav.max())
        plt.savefig('mwav_mpi.png')

    # Choose how to split frequencies to ranks
    nf = par["nfmax"]
    nf_rank = local_split((nf, ), MPI.COMM_WORLD, Partition.SCATTER, 0)
    nf_ranks = np.concatenate(MPI.COMM_WORLD.allgather(nf_rank))
    ifin_rank = np.insert(np.cumsum(nf_ranks)[:-1] , 0, 0)[rank]
    ifend_rank = np.cumsum(nf_ranks)[rank]
    print(f'Rank: {rank}, nf: {nf_rank}, ifin: {ifin_rank}, ifend: {ifend_rank}')

    # Extract batch of frequency slices (in practice, this will be directly read from input file)
    G = Gwav_fft[ifin_rank:ifend_rank].astype(cdtype)
    print(f'Rank: {rank}, G: {G.shape}')

    # Define operator
    Fop = MPIMDC((1.0 * 0.004 * np.sqrt(par["nt"])) * G, nt=2 * par["nt"] - 1, nv=nvs, nfreq=nf,
                 dt=0.004, dr=1.0, twosided=True, fftengine="scipy", prescaled=True)

    # Apply forward
    x = pylops_mpi.DistributedArray(global_shape=(2 * par["nt"] - 1) * par["nx"] * nvs,
                                    partition=Partition.BROADCAST,
                                    dtype=dtype)
    x[:] = m.astype(dtype).ravel()
    xloc = x.asarray()

    y = Fop @ x
    yloc = y.asarray().real

    if rank == 0:
        plt.figure(figsize=(20, 6))
        plt.imshow(yloc.reshape(2 * par["nt"] - 1, par["ny"], nvs).transpose(0, 2, 1).reshape(2 * par["nt"] - 1, -1),
                   aspect="auto", interpolation="nearest",
                   cmap="gray", vmin=-yloc.max(), vmax=yloc.max())
        plt.savefig('data_mpi.png')

    # Compare with serial computation
    if rank == 0:
        Fop_ = MDC((1.0 * 0.004 * np.sqrt(par["nt"])) * Gwav_fft, nt=2 * par["nt"] - 1, nv=nvs,
                   dt=0.004, dr=1.0, twosided=True, prescaled=True)
        y_ = Fop_ @ m.ravel()

        print('Forward check', np.allclose(yloc, y_, atol=1e-6))
        print(yloc[:10], y_[:10])

        plt.figure(figsize=(20, 6))
        plt.imshow(y_.reshape(2 * par["nt"] - 1, par["ny"], nvs).transpose(0, 2, 1).reshape(2 * par["nt"] - 1, -1),
                   aspect="auto", interpolation="nearest",
                   cmap="gray", vmin=-y_.max(), vmax=y_.max())
        plt.savefig('data.png')

        plt.figure(figsize=(20, 6))
        plt.imshow((yloc-y_).reshape(2 * par["nt"] - 1, par["ny"], nvs).transpose(0, 2, 1).reshape(2 * par["nt"] - 1, -1),
                   aspect="auto", interpolation="nearest",
                   cmap="gray", vmin=-0.1*y_.max(), vmax=0.1*y_.max())
        plt.savefig('data_err.png')

    # Apply adjoint
    xadj = Fop.H @ y
    xadjloc = xadj.asarray().real

    if rank == 0:
        plt.figure(figsize=(20, 6))
        plt.imshow(xadjloc.reshape(2 * par["nt"] - 1, par["nx"], nvs).transpose(0, 2, 1).reshape(2 * par["nt"] - 1, -1),
                   aspect="auto", interpolation="nearest",
                   cmap="gray", vmin=-xadjloc.max(), vmax=xadjloc.max())
        plt.savefig('adj_mpi.png')

    # Compare with serial computation
    if rank == 0:
        xadj_ = Fop_.H @ y_

        print('Adjoint check', np.allclose(xadjloc, xadj_, atol=1e-6))
        print(xadjloc[:10], xadj_[:10])

        plt.figure(figsize=(20, 6))
        plt.imshow(xadj_.reshape(2 * par["nt"] - 1, par["nx"], nvs).transpose(0, 2, 1).reshape(2 * par["nt"] - 1, -1),
                   aspect="auto", interpolation="nearest",
                   cmap="gray", vmin=-xadj_.max(), vmax=xadj_.max())
        plt.savefig('adj.png')

        plt.figure(figsize=(20, 6))
        plt.imshow((xadjloc - xadj_).reshape(2 * par["nt"] - 1, par["nx"], nvs).transpose(0, 2, 1).reshape(2 * par["nt"] - 1, -1),
                   aspect="auto", interpolation="nearest",
                   cmap="gray", vmin=-0.1*xadj_.max(), vmax=0.1*xadj_.max())
        plt.savefig('adj_err.png')

    # Inverse
    x0 = pylops_mpi.DistributedArray(global_shape=(2 * par["nt"] - 1) * par["nx"] * nvs,
                                     partition=Partition.BROADCAST,
                                     dtype=cdtype)
    x0[:] = 0

    xinv = pylops_mpi.cgls(Fop, y, x0=x0, niter=50, show=True if rank == 0 else False)[0]
    xinvloc = xinv.asarray().real

    if rank == 0:
        plt.figure(figsize=(20, 6))
        plt.imshow(xinvloc.reshape(2 * par["nt"] - 1, par["nx"], nvs).transpose(0, 2, 1).reshape(2 * par["nt"] - 1, -1),
                   aspect="auto", interpolation="nearest",
                   cmap="gray", vmin=-xinvloc.max(), vmax=xinvloc.max())
        plt.savefig('inv_mpi.png')

    # Compare with serial computation
    if rank == 0:
        xinv_ = cgls(Fop_, y_, niter=50, show=True)[0].ravel()

        print('Inv check', np.allclose(xinvloc, xinv_, atol=1e-6))
        print(xinvloc[:10], xinv_[:10])

        plt.figure(figsize=(20, 6))
        plt.imshow(xinv_.reshape(2 * par["nt"] - 1, par["nx"], nvs).transpose(0, 2, 1).reshape(2 * par["nt"] - 1, -1),
                   aspect="auto", interpolation="nearest",
                   cmap="gray", vmin=-xinv_.max(), vmax=xinv_.max())
        plt.savefig('inv.png')

        print(xinvloc.shape, xinv_.shape)
        plt.figure(figsize=(20, 6))
        plt.imshow((xinvloc - xinv_).reshape(2 * par["nt"] - 1, par["nx"], nvs).transpose(0, 2, 1).reshape(2 * par["nt"] - 1, -1),
                   aspect="auto", interpolation="nearest",
                   cmap="gray", vmin=-0.1*xinv_.max(), vmax=0.1*xinv_.max())
        plt.savefig('inv_err.png')


if __name__ == '__main__':
    run()