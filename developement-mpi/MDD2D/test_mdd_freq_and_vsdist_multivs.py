r"""
Test MDD with synthetic seismic data distributing data over frequencies and virtual sources with multiple virtual sources

Run as: export OMP_NUM_THREADS=4; export MKL_NUM_THREADS=4; export NUMBA_NUM_THREADS=4; mpiexec -n 6 python test_mdd_freq_and_vsdist_multivs.py  --nfranks 2 --nvsranks 3
"""
import sys
import time
import argparse
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
from MDC2Ddistr import MPIMDC2Ddistr
from utils import *


def run(parser):
    parser.add_argument('--nfranks', type=int, default=2, help="Number of ranks for frequencies")
    parser.add_argument("--nvsranks", type=int, default=2, help="Number of ranks for virtual sources")
    args = parser.parse_args()

    # MPI and other input parameters
    comm = MPI.COMM_WORLD
    size = comm.Get_size() # number of nodes
    rank = comm.Get_rank() # rank of current node
    dtype = np.float32
    cdtype = np.complex64

    # Check rank consistency
    if args.nfranks * args.nvsranks != size:
        raise ValueError(f"Number of ranks ({size}) must be equal to number of frequency ranks times"
                         f"number of virtual sources ranks ({args.nfranks} * {args.nvsranks} = {args.nfranks * args.nvsranks})")

    # Turn 1d ranks into 2d ranks
    rankf, rankvs = np.unravel_index(rank, (args.nfranks, args.nvsranks))
    rankf_all = np.array(MPI.COMM_WORLD.allgather(rankf))
    rankvs_all = np.array(MPI.COMM_WORLD.allgather(rankvs))
    if rank == 0:
        print(f'Rank {rank}: rankf_all {rankf_all}, rankvs_all {rankvs_all}')
    pause(comm)
    print(f'Rank {rank}: rankf {rankf}, rankvs {rankvs}')
    pause(comm)

    # Create subcomms
    subcomm_f = comm.Split(color=rankf, key=rank)
    print('Rank', rank, ': Subcomm_f rank', subcomm_f.Get_rank(), 'color', rankf)
    pause(comm)
    subcomm_vs = comm.Split(color=rankvs, key=rank)
    print('Rank', rank, ': Subcomm_vs rank', subcomm_vs.Get_rank(), 'color', rankvs)
    pause(comm)

    # Input parameters
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
    nvs = 21
    t0_m = 0.2
    vrms_m = 1100.0
    amp_m = 1.0

    t0_G = (0.2, 0.5, 0.7)
    vrms_G = (1200.0, 1500.0, 2000.0)
    amp_G = (1.0, 0.6, 0.5)

    # Choose how to split frequencies to ranks
    nf = par["nfmax"]
    nf_ranks, ifin_ranks, ifend_ranks, nf_rank, ifin_rank, ifend_rank = \
        local_split_startend(nf, rank, subcomm_vs, args.nvsranks, rep=np.repeat)
    if rank == 0:
        print(f'Rank {rank}: nf_ranks {nf_ranks}, ifin_ranks: {ifin_ranks}, ifend_ranks: {ifend_ranks}')
    pause(comm)
    print(f'Rank: {rank}: nf: {nf_rank}, ifin: {ifin_rank}, ifend: {ifend_rank}')
    pause(comm)

    # Choose how to split virtual sources to ranks
    nvs_ranks, ivsin_ranks, ivsend_ranks, nvs_rank, ivsin_rank, ivsend_rank = \
        local_split_startend(nvs, rank, subcomm_f, args.nfranks, rep=np.tile)
    if rank == 0:
        print(f'Rank {rank}: nvs_ranks: {ivsin_ranks}, ivsin_ranks: {ivsin_ranks}, ivsend_ranks: {ivsend_ranks}')
    pause(comm)
    print(f'Rank {rank}: nvs: {nvs_rank}, ivsin: {ivsin_rank}, ivsend: {ivsend_rank}')
    pause(comm)

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

    # Display model with wavelet
    if rank == 0:
        plt.figure(figsize=(20, 6))
        plt.imshow(mwav.reshape(2 * par["nt"] - 1, par["nx"], nvs).transpose(0, 2, 1).reshape(2 * par["nt"] - 1, -1),
                   aspect="auto", interpolation="nearest",
                   cmap="gray", vmin=-mwav.max(), vmax=mwav.max())
        plt.savefig('mwav_distrfvs_mpi.png')

    # Extract batch of frequency slices (in practice, this will be directly read from input file)
    G = Gwav_fft[ifin_rank:ifend_rank].astype(cdtype)
    print(f'Rank: {rank}, G: {G.shape}')
    pause(comm)

    # Extract batch of virtual points (in practice, this will be directly read from input file)
    mrank = m[..., ivsin_rank:ivsend_rank].astype(dtype)
    print(f'Rank: {rank}, m: {mrank.shape}')
    pause(comm)

    # Define operator
    Fop = MPIMDC2Ddistr((1.0 * 0.004 * np.sqrt(par["nt"])) * G, nt=2 * par["nt"] - 1, nv=nvs, nfreq=nf,
                        subcomm_f=subcomm_f, subcomm_v=subcomm_vs, mask=rankf_all,
                        dt=0.004, dr=1.0, twosided=True, fftengine="scipy",
                        usematmul=True, prescaled=True)

    # Apply forward
    x = pylops_mpi.DistributedArray(global_shape=(2 * par["nt"] - 1) * par["nx"] * nvs * args.nfranks,
                                    local_shapes=[(2 * par["nt"] - 1) * par["nx"] * nvsr for nvsr in nvs_ranks],
                                    partition=Partition.SCATTER, mask=rankf_all,
                                    dtype=dtype)
    x[:] = mrank.astype(dtype).ravel()

    y = Fop @ x
    yloc = y.asarray().real

    # Reorganize yloc to get individual portions...
    yloc1 = reorganize_distributed_2d(yloc, ((2 * par["nt"] - 1), par["ny"], nvs),
                                      y.local_shapes, ivsin_ranks, ivsend_ranks)
    """
    # Checking that the first and second repetitions of a masked array are identical
    yloc2 = np.zeros(((2 * par["nt"] - 1), par["ny"], nvs))
    for ivs in range(args.nvsranks, 2 * args.nvsranks):
        # print('y.local_shapes', y.local_shapes)
        # print('ivs', ivs, 0 if ivs == 0 else np.sum([ls[0] for ls in y.local_shapes[:ivs]]), np.sum([ls[0] for ls in y.local_shapes[:ivs + 1]]))
        yloc_tmp = yloc[0 if ivs == 0 else np.sum([ls[0] for ls in y.local_shapes[:ivs]]): np.sum(
            [ls[0] for ls in y.local_shapes[:ivs + 1]])].reshape((2 * par["nt"] - 1), par["ny"], -1)
        yloc2[:, :, ivsin_ranks[ivs - args.nvsranks]:ivsend_ranks[ivs - args.nvsranks]] = yloc_tmp
    print('np.allclose(yloc1, yloc2)', np.allclose(yloc1, yloc2))
    """

    # Display data
    if rank == 0:
        plt.figure(figsize=(20, 6))
        plt.imshow(yloc1.transpose(0, 2, 1).reshape(2 * par["nt"] - 1, -1),
                   aspect="auto", interpolation="nearest",
                   cmap="gray", vmin=-0.1*yloc.max(), vmax=0.1*yloc.max())
        plt.savefig('data_distrfvs_mpi.png')

    # Compare with serial computation
    if rank == 0:
        Fop_ = MDC((1.0 * 0.004 * np.sqrt(par["nt"])) * Gwav_fft, nt=2 * par["nt"] - 1, nv=nvs,
                   dt=0.004, dr=1.0, twosided=True, prescaled=True)
        y_ = Fop_ @ m.ravel()
        plt.figure(figsize=(20, 6))
        plt.imshow(y_.reshape(2 * par["nt"] - 1, par["ny"], nvs).transpose(0, 2, 1).reshape(2 * par["nt"] - 1, -1),
                   aspect="auto", interpolation="nearest",
                   cmap="gray", vmin=-0.1*y_.max(), vmax=0.1*y_.max())
        plt.savefig('data_distrfvs.png')

        plt.figure(figsize=(20, 6))
        plt.imshow((yloc1 - y_.reshape(2 * par["nt"] - 1, par["ny"], nvs)).transpose(0, 2, 1).reshape(2 * par["nt"] - 1, -1),
                   aspect="auto", interpolation="nearest",
                   cmap="gray", vmin=-0.1*y_.max(), vmax=0.1*y_.max())
        plt.savefig('data_distrfvs_err.png')

    # Apply adjoint
    xadj = Fop.H @ y
    xadjloc = xadj.asarray().real

    # Reorganize xadjloc
    xadjloc1 = reorganize_distributed_2d(xadjloc, ((2 * par["nt"] - 1), par["nx"], nvs),
                                         xadj.local_shapes, ivsin_ranks, ivsend_ranks)

    # Display adjoint
    if rank == 0:
        plt.figure(figsize=(20, 6))
        plt.imshow(xadjloc1.transpose(0, 2, 1).reshape(2 * par["nt"] - 1, -1),
                   aspect="auto", interpolation="nearest",
                   cmap="gray", vmin=-0.1*xadjloc.max(), vmax=0.1*xadjloc.max())
        plt.savefig('adj_distrfvs_mpi.png')

    # Compare with serial computation
    if rank == 0:
        xadj_ = Fop_.H @ y_

        plt.figure(figsize=(20, 6))
        plt.imshow(xadj_.reshape(2 * par["nt"] - 1, par["nx"], nvs).transpose(0, 2, 1).reshape(2 * par["nt"] - 1, -1),
                   aspect="auto", interpolation="nearest",
                   cmap="gray", vmin=-0.1*xadj_.max(), vmax=0.1*xadj_.max())
        plt.savefig('adj.png')

        plt.figure(figsize=(20, 6))
        plt.imshow((xadjloc1 - xadj_.reshape(2 * par["nt"] - 1, par["nx"], nvs)).transpose(0, 2, 1).reshape(2 * par["nt"] - 1, -1),
                   aspect="auto", interpolation="nearest",
                   cmap="gray", vmin=-0.1*xadj_.max(), vmax=0.1*xadj_.max())
        plt.savefig('adj_distrfvs_err.png')

    # Inverse
    x0 = pylops_mpi.DistributedArray(global_shape=(2 * par["nt"] - 1) * par["nx"] * nvs * args.nfranks,
                                     local_shapes=[(2 * par["nt"] - 1) * par["nx"] * nvsr for nvsr in nvs_ranks],
                                     partition=Partition.SCATTER, mask=rankf_all,
                                     dtype=cdtype)
    x0[:] = 0

    xinv = pylops_mpi.cgls(Fop, y, x0=x0, niter=50, show=True if rank == 0 else False)[0]
    xinvloc = xinv.asarray().real

    # Reorganize xinvloc
    xinvloc1 = reorganize_distributed_2d(xinvloc, ((2 * par["nt"] - 1), par["nx"], nvs),
                                         xinv.local_shapes, ivsin_ranks, ivsend_ranks)

    # Display inverse
    if rank == 0:
        plt.figure(figsize=(20, 6))
        plt.imshow(xinvloc1.transpose(0, 2, 1).reshape(2 * par["nt"] - 1, -1),
                   aspect="auto", interpolation="nearest",
                   cmap="gray", vmin=-xinvloc.max(), vmax=xinvloc.max())
        plt.savefig('inv_distrfvs_mpi.png')

    # Compare with serial computation
    if rank == 0:
        xinv_ = cgls(Fop_, y_, niter=50, show=True)[0].ravel()

        plt.figure(figsize=(20, 6))
        plt.imshow(xinv_.reshape(2 * par["nt"] - 1, par["nx"], nvs).transpose(0, 2, 1).reshape(2 * par["nt"] - 1, -1),
                   aspect="auto", interpolation="nearest",
                   cmap="gray", vmin=-xinv_.max(), vmax=xinv_.max())
        plt.savefig('inv.png')

        plt.figure(figsize=(20, 6))
        plt.imshow((xinvloc1 - xinv_.reshape(2 * par["nt"] - 1, par["nx"], nvs)).transpose(0, 2, 1).reshape(2 * par["nt"] - 1, -1),
                   aspect="auto", interpolation="nearest",
                   cmap="gray", vmin=-0.1*xinv_.max(), vmax=0.1*xinv_.max())
        plt.savefig('inv_err.png')


if __name__ == '__main__':
    description = '2D Multi-Dimensional Deconvolution - distributing data over frequencies and virtual sources'
    run(argparse.ArgumentParser(description=description))
