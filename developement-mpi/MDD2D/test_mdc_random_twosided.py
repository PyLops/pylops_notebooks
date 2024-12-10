r"""
Test MDC operator with random numbers

Run as: export OMP_NUM_THREADS=4; export MKL_NUM_THREADS=4; export NUMBA_NUM_THREADS=4; mpiexec -n 8 python test_mdc_random.py
"""

import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import pylops_mpi

from pylops.waveeqprocessing import MDC

from mpi4py import MPI
from pylops_mpi.DistributedArray import local_split, Partition
from pylops_mpi.waveeqprocessing.MDC import MPIMDC


def run():
    comm = MPI.COMM_WORLD
    size = comm.Get_size() # number of nodes
    rank = comm.Get_rank() # rank of current node
    dtype = np.float32

    # Create part of G at each node (this will be eventually be read from file)
    # G will be of size ns x ny x nx
    # x will be of size 2*nt-1 x nx x nv
    # y will be of size 2*nt-1 x ny x nv
    # nt, ns = 1001, 501
    # ny, nx, nv = 1001, 501, 201
    nt, ns = 1201, 801
    ny, nx, nv = 501, 301, 101

    if rank == 0:
        print(f'Testing MDC with {size} ranks')
        print(f'G = {ns} x {ny} x {nx}')
        print(f'x = {2*nt-1} x {nx} x {nv}')
        print(f'y = {2*nt-1} x {ny} x {nv}')
        print(f'-----------------------------\n\n\n')
        sys.stdout.flush()

    # Choose how to split sources to ranks
    ns_rank = local_split((ns, ), MPI.COMM_WORLD, Partition.SCATTER, 0)
    ns_ranks = np.concatenate(MPI.COMM_WORLD.allgather(ns_rank))
    isin_rank = np.insert(np.cumsum(ns_ranks)[:-1], 0, 0)[rank]
    isend_rank = np.cumsum(ns_ranks)[rank]
    print(f'Rank: {rank}, ns: {ns_rank}, isin: {isin_rank}, isend: {isend_rank}')
    sys.stdout.flush()

    G = np.random.normal(0., 1., (ns_rank[0], ny, nx)).astype(dtype) + \
        1j * np.random.normal(0., 1., (ns_rank[0], ny, nx)).astype(dtype)
    print(f'Rank: {rank}, G: {G.shape}')
    sys.stdout.flush()


    G_ = np.vstack(comm.allgather(G))
    if rank == 0:
        print(f'Rank: {rank}, G_: {G_.shape}')
    else:
        del G_

    # Define operator
    Fop = MPIMDC(G, nt=2*nt-1, nv=nv, nfreq=ns, dt=1, dr=1,
                 usematmul=True, saveGt=False, twosided=True)
    
    # Define distributed array for input
    x = pylops_mpi.DistributedArray(global_shape=(2*nt-1) * nx * nv, 
                                    partition=Partition.BROADCAST,
                                    dtype=dtype)
    x[:] = np.random.normal(0., 1., (2*nt-1, nx, nv)).astype(dtype).ravel()
    xloc = x.asarray()

    # Apply forward
    if rank == 0:
        tstart = time.perf_counter()
    y = Fop @ x
    yloc = y.asarray().real
    comm.barrier()
    if rank == 0:
        tstop = time.perf_counter()
        print("MPIMDCforward - Elapsed time:", tstop - tstart)
        sys.stdout.flush()

    if rank == 0:
        plt.figure()
        plt.imshow(yloc.reshape(2 * nt - 1, ny, nv)[..., 0], aspect="auto", interpolation="nearest",
                   cmap="gray", vmin=-yloc.max(), vmax=yloc.max())
        plt.savefig('datarand_mpi.png')

    # Compare with serial computation
    if rank == 0:
        Fop_ = MDC(G_, nt=2*nt-1, nv=nv, dt=1, dr=1,
                   usematmul=True, saveGt=False, twosided=True)
        tstart = time.perf_counter()
        y_ = (Fop_ @ xloc).real
        tstop = time.perf_counter()
        print("MDCforward - Elapsed time:", tstop - tstart)

        print('Forward check', np.allclose(yloc, y_))
        print(yloc[:10], y_[:10])

        plt.figure()
        plt.imshow(y_.reshape(2 * nt - 1, ny, nv)[..., 0], aspect="auto", interpolation="nearest",
                   cmap="gray", vmin=-y_.max(), vmax=y_.max())
        plt.savefig('datarand.png')

if __name__ == '__main__':
    run()