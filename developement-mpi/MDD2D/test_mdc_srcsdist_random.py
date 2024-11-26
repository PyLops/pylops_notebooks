r"""
Test MDC operator with random numbers distributing data over sources

Run as: export OMP_NUM_THREADS=4; export MKL_NUM_THREADS=4; export NUMBA_NUM_THREADS=4; mpiexec -n 8 python test_mdc_srcsdist_random.py
"""

import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import pylops_mpi

from pylops.basicoperators import Transpose
from pylops.waveeqprocessing import MDC

from mpi4py import MPI
from pylops_mpi.DistributedArray import local_split, Partition
from pylops_mpi.utils.dottest import dottest as mpidottest


def run():
    comm = MPI.COMM_WORLD
    size = comm.Get_size() # number of nodes
    rank = comm.Get_rank() # rank of current node
    dtype = np.float64
    cdtype = np.complex128
    rng = np.random.default_rng()

    # Create part of G at each node (this will be eventually be read from file)
    # G will be of size ns x ny x nx
    # x will be of size nt x nx x nv
    # y will be of size nt x ny x nv
    # nt, ns = 1001, 501
    # ny, nx, nv = 1001, 501, 201
    nt, ns = 1201, 401
    ny, nx, nv = 501, 301, 101

    if rank == 0:
        print(f'Testing MDC with {size} ranks')
        print(f'G = {ns} x {ny} x {nx}')
        print(f'x = {nt} x {nx} x {nv}')
        print(f'y = {nt} x {ny} x {nv}')
        print(f'-----------------------------\n\n\n')
        sys.stdout.flush()

    # Choose how to split sources to ranks
    ny_rank = local_split((ny, ), MPI.COMM_WORLD, Partition.SCATTER, 0)
    ny_ranks = np.concatenate(MPI.COMM_WORLD.allgather(ny_rank))
    iyin_rank = np.insert(np.cumsum(ny_ranks)[:-1], 0, 0)[rank]
    iyend_rank = np.cumsum(ny_ranks)[rank]
    print(f'Rank: {rank}, ns: {ny_rank}, iyin: {iyin_rank}, iyend: {iyend_rank}')
    sys.stdout.flush()

    np.random.seed(10)
    G = np.zeros((ns, ny_rank[0], nx), dtype=cdtype)
    for iy_rank in range(ny_rank[0]):
        G[:, iy_rank] = (rng.standard_normal(size=(ns, nx), dtype=dtype) + \
                        1j * rng.standard_normal(size=(ns, nx), dtype=dtype))
    # G = (rng.standard_normal(size=(ns, ny_rank[0], nx), dtype=dtype) + \
    #    1j * rng.standard_normal(size=(ns, ny_rank[0], nx), dtype=dtype))
    print(f'Rank: {rank}, G: {G.shape}')
    sys.stdout.flush()

    G_ = np.concatenate(comm.allgather(G), axis=1)
    if rank == 0:
        print(f'Rank: {rank}, G_: {G_.shape}')
    else:
        del G_

    # Define operator
    Fop = MDC(G, nt=nt, nv=nv, dt=1, dr=1,
              usematmul=True, saveGt=False,
              twosided=False, prescaled=True,
              fftengine="scipy")
    Top = Transpose(dims=(nt, ny_rank[0], nv), axes=(1, 0, 2), dtype=dtype)
    Foptot = pylops_mpi.MPIVStack(ops=[Top * Fop, ])

    # Define distributed array for input
    np.random.seed(10)
    x = pylops_mpi.DistributedArray(global_shape=nt * nx * nv,
                                    partition=Partition.UNSAFE_BROADCAST,
                                    dtype=dtype)
    x[:] = np.random.normal(0., 1., (nt, nx, nv)).astype(dtype).ravel()
    xloc = x.asarray()

    # Apply forward
    if rank == 0:
        tstart = time.perf_counter()
    y = Foptot @ x
    yloc = y.asarray().real.reshape(ny, nt, nv).transpose(1, 0, 2).ravel()
    comm.barrier()
    if rank == 0:
        tstop = time.perf_counter()
        print("MPIMDCforward - Elapsed time:", tstop - tstart)
        print("MPIMDCforward - Dtype output:", yloc.dtype)
        sys.stdout.flush()

    # Dot test
    mpidottest(Foptot, x, y, raiseerror=False, verb=True)

    # Compare with serial computation
    if rank == 0:
        Fop_ = MDC(G_, nt=nt, nv=nv, dt=1, dr=1,
                   usematmul=True, saveGt=False,
                   twosided=False, prescaled=True,
                   fftengine="scipy")
        tstart = time.perf_counter()
        y_ = (Fop_ @ xloc).real
        tstop = time.perf_counter()
        print("MDCforward - Elapsed time:", tstop - tstart)
        print("MDCforward - Dtype output:", y_.dtype)

        print('Forward check', np.allclose(yloc, y_))
        print('Forward check', np.linalg.norm(yloc - y_) / np.linalg.norm(yloc))
        print(yloc[:10], y_[:10])

if __name__ == '__main__':
    run()