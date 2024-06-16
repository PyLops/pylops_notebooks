r"""
Test MDC operator with random numbers

Run as: export OMP_NUM_THREADS=4; export MKL_NUM_THREADS=4; export NUMBA_NUM_THREADS=4; mpiexec -n 2 python test_mdc_random.py
"""
#!/usr/bin/env python
# coding: utf-8
#
# Fredholm object with MPI4PY
#
# Run: mpiexec -n 4 python fredholmmpi.py
#

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
    cdtype = np.complex64

    # Create part of G at each node (this will be eventually be read from file)
    ny, nx, nv, ns = 101, 81, 1, 45
    
    # Choose how to split sources to ranks
    ns_rank = local_split((ns, ), MPI.COMM_WORLD, Partition.SCATTER, 0)
    ns_ranks = np.concatenate(MPI.COMM_WORLD.allgather(ns_rank))
    isin_rank = np.insert(np.cumsum(ns_ranks)[:-1] , 0, 0)[rank]
    isend_rank = np.cumsum(ns_ranks)[rank]
    print(f'Rank: {rank}, ns: {ns_rank}, isin: {isin_rank}, isend: {isend_rank}')

    G = np.random.normal(0., 1., (ns_rank[0], ny, nx)).astype(dtype) + \
        1j * np.random.normal(0., 1., (ns_rank[0], ny, nx)).astype(dtype)
    print(f'Rank: {rank}, G: {G.shape}')

    G_ = np.vstack(comm.allgather(G))
    print(f'Rank: {rank}, G_: {G_.shape}')
        
    # Define operator
    nt = 201
    Fop = MPIMDC(G, nt=2*nt-1, nv=nv, nfreq=ns, dt=1, dr=1,
                 twosided=True)
    
    # Apply forward
    x = pylops_mpi.DistributedArray(global_shape=(2*nt-1) * nx * nv, 
                                    partition=Partition.BROADCAST,
                                    dtype=dtype)
    x[:] = np.random.normal(0., 1., (2*nt-1, nx, nv)).astype(dtype).ravel()
    xloc = x.asarray()

    y = Fop @ x
    yloc = y.asarray()
        
    if rank == 0:
        plt.figure()
        plt.imshow(yloc.reshape(2 * nt - 1, ny), aspect="auto", interpolation="nearest",
                   cmap="gray", vmin=-yloc.max(), vmax=yloc.max())
        plt.savefig('datarand_mpi.png')

    # Compare with serial computation
    if rank == 0:
        Fop_ = MDC(G_, nt=2*nt-1, nv=nv, dt=1, dr=1,
                   twosided=True)
        y_ = Fop_ @ xloc
        print('Forward check', np.allclose(yloc, y_))
        print(yloc[:10], y_[:10])

        plt.figure()
        plt.imshow(y_.reshape(2 * nt - 1, ny), aspect="auto", interpolation="nearest",
                   cmap="gray", vmin=-y_.max(), vmax=y_.max())
        plt.savefig('datarand.png')

if __name__ == '__main__':
    run()