r"""
Test Fredholm operator

Run as: export OMP_NUM_THREADS=4; export MKL_NUM_THREADS=4; export NUMBA_NUM_THREADS=4; mpiexec -n 2 python test_fredholm.py
"""
#!/usr/bin/env python
# coding: utf-8
#
# Fredholm object with MPI4PY
#
# Run: mpiexec -n 4 python fredholmmpi.py
#

import numpy as np
import pylops_mpi

from pylops.signalprocessing import Fredholm1

from mpi4py import MPI
from pylops_mpi.DistributedArray import local_split, Partition
from pylops_mpi.signalprocessing.Fredholm1 import MPIFredholm1


def run():
    comm = MPI.COMM_WORLD
    size = comm.Get_size() # number of nodes
    rank = comm.Get_rank() # rank of current node
    dtype = np.float64
    usematmul = False
    
    # Create part of G at each node (this will be eventually be read from file)
    ny, nx, nv, ns = 101, 81, 1, 45
    
    # Choose how to split sources to ranks
    ns_rank = local_split((ns, ), MPI.COMM_WORLD, Partition.SCATTER, 0)
    ns_ranks = np.concatenate(MPI.COMM_WORLD.allgather(ns_rank))
    isin_rank = np.insert(np.cumsum(ns_ranks)[:-1] , 0, 0)[rank]
    isend_rank = np.cumsum(ns_ranks)[rank]
    print(f'Rank: {rank}, ns: {ns_rank}, isin: {isin_rank}, isend: {isend_rank}')

    G = np.random.normal(0., 1., (ns_rank[0], ny, nx)).astype(dtype)
    print(f'Rank: {rank}, G: {G.shape}')

    G_ = np.vstack(comm.allgather(G))
    print(f'Rank: {rank}, G_: {G_.shape}')
        
    # Define operator
    Fop = MPIFredholm1(G, nv, ns, usematmul=usematmul, dtype=dtype)
    
    # Apply forward
    x = pylops_mpi.DistributedArray(global_shape=ns * nx * nv, 
                                    partition=Partition.BROADCAST,
                                    dtype=dtype)
    x[:] = np.random.normal(0., 1., (ns, nx, nv)).astype(dtype).ravel()
    xloc = x.asarray()

    y = Fop @ x
    yloc = y.asarray()

    # Compare with serial computation
    if rank == 0:
        Fop_ = Fredholm1(G_, nv, ns, usematmul=usematmul, dtype=dtype)
        y_ = Fop_ @ xloc
        print('Forward check', np.allclose(yloc, y_))

    
    # Apply adjoint
    xadj = Fop.H @ y
    xadjloc = xadj.asarray()

    # Compare with serial computation
    if rank == 0:
        xadj_ = Fop_.H @ yloc
        print('Adjoint check', np.allclose(xadjloc, xadj_))

if __name__ == '__main__':
    run()