r"""
Test distributed arrays

Run as: export OMP_NUM_THREADS=4; export MKL_NUM_THREADS=4; export NUMBA_NUM_THREADS=4; mpiexec -n 4 python Test_distr.py
"""

import os
import sys
import time
import numpy as np
import pylops_mpi

from mpi4py import MPI

plt.close("all")

def run():
    comm = MPI.COMM_WORLD
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()

    tic = time.perf_counter()  
    
    nil, nxl, nt = 400, 1000, 1025
    dtype = np.float32

    # Create distributed data
    d_dist = pylops_mpi.DistributedArray(global_shape=nil * nxl * nt, dtype=dtype)
    print(rank, d_dist.local_shape)
    d_dist[:] = np.ones(d_dist.local_shape, dtype=dtype)
    print(rank, nil, nxl, nt)
    d1 = d_dist.asarray()#.reshape((nil, nxl, nt))


if __name__ == '__main__':
    run()