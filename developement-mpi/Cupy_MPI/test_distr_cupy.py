r"""
Test distributed cupy arrays (see https://github.com/mpi4py/mpi4py/blob/master/demo/cuda-aware-mpi/use_cupy.py)

Run as: module load cuda/11.5.0/gcc-7.5.0-syen6pj; export OMP_NUM_THREADS=4; export MKL_NUM_THREADS=4; export NUMBA_NUM_THREADS=4; mpiexec -n 2 python test_distr_cupy.py
"""

import os
import sys
import time
import numpy as np
import cupy as cp
import pylops
import pylops_mpi

from mpi4py import MPI


def run():
    comm = MPI.COMM_WORLD
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    cp.cuda.Device(device=rank).use()

    tic = time.perf_counter()  
    
    nxl, nt = 20, 20
    dtype = np.float32

    # Create distributed data (broadcast)
    d_dist = pylops_mpi.DistributedArray(global_shape=nxl * nt,                                   
                                         partition=pylops_mpi.Partition.BROADCAST,
                                         engine="cupy", dtype=dtype)
    print('Local shape', rank, d_dist.local_shape)
    d_dist[:] = cp.ones(d_dist.local_shape, dtype=dtype)
    print(rank, nxl, nt)
    d1 = d_dist.asarray()
    print(rank, type(d1), d1.device)
    sys.stdout.flush()
    comm.barrier()

    # Create and apply VStack operator
    if rank == 0:
        print("Create and apply VStack operator")
    
    Sop = pylops.MatrixMult(cp.ones((nxl, nxl)), otherdims=(nt, ))
    HOp = pylops_mpi.MPIVStack(ops=[Sop, ])
    y_dist = HOp @ d_dist
    y = y_dist.asarray()
    print(rank, type(y), y.device)
    sys.stdout.flush()

    # Create distributed data (scatter)
    d_dist = pylops_mpi.DistributedArray(global_shape=nxl * nt, 
                                         engine="cupy", dtype=dtype)
    print('Local shape', rank, d_dist.local_shape)
    d_dist[:] = cp.ones(d_dist.local_shape, dtype=dtype)
    print(rank, nxl, nt)
    d1 = d_dist.asarray()
    print(rank, type(d1), d1.device)
    
    # Create and apply HStack operator
    if rank == 0:
        print("Create and apply HStack operator")
    
    Sop = pylops.MatrixMult(cp.ones((nxl, nxl // 2)), otherdims=(nt, ))
    HOp = pylops_mpi.MPIHStack(ops=[Sop, ])
    y_dist = HOp @ d_dist
    y = y_dist.asarray()
    print(rank, y, type(y), y.device)
    sys.stdout.flush()

    if rank == 0:
        Sop = pylops.MatrixMult(cp.ones((nxl, nxl // 2)), otherdims=(nt, ))
        HOp = pylops.HStack(ops=[Sop, ] * size, forceflat=True)
        y1 = HOp @ d1.ravel()
        print(y1, type(y1), y1.device)
        print(y.shape, y1.shape)
        print('HStack against pylops', y - y1)
    
    # Stacked arrays
    subarr1 = pylops_mpi.DistributedArray(global_shape=size * 10,
                                          engine="cupy", partition=pylops_mpi.Partition.SCATTER,
                                          axis=0)
    subarr2 = pylops_mpi.DistributedArray(global_shape=size * 4,
                                          engine="cupy", partition=pylops_mpi.Partition.SCATTER,
                                          axis=0)
    # Filling the local arrays
    subarr1[:], subarr2[:] = cp.ones(subarr1.local_shape), 2 * cp.ones(subarr2.local_shape)
 
    arr1 = pylops_mpi.StackedDistributedArray([subarr1, subarr2])
    if rank == 0:
        print('Stacked array:', arr1)

    # Extract and print full array
    full_arr1 = arr1.asarray()
    if rank == 0:
        print('Full array:', full_arr1, type(full_arr1))

    # Modify the part of the first array in rank0
    if rank == 0:
        arr1[0][:] = 10 * cp.ones(subarr1.local_shape)
    full_arr1 = arr1.asarray()
    if rank == 0:
        print('Modified full array:', full_arr1, type(full_arr1), full_arr1.device)


if __name__ == '__main__':
    run()