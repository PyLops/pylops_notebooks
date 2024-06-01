r"""
Test distributed cupy arrays (see https://github.com/mpi4py/mpi4py/blob/master/demo/cuda-aware-mpi/use_cupy.py)

Run as: module load cuda/11.5.0/gcc-7.5.0-syen6pj; export OMP_NUM_THREADS=4; export MKL_NUM_THREADS=4; export NUMBA_NUM_THREADS=4; mpiexec -n 2 python test_distr_cupy_1.py
"""

import os
import sys
import time
import numpy as np
import cupy as cp
import pylops
import pylops_mpi

from mpi4py import MPI
from pylops_mpi.DistributedArray import local_split, Partition


def run():
    comm = MPI.COMM_WORLD
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    cp.cuda.Device(device=rank).use()

    # Defining the global shape of the distributed array
    global_shape = (10, 5)
    
    # Scatter 1st axis
    arr = pylops_mpi.DistributedArray(global_shape=global_shape,
                                      partition=pylops_mpi.Partition.SCATTER,
                                      engine='cupy', axis=0)
    # Filling the local arrays
    arr[:] = cp.arange(arr.local_shape[0] * arr.local_shape[1] * arr.rank,
                       arr.local_shape[0] * arr.local_shape[1] * (arr.rank + 1)).reshape(arr.local_shape)
    print(rank, arr.local_array.shape, arr.local_array.device)
    sys.stdout.flush()
    comm.barrier()

    # Scatter 2nd axis
    arr = pylops_mpi.DistributedArray(global_shape=global_shape,
                                      partition=pylops_mpi.Partition.SCATTER,
                                      engine='cupy', axis=1)
    # Filling the local arrays
    arr[:] = cp.arange(arr.local_shape[0] * arr.local_shape[1] * arr.rank,
                    arr.local_shape[0] * arr.local_shape[1] * (arr.rank + 1)).reshape(arr.local_shape)
    print(rank, arr.local_array.shape, arr.local_array.device)
    sys.stdout.flush()
    comm.barrier()

    # Choosing local shape
    local_shape = local_split(global_shape, MPI.COMM_WORLD, Partition.SCATTER, 0)
    # Assigning local_shapes(List of tuples)
    local_shapes = ((6, 5), (4, 5)) # MPI.COMM_WORLD.allgather(local_shape)
    arr = pylops_mpi.DistributedArray(global_shape=global_shape, 
                                      local_shapes=local_shapes, 
                                      engine='cupy', axis=0)
    arr[:] = cp.arange(arr.local_shape[0] * arr.local_shape[1] * arr.rank,
                    arr.local_shape[0] * arr.local_shape[1] * (arr.rank + 1)).reshape(arr.local_shape)
    print(rank, arr.local_array.shape, arr.local_array.device)
    sys.stdout.flush()
    comm.barrier()

    # Convert cupy to DistributedArray
    n = global_shape[0] * global_shape[1]
    # Array to be distributed
    array = cp.arange(n) / float(n)
    arr1 = pylops_mpi.DistributedArray.to_dist(x=array.reshape(global_shape), axis=1)
    array = array / 2.0
    arr2 = pylops_mpi.DistributedArray.to_dist(x=array.reshape(global_shape), axis=1)
    print(rank, arr1.local_array, arr1.local_array.shape, arr1.local_array.device)
    sys.stdout.flush()
    comm.barrier()

    # Scaling
    scale_arr = .5 * arr1
    print(rank, scale_arr.local_array, 
          scale_arr.local_array.shape, 
          scale_arr.local_array.device)
    sys.stdout.flush()
    comm.barrier()

    # Sum
    sum_arr = arr1 + arr2
    print(rank, sum_arr.local_array, 
          sum_arr.local_array.shape, 
          sum_arr.local_array.device)
    sys.stdout.flush()
    comm.barrier()

    # Element-wise In-place Addition
    sum_arr += arr2
    print(rank, sum_arr.local_array.shape, 
          sum_arr.local_array.device)
    sys.stdout.flush()
    comm.barrier()
   
    # Subtraction
    diff_arr = arr1 - arr2
    print(rank, diff_arr.local_array.shape, 
          diff_arr.local_array.device)
    
    # Product
    mult_arr = arr1 * arr2
    print(rank, mult_arr.local_array.shape, 
          mult_arr.local_array.device)
    

if __name__ == '__main__':
    run()