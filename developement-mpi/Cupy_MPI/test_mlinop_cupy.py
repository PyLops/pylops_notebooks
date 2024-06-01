r"""
Test mplinop with cupy arrays (see https://github.com/mpi4py/mpi4py/blob/master/demo/cuda-aware-mpi/use_cupy.py)

Run as: mpiexec -n 2 python test_mlinop_cupy.py
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

      Ny, Nx = 11, 22
      Fop = pylops.FirstDerivative(dims=(Ny, Nx), axis=0, dtype=np.float64)
      Mop = pylops_mpi.asmpilinearoperator(Op=Fop)
      print(Mop)

      x = pylops_mpi.DistributedArray(global_shape=Ny * Nx, engine='cupy',
                                      partition=pylops_mpi.Partition.BROADCAST)
      x[:] = cp.ones(x.local_shape)
      y_dist = Mop @ x
      y = y_dist.asarray()
      if rank == 0:
            print(f'y: {y_dist}, {type(y)}')

      Sop = pylops.SecondDerivative(dims=(Ny, Nx), axis=0, dtype=np.float64)
      VStack = pylops_mpi.MPIVStack(ops=[(rank + 1) * Sop, ])
      FullOp = VStack @ Mop

      X = cp.zeros(shape=(Ny, Nx))
      X[Ny // 2, Nx // 2] = 1
      X1 = X.ravel()
      x = pylops_mpi.DistributedArray(global_shape=Ny * Nx, engine='cupy',
                                      partition=pylops_mpi.Partition.BROADCAST)
      x[:] = X1
      y_dist = FullOp @ x
      y = y_dist.asarray().reshape((size * Ny, Nx))
      if rank == 0:
            print(f'y: {y_dist}, {type(y)}')

      x = pylops_mpi.DistributedArray(global_shape=size * Ny * Nx, engine='cupy',
                                      partition=pylops_mpi.Partition.SCATTER)
      x[:] = X1
      y_dist = FullOp.H @ x
      y = y_dist.asarray().reshape((Ny, Nx))
      if rank == 0:
            print(f'y: {y_dist}, {type(y)}')

if __name__ == '__main__':
      run()