r"""
Test derivatives with cupy arrays (see https://github.com/mpi4py/mpi4py/blob/master/demo/cuda-aware-mpi/use_cupy.py)

Run as: mpiexec -n 2 python test_derivative_cupy.py
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

      nx, ny = 11, 21
      x = cp.zeros((nx, ny))
      x[nx // 2, ny // 2] = 1.0

      # First derivative      
      Fop = pylops.FirstDerivative((nx, ny), axis=0, kind='forward',
                                   edge=False, order=5, dtype=np.float64)
      Fopmpi = pylops_mpi.MPIFirstDerivative((nx, ny), kind='forward',
                                             edge=False, order=5, dtype=np.float64)
      x_dist = pylops_mpi.DistributedArray.to_dist(x=x.flatten())
      y_dist = Fopmpi @ x_dist
      y = y_dist.asarray().reshape((nx, ny))
      
      if rank == 0:
            print('Check equivalency FirstDerivative forward with PyLops:', cp.allclose(y, Fop @ x))
      
      xadj_dist = Fopmpi.H @ x_dist
      xadj = xadj_dist.asarray().reshape((nx, ny))

      if rank == 0:
            print('Check equivalency FirstDerivative adjoint with PyLops:', cp.allclose(xadj, Fop.H @ x))
      
      # Second derivative      
      Fop = pylops.SecondDerivative((nx, ny), axis=0, dtype=np.float64)
      Fopmpi = pylops_mpi.MPISecondDerivative((nx, ny), dtype=np.float64)
      x_dist = pylops_mpi.DistributedArray.to_dist(x=x.flatten())
      y_dist = Fopmpi @ x_dist
      y = y_dist.asarray().reshape((nx, ny))
      
      if rank == 0:
            print('Check equivalency SecondDerivative forward with PyLops:', cp.allclose(y, Fop @ x))
      
      xadj_dist = Fopmpi.H @ x_dist
      xadj = xadj_dist.asarray().reshape((nx, ny))

      if rank == 0:
            print('Check equivalency SecondDerivative adjoint with PyLops:', cp.allclose(xadj, Fop.H @ x))
      
      # Laplacian
      Fop = pylops.Laplacian((nx, ny), weights=(1, 1), dtype=np.float64)
      Fopmpi = pylops_mpi.MPILaplacian(dims=(nx, ny), weights=(1, 1), dtype=np.float64)
      x_dist = pylops_mpi.DistributedArray.to_dist(x=x.flatten())
      y_dist = Fopmpi @ x_dist
      y = y_dist.asarray().reshape((nx, ny))
      
      if rank == 0:
            print('Check equivalency Laplacian forward with PyLops:', cp.allclose(y, Fop @ x))
      
      xadj_dist = Fopmpi.H @ x_dist
      xadj = xadj_dist.asarray().reshape((nx, ny))

      if rank == 0:
            print('Check equivalency Laplacian adjoint with PyLops:', cp.allclose(xadj, Fop.H @ x))
      
      # Gradient
      Fop = pylops.Gradient((nx, ny), dtype=np.float64)
      Fopmpi = pylops_mpi.MPIGradient(dims=(nx, ny), dtype=np.float64)
      x_dist = pylops_mpi.DistributedArray.to_dist(x=x.flatten())
      y_dist = Fopmpi @ x_dist
      y = y_dist.asarray().reshape((2, nx, ny))
      
      if rank == 0:
            print('Check equivalency Gradient forward with PyLops:', cp.allclose(y, Fop @ x))
      
      xadj_dist = Fopmpi.H @ y_dist
      xadj = xadj_dist.asarray().reshape((nx, ny))

      if rank == 0:
            print('Check equivalency Gradient adjoint with PyLops:', cp.allclose(xadj, Fop.H @ y))
     
if __name__ == '__main__':
      run()