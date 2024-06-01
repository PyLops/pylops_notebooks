r"""
Test CG/CGLS with distributed cupy arrays (see https://github.com/mpi4py/mpi4py/blob/master/demo/cuda-aware-mpi/use_cupy.py)

Run as: mpiexec -n 2 python test_cg_cgls.py
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

    N, M = 200, 100
    Mop = pylops.MatrixMult(A=cp.random.normal(0, 1, (N, M)))
    BDiag = pylops_mpi.MPIBlockDiag(ops=[Mop, ], dtype=np.float64)

    x = pylops_mpi.DistributedArray(size * M, engine="cupy", dtype=np.float64)
    x[:] = cp.ones(M, dtype=np.float64)
    x_array = x.asarray()
    y = BDiag @ x

    # CG
    NormEqOp = BDiag.H @ BDiag
    ynorm = BDiag.H @ y
    x0 = pylops_mpi.DistributedArray(BDiag.shape[1], engine="cupy", dtype=np.float64)
    x0[:] = 0  
    xcg = pylops_mpi.optimization.basic.cg(NormEqOp, ynorm, x0=x0, niter=100, show=True)[0]
    xcg_array = xcg.asarray()
    if rank == 0:
        print('CG error', np.linalg.norm(x_array - xcg_array))

    # CGLS
    x0 = pylops_mpi.DistributedArray(BDiag.shape[1], engine="cupy", dtype=np.float64)
    x0[:] = 0
    xcgls = pylops_mpi.cgls(BDiag, y, x0=x0, niter=100, tol=1e-10, show=True)[0]
    xcgls_array = xcgls.asarray()
    if rank == 0:
        print('CGLS error', np.linalg.norm(x_array - xcgls_array))


if __name__ == '__main__':
    run()