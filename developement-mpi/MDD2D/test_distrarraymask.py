r"""
Test operations on DistributedArray with mask

Run as: export OMP_NUM_THREADS=4; export MKL_NUM_THREADS=4; export NUMBA_NUM_THREADS=4; mpiexec -n 8 python test_distrarraymask.py
"""

import numpy as np
import pylops_mpi

from mpi4py import MPI
from pylops_mpi.DistributedArray import Partition


def run():
    comm = MPI.COMM_WORLD
    size = comm.Get_size() # number of nodes
    rank = comm.Get_rank() # rank of current node
    dtype = np.float64

    # Define array
    n = 10

    # Create mask
    mask = np.repeat(np.arange(size // 4), 4)
    print(mask)

    # Apply forward
    x = pylops_mpi.DistributedArray(global_shape=n * size,
                                    partition=Partition.SCATTER,
                                    mask=mask,
                                    dtype=dtype)
    x[:] = (rank + 1) * np.ones(n)
    xloc = x.asarray()

    # Dot product
    dot = x.dot(x)

    # Norm
    norm = x.norm(ord=2)

    # Compare with serial computation
    if rank == 0:
        dotloc = np.dot(xloc[:n * size // 2], xloc[:n * size // 2])
        print('Dot check', np.allclose(dot, dotloc))
        normloc = np.linalg.norm(xloc[:n * size // 2], ord=2)
        print('Norm check', np.allclose(norm, normloc))

if __name__ == '__main__':
    run()