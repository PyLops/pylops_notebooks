r"""
Utility functions for 2D-distribution
"""

import sys
import time
import numpy as np

from pylops_mpi.DistributedArray import local_split, Partition


def pause(comm, t=4):
    """Flush stdout and pause code execution

    Parameters
    ----------
    comm : :obj:`mpi4py.MPI.Comm`
        MPI Base Communicator.
    t : :obj:`int`, optional
        Number of second to pause

    Returns
    -------
    local_shape : :obj:`tuple`
        Shape of the local array.
    """
    sys.stdout.flush()
    comm.barrier()
    time.sleep(t)


def local_split_customranks(global_shape, nranks, rank, partition, axis):
    """Compute the local shape from the global shape given a chosen number of ranks

    Parameters
    ----------
    global_shape : :obj:`tuple`
        Shape of the global array.
    nranks : :obj:`int`
        Number of ranks to split global_shape equally in
    rank : :obj:`int`
        Rank
    partition : :obj:`Partition`
        Type of partition.
    axis : :obj:`int`
        Axis of distribution

    Returns
    -------
    local_shape : :obj:`tuple`
        Shape of the local array.
    """
    if partition == Partition.BROADCAST:
        local_shape = global_shape
    # Split the array
    else:
        local_shape = list(global_shape)
        if rank < (global_shape[axis] % nranks):
            local_shape[axis] = global_shape[axis] // nranks + 1
        else:
            local_shape[axis] = global_shape[axis] // nranks
    return tuple(local_shape)


def local_split_startend(n, rank, subcomm, nrank_rep, rep=np.repeat):
    """Compute the start and end indices of a local split
    with subcommunicators (for 2D distribution)

    Parameters
    ----------
    n : :obj:`int`
        Lenght of array to split
    rank : :obj:`int`
        Rank
    subcomm : :obj:`mpi4py.MPI.Comm`
        MPI Sub-Communicator that creates a communication between
        groups of ranks
    nrank_rep : :obj:`int`
        Number of repetitions to apply to a Sub-Communicator
        list of local shapes to obtain a list of local
        shapes for the base MPI Communicator.
    rep : :obj:`funct`
        Numpy function used to replicate the Sub-Communicator
        list of local shapes (must be ``np.repeat`` or ``np.tile``)

    Returns
    -------
    n_ranks : :obj:`list`
        Shape of the local arrays.
    iin_ranks : :obj:`list`
        Start indices of a local split
    iend_ranks : :obj:`list`
        End indices of a local split
    n_rank : :obj:`int`
        Shape of the local array at chosen rank
    iin_rank : :obj:`int`
        Start index of a local split at chosen rank
    iend_rank : :obj:`int`
        End index of a local split at chosen rank

    """
    n_rank = local_split((n,), subcomm, Partition.SCATTER, 0)
    n_ranks = np.concatenate(subcomm.allgather(n_rank))
    iin_ranks = np.insert(np.cumsum(n_ranks)[:-1], 0, 0)
    iend_ranks = np.cumsum(n_ranks)
    n_ranks = rep(n_ranks, nrank_rep)
    iin_rank = rep(iin_ranks, nrank_rep)[rank]
    iend_rank = rep(iend_ranks, nrank_rep)[rank]
    return n_ranks, iin_ranks, iend_ranks, n_rank, iin_rank, iend_rank


def reorganize_distributed_2d(x, shape, local_shapes, istart, iend):
    """Reorganized array distributed in a masked fashion from flattened to n-dimensional.

    Parameters
    ----------
    x : :obj:`numpy.ndarray`
        Flattened array
    shape : :obj:`tuple`
        Shape of output array
    local_shapes : :obj:`tuple`
        Shapes of each of the local arrays
    istart : :obj:`tuple`
        Start indices for the n-dimensional array
    iend : :obj:`tuple`
        End indices for the n-dimensional array

    Returns
    -------
    x1 : :obj:`numpy.ndarray`
        Reshaped array

    """
    x1 = np.zeros(shape)
    for ivs in range(len(istart)):
        xtmp = x[0 if ivs == 0 else np.sum([ls[0] for ls in local_shapes[:ivs]]):
                 np.sum([ls[0] for ls in local_shapes[:ivs + 1]])].reshape(shape[0], shape[1], -1)
        x1[..., istart[ivs]:iend[ivs]] = xtmp
    return x1