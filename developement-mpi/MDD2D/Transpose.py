from typing import Optional, Union
import numpy as np
from mpi4py import MPI

from pylops.utils.typing import DTypeLike, NDArray, InputDimsLike
# from pylops.utils.backend import get_normalize_axis_index

from pylops_mpi import (
    DistributedArray,
    MPILinearOperator,
)
from pylops_mpi.DistributedArray import local_split, Partition


class Transpose(MPILinearOperator):
    r"""Transpose operator.

    Transpose the last two axes of a multi-dimensional array distributed across
    one of the axes to be transposed.

    Parameters
    ----------
    dims : :obj:`tuple`, optional
        Number of samples for each dimension
    axis : :obj:`int`, optional
        Axis along which the input array is distributed (must be either -2 or -1)
    base_comm : :obj:`mpi4py.MPI.Comm`, optional
        MPI Base Communicator. Defaults to ``mpi4py.MPI.COMM_WORLD``.
    dtype : :obj:`str`, optional
        Type of elements in input array.

    Attributes
    ----------
    shape : :obj:`tuple`
        Operator shape

    Raises
    ------
    NotImplementedError
        If the size of the first dimension of ``G`` is equal to 1 in any of the ranks

    Notes
    -----
    The Transpose operator reshapes the input model into a multi-dimensional
    array of size ``dims`` and transposes (or swaps) its axes as defined
    in ``axes``.

    Similarly, in adjoint mode the data is reshaped into a multi-dimensional
    array whose size is a permuted version of ``dims`` defined by ``axes``.
    The array is then rearragned into the original model dimensions ``dims``.

    Note that since the multi-dimensional array is distributed across one of
    the two axes that are transposed, all ranks must broadcast their portion
    of the array and at the same time must fill their part of the transposed
    array using portions of the portions of the array received by other ranks.

    """

    def __init__(
        self,
        dims: Union[int, InputDimsLike],
        axis: Optional[int] = -1,
        base_comm: Optional[MPI.Comm ] = MPI.COMM_WORLD,
        dtype: Optional[DTypeLike] = "float64",
    ) -> None:
        self.dims = dims
        self.ndim = len(dims)
        self.nextra = int(np.prod(self.dims[:-2])) # total number of elements in extra dimensions
        self.axis = self.ndim + axis # get_normalize_axis_index()(axis, self.ndim)
        self.base_comm = base_comm
        self.rank = base_comm.Get_rank()
        self.size = base_comm.Get_size()

        # define how axis is split across ranks
        nrank = local_split((self.dims[self.axis], ), self.base_comm, Partition.SCATTER, 0)
        self.nranks = np.concatenate(self.base_comm.allgather(nrank))
        self.in_ranks = np.insert(np.cumsum(self.nranks)[:-1], 0, 0)
        self.end_ranks = np.cumsum(self.nranks)

        shape = (np.prod(self.dims), np.prod(self.dims))
        super().__init__(shape=shape, dtype=np.dtype(dtype), base_comm=base_comm)

    def _matvec_1(self, x: DistributedArray) -> DistributedArray:
        y = x.zeros_like()

        for irank in range(self.size):
            xbt = self.base_comm.bcast(x.local_array, root=irank)
            xbt = xbt.reshape(int(np.prod(self.dims[:-2])), self.dims[-2], self.nranks[irank]).swapaxes(-1, -2)
            for iextra in range(self.nextra):
                y[self.in_ranks[irank] * self.nranks[self.rank] + iextra * self.dims[-2] * self.nranks[self.rank]:
                  self.end_ranks[irank] * self.nranks[self.rank] + iextra * self.dims[-2] * self.nranks[self.rank]] = \
                    xbt[iextra, :, self.in_ranks[self.rank]: self.end_ranks[self.rank]].ravel()
        return y

    def _matvec_2(self, x: DistributedArray) -> DistributedArray:
        pass

    def _matvec(self, x: DistributedArray) -> DistributedArray:
        if self.axis == self.ndim - 1:
            return self._matvec_1(x)
        else:
            return self._matvec_2(x)

    def _rmatvec(self, x: NDArray) -> NDArray:
        return self._matvec(x)