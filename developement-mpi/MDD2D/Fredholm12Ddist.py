import numpy as np

from mpi4py import MPI
from pylops.utils.backend import get_module
from pylops.utils.typing import DTypeLike, NDArray

from pylops_mpi import (
    DistributedArray,
    MPILinearOperator,
)
from pylops_mpi.DistributedArray import local_split, Partition


class MPIFredholm12Ddist(MPILinearOperator):
    r"""Fredholm integral of first kind.

    Implement a multi-dimensional Fredholm integral of first kind distributed
    across both the first dimension and additional dimension

    Parameters
    ----------
    G : :obj:`numpy.ndarray`
        Multi-dimensional convolution kernel of size
        :math:`[n_{\text{slice}} \times n_x \times n_y]`
    sub_comm_sl : :obj:`mpi4py.MPI.Comm`, optional
        MPI Sub-Communicator that creates a communication between
        all ranks sharing the same frequency
    sub_comm_z : :obj:`mpi4py.MPI.Comm`, optional
        MPI Sub-Communicator that creates a communication between
        all ranks sharing the same additional dimension
    nz : :obj:`int`, optional
        Additional dimension of model
    saveGt : :obj:`bool`, optional
        Save ``G`` and ``G.H`` to speed up the computation of adjoint
        (``True``) or create ``G.H`` on-the-fly (``False``)
        Note that ``saveGt=True`` will double the amount of required memory
    usematmul : :obj:`bool`, optional
        Use :func:`numpy.matmul` (``True``) or for-loop with :func:`numpy.dot`
        (``False``). As it is not possible to define which approach is more
        performant (this is highly dependent on the size of ``G`` and input
        arrays as well as the hardware used in the computation), we advise users
        to time both methods for their specific problem prior to making a
        choice.
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
    A multi-dimensional Fredholm integral of first kind can be expressed as

    .. math::

        d(k, x, z) = \int{G(k, x, y) m(k, y, z) \,\mathrm{d}y}
        \quad \forall k=1,\ldots,n_{slice}

    on the other hand its adjoint is expressed as

    .. math::

        m(k, y, z) = \int{G^*(k, y, x) d(k, x, z) \,\mathrm{d}x}
        \quad \forall k=1,\ldots,n_{\text{slice}}

    This integral is implemented in a distributed fashion, where ``G``
    is split across ranks along its first dimension. The inputs
    of both the forward and adjoint are distributed arrays with broadcast partion:
    each rank takes a portion of such arrays, computes a partial integral, and
    the resulting outputs are then gathered by all ranks to return a
    distributed arrays with broadcast partion.

    """

    def __init__(
        self,
        G: NDArray,
        sub_comm_sl: MPI.Comm, # should take first subgroup as 0, 1, 2, nvs - 1
        sub_comm_z: MPI.Comm, # should take first subgroup as 0, nvs, 2nvs, ...
        nz: int = 1,
        saveGt: bool = False,
        usematmul: bool = True,
        base_comm: MPI.Comm = MPI.COMM_WORLD,
        dtype: DTypeLike = "float64",
    ) -> None:
        self.sub_comm_sl = sub_comm_sl
        self.sub_comm_z = sub_comm_z
        self.rank = base_comm.Get_rank()
        self.rank_sl = sub_comm_sl.Get_rank()
        self.rank_z = sub_comm_z.Get_rank()
        self.rank_slall = np.array(base_comm.allgather(self.rank_sl))

        self.nsl, self.nx, self.ny = G.shape
        self.nz = nz
        self.nsls = self.sub_comm_z.allgather(self.nsl)
        self.nsls1 = base_comm.allgather(self.nsl)
        # print('self.nsls', self.nsls, self.nsls1)
        if base_comm.Get_rank() == 0 and 1 in self.nsls:
            raise NotImplementedError(f'All ranks must have at least 2 or more '
                                      f'elements in the first dimension: '
                                      f'local split is instead {self.nsls}...')
        self.nslstot = self.sub_comm_z.allreduce(self.nsl)
        self.islstart = np.insert(np.cumsum(self.nsls)[:-1], 0, 0)
        self.islend = np.cumsum(self.nsls)
        self.nz_rank = local_split((nz, ), self.sub_comm_sl, Partition.SCATTER, 0)[0]
        self.nz_ranks = base_comm.allgather(self.nz_rank)
        # print('self.nz_ranks', self.nz_ranks)
        self.dims = (self.nslstot, self.ny, self.nz)
        self.dimsd = (self.nslstot, self.nx, self.nz)
        self.dims_rank = (self.nslstot, self.ny, self.nz_rank)
        self.dimsd_rank = (self.nslstot, self.nx, self.nz_rank)
        shape = (np.prod(self.dimsd) * self.sub_comm_z.Get_size(),
                 np.prod(self.dims) * self.sub_comm_z.Get_size())
        super().__init__(shape=shape, dtype=np.dtype(dtype), base_comm=base_comm)

        self.G = G
        if saveGt:
            self.GT = G.transpose((0, 2, 1)).conj()
        self.usematmul = usematmul

    def _matvec(self, x: DistributedArray) -> DistributedArray:
        ncp = get_module(x.engine)
        if x.partition is not Partition.SCATTER:
            raise ValueError(f"x should have partition={Partition.SCATTER}, {x.partition} != {Partition.SCATTER}")
        y = DistributedArray(global_shape=self.shape[0],
                             local_shapes=[self.nslstot * self.nx * nz for nz in self.nz_ranks],
                             partition=Partition.SCATTER, mask=self.rank_slall,
                             engine=x.engine, dtype=self.dtype)
        x = x.local_array.reshape(self.dims_rank).squeeze()
        x = x[self.islstart[self.rank_z]:self.islend[self.rank_z]]
        # apply matmul for portion of the rank of interest
        if self.usematmul:
            if self.nz == 1:
                x = x[..., ncp.newaxis]
            y1 = ncp.matmul(self.G, x)
        else:
            y1 = ncp.squeeze(ncp.zeros((self.nsls[self.rank], self.nx, self.nz), dtype=self.dtype))
            for isl in range(self.nsls[self.rank]):
                y1[isl] = ncp.dot(self.G[isl], x[isl])
        # gather results
        y[:] = np.vstack(self.sub_comm_z.allgather(y1)).ravel()
        return y

    def _rmatvec(self, x: NDArray) -> NDArray:
        ncp = get_module(x.engine)
        if x.partition is not Partition.SCATTER:
            raise ValueError(f"x should have partition={Partition.SCATTER}, {x.partition} != {Partition.SCATTER}")
        y = DistributedArray(global_shape=self.shape[1],
                             local_shapes=[self.nslstot * self.ny * nz for nz in self.nz_ranks],
                             partition=Partition.SCATTER, mask=self.rank_slall,
                             engine=x.engine, dtype=self.dtype)
        x = x.local_array.reshape(self.dimsd_rank).squeeze()
        x = x[self.islstart[self.rank_z]:self.islend[self.rank_z]]
        # apply matmul for portion of the rank of interest
        if self.usematmul:
            if self.nz == 1:
                x = x[..., ncp.newaxis]
            if hasattr(self, "GT"):
                y1 = ncp.matmul(self.GT, x)
            else:
                y1 = (
                    ncp.matmul(x.transpose(0, 2, 1).conj(), self.G)
                    .transpose(0, 2, 1)
                    .conj()
                )
        else:
            y1 = ncp.squeeze(ncp.zeros((self.nsls[self.rank], self.ny, self.nz), dtype=self.dtype))
            if hasattr(self, "GT"):
                for isl in range(self.nsls[self.rank]):
                    y1[isl] = ncp.dot(self.GT[isl], x[isl])
            else:
                for isl in range(self.nsl):
                    y1[isl] = ncp.dot(x[isl].T.conj(), self.G[isl]).T.conj()

        # gather results
        y[:] = np.vstack(self.sub_comm_z.allgather(y1)).ravel()
        return y
