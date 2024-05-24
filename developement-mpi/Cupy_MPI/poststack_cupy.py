r"""
Distributed poststack inversion with cupy arrays

Run as: [module load cuda/11.5.0/gcc-7.5.0-syen6pj;] export OMP_NUM_THREADS=4; export MKL_NUM_THREADS=4; export NUMBA_NUM_THREADS=4; mpiexec -n 2 python poststack_cupy.py
"""

import os
import sys
import time
import numpy as np
import cupy as cp
import pylops
import matplotlib.pyplot as plt

from scipy.signal import filtfilt
from pylops.avo import PoststackLinearModelling
from pylops.basicoperators import Transpose
from pylops.utils.wavelets import ricker

from mpi4py import MPI

import pylops_mpi


def run():
    comm = MPI.COMM_WORLD
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    cp.cuda.Device(device=rank).use()

    # Model
    model = np.load("../../data/avo/poststack_model.npz")
    x, z, m = model['x'][::3], model['z'], np.log(model['model'])[:, ::3]

    # Making m a 3D model
    ny_i = 20  # size of model in y direction for rank i
    y = np.arange(ny_i)
    m3d_i = np.tile(m[:, :, np.newaxis], (1, 1, ny_i)).transpose((2, 1, 0))
    ny_i, nx, nz = m3d_i.shape

    # Size of y at all ranks
    ny = MPI.COMM_WORLD.allreduce(ny_i)

    # Smooth model
    nsmoothy, nsmoothx, nsmoothz = 5, 30, 20
    mback3d_i = filtfilt(np.ones(nsmoothy) / float(nsmoothy), 1, m3d_i, axis=0)
    mback3d_i = filtfilt(np.ones(nsmoothx) / float(nsmoothx), 1, mback3d_i, axis=1)
    mback3d_i = filtfilt(np.ones(nsmoothz) / float(nsmoothz), 1, mback3d_i, axis=2)

    # Wavelet
    dt = 0.004
    t0 = np.arange(nz) * dt
    ntwav = 41
    wav = ricker(t0[:ntwav // 2 + 1], 15)[0]

    # Collecting all the m3d and mback3d at all ranks
    m3d = np.concatenate(MPI.COMM_WORLD.allgather(m3d_i))
    mback3d = np.concatenate(MPI.COMM_WORLD.allgather(mback3d_i))

    # Create flattened model data
    m3d_dist = pylops_mpi.DistributedArray(global_shape=ny * nx * nz, engine='cupy')
    m3d_dist[:] = cp.asarray(m3d_i.astype(np.float32).flatten())

    # Create flattened smooth model data
    mback3d_dist = pylops_mpi.DistributedArray(global_shape=ny * nx * nz, engine='cupy')
    mback3d_dist[:] = cp.asarray(mback3d_i.astype(np.float32).flatten())

    # LinearOperator PostStackLinearModelling
    PPop = PoststackLinearModelling(cp.asarray(wav.astype(np.float32)), 
                                    nt0=nz, spatdims=(ny_i, nx))
    Top = Transpose((ny_i, nx, nz), (2, 0, 1))
    BDiag = pylops_mpi.basicoperators.MPIBlockDiag(ops=[Top.H @ PPop @ Top, ])

    # Data
    d_dist = BDiag @ m3d_dist
    d_local = d_dist.local_array.reshape((ny_i, nx, nz))
    d = cp.asnumpy(d_dist.asarray().reshape((ny, nx, nz)))
    d_0_dist = BDiag @ mback3d_dist
    d_0 = cp.asnumpy(d_dist.asarray().reshape((ny, nx, nz)))

    # Inversion using CGLS solver
    minv3d_iter_dist = pylops_mpi.optimization.basic.cgls(BDiag, d_dist, x0=mback3d_dist, niter=100, show=True)[0]
    minv3d_iter = cp.asnumpy(minv3d_iter_dist.asarray().reshape((ny, nx, nz)))

    if rank == 0:
        # Check the distributed implementation gives the same result
        # as the one running only on rank0
        PPop0 = PoststackLinearModelling(wav, nt0=nz, spatdims=(ny, nx))
        d0 = (PPop0 @ m3d.transpose(2, 0, 1)).transpose(1, 2, 0)
        d0_0 = (PPop0 @ m3d.transpose(2, 0, 1)).transpose(1, 2, 0)

        # Check the two distributed implementations give the same modelling results
        print('Distr == Local', np.allclose(d, d0))
        print('Smooth Distr == Local', np.allclose(d_0, d0_0))

        # Visualize
        fig, axs = plt.subplots(nrows=6, ncols=3, figsize=(9, 14), constrained_layout=True)
        axs[0][0].imshow(m3d[5, :, :].T, cmap="gist_rainbow", vmin=m.min(), vmax=m.max())
        axs[0][0].set_title("Model x-z")
        axs[0][0].axis("tight")
        axs[0][1].imshow(m3d[:, 200, :].T, cmap="gist_rainbow", vmin=m.min(), vmax=m.max())
        axs[0][1].set_title("Model y-z")
        axs[0][1].axis("tight")
        axs[0][2].imshow(m3d[:, :, 220].T, cmap="gist_rainbow", vmin=m.min(), vmax=m.max())
        axs[0][2].set_title("Model y-z")
        axs[0][2].axis("tight")

        axs[1][0].imshow(mback3d[5, :, :].T, cmap="gist_rainbow", vmin=m.min(), vmax=m.max())
        axs[1][0].set_title("Smooth Model x-z")
        axs[1][0].axis("tight")
        axs[1][1].imshow(mback3d[:, 200, :].T, cmap="gist_rainbow", vmin=m.min(), vmax=m.max())
        axs[1][1].set_title("Smooth Model y-z")
        axs[1][1].axis("tight")
        axs[1][2].imshow(mback3d[:, :, 220].T, cmap="gist_rainbow", vmin=m.min(), vmax=m.max())
        axs[1][2].set_title("Smooth Model y-z")
        axs[1][2].axis("tight")

        axs[2][0].imshow(d[5, :, :].T, cmap="gray", vmin=-1, vmax=1)
        axs[2][0].set_title("Data x-z")
        axs[2][0].axis("tight")
        axs[2][1].imshow(d[:, 200, :].T, cmap='gray', vmin=-1, vmax=1)
        axs[2][1].set_title('Data y-z')
        axs[2][1].axis('tight')
        axs[2][2].imshow(d[:, :, 220].T, cmap='gray', vmin=-1, vmax=1)
        axs[2][2].set_title('Data x-y')
        axs[2][2].axis('tight')

        axs[3][0].imshow(minv3d_iter[5, :, :].T, cmap="gist_rainbow", vmin=m.min(), vmax=m.max())
        axs[3][0].set_title("Inverted Model iter x-z")
        axs[3][0].axis("tight")
        axs[3][1].imshow(minv3d_iter[:, 200, :].T, cmap='gist_rainbow', vmin=m.min(), vmax=m.max())
        axs[3][1].set_title('Inverted Model iter y-z')
        axs[3][1].axis('tight')
        axs[3][2].imshow(minv3d_iter[:, :, 220].T, cmap='gist_rainbow', vmin=m.min(), vmax=m.max())
        axs[3][2].set_title('Inverted Model iter x-y')
        axs[3][2].axis('tight')

        plt.savefig('poststack')
if __name__ == '__main__':
    run()