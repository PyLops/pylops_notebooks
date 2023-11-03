r"""
Kirchhoff migration of 2D Volve dataset distributed over sources. 

This example is used to showcase how PyLops-mpi can also be used to easily
parallelize standard seismic processing/imaging tasks provided they can be
written via PyLops operators

Run as: export OMP_NUM_THREADS=4; export MKL_NUM_THREADS=4; export NUMBA_NUM_THREADS=4; mpiexec -n 10 python Kirchhoff_Volve.py 
"""

import numpy as np
import pylops_mpi
import time
import warnings
warnings.filterwarnings('ignore')

from scipy.signal import filtfilt
from matplotlib import pyplot as plt
from mpi4py import MPI

from pylops.waveeqprocessing.kirchhoff import Kirchhoff
from pylops_mpi.DistributedArray import local_split, Partition

plt.close("all")
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()

tic = time.perf_counter()  

if rank == 0:
    print(f'Distributed Kirchhoff migration of 2D Volve data ({size} ranks)')

# Model and Geometry
inputvolve = np.load('/home/ravasim/Data/VolveSynthetic/Data/Velocity/Velocities.npz')
vel = inputvolve["vback"].T
nx, nz = vel.shape

x = inputvolve["x"]
z = inputvolve["z"]
s = inputvolve["recs"].T
r = inputvolve["recs"].T
ns, nr = s.shape[1], r.shape[1]
nx, nz = x.size, z.size

if rank == 0:
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    im = ax.imshow(vel.T, cmap='jet', vmin=1000, vmax=5000, extent=(x[0], x[-1], z[-1], z[0]))
    ax.plot(r[0], r[1], '.w', ms=20)
    ax.set_xlabel(r'X [m]')
    ax.set_ylabel(r'Z [m]')
    ax.set_title(r'VP')
    ax.axis('tight')
    ax.set_xlim(2800, 9200)
    ax.set_ylim(4000, 0)
    plt.colorbar(im)
    plt.savefig('Model.png')

# Choose how to split sources to ranks
ns_rank = local_split((ns, ), MPI.COMM_WORLD, Partition.SCATTER, 0)
ns_ranks = np.concatenate(MPI.COMM_WORLD.allgather(ns_rank))
isin_rank = np.insert(np.cumsum(ns_ranks)[:-1] , 0, 0)[rank]
isend_rank = np.cumsum(ns_ranks)[rank]
print(f'Rank: {rank}, ns: {ns_rank}, isin: {isin_rank}, isend: {isend_rank}')

# Data
f = np.load('/home/ravasim/Documents/2021/Projects/MDD-StochasticSolvers/data/mdd_volvereal_multimasked_iter40_batch32_all.npz')

t = f['t']
dt = t[1]
nt = t.size
drank = f['Rnsgd'][:, isin_rank:isend_rank].transpose(1, 2, 0)
srank = s[:, isin_rank:isend_rank]

# Convert from global to local grid
xorig = x[0]
x -= xorig
dx, dz = x[1], z[1]

srank[0] -= xorig
r[0] -= xorig

# Compute traveltimes
if rank == 0:
    print('Compute traveltimes...')
trav_srcs_eik, _, _, _, _, _ = \
    Kirchhoff._traveltime_table(z, x, srank, r[:, :1], vel, mode='eikonal')
trav_recs_eik = np.concatenate(MPI.COMM_WORLD.allgather(trav_srcs_eik.T)).T

if rank == 0:
    fig, axs = plt.subplots(3, 1, sharey=True, figsize=(14, 18))
    axs[0].imshow(trav_srcs_eik[:, 0].reshape((nx, nz)).T, cmap='tab10', 
                extent = (x[0], x[-1], z[-1], z[0]))
    axs[0].scatter(srank[0, 0], srank[1, 0], marker='*', s=150, c='r', edgecolors='k')
    axs[0].axis('tight')
    axs[0].set_xlabel('x [m]')
    axs[0].set_ylabel('y [m]')
    axs[0].set_title('Source traveltime')
    axs[0].set_ylim(z[-1], z[0])
    axs[1].imshow(trav_recs_eik[:, -10].reshape((nx, nz)).T, cmap='tab10', 
                  extent = (x[0], x[-1], z[-1], z[0]))
    axs[1].scatter(r[0, -10], r[1, -10], marker='v', s=150, c='b', edgecolors='k')
    axs[1].axis('tight')
    axs[1].set_xlabel('x [m]')
    axs[1].set_title('Receiver traveltime')
    axs[1].set_ylim(z[-1], z[0])
    axs[2].imshow(trav_srcs_eik[:, 10].reshape((nx, nz)).T + trav_recs_eik[:, -10].reshape((nx, nz)).T, 
                cmap='tab10', extent = (x[0], x[-1], z[-1], z[0]))
    axs[2].scatter(srank[0, 10], srank[1, 10], marker='*', s=150, c='r', edgecolors='k')
    axs[2].scatter(r[0, -10], r[1, -10], marker='v', s=150, c='b', edgecolors='k')
    axs[2].axis('tight')
    axs[2].set_xlabel('x [m]')
    axs[2].set_title('Src+rec traveltime')
    axs[2].set_ylim(z[-1], z[0])
    plt.savefig('Travs.png')

# Wavelet
wav, wavc = np.zeros(81), 41
wav[wavc] = 1.

# Kirchhoff operator
KOp = Kirchhoff(z, x, t, srank, r, vel, wav, wavc, dynamic=False, 
                trav=(trav_srcs_eik, trav_recs_eik),
                mode='byot', engine='numba')

KOptot = pylops_mpi.MPIVStack(ops=[KOp, ])

ddist = pylops_mpi.DistributedArray(global_shape=ns * nr * nt, partition=pylops_mpi.Partition.SCATTER,
                                    local_shapes=tuple([(ns_ranks[r] * nr * nt, ) for r in range(size)]))
ddist[:] = drank.flatten()
#print(f'Rank: {rank}, KOptot: {KOptot}, ddist: {ddist}')

# Imaging
if rank == 0:
    print('Perform imaging...')
ticim = time.perf_counter()  
image = KOptot.H * ddist
image = image.asarray().reshape(nx, nz)
tocim = time.perf_counter()  

if rank == 0:
    clip = 5e0
    gain = np.sqrt(z)

    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.imshow(image.T * gain[:, np.newaxis], cmap='gray', vmin=-clip, vmax=clip, 
            interpolation='sinc', extent=(x[0]+xorig, x[-1]+xorig, z[-1], z[0]))
    ax.plot(r[0]+xorig, r[1], '.w', ms=20)
    ax.set_xlabel(r'X [m]')
    ax.set_ylabel(r'Z [m]')
    ax.set_title(r'Image')
    ax.axis('tight')
    ax.set_xlim(2800, 9200)
    ax.set_ylim(4000, 0)
    plt.savefig('Image.png')

toc = time.perf_counter()  

if rank == 0:
    print(f'Imaging elapsed time: {tocim-ticim} s')
    print(f'Total elapsed time: {toc-tic} s')

    f = open("Kirchhoff_Volve_timings.txt", "a")
    f.write(f"{size} \t {tocim-ticim:.3f} \t {toc-tic:.3f}\n")
    f.close()