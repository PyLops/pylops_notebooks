r"""
LSRTM of Marmousi dataset distributed over sources. 

This example is used to showcase how PyLops-mpi can also be used to easily
parallelize standard seismic processing/imaging tasks provided they can be
written via PyLops operators

Run as: export DEVITO_LANGUAGE=openmp; export DEVITO_MPI=0; export OMP_NUM_THREADS=12; export MKL_NUM_THREADS=12; export NUMBA_NUM_THREADS=12; mpiexec -n 4 python LSRTM_Marmousi.py 
"""

import os
import time
import numpy as np
import scipy as sp
import pylops_mpi

from matplotlib import pyplot as plt
from mpi4py import MPI

from devito import configuration
from examples.seismic import Model

from pylops.waveeqprocessing.twoway import AcousticWave2D
from pylops_mpi.DistributedArray import local_split, Partition
from pylops_mpi.DistributedArray import local_split, Partition

comm = MPI.COMM_WORLD
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()

configuration['log-level'] = 'ERROR'
plt.close("all")

tic = time.perf_counter()  

if rank == 0:
    print(f'Distributed LSRTM of Marmousi data ({size} ranks)')

#model parameter
origshape = (1360, 280)
shape = (500, 280)
spacing = (12.5, 12.5)
origin = (0, 0)

#geometry arrange
ns = 10
nr = 200

# other modelling params
nbl=100 # Number of boundary layers around the domain
space_order=6 # Space order of the simulation
f0 = 15 # Source peak frequency
tn=5000 # Total simulation time

data_path = "../../data/lsm/marmousi.bin"
fig_path = '.'

# Get true model
#if rank == 0:
vp = 1e-3 * np.fromfile(data_path, dtype='float32', sep="") # devito expects velocity to have km/s as units
vp = vp.reshape(origshape)[:shape[0],:shape[1]]
#true_model = Model(space_order=space_order, vp=vp, origin=origin, shape=shape, #subdomain=(shape, ), 
#                   dtype=np.float32, spacing=spacing, bcs="damp")

# Smooth model
v0 = sp.ndimage.gaussian_filter(vp, sigma=10)    
v0[:,:36] = 1.5 # Do not smooth water layer

# Born perturbation from m - m0
dm = (vp**(-2) - v0**(-2))

if rank == 0:
    plt.figure(figsize=(14, 5))
    plt.imshow(dm.T, cmap='gray', vmin=-1e-1, vmax=1e-1)
    plt.title('Dm')
    plt.axis('tight')
    plt.colorbar()
    plt.savefig('ModelPert.png')

# Choose how to split sources to ranks
ns_rank = local_split((ns, ), MPI.COMM_WORLD, Partition.SCATTER, 0)
ns_ranks = np.concatenate(MPI.COMM_WORLD.allgather(ns_rank))
isin_rank = np.insert(np.cumsum(ns_ranks)[:-1] , 0, 0)[rank]
isend_rank = np.cumsum(ns_ranks)[rank]
print(f'Rank: {rank}, ns: {ns_rank}, isin: {isin_rank}, isend: {isend_rank}')

# Modelling operator
src_x = np.linspace(0, shape[0]*spacing[0], num=ns)
src_z = 20. 

rec_x = np.linspace(0, shape[0]*spacing[0], num=nr)
rec_z = 20.

src_x_rank = src_x[isin_rank:isend_rank]

Aop = AcousticWave2D(shape, origin, spacing, v0 * 1e3,
                     src_x_rank, src_z, rec_x, rec_z, 0., 
                     tn, 'Ricker',
                     space_order=6, nbl=nbl, f0=f0,
                     dtype="float32", name="A")

Aoptop = pylops_mpi.MPIVStack(ops=[Aop, ])

dmdist = pylops_mpi.DistributedArray(global_shape=np.prod(shape), partition=pylops_mpi.Partition.BROADCAST)
dmdist[:] = dm.flatten()

# Data creation
if rank == 0: 
    print('Start modelling...')
comm.Barrier()

dobs = Aoptop @ dmdist

plt.figure(figsize=(10, 9))
plt.imshow(dobs.local_array.reshape(Aop.geometry.nsrc, Aop.geometry.nrec, Aop.geometry.nt)[Aop.geometry.nsrc//2].T, 
           cmap='gray', vmin=-0.1, vmax=0.1)
plt.title('FD modelling')
plt.axis('tight')
plt.savefig(os.path.join(fig_path, f'Data_r{rank}.png'))

# Imaging
if rank == 0: 
    print('Start imaging...')
comm.Barrier()

dmrtm = Aoptop.H @ dobs
dmrtm = dmrtm.asarray().reshape(shape)

if rank == 0: 
    plt.figure(figsize=(14, 6))
    plt.imshow(np.diff(dmrtm, axis=1).T, cmap='gray', vmin=-3e2, vmax=3e2)
    plt.title('Dm - RTM')
    plt.axis('tight')
    plt.savefig(os.path.join(fig_path, f'RTM_Image.png'))

# LSRTM
if rank == 0: 
    print('Start LS imaging...')
comm.Barrier()

x0 = pylops_mpi.DistributedArray(Aoptop.shape[1], partition=pylops_mpi.Partition.BROADCAST)
x0[:] = 0
dminv = pylops_mpi.cgls(Aoptop, dobs, x0=x0, niter=50, show=True)[0]
dminv = dminv.asarray().reshape(shape)

if rank == 0: 
    plt.figure(figsize=(14, 6))
    plt.imshow(dminv.T, cmap='gray', vmin=-1e-1, vmax=1e-1)
    plt.title('Dm - LSRTM')
    plt.axis('tight')
    plt.savefig(os.path.join(fig_path, f'LSRTM_Image.png'))
