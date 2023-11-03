r"""
Kirchhoff migration of 3D Volve dataset distributed over sources. 

This example is used to showcase how PyLops-mpi can also be used to easily
parallelize standard seismic processing/imaging tasks provided they can be
written via PyLops operators

Run as: export OMP_NUM_THREADS=4; export MKL_NUM_THREADS=4; export NUMBA_NUM_THREADS=4; mpiexec -n 20 python Kirchhoff_Volve.py 5370 5670 
"""

import os
import sys
import numpy as np
import pylops_mpi
import time
import segyio
import warnings
warnings.filterwarnings('ignore')

from matplotlib import pyplot as plt
from mpi4py import MPI

from scipy.interpolate import RegularGridInterpolator
from pylops.waveeqprocessing.kirchhoff import Kirchhoff
from pylops_mpi.DistributedArray import local_split, Partition
from segyshot import SegyShot
from visual import explode_volume

plt.close("all")

def run(ishotin, nshots):
    comm = MPI.COMM_WORLD
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()

    tic = time.perf_counter()  

    # Geometry selection
    #ishotin, ishotend = 5500, 5540 # supersmall
    #ishotin, ishotend = 5370, 5670 # 2d line
    #ishotin, ishotend =  5000, 6000 # small
    #ishotin, ishotend = 2880, 6715 # medium
    #ishotin, ishotend = 2400, 9596 # large
    ishotend = ishotin + nshots
    ivxmax = 22
    itmax = 2000

    if rank == 0:
        print(f'Distributed Kirchhoff migration of 3D Volve data ({size} ranks, {ishotin}-{ishotend} shots)')

    # Create folder to save figures
    figdir = f'Figs_sin{ishotin}_send{ishotend}'
    if rank == 0:
        if not os.path.exists(figdir):
            os.makedirs(figdir)
    
    # Velocity Model
    velfilepath = '/home/ravasim/Data/Volve/ST10010ZC11-MIG-VEL.MIG_VEL.VELOCITY.3D.JS-017527.segy'

    fvmod = segyio.open(velfilepath)
    vmod = segyio.cube(velfilepath)
    zvel = fvmod.samples

    # Data
    if rank == 0:
        print(f'Interpret data')

    filepath = '/home/ravasim/Data/Volve/'
    inputfile = os.path.join(filepath, 'ST10010_1150780_40203.sgy')
    f = segyio.open(inputfile, ignore_geometry=True)

    sg = SegyShot(inputfile, components=['P', 'VZ', 'VX', 'VY'])
    sg.interpret()

    _,_,_,_, (xvel_local, yvel_local) = \
        sg.rotategeometry(velfile=velfilepath, plotflag=1)

    # Select sources and portion of velocity model
    #sg.resetsrcs()
    #sg.selectsrcs(ishotin, ishotend, plotflag=False)

    # Extract only useful part of velocity model
    xvel_localgrid = xvel_local.reshape(vmod.shape[:2])[:ivxmax]
    yvel_localgrid = yvel_local.reshape(vmod.shape[:2])[:ivxmax]
    vel_local = vmod[:ivxmax]

    if rank == 0:
        explode_volume(vel_local.transpose(2, 1, 0), t=50, y=11,
                    cmap='jet', clipval=(1000, 5000), figsize=(15, 5))
        plt.savefig(os.path.join(figdir, 'Velocity.png'), dpi=300)

    # Local grid
    nx, ny = xvel_localgrid.T.shape
    nz = len(zvel)
    x0 = xvel_localgrid.min()
    y0 = yvel_localgrid.min()

    dx = np.round(np.mean(np.abs(np.diff(xvel_localgrid[0]))))
    dy = np.round(np.mean(np.abs(np.diff(yvel_localgrid[:, 0]))))

    xvel_local = np.arange(nx) * dx
    yvel_local = np.arange(ny) * dy

    xvel_localgrid, yvel_localgrid = np.meshgrid(xvel_local, yvel_local, indexing='ij')

    # Select sources and portion of velocity model
    sg.resetsrcs()
    sg.selectsrcs(ishotin, ishotend, plotflag=False)

    srcx_local = sg.srcx_local[sg.selected_src] - x0
    srcy_local = sg.srcy_local[sg.selected_src] - y0
    srcz_local = sg.srcz[sg.selected_src]
    src_local = np.vstack([srcy_local, srcx_local, srcz_local])

    recx_local = sg.recx_local - x0
    recy_local = sg.recy_local - y0
    recz_local = sg.recz
    rec_local = np.vstack([recy_local, recx_local, recz_local])

    ns, nr = src_local.shape[1], rec_local.shape[1]

    if rank == 0:
        plt.figure(figsize=(15, 12))
        plt.scatter(xvel_localgrid.ravel(), yvel_localgrid.ravel(), color='k')
        plt.scatter(srcx_local, srcy_local, color='r')
        plt.scatter(recx_local, recy_local, color='b');
        plt.savefig(os.path.join(figdir, 'Geometry.png'), dpi=300)

    # Interpolate velocity model
    dy3d, dx3d, dz3d = 20, 20, 20
    ny3d, nx3d, nz3d = int(yvel_local[-1]//dy3d), int(xvel_local[-1]//dx3d), int(zvel[-1]//dz3d)

    yvel_local3d = np.arange(ny3d) * dy3d
    xvel_local3d = np.arange(nx3d) * dx3d
    zvel_local3d = np.arange(nz3d) * dz3d

    Y3d, X3d, Z3d = np.meshgrid(yvel_local3d, xvel_local3d, zvel_local3d, indexing='ij')
    YXZ3d = np.vstack((Y3d.ravel(), X3d.ravel(), Z3d.ravel())).T

    vel_interp3d = np.zeros(ny3d * nx3d * nz3d, dtype=np.float32)
    interpolator = RegularGridInterpolator((yvel_local, xvel_local, zvel), 
                                        vel_local, bounds_error=False, fill_value=0)
    for i in range(0, ny3d*nx3d*nz3d, nx3d*nz3d):
        vel_interp3d[i:i+nx3d*nz3d] = interpolator(YXZ3d[i:i+nx3d*nz3d]).astype(np.float32)
    vel_interp3d = vel_interp3d.reshape(ny3d, nx3d, nz3d)

    if rank == 0:
        explode_volume(vel_interp3d.transpose(2, 1, 0), t=75, y=25,
                    cmap='jet', clipval=(1000, 5000), figsize=(15, 5))
        plt.savefig(os.path.join(figdir, 'Velocity_interp.png'), dpi=300)

    # Choose how to split sources to ranks
    ns_rank = local_split((ns, ), comm, Partition.SCATTER, 0)
    ns_ranks = np.concatenate(comm.allgather(ns_rank))
    isin_rank = np.insert(np.cumsum(ns_ranks)[:-1] , 0, 0)[rank]
    isend_rank = np.cumsum(ns_ranks)[rank]
    print(f'Rank: {rank}, ns: {ns_rank}, isin: {isin_rank}, isend: {isend_rank}')

    # Choose how to split recs to ranks
    nr_rank = local_split((nr, ), comm, Partition.SCATTER, 0)
    nr_ranks = np.concatenate(comm.allgather(nr_rank))
    irin_rank = np.insert(np.cumsum(nr_ranks)[:-1] , 0, 0)[rank]
    irend_rank = np.cumsum(nr_ranks)[rank]
    print(f'Rank: {rank}, nr: {nr_rank}, irin: {irin_rank}, irend: {irend_rank}')

    comm.Barrier()

    # Extract data
    if rank == 0:
        print('Loading data...', flush=True)

    t = sg.t[:itmax]
    dt = t[1]
    drank = np.concatenate([sg.get_shotgather(ishot)['P'][np.newaxis, :, :itmax] for ishot in range(isin_rank + ishotin, isend_rank + ishotin)])
    srank = src_local[:, isin_rank:isend_rank]
    rrank = rec_local[:, irin_rank:irend_rank]

    fig, ax = plt.subplots(1, 1, figsize=(15, 3))
    ax.imshow(drank.reshape(drank.shape[0]*drank.shape[1], -1).T, cmap='gray', vmin=-1e5, vmax=1e5)
    ax.set_title('Data')
    ax.axis('tight')
    plt.savefig(os.path.join(figdir, f'Data_r{rank}.png'))

    # Remove direct arrival from data
    vwater = 1500
    drank_nodirect = drank.copy()
    for ishot in range(ns_rank[0]):
        direct = np.sqrt(np.sum((srank[:, ishot:ishot+1]-rec_local)**2, axis=0)) / vwater
        for irec in range(nr):
            drank_nodirect[ishot, irec, :int(direct[irec]/dt+200)] = 0

    fig, ax = plt.subplots(1, 1, figsize=(15, 3))
    ax.imshow(drank_nodirect.reshape(drank.shape[0]*drank.shape[1], -1).T, cmap='gray', vmin=-1e5, vmax=1e5)
    ax.set_title('Data without direct')
    ax.axis('tight')
    plt.savefig(os.path.join(figdir, f'Datanodir_r{rank}.png'))

    comm.Barrier()

    # Compute traveltimes
    if rank == 0:
        print('Compute traveltimes...', flush=True)
    trav_srcs_eik, trav_recsrank_eik, _, _, _, _ = \
        Kirchhoff._traveltime_table(zvel_local3d, xvel_local3d, srank, rrank, 
                                    vel_interp3d, y=yvel_local3d, mode='eikonal')
    trav_recsrank_eik = trav_recsrank_eik.astype('float32').T.flatten()

    # works only with constant nr_rank (and small arrays)... never use
    #trav_recs_eik = np.concatenate(comm.allgather(trav_recsrank_eik)).reshape(nr, -1).T
    #trav_recs_eik = trav_recs_eik.reshape(nr, -1).T

    # works only with constant nr_rank
    # trav_recs_eik = np.zeros(trav_recsrank_eik.size * size, dtype='float32') # working with constant nr_rank
    #comm.Allgather([trav_recsrank_eik,  MPI.FLOAT], [trav_recs_eik, MPI.FLOAT])
    #trav_recs_eik = trav_recs_eik.reshape(nr, -1).T

    # variable size nr_rank
    ntrav_rank = nr_ranks * ny3d * nx3d * nz3d
    itravrin_rank = np.insert(np.cumsum(nr_ranks * ny3d * nx3d * nz3d)[:-1] , 0, 0)
    trav_recs_eik = np.zeros(ny3d * nx3d * nz3d * nr, dtype='float32') 
    comm.Allgatherv(trav_recsrank_eik, [trav_recs_eik, ntrav_rank, itravrin_rank, MPI.FLOAT])
    trav_recs_eik = trav_recs_eik.reshape(nr, -1).T

    if rank == 0:
        print('trav_srcs_eik', trav_srcs_eik.max())
        print('trav_recs_eik', trav_recs_eik.max())
        explode_volume(trav_srcs_eik[:, ns_rank[0]//2].reshape(ny3d, nx3d, nz3d).transpose(2, 1, 0), cmap='tab10', t=20, figsize=(15, 5))
        plt.savefig(os.path.join(figdir, 'Trav_src.png'))
        explode_volume(trav_recs_eik[:, nr//2].reshape(ny3d, nx3d, nz3d).transpose(2, 1, 0), cmap='tab10', t=20, figsize=(15, 5))
        plt.savefig(os.path.join(figdir, 'Trav_rec.png'))

    # Wavelet
    wav, wavc = np.zeros(81), 41
    wav[wavc] = 1.

    # Kirchhoff operator
    if rank == 0:
        print('Create operator...')
    KOp = Kirchhoff(zvel_local3d, xvel_local3d, t, srank, rec_local, vel_interp3d, 
                    wav, wavc, y=yvel_local3d, dynamic=False, 
                    trav=(trav_srcs_eik, trav_recs_eik), mode='byot', engine='numba')

    KOptot = pylops_mpi.MPIVStack(ops=[KOp, ])

    ddist = pylops_mpi.DistributedArray(global_shape=ns * nr * itmax, partition=pylops_mpi.Partition.SCATTER,
                                        local_shapes=tuple([(ns_ranks[r] * nr * itmax, ) for r in range(size)]))
    ddist[:] = drank_nodirect.flatten()
    #print(f'Rank: {rank}, KOptot: {KOptot}, ddist: {ddist}')

    # Imaging
    if rank == 0:
        print('Perform imaging...')
        
    ticim = time.perf_counter()  
    image = KOptot.H * ddist
    image = image.asarray().reshape(ny3d, nx3d, nz3d)
    tocim = time.perf_counter()  

    if rank == 0:
        np.save(f'Image_sin{ishotin}_send{ishotend}', image)    

        explode_volume(image.transpose(2, 1, 0), cmap='gray', t=75, y=25,
                    clipval=(-0.2*np.abs(image).max(), 0.2*np.abs(image).max()), figsize=(15, 5))
        plt.savefig(os.path.join(figdir, f'Image_sin{ishotin}_send{ishotend}.png'), dpi=300)
        
    toc = time.perf_counter()  

    if rank == 0:
        print(f'Imaging elapsed time: {tocim-ticim} s')
        print(f'Total elapsed time: {toc-tic} s')

        f = open("Kirchhoff_Volve_timings.txt", "a")
        f.write(f"{size} \t {tocim-ticim:.3f} \t {toc-tic:.3f}\n")
        f.close()


if __name__ == '__main__':
    ishotin = int(sys.argv[1])
    ishotend = int(sys.argv[2]) 
    run(ishotin, ishotend)