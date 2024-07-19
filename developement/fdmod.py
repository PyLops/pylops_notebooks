import numpy as np
import devito

from examples.seismic import Model
from examples.seismic import AcquisitionGeometry
from examples.seismic.acoustic import AcousticWaveSolver
from examples.seismic import Receiver
from examples.seismic import plot_velocity, plot_shotrecord
from pylops.waveeqprocessing.twoway import AcousticWave2D

devito.configuration['log-level'] = 'ERROR'


def fdmodelling(vel, dx, dz, dt, sx, sz, rx, rz, tn, f0, v0, isrc, nt):
    nbl = 300 # Number of boundary layers around the domain
    space_order = 6 # Space order of the simulation
    nx, nz = vel.shape
    
    model = Model(space_order=space_order, vp=vel, origin=(0, 0), shape=(nx, nz),
                  dtype=np.float32, spacing=(dx, dz), nbl=nbl, bcs="damp")

    Aop = AcousticWave2D((nx, nz), (0, 0), (dx, dz), vel,
                         sx, sz, rx, rz, 0., tn, 'Ricker',
                         space_order=space_order, nbl=nbl, f0=f0,
                         dtype="float32", name="A")
    geometry = AcquisitionGeometry(
                Aop.model,
                Aop.geometry.rec_positions,
                Aop.geometry.src_positions[isrc:isrc+1, :],
                Aop.geometry.t0,
                Aop.geometry.tn,
                f0=Aop.geometry.f0,
                src_type=Aop.geometry.src_type)
    
    solver = AcousticWaveSolver(Aop.model, geometry, space_order=Aop.space_order)
    #d_fd = solver.forward()[0]
    d_fd, u_fd, _ = solver.forward(save=True)
    d_fd = np.array(d_fd.resample(dt * 1e3).data)[:nt].T

    # Direct wave masking
    # direct = np.sqrt((sx[isrc]-rx)**2 + (sz[isrc]-rz)**2) / v0
    # directmask = np.zeros((nr, nt))
    # for ir in range(nr):
    #     directmask[ir, int((direct[ir] + 0.2) // dt):] = 1.
    # d_fd = d_fd * directmask

    # Remove wavelet
    wcenter = np.argmax(geometry.src.resample(dt * 1e3).data)
    d_fd = np.pad(d_fd[..., wcenter:], ((0, 0), (0, wcenter)))
    
    wcenter = np.argmax(geometry.src.data)
    u_fd = np.array(u_fd.data[wcenter:, nbl:-nbl, nbl:-nbl])
   
    return d_fd, u_fd, geometry.dt*1e-3

def refl_angle(thetas, vel):
    # Compute full angle reflectivity and averaged reflectivity
    thetas0 = thetas.copy()
    nthetas = len(thetas0)
    nx, nz = vel.shape
    
    thetas1 = np.zeros((nthetas, nx, nz))
    for itheta, theta0 in enumerate(thetas0):
        thetas1[itheta, :, :-1] = np.rad2deg(np.arcsin(np.sin(np.deg2rad(theta0)) * vel[:, 1:]/vel[:, :-1]))
    thetas1[:, :, -1] = thetas1[:, :, -2]

    refls = np.zeros((nthetas, nx, nz))
    for itheta, theta0 in enumerate(thetas0):
        refls[itheta, :, :-1] = (vel[:, 1:] * np.cos(np.deg2rad(theta0)) - 
                                 vel[:, :-1] * np.cos(np.deg2rad(thetas1[itheta, :, :-1]))) / \
                                (vel[:, 1:] * np.cos(np.deg2rad(theta0)) +
                                 vel[:, :-1] * np.cos(np.deg2rad(thetas1[itheta, :, :-1])))
    refls[np.isnan(refls)] = 0
    refls[:, :, -1] = refls[:, :, -2]

    refl_av = (1/nthetas) * np.sum(refls * np.cos(np.deg2rad(thetas0[:, np.newaxis, np.newaxis])), axis=0)

    return refls, refl_av