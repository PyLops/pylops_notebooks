import os
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import devito

from examples.seismic import Model
from examples.seismic import AcquisitionGeometry
from examples.seismic.acoustic import AcousticWaveSolver
from examples.seismic import Receiver
from examples.seismic import plot_velocity, plot_shotrecord
#from model import Model

from pylops.basicoperators import *
from pylops.waveeqprocessing.twoway import AcousticWave2D
from pylops.optimization.sparsity import fista
from pylops.utils import dottest

devito.configuration['log-level'] = 'ERROR'
figdir = '.'

#model parameter
origshape = (1360, 280)
shape = (500, 280)
spacing = (12.5, 12.5)
origin = (0, 0)

#geometry arrange
nsrc = 2
nrec = 200

# other modelling params
nbl=100 # Number of boundary layers around the domain
space_order=6 # Space order of the simulation
f0 = 15 # Source peak frequency
tn=5000 # Total simulation time

data_path = "../../data/lsm/marmousi.bin"

# Get true model
#vp = 1e-3 * np.fromfile(data_path, dtype='float32', sep="") # devito expects velocity to have km/s as units
#vp = vp.reshape(origshape)[:shape[0],:shape[1]]
vp = np.ones(shape)
true_model = Model(space_order=space_order, vp=vp, origin=origin, shape=shape,
                   dtype=np.float32, spacing=spacing, nbl=nbl, bcs="damp")

# Get smooth model
v0 = scipy.ndimage.gaussian_filter(vp, sigma=10)    
v0[:,:36] = 1.5 # Do not smooth water layer
smooth_model = Model(space_order=space_order, vp=v0, origin=origin, shape=shape,
                     dtype=np.float32, spacing=spacing, nbl=nbl, bcs="damp")

# display
plot_velocity(true_model)
plot_velocity(smooth_model)

# Compute initial born perturbation from m - m0
dm = (vp**(-2) - v0**(-2))

plt.figure(figsize=(14, 5))
plt.imshow(dm.T, 
           cmap='jet', vmin=-1e-1, vmax=1e-1)
plt.title('Dm')
plt.axis('tight')
plt.colorbar()

src_x = np.linspace(0, shape[0]*spacing[0], num=nsrc)
src_z = 20. 

rec_x = np.linspace(0, shape[0]*spacing[0], num=nrec)
rec_z = 20.

Aop = AcousticWave2D(shape, origin, spacing, v0 * 1e3,
                     src_x, src_z, rec_x, rec_z, 0., tn, 'Ricker',
                     space_order=6, nbl=nbl, f0=f0,
                     dtype="float32", name="A")

dobs = Aop @ dm

plt.figure(figsize=(10, 9))
plt.imshow(dobs[Aop.geometry.nsrc//2].reshape(Aop.geometry.nrec, Aop.geometry.nt).T, 
           cmap='gray', vmin=-0.1, vmax=0.1)
plt.title('FD modelling')
plt.axis('tight')
plt.savefig(os.path.join(figdir, f'Data.png'))
