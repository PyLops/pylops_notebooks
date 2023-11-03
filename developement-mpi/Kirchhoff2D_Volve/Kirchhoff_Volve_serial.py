r"""
Kirchhoff migration of 2D Volve dataset distributed over sources - serial benchmark 

Run as: export OMP_NUM_THREADS=4; export MKL_NUM_THREADS=4; export NUMBA_NUM_THREADS=4; python Kirchhoff_Volve_serial.py 
"""

import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

from scipy.signal import filtfilt
from matplotlib import pyplot as plt

from pylops.waveeqprocessing.kirchhoff import Kirchhoff

tic = time.perf_counter()  

print('Kirchhoff migration of 2D Volve data')

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
plt.savefig('Model_serial.png')


# Data
f = np.load('/home/ravasim/Documents/2021/Projects/MDD-StochasticSolvers/data/mdd_volvereal_multimasked_iter40_batch32_all.npz')

t = f['t']
dt = t[1]
nt = t.size
d = f['Rnsgd'].transpose(1, 2, 0)

# Convert from global to local grid
xorig = x[0]
x -= xorig
dx, dz = x[1], z[1]

s[0] -= xorig
r[0] -= xorig

# Compute traveltimes
print('Compute traveltimes...')
trav_srcs_eik, _, _, _, _, _ = \
    Kirchhoff._traveltime_table(z, x, s, r[:, :1], vel, mode='eikonal')
trav_recs_eik = trav_srcs_eik.copy()

fig, axs = plt.subplots(3, 1, sharey=True, figsize=(14, 18))
axs[0].imshow(trav_srcs_eik[:, 0].reshape((nx, nz)).T, cmap='tab10', 
            extent = (x[0], x[-1], z[-1], z[0]))
axs[0].scatter(s[0, 0], s[1, 0], marker='*', s=150, c='r', edgecolors='k')
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
axs[2].scatter(s[0, 10], s[1, 10], marker='*', s=150, c='r', edgecolors='k')
axs[2].scatter(r[0, -10], r[1, -10], marker='v', s=150, c='b', edgecolors='k')
axs[2].axis('tight')
axs[2].set_xlabel('x [m]')
axs[2].set_title('Src+rec traveltime')
axs[2].set_ylim(z[-1], z[0])
plt.savefig('Travs_serial.png')

# Wavelet
wav, wavc = np.zeros(81), 41
wav[wavc] = 1.

# Kirchhoff operator
KOp = Kirchhoff(z, x, t, s, r, vel, wav, wavc, dynamic=False, 
                trav=(trav_srcs_eik, trav_recs_eik),
                mode='byot', engine='numba')

# Imaging
print('Perform imaging...')
ticim = time.perf_counter()  
image = KOp.H * d
image = image.reshape(nx, nz)
tocim = time.perf_counter()  

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
plt.savefig('Image_serial.png')

toc = time.perf_counter()  

print(f'Imaging elapsed time: {tocim-ticim} s')
print(f'Total elapsed time: {toc-tic} s')

f = open("Kirchhoff_Volve_timings.txt", "a")
f.write(f"{1} \t {tocim-ticim:.3f} \t {toc-tic:.3f}\n")
f.close()