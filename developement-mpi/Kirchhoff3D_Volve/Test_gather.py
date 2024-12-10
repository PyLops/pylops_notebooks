import os
import numpy as np
import pylops_mpi
import time
import segyio
import warnings
warnings.filterwarnings('ignore')

from scipy.signal import filtfilt
from matplotlib import pyplot as plt
from mpi4py import MPI

from pylops.waveeqprocessing.kirchhoff import Kirchhoff
from pylops_mpi.DistributedArray import local_split, Partition
from segyshot import SegyShot
from visual import explode_volume

plt.close("all")
comm = MPI.COMM_WORLD
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()


trav_recsrank_eik = np.ones((20000, 10)).T.flatten()
trav_recs_eik = np.concatenate(comm.All(trav_recsrank_eik)).reshape(20000*size, 10)
print('trav_recs_eik', trav_recs_eik.shape)