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

nr_rank = 10 if rank < size//2 else 11
nr = np.array(comm.allgather(nr_rank))
if rank == 0:
    print('nr', nr)
trav_recsrank_eik = rank*np.ones(nr_rank, dtype='float32')
trav_recs_eik = np.zeros(np.sum(nr), dtype='float32')
ntrav_rank = nr
itravrin_rank = np.insert(np.cumsum(nr)[:-1],0,0)
print(ntrav_rank, itravrin_rank)
comm.Allgatherv(trav_recsrank_eik, [trav_recs_eik, ntrav_rank, itravrin_rank, MPI.FLOAT])
print(trav_recs_eik, trav_recs_eik.shape)