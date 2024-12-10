
from matplotlib import pyplot as plt
import numpy as np
from mpi4py import MPI

plt.close("all")
np.random.seed(42)
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()

a = 1