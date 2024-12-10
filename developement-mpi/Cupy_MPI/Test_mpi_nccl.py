r"""
Test NCCL for distributed computing between GPUs (no passing data via MPI)

Run as: module load cuda/11.5.0/gcc-7.5.0-syen6pj; export NCCL_DEBUG="WARN"; mpiexec -n 2 python Test_nccl.py
"""

from mpi4py import MPI
import cupy as cp
import cupy.cuda.nccl as nccl

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Number of GPUs per node (assuming the same for all nodes)
local_gpus = 1

# Create a unique NCCL ID on rank 0 and broadcast it
if rank == 0:
    nccl_id = nccl.get_unique_id()
else:
    nccl_id = None

nccl_id = comm.bcast(nccl_id, root=0)

# Initialize CUDA devices
device = cp.cuda.Device(rank)

# Initialize arrays on each device
with cp.cuda.Device(device):
    array = cp.array([1, 2, 3, 4], dtype=cp.float32)
    print(rank, array, array.device, array + array)

# Initialize NCCL communicators (does not work)
comms = nccl.NcclCommunicator(size * local_gpus, nccl_id, rank)

"""
# Perform all-reduce operation
for i, d in enumerate(devices):
    with d:
        comms[i].allReduce(arrays[i].data.ptr, arrays[i].data.ptr, arrays[i].size, nccl.NCCL_FLOAT32, nccl.NCCL_SUM, cp.cuda.Stream.null.ptr)

# Synchronize to ensure all operations are complete
for d in devices:
    with d:
        cp.cuda.Stream.null.synchronize()

# Verify the result
for i, d in enumerate(devices):
    with d:
        print(f"Node {rank}, Device {i}: {arrays[i]}")
"""
# Finalize MPI
MPI.Finalize()
