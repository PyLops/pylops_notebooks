#!/bin/sh
# Runner for Poststack_Volve.py 

#export OMP_NUM_THREADS=4; export MKL_NUM_THREADS=4; export NUMBA_NUM_THREADS=4; mpiexec -n 4 python Poststack_Volve.py
export OMP_NUM_THREADS=2; export MKL_NUM_THREADS=2; export NUMBA_NUM_THREADS=2; mpiexec -n 16 python Poststack_Volve.py