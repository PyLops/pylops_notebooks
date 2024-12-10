#!/bin/sh
# Runner for Poststack_Volve_cupy2.py

export OMP_NUM_THREADS=2; export MKL_NUM_THREADS=16; export NUMBA_NUM_THREADS=16; mpiexec -n 2 python Poststack_Volve_cupy2.py