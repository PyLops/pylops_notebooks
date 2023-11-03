#!/bin/sh
# Runner for Kirchhoff_Volve.py with increasing ranks

for rank in {4..20..1}
do
  export OMP_NUM_THREADS=4; export MKL_NUM_THREADS=4; export NUMBA_NUM_THREADS=4; mpiexec -n ${rank} python Kirchhoff_Volve.py 
done