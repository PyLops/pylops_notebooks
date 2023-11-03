#!/bin/sh
# Runner for Kirchhoff_Volve.py for different groups of sources

for srcin in {3080..9596..300}
do
  export OMP_NUM_THREADS=4; export MKL_NUM_THREADS=4; export NUMBA_NUM_THREADS=4; mpiexec -n 17 python Kirchhoff_Volve.py ${srcin} 300 
done
