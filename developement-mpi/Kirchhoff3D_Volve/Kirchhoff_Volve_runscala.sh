#!/bin/sh
# Runner for Kirchhoff_Volve.py for scalability test

for nranks in {6..12..2}
do
  export OMP_NUM_THREADS=2; export MKL_NUM_THREADS=2; export NUMBA_NUM_THREADS=2; mpiexec -n ${nranks} python Kirchhoff_Volve.py 5370 300 
done
