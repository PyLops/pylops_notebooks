#!/bin/sh
# Running jupyter-notebook with env variables set up to ensure BLAS/MKL to run in 
# serial mode and avoid oversubscribingthreads

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

jupyter-notebook

