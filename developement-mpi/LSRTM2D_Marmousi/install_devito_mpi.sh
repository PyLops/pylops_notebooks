#!/bin/bash
# 
# Installer for Devito CPU environment
# 
# Run: ./install_devito.sh 
#
# M. Ravasi, 29/03/2021

echo 'Creating Devito CPU environment'

module load gcc/9.4.0/gcc-7.5.0-rukpabt 
module load mpich/3.4.2/gcc-7.5.0-oorergi  

# create conda env
conda env create -f environment.yml
source ~/miniconda3/etc/profile.d/conda.sh
conda activate devito_mpi
echo 'Created and activated environment:' $(which python)

# install devito
pip install git+https://github.com/devitocodes/devito@master

# check packages work as expected
echo 'Checking devito version and running a command...'
python -c 'import devito; print(devito.__version__)'

echo 'Done!'