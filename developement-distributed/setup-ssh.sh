#!/usr/bin/env bash
#
# Setup dask server via ssh. 
# Run as ./setup-ssh.sh ${SCHEDULER}

dask-ssh --hostfile hostfile.txt --scheduler $1
