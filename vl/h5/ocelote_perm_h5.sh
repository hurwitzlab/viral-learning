#!/bin/sh

#PBS -N perm_h5
#PBS -m bea
#PBS -M jklynch@email.arizona.edu
#PBS -W group_list=bhurwitz
#PBS -q standard
#PBS -l select=1:ncpus=28:mem=168gb
#PBS -l walltime=04:00:00

source activate ktf

export OMP_NUM_THREADS=28

cd ~/project/viral-learning/vl

python h5/perm_h5.py /extra/jklynch/viral-learning/training_testing.h5

#python h5/norm_h5.py /extra/jklynch/viral-learning/perm_training_testing.h5
