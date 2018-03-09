#!/bin/sh

#SBATCH -A iPlant-Collabs
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 48:00:00
#SBATCH -p skx-normal
#SBATCH -J perm_h5
#SBATCH --mail-type BEGIN,END,FAIL
#SBATCH --mail-user jklynch@email.arizona.edu

source activate ktf

export OMP_NUM_THREADS=48

`pwd`
cd $WORK/project/viral-learning/vl

cp $WORK/project/viral-learning/data/training_testing.h5 /tmp

python h5/perm_h5.py /tmp/training_testing.h5

cp /tmp/perm_training_testing.h5 $WORK/project/viral-learning/perm_training_testing.h5
