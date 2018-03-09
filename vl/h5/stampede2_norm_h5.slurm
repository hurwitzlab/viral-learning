#!/bin/sh

#SBATCH -A iPlant-Collabs
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 48:00:00
#SBATCH -p skx-normal
#SBATCH -J norm_h5
#SBATCH --mail-type BEGIN,END,FAIL
#SBATCH --mail-user jklynch@email.arizona.edu

conda activate ktf

export OMP_NUM_THREADS=48

`pwd`
cd $WORK/project/viral-learning/vl

python h5/norm_h5.py $WORK/project/viral-learning/data/training_testing.h5
