#!/bin/sh
#PBS -N evaluate_layers
#PBS -m bea
#PBS -M jklynch@email.arizona.edu
#PBS -W group_list=bhurwitz
#PBS -q standard
#PBS -l select=1:ncpus=28:mem=168gb
#PBS -l walltime=12:00:00

source activate ktf

export OMP_NUM_THREADS=28

`pwd`
cd ~/project/viral-learning/vl

VIRAL_LEARNING_DIR=~/extra/jklynch/viral-learning/

time python model/layers/evaluate_network_depth.py \
    -i ${VIRAL_LEARNING_DIR}/data/perm_training_testing.h5 \
    -o model/layers/ocelote/output_dropout.pdf \
    --epoch-count 10 \
    --process-count 13
