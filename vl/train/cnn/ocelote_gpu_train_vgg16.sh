#!/bin/sh

#PBS -N gpu_train_cnn_vgg16
#PBS -m bea
#PBS -M jklynch@email.arizona.edu
#PBS -W group_list=bhurwitz
#PBS -q standard
#PBS -l select=1:ncpus=28:ngpus=1:mem=168gb
#PBS -l walltime=08:00:00

source activate ktfgpu

#h5ls /extra/jklynch/phage_prok_500.h5

cd ~/project/viral-learning/vl/train/cnn

time python train_vgg.py \
    --input-fp /extra/jklynch/phage_prok_500.h5 \
    --train phage500_training_set.fa:0:800,proc500_training_set.fasta:0:800 \
    --dev phage500_training_set.fa:800:900,proc500_training_set.fasta:800:900 \
    --test phage500_training_set.fa:90000:,proc500_training_set.fasta:90000: \
    --epochs 2
