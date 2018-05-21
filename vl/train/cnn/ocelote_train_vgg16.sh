#!/bin/sh

source activate ktf

h5ls /extra/jklynch/phage_prok_500.h5

python train_vgg.py \
    --input-fp /extra/jklynch/phage_prok_500.h5 \
    --train phage500_training_set.fa:0:80000,proc500_training_set.fasta:0:80000 \
    --dev phage500_training_set.fa:80000:90000,proc500_training_set.fasta:80000:90000 \
    --test phage500_training_set.fa:90000:,proc500_training_set.fasta:90000: \
    --epochs 2
