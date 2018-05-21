#!/bin/sh

h5ls /home/jklynch/host/project/viral-learning/data/phage_prok_500.h5

python train_vgg.py \
    --input-fp /home/jklynch/host/project/viral-learning/data/phage_prok_500.h5 \
    --train phage500_training_set.fa:0:1000,proc500_training_set.fasta:0:1000 \
    --dev phage500_training_set.fa:80000:81000,proc500_training_set.fasta:80000:81000 \
    --test phage500_training_set.fa:90000:,proc500_training_set.fasta:90000: \
    --epochs 2
