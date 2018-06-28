#!/bin/sh

h5ls /home/jklynch/host/project/viral-learning/data/phage_prok_500.h5

python synth_train_vgg.py \
    --seed 1 \
    --bp 5000 \
    --batch-size 64 \
    --train phage:10000,proka:10000 \
    --dev phage:1000,proka:1000 \
    --test phage:1000:,proka:1000 \
    --epochs 5
