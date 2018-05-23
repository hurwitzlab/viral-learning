#!/bin/sh

python train_nn_meta_genome.py \
    --input-fp /home/jklynch/host/project/viral-learning/data/perm_training_testing.h5 \
    --training-sample-count 100000 \
    --dev-sample-count 2000 \
    --half-batch-count 50 \
    --epoch-count 5
