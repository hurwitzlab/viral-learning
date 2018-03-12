#!/bin/bash

source activate ktf

VIRAL_LEARNING_DIR=~/host/project/viral-learning/

time python ${VIRAL_LEARNING_DIR}/vl/model/layers/evaluate_network_depth.py \
    -i ${VIRAL_LEARNING_DIR}/data/perm_training_testing.h5 \
    -o ${VIRAL_LEARNING_DIR}/vl/model/layers/minty/output_dropout.pdf \
    --epoch-count 10 \
    --process-count 2 \
    --verbosity 1