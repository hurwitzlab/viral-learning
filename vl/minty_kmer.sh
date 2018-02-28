#!/bin/sh

DATA_DIR=../data

python kmer_nn_h5_generator.py \
    ${DATA_DIR}/training_testing.h5 \
    /clean-bact/training1/extract/kmers/kmer_file1 \
    /clean-vir/training1/extract/kmers/kmer_file1 \
    5 \
    1000
