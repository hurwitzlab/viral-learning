#!/bin/sh

DATA_DIR=../data

python kmer_nn_h5_generator.py \
    ${DATA_DIR}/bact_kmer_file1.h5 \
    ${DATA_DIR}/bact_kmer_file2.h5 \
    ${DATA_DIR}/vir_kmer_file1.h5 \
    ${DATA_DIR}/vir_kmer_file2.h5 \
    100
