#!/bin/sh

DATA_DIR=/extra/jklynch/viral-learning

python kmer_nn_h5_generator.py \
    ${DATA_DIR}/clean-bact/training1/extract/kmers/kmer_file1.h5 \
    ${DATA_DIR}/bact_marinePatric/extract_bact_10000/kmers/kmer_file1.h5 \
    ${DATA_DIR}/clean-vir/training1/extract/kmers/kmer_file1.h5 \
    ${DATA_DIR}/vir_marinePatric/extract_vir_10000/kmers/kmer_file1.h5 \
    10 \
    200
