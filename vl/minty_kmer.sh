#!/bin/sh

DATA_DIR=../data

python kmer_nn_np_generator.py \
    ${DATA_DIR}/bact_kmer_file1.fasta.tab.gz \
    ${DATA_DIR}/bact_kmer_file2.fasta.tab.gz \
    ${DATA_DIR}/vir_kmer_file1.fasta.tab.gz \
    ${DATA_DIR}/vir_kmer_file2.fasta.tab.gz \
    200
