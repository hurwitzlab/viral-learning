#!/bin/sh

DATA_DIR=/rsgrps/bhurwitz/alise/my_data/Machine_learning/size_read/contigs_training/set/cleanSet_Centrifuge

python kmer_nn_np_generator.py \
    ${DATA_DIR}/clean-bact/training1/extract/kmers/kmer_file1.fasta.tab \
    ${DATA_DIR}/clean-bact/training1/extract/kmers/kmer_file2.fasta.tab \
    ${DATA_DIR}/clean-vir/training1/extract/kmers/kmer_file1.fasta.tab \
    ${DATA_DIR}/clean-vir/training1/extract/kmers/kmer_file2.fasta.tab \
    100