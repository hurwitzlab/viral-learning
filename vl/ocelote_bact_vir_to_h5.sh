source activate ktf

TRAINING_DATA_DIR=/rsgrps/bhurwitz/alise/my_data/Machine_learning/size_read/contigs_training/set/cleanSet_Centrifuge

H5_DATA_DIR=/extra/jklynch/viral-learning

time \
  python to_h5.py \
    ${TRAINING_DATA_DIR}/clean-bact/training1/extract/kmers/kmer_file1.fasta.tab \
    ${H5_DATA_DIR}/clean-bact/training1/extract/kmers/kmer_file1.h5 \
    bacteria \
    200000 &

time \
  python to_h5.py \
    ${TRAINING_DATA_DIR}/clean-bact/training1/extract/kmers/kmer_file2.fasta.tab \
    ${H5_DATA_DIR}/clean-bact/training1/extract/kmers/kmer_file2.h5 \
    bacteria \
    200000 &

time \
  python to_h5.py \
    ${TRAINING_DATA_DIR}/clean-vir/training1/extract/kmers/kmer_file1.fasta.tab \
    ${H5_DATA_DIR}/clean-vir/training1/extract/kmers/kmer_file1.h5 \
    virus \
    200000 &

time \
  python to_h5.py \
    ${TRAINING_DATA_DIR}/clean-vir/training1/extract/kmers/kmer_file2.fasta.tab \
    ${H5_DATA_DIR}/clean-vir/training1/extract/kmers/kmer_file2.h5 \
    virus \
    200000 &

