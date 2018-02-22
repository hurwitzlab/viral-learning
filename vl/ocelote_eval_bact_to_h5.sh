source activate ktf

EVAL_DATA_DIR=/rsgrps/bhurwitz/alise/my_data/Machine_learning/size_read/eval_set

H5_DATA_DIR=/extra/jklynch/viral-learning

time \
  python to_h5.py \
    ${EVAL_DATA_DIR}/bact_marinePatric/extract_bact_100/kmers/kmer_file1.fasta.tab \
    ${H5_DATA_DIR}/bact_marinePatric/extract_bact_100/kmers/kmer_file1.h5 \
    bacteria \
    5000 &

time \
  python to_h5.py \
    ${EVAL_DATA_DIR}/bact_marinePatric/extract_bact_200/kmers/kmer_file1.fasta.tab \
    ${H5_DATA_DIR}/bact_marinePatric/extract_bact_200/kmers/kmer_file1.h5 \
    bacteria \
    5000 &

time \
  python to_h5.py \
    ${EVAL_DATA_DIR}/bact_marinePatric/extract_bact_500/kmers/kmer_file1.fasta.tab \
    ${H5_DATA_DIR}/bact_marinePatric/extract_bact_500/kmers/kmer_file1.h5 \
    bacteria \
    5000 &

time \
  python to_h5.py \
    ${EVAL_DATA_DIR}/bact_marinePatric/extract_bact_1000/kmers/kmer_file1.fasta.tab \
    ${H5_DATA_DIR}/bact_marinePatric/extract_bact_1000/kmers/kmer_file1.h5 \
    bacteria \
    5000 &

time \
  python to_h5.py \
    ${EVAL_DATA_DIR}/bact_marinePatric/extract_bact_5000/kmers/kmer_file1.fasta.tab \
    ${H5_DATA_DIR}/bact_marinePatric/extract_bact_5000/kmers/kmer_file1.h5 \
    bacteria \
    5000 &

time \
  python to_h5.py \
    ${EVAL_DATA_DIR}/bact_marinePatric/extract_bact_10000/kmers/kmer_file1.fasta.tab \
    ${H5_DATA_DIR}/bact_marinePatric/extract_bact_10000/kmers/kmer_file1.h5 \
    bacteria \
    5000 &

