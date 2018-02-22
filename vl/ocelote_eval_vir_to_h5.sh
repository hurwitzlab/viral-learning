source activate ktf

EVAL_DATA_DIR=/rsgrps/bhurwitz/alise/my_data/Machine_learning/size_read/eval_set

H5_DATA_DIR=/extra/jklynch/viral-learning

time \
  python to_h5.py \
    ${EVAL_DATA_DIR}/vir_marinzPatric/extract_vir_100/kmers/kmer_file1.fasta.tab \
    ${H5_DATA_DIR}/vir_marinePatric/extract_vir_100/kmers/kmer_file1.h5 \
    virus \
    5000 &

time \
  python to_h5.py \
    ${EVAL_DATA_DIR}/vir_marinzPatric/extract_vir_200/kmers/kmer_file1.fasta.tab \
    ${H5_DATA_DIR}/vir_marinePatric/extract_vir_200/kmers/kmer_file1.h5 \
    virus \
    5000 &

time \
  python to_h5.py \
    ${EVAL_DATA_DIR}/vir_marinzPatric/extract_vir_500/kmers/kmer_file1.fasta.tab \
    ${H5_DATA_DIR}/vir_marinePatric/extract_vir_500/kmers/kmer_file1.h5 \
    virus \
    5000 &

time \
  python to_h5.py \
    ${EVAL_DATA_DIR}/vir_marinzPatric/extract_vir_1000/kmers/kmer_file1.fasta.tab \
    ${H5_DATA_DIR}/vir_marinePatric/extract_vir_1000/kmers/kmer_file1.h5 \
    virus \
    5000 &

time \
  python to_h5.py \
    ${EVAL_DATA_DIR}/vir_marinzPatric/extract_vir_5000/kmers/kmer_file1.fasta.tab \
    ${H5_DATA_DIR}/vir_marinePatric/extract_vir_5000/kmers/kmer_file1.h5 \
    virus \
    5000 &

time \
  python to_h5.py \
    ${EVAL_DATA_DIR}/vir_marinzPatric/extract_vir_10000/kmers/kmer_file1.fasta.tab \
    ${H5_DATA_DIR}/vir_marinePatric/extract_vir_10000/kmers/kmer_file1.h5 \
    virus \
    5000 &

