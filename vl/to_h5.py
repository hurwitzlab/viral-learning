"""
Read Alise's training and testing data files and smash them all into one H5 file.
"""
import os
import time

import h5py
import numpy as np


data_source_dir = '/rsgrps/bhurwitz/alise/my_data/Machine_learning/size_read/contigs_training/set/cleanSet_Centrifuge'


def write_all_training_and_testing_data():
    h5_fp = '/extra/jklynch/viral-learning/training_testing.h5'
    with h5py.File(h5_fp, 'w') as h5_file:
        h5_file.create_group('/clean-bact/training1/extract/kmers')
        h5_file.create_group('/clean-vir/training1/extract/kmers')
        h5_file.create_group('/vir_marinePatric/extract_vir_100/kmers')
        h5_file.create_group('/vir_marinePatric/extract_vir_200/kmers')
        h5_file.create_group('/vir_marinePatric/extract_vir_500/kmers')
        h5_file.create_group('/vir_marinePatric/extract_vir_1000/kmers')
        h5_file.create_group('/vir_marinePatric/extract_vir_5000/kmers')
        h5_file.create_group('/vir_marinePatric/extract_vir_10000/kmers')

        training_line_count = 1000

        read_tsv_write_h5_group(
            tsv_fp=os.path.join(data_source_dir, 'clean-bact/training1/extract/kmers/kmer_file1.fasta.tab'),
            h5_file=h5_file,
            dset_name='clean-bact/training1/extract/kmers/kmer_file1',
            line_count=training_line_count)

        read_tsv_write_h5_group(
            tsv_fp=os.path.join(data_source_dir, 'clean-bact/training1/extract/kmers/kmer_file2.fasta.tab'),
            h5_file=h5_file,
            dset_name='clean-bact/training1/extract/kmers/kmer_file2',
            line_count=training_line_count)

        read_tsv_write_h5_group(
            tsv_fp=os.path.join(data_source_dir, 'clean-vir/training1/extract/kmers/kmer_file1.fasta.tab'),
            h5_file=h5_file,
            dset_name='clean-vir/training1/extract/kmers/kmer_file1.fasta.tab',
            line_count=training_line_count)

        read_tsv_write_h5_group(
            tsv_fp=os.path.join(data_source_dir, 'clean-bact/training1/extract/kmers/kmer_file2.fasta.tab'),
            h5_file=h5_file,
            dset_name='clean-vir/training1/extract/kmers/kmer_file2.fasta.tab',
            line_count=training_line_count)


def read_tsv_write_h5_group(tsv_fp, h5_file, dset_name, line_count):
    t0 = time.time()
    with open(tsv_fp, 'rt') as input_file:
        input_header = input_file.readline().strip().split('\t')
        # do not store the first and last columns
        # store only kmer counts
        # the first is a string, the last is read_type
        dset_shape = (line_count, len(input_header)-2)
        print('dataset shape is {}'.format(dset_shape))
        dset = h5_file.create_dataset(
            dset_name,
            dset_shape,
            # I tried np.float32 to save space but very little space was saved
            # 139MB vs 167MB for 5000 rows
            dtype=np.float64,
            # write speed and compression are best with 1-row chunks?
            chunks=(1, dset_shape[1]),
            compression='gzip')
        for i, line in enumerate(input_file):
            if i >= dset.shape[0]:
                break
            dset[i, :] = [float(d) for d in line.strip().split('\t')[1:-1]]

        print('wrote {} rows in {:5.2f}s'.format(dset.shape[0], time.time()-t0))


t0 = time.time()
write_all_training_and_testing_data()
print('wrote all training and testing data in {:5.2f}s'.format(time.time()-t0))
