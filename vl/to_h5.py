"""
Read Alise's training and testing data files and smash them all into one H5 file.
"""
import glob
import os
import time

import h5py
import numpy as np


data_source_dir = '/rsgrps/bhurwitz/alise/my_data/Machine_learning/size_read'
data_target_fp = '/extra/jklynch/viral-learning/training_testing.h5'

def write_all_training_and_testing_data():
    h5_fp = data_target_fp
    with h5py.File(h5_fp, 'w') as h5_file:
        h5_file.create_group('/clean-bact/training1/extract/kmers')
        h5_file.create_group('/clean-vir/training1/extract/kmers')
        h5_file.create_group('/vir_marinePatric/extract_vir_100/kmers')
        h5_file.create_group('/vir_marinePatric/extract_vir_200/kmers')
        h5_file.create_group('/vir_marinePatric/extract_vir_500/kmers')
        h5_file.create_group('/vir_marinePatric/extract_vir_1000/kmers')
        h5_file.create_group('/vir_marinePatric/extract_vir_5000/kmers')
        h5_file.create_group('/vir_marinePatric/extract_vir_10000/kmers')

        training_line_count = 100

        read_tsv_write_h5_group(
            tsv_fp_list=(os.path.join(data_source_dir, 'contigs_training/set/cleanSet_Centrifuge/clean-bact/training1/extract/kmers/kmer_file1.fasta.tab'), ),
            h5_file=h5_file,
            dset_name='clean-bact/training1/extract/kmers/kmer_file1',
            line_count=training_line_count)

        read_tsv_write_h5_group(
            tsv_fp_list=(os.path.join(data_source_dir, 'contigs_training/set/cleanSet_Centrifuge/clean-bact/training1/extract/kmers/kmer_file2.fasta.tab'), ),
            h5_file=h5_file,
            dset_name='clean-bact/training1/extract/kmers/kmer_file2',
            line_count=training_line_count)

        read_tsv_write_h5_group(
            tsv_fp_list=(os.path.join(data_source_dir, 'contigs_training/set/cleanSet_Centrifuge/clean-vir/training1/extract/kmers/kmer_file1.fasta.tab'), ),
            h5_file=h5_file,
            dset_name='clean-vir/training1/extract/kmers/kmer_file1',
            line_count=training_line_count)

        read_tsv_write_h5_group(
            tsv_fp_list=(os.path.join(data_source_dir, 'contigs_training/set/cleanSet_Centrifuge/clean-bact/training1/extract/kmers/kmer_file2.fasta.tab'), ),
            h5_file=h5_file,
            dset_name='clean-vir/training1/extract/kmers/kmer_file2',
            line_count=training_line_count)

        testing_line_count = 5000

        for read_length in (100, 200, 500, 1000, 5000, 10000):
            read_tsv_write_h5_group(
                tsv_fp_list=glob.glob(os.path.join(data_source_dir, 'eval_set/bact_marinePatric/extract_bact_{}/kmers/kmer_file*.fasta.tab'.format(read_length))),
                h5_file=h5_file,
                dset_name='bact_marinePatric/extract_bact_{}/kmers/kmer_file1'.format(read_length),
                line_count=testing_line_count)

        for read_length in (100, 200, 500, 1000, 5000, 10000):
            read_tsv_write_h5_group(
                tsv_fp_list=glob.glob(os.path.join(data_source_dir, 'eval_set/vir_marinzPatric/extract_vir_{}/kmers/kmer_file*.fasta.tab'.format(read_length))),
                h5_file=h5_file,
                dset_name='vir_marinePatric/extract_vir_{}/kmers/kmer_file1'.format(read_length),
                line_count=testing_line_count)


# https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
# for a new value newValue, compute the new count, new mean, the new M2.
# mean accumulates the mean of the entire dataset
# M2 aggregates the squared distance from the mean
# count aggregates the number of samples seen so far
def update(existingAggregate, newValue):
    (count, mean, M2) = existingAggregate
    count = count + 1
    delta = newValue - mean
    mean = mean + delta / count
    delta2 = newValue - mean
    M2 = M2 + delta * delta2

    return (count, mean, M2)


# retrieve the mean and variance from an aggregate
def finalize(existingAggregate):
    (count, mean, M2) = existingAggregate
    (mean, variance) = (mean, M2/(count - 1))
    if count < 2:
        return float('nan')
    else:
        return (mean, variance)


def read_tsv_write_h5_group(tsv_fp_list, h5_file, dset_name, line_count):
    t0 = time.time()
    dataset_row = 0

    mean_var_state = None

    for i, tsv_fp in enumerate(tsv_fp_list):
        print('reading "{}"'.format(tsv_fp))
        with open(tsv_fp, 'rt') as input_file:
            input_header = input_file.readline().strip().split('\t')
            if i == 0:
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

            for line in input_file:
                if dataset_row >= dset.shape[0]:
                    break
                dset[dataset_row, :] = np.asarray([float(d) for d in line.strip().split('\t')[1:-1]])
                dataset_row += 1

            print('wrote {} rows in {:5.2f}s'.format(dset.shape[0], time.time()-t0))


t0 = time.time()
write_all_training_and_testing_data()
print('wrote all training and testing data in {:5.2f}s to {}'.format(time.time()-t0, data_target_fp))
