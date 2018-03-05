"""
Read Alise's training and testing data files and smash them all into one H5 file.

This script is hard-coded to run on ocelote.
"""
import glob
import os
import sys
import time

import h5py
import numpy as np
import pandas as pd


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

        training_line_count = int(sys.argv[1])

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
                tsv_fp_list=sorted(glob.glob(os.path.join(data_source_dir, 'eval_set/bact_marinePatric/extract_bact_{}/kmers/kmer_file*.fasta.tab'.format(read_length)))),
                h5_file=h5_file,
                dset_name='bact_marinePatric/extract_bact_{}/kmers/kmer_file1'.format(read_length),
                line_count=testing_line_count)

        for read_length in (100, 200, 500, 1000, 5000, 10000):
            read_tsv_write_h5_group(
                tsv_fp_list=sorted(glob.glob(os.path.join(data_source_dir, 'eval_set/vir_marinzPatric/extract_vir_{}/kmers/kmer_file*.fasta.tab'.format(read_length)))),
                h5_file=h5_file,
                dset_name='vir_marinePatric/extract_vir_{}/kmers/kmer_file1'.format(read_length),
                line_count=testing_line_count)


def read_tsv_write_h5_group(tsv_fp_list, h5_file, dset_name, line_count):
    t0 = time.time()
    dataset_row = 0

    for i, tsv_fp in enumerate(tsv_fp_list):
        print('reading "{}"'.format(tsv_fp))
        with open(tsv_fp, 'rt') as input_file:
            #input_header = input_file.readline().strip().split('\t')

            input_reader = pd.read_table(input_file, chunksize=100000, usecols=range(1, 32768+1))
            #for line in input_file:
            t0 = time.time()
            for chunk in input_reader:
                t1 = time.time()
                print('read chunk in {:5.2f}s'.format(t1-t0))
                if i == 0:
                    # do not store the first and last columns
                    # store only kmer counts
                    # the first is a string, the last is read_type
                    print('chunk shape: {}'.format(chunk.shape))
                    print('chunk values.shape: {}'.format(chunk.values.shape))
                    #print(chunk[:5, :5])
                    print(chunk.values[:5, :5])
                    dset_shape = (line_count, chunk.shape[1])
                    print('dataset shape is {}'.format(dset_shape))
                    dset = h5_file.require_dataset(
                        dset_name,
                        dset_shape,
                        maxshape=dset_shape,  # make the dataset shrinkable but not enlargeable
                        # I tried np.float32 to save space but very little space was saved
                        # 139MB vs 167MB for 5000 rows?
                        dtype=np.float64,
                        # write speed and compression are best with 1-row chunks?
                        chunks=(1, dset_shape[1]),
                        compression='gzip')

                #dset[dataset_row, :] = np.asarray([float(d) for d in line.strip().split('\t')[1:-1]])
                dset[dataset_row:(dataset_row+chunk.shape[0]), :] = chunk.values
                #dataset_row += 1
                dataset_row += chunk.shape[0]
                t0 = time.time()
                print('wrote chunk in {:52f}s'.format(t0-t1))
                if dataset_row >= dset.shape[0]:
                    print('dset {} is full'.format(dset.name))
                    print('  dataset_row {}'.format(dataset_row))
                    print('  dset.shape  {}'.format(dset.shape))
                    break

    print('wrote {} rows in {:5.2f}s'.format(dataset_row, time.time()-t0))

    if dataset_row == dset_shape[0]:
        print('dataset size is correct')
    elif dataset_row < dset_shape[0]:
        print('shrinking dataset from shape {} to {}'.format(dset_shape, (dataset_row, dset_shape[1])))
        dset.resize((dataset_row, dset_shape[1]))
    else:
        # should not happen
        raise Exception()



t0 = time.time()
write_all_training_and_testing_data()
print('wrote all training and testing data in {:5.2f}s to {}'.format(time.time()-t0, data_target_fp))
