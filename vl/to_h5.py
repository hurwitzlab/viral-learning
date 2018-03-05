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


data_source_dir = '/rsgrps/bhurwitz/alise/my_data/Machine_learning/size_read'
data_target_fp = '/extra/jklynch/viral-learning/training_testing.h5'

def create_dataset(h5_file, name, shape):
    dset = h5_file.create_dataset(
        name,
        shape,
        maxshape=(None, shape[1]),
        # I tried np.float32 to save space but very little space was saved
        # 139MB vs 167MB for 5000 rows?
        dtype=np.float64,
        # write speed and compression are best with 1-row chunks?
        chunks=(1, shape[1]),
        compression='gzip')

    return dset

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
        column_count = 32768

        read_tsv_write_h5_group(
            tsv_fp_list=(os.path.join(data_source_dir, 'contigs_training/set/cleanSet_Centrifuge/clean-bact/training1/extract/kmers/kmer_file1.fasta.tab'), ),
            dset=create_dataset(h5_file, name='clean-bact/training1/extract/kmers/kmer_file1', shape=(training_line_count, column_count)),
            chunksize=10000)

        read_tsv_write_h5_group(
            tsv_fp_list=(os.path.join(data_source_dir, 'contigs_training/set/cleanSet_Centrifuge/clean-bact/training1/extract/kmers/kmer_file2.fasta.tab'), ),
            dset=create_dataset(h5_file, name='clean-bact/training1/extract/kmers/kmer_file2', shape=(training_line_count, column_count)),
            chunksize=10000)

        read_tsv_write_h5_group(
            tsv_fp_list=(os.path.join(data_source_dir, 'contigs_training/set/cleanSet_Centrifuge/clean-vir/training1/extract/kmers/kmer_file1.fasta.tab'), ),
            dset=create_dataset(h5_file, name='clean-vir/training1/extract/kmers/kmer_file1', shape=(training_line_count, column_count)),
            chunksize=10000)

        read_tsv_write_h5_group(
            tsv_fp_list=(os.path.join(data_source_dir, 'contigs_training/set/cleanSet_Centrifuge/clean-bact/training1/extract/kmers/kmer_file2.fasta.tab'), ),
            dset=create_dataset(h5_file, name='clean-vir/training1/extract/kmers/kmer_file2', shape=(training_line_count, column_count)),
            chunksize=10000)

        for read_length in (100, 200, 500, 1000, 5000, 10000):
            read_tsv_write_h5_group(
                tsv_fp_list=sorted(glob.glob(os.path.join(data_source_dir, 'eval_set/bact_marinePatric/extract_bact_{}/kmers/kmer_file*.fasta.tab'.format(read_length)))),
                dset=create_dataset(
                    h5_file,
                    name='bact_marinePatric/extract_bact_{}/kmers/kmer_file1'.format(read_length),
                    shape=(training_line_count, column_count)),
                chunksize=5000)

        for read_length in (100, 200, 500, 1000, 5000, 10000):
            read_tsv_write_h5_group(
                tsv_fp_list=sorted(glob.glob(os.path.join(data_source_dir, 'eval_set/vir_marinzPatric/extract_vir_{}/kmers/kmer_file*.fasta.tab'.format(read_length)))),
                dset=create_dataset(
                    h5_file,
                    name='vir_marinePatric/extract_vir_{}/kmers/kmer_file1'.format(read_length),
                    shape=(training_line_count, column_count)),
                chunksize=5000)

        print('finished writing {} in {:5.2f}s'.format(h5_fp, time.time() - t0))
        print('  file size is {:5.2f}MB'.format(os.path.getsize(h5_fp) / 1e6))


def read_chunk(f_, shape):
    chunk_ = np.zeros(shape)
    chunk_i = 0
    for line in f_:
        chunk_[chunk_i, :] = [float(f) for f in line.rstrip().split('\t')[1:-1]]
        chunk_i += 1
        if chunk_i == shape[0]:
            # end of a chunk!
            yield chunk_
            chunk_i = 0

    if chunk_i > 0:
        # yield a partial chunk
        yield chunk_[:chunk_i, :]


def read_tsv_write_h5_group(tsv_fp_list, dset, chunksize):
    t0 = time.time()

    si = 0
    sj = 0
    for i, fp in enumerate(tsv_fp_list):
        print('reading "{}"'.format(fp))
        with open(fp, 'rt') as f:
            print('reading "{}" with size {:5.2f}MB'.format(fp, os.path.getsize(fp) / 1e6))
            header_line = f.readline()
            print('  header : "{}"'.format(header_line[:30]))
            column_count = len(header_line.strip().split('\t'))
            print('  header has {} columns'.format(column_count))

            t00 = time.time()
            for i, chunk in enumerate(read_chunk(f, shape=(chunksize, dset.shape[1]))):
                t11 = time.time()
                sj = si + chunk.shape[0]
                print('read chunk {} with shape {} in {:5.2f}s ({} rows total)'.format(
                    i, chunk.shape, t11 - t00, sj))
                dset[si:sj, :] = chunk
                si = sj
                t00 = time.time()
                print('  wrote chunk in {:5.2f}s'.format(t00 - t11))

                if sj >= dset.shape[0]:
                    print('dataset {} is full'.format(dset.name))
                    break

            if sj >= dset.shape[0]:
                print('dataset {} is full'.format(dset.name))
                break

    print('read {} rows'.format(si))
    print('dataset "{}" has shape {}'.format(dset.name, dset.shape))
    if sj < dset.shape[0]:
        new_shape = (sj, dset.shape[1])
        print('resizing dataset from {} to {}'.format(dset.shape, new_shape))
        dset.resize(new_shape)


t0 = time.time()
write_all_training_and_testing_data()
print('wrote all training and testing data in {:5.2f}s to {}'.format(time.time()-t0, data_target_fp))
