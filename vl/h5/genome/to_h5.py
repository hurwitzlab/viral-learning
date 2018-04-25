"""
Read Alise's training and testing data files and smash them all into one H5 file.

This script is hard-coded to run on ocelote.
"""
import itertools
import os
import time

import h5py
import numpy as np


# 500, 1000, 5000pb
# Phage, Proc
# 4, 6, 8 mers
# kmer_file1.fasta.tab, kmer_file2.fasta.tab, ..., kmer_file10.fasta.tab
data_file_path_tmpl = '/rsgrps/bhurwitz/alise/my_data/Riveal_exp/Models/RefSeq_based_models/Prok_Phages_models/size_{}pb/training_set/{}_trainingset/extract_kmers/kmers{}/kmer_file{}.fasta.tab'

output_file_path_tmpl = '/extra/jklynch/viral-learning/riveal_refseq_prok_phage_{}pb_kmers{}.h5'


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
    for pb, k in itertools.product((500, 1000, 5000), (4, 6, 8)):
        t0 = time.time()

        h5_fp = output_file_path_tmpl.format(pb, k)
        print('writing file {}'.format(h5_fp))
        with h5py.File(h5_fp, 'w') as h5_file:

            # read the first line of one file to get the number of columns
            # header lines look like this:
            #   seq_id  AAAA    AAAT    AAAG    AAAC    AATA    AATT ... GCCG    GCCC    read_type
            with open(data_file_path_tmpl.format(pb, 'Phage', k, 1)) as data_file:
                headers = data_file.readline().strip().split('\t')
                kmer_count = len(headers) - 2

            for organism in ('Phage', 'Proc'):
                # dataset names look like /Phage/500pb/kmers4
                dset_name = '/{}/{}pb/kmers{}'.format(organism, pb, k)
                dset = create_dataset(h5_file, dset_name, shape=(10 * 10000, kmer_count))

                file_list = [data_file_path_tmpl.format(pb, organism, k, n+1) for n in range(10)]
                read_tsv_write_h5_group(file_list, dset, chunksize=1000)
            print('finished writing {} in {:5.2f}s'.format(h5_fp, time.time() - t0))
            print('  file size is {:5.2f}MB'.format(os.path.getsize(h5_fp) / 1e6))


def read_chunk(f_, shape):
    seq_ids = []
    chunk_ = np.zeros(shape)
    chunk_i = 0
    for line in f_:
        row_elements = line.rstrip().split('\t')
        seq_ids.append(row_elements[0])
        chunk_[chunk_i, :] = [float(f) for f in row_elements[1:-1]]
        chunk_i += 1
        if chunk_i == shape[0]:
            # end of a chunk!
            yield seq_ids, chunk_
            chunk_i = 0
            seq_ids = []

    if chunk_i > 0:
        # yield a partial chunk
        yield seq_ids, chunk_[:chunk_i, :]


def read_tsv_write_h5_group(tsv_fp_list, dset, chunksize):
    t0 = time.time()

    si = 0
    sj = 0
    for i, fp in enumerate(tsv_fp_list):
        if not os.path.exists(fp):
            print('WARNING: file "{}" does not exist'.format(fp))
        else:
            with open(fp, 'rt') as f:
                print('reading "{}" with size {:5.2f}MB'.format(fp, os.path.getsize(fp) / 1e6))
                header_line = f.readline()
                print('  header : "{}"'.format(header_line[:30]))
                column_count = len(header_line.strip().split('\t'))
                print('  header has {} columns'.format(column_count))

                t00 = time.time()
                for i, (seq_ids, chunk) in enumerate(read_chunk(f, shape=(chunksize, dset.shape[1]))):
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

    print('read {} rows in {:5.2f}s'.format(si, time.time()-t0))
    print('dataset "{}" has shape {}'.format(dset.name, dset.shape))
    if sj < dset.shape[0]:
        new_shape = (sj, dset.shape[1])
        print('resizing dataset from {} to {}'.format(dset.shape, new_shape))
        dset.resize(new_shape)


t0_ = time.time()
write_all_training_and_testing_data()
print('Done in {:5.2f}s.'.format(time.time()-t0_))
