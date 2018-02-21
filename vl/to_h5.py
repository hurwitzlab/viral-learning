import gzip
import sys
import time

import h5py
import numpy as np


input_fp = sys.argv[1]  # '../data/bact_kmer_file1.fasta.tab.gz'
h5_fp = sys.argv[2]  #'../data/bact_kmer_file1.h5'
dset_name = sys.argv[3]
row_count = int(sys.argv[4])


def read_tsv_write_h5(tsv_fp, h5_fp, dset_name, line_count):
    t0 = time.time()
    with h5py.File(h5_fp, 'w') as h5_file, gzip.open(tsv_fp, 'rt') as input_file:
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
read_tsv_write_h5(input_fp, h5_fp, dset_name, row_count)
print('wrote {} rows to {} in {:5.2f}s'.format(row_count, h5_fp, time.time()-t0))
