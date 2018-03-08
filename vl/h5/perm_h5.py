import os.path
import sys
import time

import h5py

import numpy as np


def permute_datasets(input_h5_fp, perm_h5_fp):
    dset_paths = []
    def find_data(name, obj):
        if hasattr(obj, 'dtype'):
            print('found dataset "{}"'.format(name))
            dset_paths.append(obj.name)
        else:
            pass

    print('reading "{}"'.format(input_h5_fp))
    with h5py.File(input_h5_fp, 'r', libver='latest', swmr=True) as input_h5_file:
        print('writing permuted data to "{}"'.format(perm_h5_fp))
        with h5py.File(perm_h5_fp, 'w') as perm_h5_file:
            input_h5_file.visititems(find_data)

            for dset_path in dset_paths:
                dset = input_h5_file[dset_path]
                print('  permuting "{}"'.format(dset.name))

                permuted_dset = perm_h5_file.require_dataset(
                    name=dset.name,
                    shape=dset.shape,
                    dtype=dset.dtype,
                    chunks=(1, dset.shape[1]),
                    compression='gzip',
                    compression_opts=9,
                    shuffle=True)

                permuted_index = np.random.permutation(dset.shape[0])

                t0 = time.time()
                n = 10000
                for i in range(0, dset.shape[0], n):
                    j = min(i + n, dset.shape[0])
                    t00 = time.time()
                    permuted_dset[i:j, :] = dset[sorted(permuted_index[i:j]), :]
                    print('  permuted slice {}:{} in {:5.2f}s'.format(i, j, time.time()-t00))

                print('permuted dset {} in {:5.2f}s'.format(dset.name, time.time()-t0))


def main():
    input_h5_fp = sys.argv[1]  # '../data/training_testing.h5'
    print(input_h5_fp)

    with h5py.File(input_h5_fp, 'r') as input_h5_file:
        print(list(input_h5_file['/clean-bact/training1/extract'].items()))


    input_h5_dp, input_h5_name = os.path.split(input_h5_fp)
    perm_h5_fp = os.path.join(input_h5_dp, 'perm_' + input_h5_name)
    permute_datasets(input_h5_fp, perm_h5_fp)


if __name__ == '__main__':
    main()