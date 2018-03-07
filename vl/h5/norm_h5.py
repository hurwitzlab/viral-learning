import os.path
import sys
import time

import h5py

import numpy as np


def calculate_mean_variance(dsets):
    """
    Given a list of datasets calculate the mean and variance for all rows in all datasets.
    
    Arguments:
        dsets: sequence of datasets with matching column counts
        
    Returns:
        (mean, variance): tuple of mean vector and variance vector
    """

    print('calculating mean and variance for "{}"'.format([dset.name for dset in dsets]))
    t0 = time.time()
    
    mean = np.zeros((1, dsets[0].shape[1]))
    M2 = np.zeros((1, dsets[0].shape[1]))
    count = 0
    
    for dset in dsets:
        # find the right subset size to load without running out of memory
        # if dset has more than 10,000 rows use 10,000
        # if dset has fewer than 10,000 rows load the whole dset
        dsubset = np.zeros((min(10000, dset.shape[0]), dset.shape[1]))
        print('  working on "{}"'.format(dset.name))
        for n in range(0, dset.shape[0], dsubset.shape[0]):
            m = min(n + dsubset.shape[0], dset.shape[0])
            dset.read_direct(dsubset, source_sel=np.s_[n:m, :])

            t00 = time.time()
            for i in range(0, dsubset.shape[0]):
                count = count + 1 
                delta = dsubset[i, :] - mean
                mean += delta / count
                delta2 = dsubset[i, :] - mean
                M2 += delta * delta2
            print('    processed slice [{}:{}] {:5.2f}s'.format(n, m, time.time()-t00))

    print('  finished mean and variance in {:5.2f}s'.format(time.time()-t0))
    # return mean, variance
    return (mean, M2/(count - 1))


def normalize_datasets(input_h5_fp, norm_h5_fp):
    dset_paths = []
    def find_data(name, obj):
        if hasattr(obj, 'dtype'):
            print('found dataset "{}"'.format(name))
            dset_paths.append(obj.name)
        else:
            pass

    with h5py.File(input_h5_fp,  'r', libver='latest', swmr=True) as input_h5_file:
        input_h5_file.visititems(find_data)

        mean, variance = calculate_mean_variance((
            input_h5_file['/clean-bact/training1/extract/kmers'],
            input_h5_file['/clean-vir/training1/extract/kmers']))

        zero_mean_column_count = len(mean[mean == 0.0])
        print('{} column(s) have zero mean'.format(zero_mean_column_count))
        zero_var_column_count = len(variance[variance == 0.0])
        print('{} column(s) have zero variance'.format(zero_var_column_count))
        
        with h5py.File(norm_h5_fp, 'w') as norm_h5_file:
            print('writing normalized data to "{}"'.format(norm_h5_fp))
            
            mean_dset = norm_h5_file.require_dataset(
                name='/mean',
                shape=mean.shape,
                dtype=mean.dtype,
                chunks=mean.shape,
                compression='gzip')
            mean_dset[:, :] = mean
            
            variance_dset = norm_h5_file.require_dataset(
                name='/variance',
                shape=variance.shape,
                dtype=variance.dtype,
                chunks=variance.shape,
                compression='gzip')

            variance_dset[:, :] = variance
            
            for dset_path in dset_paths:
                dset = input_h5_file[dset_path]
                print('  normalizing "{}"'.format(dset.name))
                
                normalized_dset = norm_h5_file.require_dataset(
                    name=dset.name,
                    shape=dset.shape,
                    dtype=dset.dtype,
                    chunks=mean.shape,
                    compression='gzip',
                    compression_opts=9,
                    shuffle=True)
                
                t0 = time.time()
                n = 10000
                for i in range(0, dset.shape[0], n):
                    j = i + n
                    # maintain sparsity
                    #normalized_dset[i:j, :] = (dset[i:j, :] - mean) / variance
                    t00 = time.time()
                    normalized_dset[i:j, :] = dset[i:j, :] / variance
                    print('  normalized slice {}:{} in {:5.2f}s'.format(i, j, time.time()-t00))
                print('normalized "{}" in {:5.2f}s'.format(dset.name, time.time()-t0))


def main():
    input_h5_fp = sys.argv[1]  # '../data/training_testing.h5'
    print(input_h5_fp)

    with h5py.File(input_h5_fp, 'r') as input_h5_file:
        print(list(input_h5_file['/clean-bact/training1/extract'].items()))


    input_h5_dp, input_h5_name = os.path.split(input_h5_fp)
    norm_h5_fp = os.path.join(input_h5_dp, 'norm_' + input_h5_name)
    normalize_datasets(input_h5_fp, norm_h5_fp)


if __name__ == '__main__':
    main()