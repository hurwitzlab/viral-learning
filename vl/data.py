from itertools import zip_longest
import numpy as np
import sklearn.utils


def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def load_kmer_random_batches_h5(name, bacteria_dset, virus_dset, bacteria_subsample, virus_subsample, half_batch_size,
        shuffle_batch=True, yield_state=False):
    """
    Return batches of input and labels from a combined H5 file.

    The returned data is shuffled. This is very important. If the
    batches are returned with the first half bacteria data and
    the second half virus data the models train 'almost perfectly'
    and evaluate 'perfectly'.

    Arguments:
        training_testing_fp: file path of combined training and testing data
        bacteria_dset:       H5 dataset bacteria training data
        virus_dset:          H5 dataset virus training data
        half_batch_size:     each batch will have half_batch_size bacteria samples and half_batch_size virus samples

    Yield:
        (batch, labels) tuple of (half_batch_size*2, features) and (half_batch_size*2, 1) labels
    """
    print('reading bacteria dataset "{}" with shape {}'.format(bacteria_dset.name, bacteria_dset.shape))
    print('reading virus dataset "{}" with shape {}'.format(virus_dset.name, virus_dset.shape))

    batch_size = half_batch_size * 2
    batch_count = min(len(bacteria_subsample) // half_batch_size, len(virus_subsample) // half_batch_size)
    print('{} batches of {} samples will be yielded in each epoch'.format(batch_count, batch_size))

    if shuffle_batch:
        print('batches and labels will be shuffled')
    else:
        print('batches and labels will NOT be shuffled')

    # bacteria label is 0
    # virus label is 1
    labels = np.vstack((np.zeros((half_batch_size, 1)), np.ones((half_batch_size, 1))))

    # this is a never ending generator
    epoch = 0
    while True:
        epoch += 1
        bacteria_sample_groups = grouper(bacteria_subsample, n=half_batch_size)
        virus_sample_groups = grouper(virus_subsample, n=half_batch_size)

        step = 0
        # note that zip will terminate when it has depleted the shortest of the input iterators
        # this is the behavior we want since it happens that some virus testing sets are shorter
        # than their associated bacteria testing sets
        for bacteria_group, virus_group in zip(bacteria_sample_groups, virus_sample_groups):
            step += 1
            # H5 wants a list index to be in ascending order

            batch = np.vstack((
                bacteria_dset[sorted(bacteria_group), :],
                virus_dset[sorted(virus_group), :]))

            return_tuple = (batch, labels)
            if shuffle_batch:
                return_tuple = sklearn.utils.shuffle(*return_tuple)

            if yield_state:
                yield (*return_tuple, step, epoch)
            else:
                yield return_tuple

        print('generator "{}" epoch {} has ended'.format(name, epoch))


def load_kmer_range_batches_h5(name, bacteria_dset, virus_dset, bacteria_range, virus_range, half_batch_size,
        shuffle_batch=True, yield_state=False):
    """
    Return batches of input and labels from a combined H5 file.

    The returned data is shuffled. This is very important. If the
    batches are returned with the first half bacteria data and
    the second half virus data the models train 'almost perfectly'
    and evaluate 'perfectly'.

    Arguments:
        name                 a name for this generator to be used when printing
        bacteria_dset:       H5 dataset bacteria training data
        virus_dset:          H5 dataset virus training data
        bacteria_range:      (start, stop) indices for bacteria_dset
        virus_range:         (start, stop) indices for virus_dset
        half_batch_size:     each batch will have half_batch_size bacteria samples and half_batch_size virus samples
        shuffle_batch:       (True or False) shuffle the samples in each batch

    Yield:
        (batch, labels, step, epoch) tuple of (half_batch_size*2, features) and (half_batch_size*2, 1) labels
    """
    print('reading bacteria dataset "{}" with shape {}'.format(bacteria_dset.name, bacteria_dset.shape))
    print('reading virus dataset "{}" with shape {}'.format(virus_dset.name, virus_dset.shape))

    batch_size = half_batch_size * 2
    batch_count = min(
        (bacteria_range[1] - bacteria_range[0]) // half_batch_size,
        (virus_range[1] - virus_range[0]) // half_batch_size)
    print('{} batches of {} samples will be yielded in each epoch'.format(batch_count, batch_size))

    if shuffle_batch:
        print('batches and labels will be shuffled')
    else:
        print('batches and labels will NOT be shuffled')

    # bacteria label is 0
    # virus label is 1
    labels = np.vstack((np.zeros((half_batch_size, 1)), np.ones((half_batch_size, 1))))

    # this is a never ending generator
    epoch = 0
    while True:
        epoch += 1

        step = 0
        for bacteria_n, virus_n in zip(range(*bacteria_range, half_batch_size), range(*virus_range, half_batch_size)):
            step += 1

            bacteria_m = bacteria_n + half_batch_size
            virus_m = virus_n + half_batch_size

            batch = np.vstack((
                bacteria_dset[bacteria_n:bacteria_m, :],
                virus_dset[virus_n:virus_m, :]
            ))

            return_tuple = (batch, labels)
            if shuffle_batch:
                # yield shuffled views
                # the source arrays are not modified
                return_tuple = sklearn.utils.shuffle(*return_tuple)

            if yield_state:
                return_tuple = (*return_tuple, step, epoch)

            yield return_tuple

        print('generator "{}" epoch {} has ended'.format(name, epoch))
