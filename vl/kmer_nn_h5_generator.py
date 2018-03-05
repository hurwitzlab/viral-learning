
# coding: utf-8

# ## neural network trained on kmers using numpy
# Steps:
# 1. load data
# 2. find dimensions of the data
# 3. standardize the data?
# 4. build a model
# 5. train the model

# In[1]:
from collections import defaultdict
import itertools
import sys
import time

import h5py

import numpy as np

import sklearn.utils

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Lambda

# ### 1. Load Data

# In[2]:


def load_kmer_batches_h5(bacteria_kmer_fp, virus_kmer_fp, batch_size):
    """
    Return batches of input and labels from the specified files.

    The returned data is shuffled. This is very important. If the
    batches are returned with the first half bacteria data and
    the second half virus data the models train 'almost perfectly'
    and evaluate 'perfectly'.

    :param bacteria_kmer_fp:
    :param virus_kmer_fp:
    :param batch_size:
    :return:
    """

    with h5py.File(bacteria_kmer_fp, 'r') as bacteria_file, h5py.File(virus_kmer_fp, 'r') as virus_file:
        bacteria_dataset = bacteria_file['bacteria']
        virus_dataset = virus_file['virus']

        bacteria_batch = np.zeros((batch_size, bacteria_dataset.shape[1]))
        virus_batch = np.zeros((batch_size, virus_dataset.shape[1]))
        print('kmer batch shape is {}'.format((bacteria_batch.shape[0] * 2, bacteria_batch.shape[1])))

        # bacteria label is 0
        # virus label is 1
        labels = np.vstack((np.zeros((batch_size, 1)), np.ones((batch_size, 1))))

        for n in range(0, bacteria_dataset.shape[0], batch_size):
            source_slice = np.s_[n:n + batch_size, :]
            bacteria_dataset.read_direct(bacteria_batch, source_sel=source_slice)
            virus_dataset.read_direct(virus_batch, source_sel=source_slice)
            batch = np.vstack((bacteria_batch, virus_batch))
            # yeild shuffled views
            # the source arrays are not modified
            yield sklearn.utils.shuffle(batch, labels)


# In[3]:


def load_kmer_batches_h5_shuffle_labels(bacteria_kmer_fp, virus_kmer_fp, batch_size):
    for batch_df, labels in load_kmer_batches_h5(bacteria_kmer_fp, virus_kmer_fp, batch_size):
        shuffled_labels = sklearn.utils.shuffle(labels)
        yield batch_df, shuffled_labels


def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)


def load_kmer_batches_combined_h5(training_testing_fp, bacteria_dset_name, virus_dset_name, half_batch_size):
    """
    Return batches of input and labels from a combined H5 file.

    The returned data is shuffled. This is very important. If the
    batches are returned with the first half bacteria data and
    the second half virus data the models train 'almost perfectly'
    and evaluate 'perfectly'.

    Arguments:
        training_testing_fp: file path of combined training and testing data
        bacteria_dset_name:  H5 dataset name of bacteria training data
        virus_dset_name:     H5 dataset name of virus training data
        half_batch_size:     each batch will have half_batch_size bacteria samples and half_batch_size virus samples

    Yield:
        (batch, labels) tuple of (half_batch_size*2, features) and (half_batch_size*2, 1) labels
    """

    with h5py.File(training_testing_fp, 'r') as training_testing_file:
        bacteria_dset = training_testing_file[bacteria_dset_name]
        print('reading bacteria dataset "{}" with shape {}'.format(bacteria_dset_name, bacteria_dset.shape))

        virus_dset = training_testing_file[virus_dset_name]
        print('reading virus dataset "{}" with shape {}'.format(virus_dset_name, virus_dset.shape))

        # bacteria label is 0
        # virus label is 1
        labels = np.vstack((np.zeros((half_batch_size, 1)), np.ones((half_batch_size, 1))))

        # create separate permutations of bacteria and virus sample indices
        bacteria_sample_groups = grouper(np.random.permutation(bacteria_dset.shape[0]), n=half_batch_size)
        virus_sample_groups = grouper(np.random.permutation(virus_dset.shape[0]), n=half_batch_size)

        # note that zip will terminate when it has depleted the shortest of the input iterators
        # this is the behavior we want since it happens that some virus testing sets are shorter
        # than their associated bacteria testing sets
        for bacteria_group, virus_group in zip(bacteria_sample_groups, virus_sample_groups):
            # H5 wants a list index to be in ascending order

            batch = np.vstack((bacteria_dset[sorted(bacteria_group), :], virus_dset[sorted(virus_group), :]))
            yield sklearn.utils.shuffle(batch, labels)


def load_kmer_batches_combined_h5_shuffle_labels(training_testing_fp, bacteria_dset_name, virus_dset_name, half_batch_size):
    for batch, labels in load_kmer_batches_combined_h5(training_testing_fp, bacteria_dset_name, virus_dset_name, half_batch_size):
        shuffled_labels = sklearn.utils.shuffle(labels)
        yield batch, shuffled_labels


def build_model(input_dim, mean=None, variance=None):
    model = Sequential()
    #model.add(Dense(1, activation='sigmoid', input_dim=batch_feature_count))
    #if mean is not None and variance is not None:
    #    model.add(Lambda(lambda x: (x - mean) / variance, input_shape=(input_dim,)))
    model.add(Dense(64, activation='relu', input_dim=input_dim))
    #model.add(Dropout(rate=0.5))
    #model.add(Dense(32, activation='relu'))
    #model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def train_and_test_model(model, training_generator, testing_generators):
    """

    :param model:
    :param training_generator:
    :param testing_generators:
    :return:
    """

    training_history = defaultdict(list)

    for X, y in training_generator:
        metrics = model.train_on_batch(X, y)
        for metric_name, metric_value in zip(model.metrics_names, metrics):
            training_history[metric_name].append(metric_value)


def main():

    run_sanity_model = False
    run_model = True

    training_testing_fp = sys.argv[1]  #'../data/bact_kmer_file1.fasta.tab.gz'

    bacteria_kmer_training_dset_name = sys.argv[2]
    virus_kmer_training_dset_name = sys.argv[3]


    # In[6]:


    for batch, labels in load_kmer_batches_combined_h5(training_testing_fp, bacteria_kmer_training_dset_name, virus_kmer_training_dset_name, 10):
        print(batch[:5, :5])
        print(labels[:5])
        break


    # ### Find the dimensions of the data

    # In[7]:


    batch_feature_count = batch.shape[1]
    batch_sample_count = batch.shape[0]

    print('batch feature count : {}'.format(batch_feature_count))
    print('batch sample count  : {}'.format(batch_sample_count))

    with h5py.File(training_testing_fp) as training_testing_file:
        mean_dset = training_testing_file['/clean-bact-vir/training1/extract/kmers/kmer_file1/mean']
        variance_dset = training_testing_file['/clean-bact-vir/training1/extract/kmers/kmer_file1/variance']

        mean = np.zeros(mean_dset.shape)
        variance = np.zeros(variance_dset.shape)

        mean_dset.read_direct(mean)
        variance_dset.read_direct(variance)
        variance[variance == 0.0] = 1.0


    # the sanity model
    sanity_model = build_model(input_dim=batch_feature_count, mean=mean, variance=variance)

    model = build_model(input_dim=batch_feature_count, mean=mean, variance=variance)

    # ### 5. Train the Model

    # train
    epochs = int(sys.argv[4])
    steps = int(sys.argv[5])
    half_batch = 32

    # train with shuffled labels as sanity check
    # 500 * 64 = 32000 training samples per epoch
    if run_sanity_model:
        sanity_model.fit_generator(
            generator=load_kmer_batches_combined_h5_shuffle_labels(
                training_testing_fp,
                bacteria_kmer_training_dset_name,
                virus_kmer_training_dset_name,
                half_batch),
            epochs=2,
            steps_per_epoch=500)

        sanity_model_performance = sanity_model.evaluate_generator(
            generator=load_kmer_batches_combined_h5(
                training_testing_fp,
                '/bact_marinePatric/extract_bact_1000/kmers/kmer_file1',
                '/vir_marinePatric/extract_vir_1000/kmers/kmer_file1',
                1),
            steps=5000)

        print('sanity-check model performance:')
        for metric_name, metric_value in zip(sanity_model.metrics_names, sanity_model_performance):
            print('{}: {:5.2f}'.format(metric_name, metric_value))


    if run_model:

        training_metrics = defaultdict(list)
        testing_metrics = defaultdict(list)

        generator = load_kmer_batches_combined_h5(
            training_testing_fp,
            bacteria_kmer_training_dset_name,
            virus_kmer_training_dset_name,
            half_batch)


        for epoch in range(1, epochs+1):
            t0 = time.time()
            training_history = model.fit_generator(
                generator=generator,
                epochs=1,
                steps_per_epoch=steps)
            print('training epoch {} done in {:5.2f}s'.format(epoch, time.time()-t0))

            print(training_history.history)

            # test at the end of each epoch
            t0 = time.time()
            for read_length in (100, 200, 500, 1000, 5000, 10000):
                testing_bacteria_dset_name = '/bact_marinePatric/extract_bact_{}/kmers/kmer_file1'.format(read_length)
                testing_virus_dset_name = '/vir_marinePatric/extract_vir_{}/kmers/kmer_file1'.format(read_length)

                with h5py.File(training_testing_fp) as training_testing_file:
                    testing_steps = np.min(
                        (
                            training_testing_file[testing_bacteria_dset_name].shape[0],
                            training_testing_file[testing_virus_dset_name].shape[0]
                        )
                    )
                    print('  {} testing steps'.format(testing_steps))

                testing_history = model.evaluate_generator(
                    generator=load_kmer_batches_combined_h5(
                        training_testing_fp,
                        testing_bacteria_dset_name,
                        testing_virus_dset_name,
                        1),
                    steps=testing_steps)

                print(testing_history)

                print('test epoch {} done in {:5.2f}s'.format(epoch, time.time()-t0))
                for metric_name, metric_value in zip(model.metrics_names, testing_history):
                    print('{}: {:5.2f}'.format(metric_name, metric_value))
                    testing_metrics[metric_name].append(metric_value)


if __name__ == '__main__':
    main()