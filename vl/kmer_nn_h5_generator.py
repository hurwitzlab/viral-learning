
# coding: utf-8

# ## neural network trained on kmers using numpy
# Steps:
# 1. load data
# 2. find dimensions of the data
# 3. standardize the data?
# 4. build a model
# 5. train the model

# In[1]:
import sys
import time

import h5py

import numpy as np

import sklearn.utils

from keras.models import Sequential
from keras.layers import Dense

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


# In[4]:


bacteria_kmer_file1_fp = sys.argv[1]  #'../data/bact_kmer_file1.fasta.tab.gz'
bacteria_kmer_file2_fp = sys.argv[2]  #'../data/bact_kmer_file2.fasta.tab.gz'


# In[5]:


virus_kmer_file1_fp = sys.argv[3]  #'../data/vir_kmer_file1.fasta.tab.gz'
virus_kmer_file2_fp = sys.argv[4]  #'../data/vir_kmer_file2.fasta.tab.gz'


# In[6]:


for batch, labels in load_kmer_batches_h5(bacteria_kmer_file1_fp, virus_kmer_file1_fp, 10):
    print(batch[:5, :5])
    print(labels[:5])
    break


# ### Find the dimensions of the data

# In[7]:


batch_feature_count = batch.shape[1]
batch_sample_count = batch.shape[0]

print('batch feature count : {}'.format(batch_feature_count))
print('batch sample count  : {}'.format(batch_sample_count))


# ### 4. Build a Model
# A single hidden layer of 8 or 16 nodes gives 0.8 test accuracy on 1600/1600
# (100 steps) training samples in 2 epochs. Training takes about 15 minutes
# per epoch.

# In[8]:


sanity_model = Sequential()
sanity_model.add(Dense(64, activation='relu', input_dim=batch_feature_count))
sanity_model.add(Dense(32, activation='relu'))
sanity_model.add(Dense(16, activation='relu'))
sanity_model.add(Dense(1, activation='sigmoid'))

sanity_model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model = Sequential()
model.add(Dense(64, activation='relu', input_dim=batch_feature_count))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# ### 5. Train the Model

# train
epochs = int(sys.argv[5])
steps = int(sys.argv[6])
half_batch = 16

# train with shuffled labels as sanity check
sanity_model.fit_generator(
    generator=load_kmer_batches_h5_shuffle_labels(bacteria_kmer_file1_fp, virus_kmer_file1_fp, half_batch),
    epochs=epochs,
    steps_per_epoch=steps)

sanity_model_performance = sanity_model.evaluate_generator(
    generator=load_kmer_batches_h5(bacteria_kmer_file2_fp, virus_kmer_file2_fp, 1),
    steps=4900)

print('sanity-check model performance:')
for metric_name, metric_value in zip(sanity_model.metrics_names, sanity_model_performance):
    print('{}: {:5.2f}'.format(metric_name, metric_value))

generator = load_kmer_batches_h5(
    bacteria_kmer_file1_fp,
    virus_kmer_file1_fp,
    half_batch)
for epoch in range(1, epochs+1):
    t0 = time.time()
    model.fit_generator(
        generator=generator,
        epochs=1,
        steps_per_epoch=steps)
    print('training epoch {} done in {:5.2f}s'.format(epoch, time.time()-t0))

    # test
    t0 = time.time()
    model_performance = model.evaluate_generator(
        generator=load_kmer_batches_h5(
            bacteria_kmer_file2_fp,
            virus_kmer_file2_fp,
            1),
        steps=4900)

    print('test epoch {} done in {:5.2f}s'.format(epoch, time.time()-t0))
    for metric_name, metric_value in zip(model.metrics_names, model_performance):
        print('{}: {:5.2f}'.format(metric_name, metric_value))
