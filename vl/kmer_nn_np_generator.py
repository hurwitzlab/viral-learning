
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

import numpy as np

import pandas as pd
import sklearn.utils

from keras.models import Sequential
from keras.layers import Dense

# this does not seem to help
#import tensorflow as tf
#sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=2))
#from keras import backend as K
#K.set_session(sess)

# ### 1. Load Data

# In[2]:


def load_kmer_batches(bacteria_kmer_fp, virus_kmer_fp, batch_size):
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

    def not_read_type(column_name):
        """
        Return True if the column name is NOT 'read_type'.
        """
        return column_name != 'read_type'

    bacteria_kmer_iter = pd.read_table(
        filepath_or_buffer=bacteria_kmer_fp,
        index_col=0,
        usecols=not_read_type,
        engine='c',
        chunksize=batch_size)

    virus_kmer_iter = pd.read_table(
        filepath_or_buffer=virus_kmer_fp,
        index_col=0,
        usecols=not_read_type,
        engine='c',
        chunksize=batch_size)
    
    labels = np.vstack((np.zeros((batch_size, 1)), np.ones((batch_size, 1))))

    for bacteria_batch, virus_batch in zip(bacteria_kmer_iter, virus_kmer_iter):
        batch_df = pd.concat((bacteria_batch, virus_batch))
        yield sklearn.utils.shuffle(batch_df, labels)


# In[3]:


def load_kmer_batches_shuffle_labels(bacteria_kmer_fp, virus_kmer_fp, batch_size):
    for batch_df, labels in load_kmer_batches(bacteria_kmer_fp, virus_kmer_fp, batch_size):
        shuffled_labels = sklearn.utils.shuffle(labels)
        yield batch_df, shuffled_labels


# In[4]:


bacteria_kmer_file1_fp = sys.argv[1]  #'../data/bact_kmer_file1.fasta.tab.gz'
bacteria_kmer_file2_fp = sys.argv[2]  #'../data/bact_kmer_file2.fasta.tab.gz'


# In[5]:


virus_kmer_file1_fp = sys.argv[3]  #'../data/vir_kmer_file1.fasta.tab.gz'
virus_kmer_file2_fp = sys.argv[4]  #'../data/vir_kmer_file2.fasta.tab.gz'


# In[6]:


for batch, labels in load_kmer_batches(bacteria_kmer_file1_fp, virus_kmer_file1_fp, 10):
    print(batch.head())
    break


# ### Find the dimensions of the data

# In[7]:


batch_feature_count = batch.shape[1]
batch_sample_count = batch.shape[0]

print('batch feature count : {}'.format(batch_feature_count))
print('batch sample count  : {}'.format(batch_sample_count))


# ### 4. Build a Model

# In[8]:


sanity_model = Sequential()
sanity_model.add(Dense(16, activation='relu', input_dim=batch_feature_count))
sanity_model.add(Dense(1, activation='sigmoid'))

sanity_model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model = Sequential()
model.add(Dense(8, activation='relu', input_dim=batch_feature_count))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# ### 5. Train the Model


# train with shuffled labels as sanity check
sanity_model.fit_generator(
    generator=load_kmer_batches_shuffle_labels(bacteria_kmer_file1_fp, virus_kmer_file1_fp, 16),
    epochs=2,
    steps_per_epoch=10,
    workers=2)

sanity_model_performance = sanity_model.evaluate_generator(
    generator=load_kmer_batches(bacteria_kmer_file2_fp, virus_kmer_file2_fp, 16),
    steps=10,
    workers=2)

print('sanity-check model performance:')
for metric_name, metric_value in zip(sanity_model.metrics_names, sanity_model_performance):
    print('{}: {:5.2f}'.format(metric_name, metric_value))

# train
steps = int(sys.argv[5])

model.fit_generator(
    generator=load_kmer_batches(bacteria_kmer_file1_fp, virus_kmer_file1_fp, 16),
    epochs=2,
    steps_per_epoch=steps,
    workers=2)

# test
model_performance = model.evaluate_generator(
    generator=load_kmer_batches(bacteria_kmer_file2_fp, virus_kmer_file2_fp, 16),
    steps=steps,
    workers=2)

for metric_name, metric_value in zip(model.metrics_names, model_performance):
    print('{}: {:5.2f}'.format(metric_name, metric_value))

