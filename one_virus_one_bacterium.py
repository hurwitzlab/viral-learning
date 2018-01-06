
# coding: utf-8

# In[1]:


import gzip
from io import StringIO

import numpy as np
from numpy import random

from keras.models import Sequential
from keras.layers import Dense, Activation, Conv1D, Dropout, Embedding, GlobalMaxPooling1D
from keras.preprocessing.text import Tokenizer


# In[3]:


# read a viral genome mycobacterium_phage_shipwreck
with open('genome_virus.fasta', 'rt') as viral_genome_file:
    viral_genome_header = viral_genome_file.readline()
    viral_genome_buffer = StringIO()
    for line in viral_genome_file.readlines():
        viral_genome_buffer.write(line.strip())
viral_genome = viral_genome_buffer.getvalue()


# In[4]:


viral_genome_header


# In[5]:


print('viral genome has {} bp'.format(len(viral_genome)))
viral_genome[:100]


# In[6]:


def generate_reads(genome, read_count, read_length):
    """Yield 'read_count' random reads of length 'read_length' from 'genome'.
    """
    for n in random.randint(low=0, high=len(genome)-read_length+1, size=read_count):
        yield genome[n:n+read_length]


# In[7]:


# a small example
for r in generate_reads('ACGTAC', read_count=5, read_length=5):
    print(r)


# In[8]:


# configure a Tokenizer to convert A,C,G,T to 1,2,3,4
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts('ACGT')
print(tokenizer.word_counts)
print(tokenizer.document_count)
print(tokenizer.word_docs)


# In[9]:


# use these parameters for building the model and layers
mini_batch_size = 10
input_vector_length = 100
input_vector_count = 10000


# In[10]:


def generate_training_data(label, genome, read_count, read_length):
    """
    """
    genome_reads = list(generate_reads(
        genome, read_count=read_count, read_length=read_length))
    #print(np.asarray(viral_reads)[:5])
    genome_sequences = tokenizer.texts_to_sequences(texts=genome_reads)
    print(np.asarray(genome_sequences)[:5])
    genome_sequence_array = np.asarray(genome_sequences)
    print(genome_sequence_array)
    genome_sequence_label_array = label * np.ones((read_count,1))

    return genome_sequence_array, genome_sequence_label_array


# In[11]:


viral_training_data, viral_training_labels = generate_training_data(
    1.0, genome=viral_genome, read_count=input_vector_count, read_length=input_vector_length)
print('viral training data shape        : {}'.format(viral_training_data.shape))
print('viral training data labels shape : {}'.format(viral_training_labels.shape))
viral_validation_data, viral_validation_labels = generate_training_data(
    1.0, genome=viral_genome, read_count=input_vector_count, read_length=input_vector_length)
print('viral validation data shape        : {}'.format(viral_validation_data.shape))
print('viral validation data labels shape : {}'.format(viral_validation_labels.shape))


# Read bacterial genome and create training and testing data.

# In[12]:


# GCF_000195955.2_ASM19595v2_genomic.fna.gz
with gzip.open('genome_bacterium.fna.gz', 'rt') as bacterial_genome_file:
    bacterial_genome_header = bacterial_genome_file.readline()
    print(bacterial_genome_header)
    bacterial_genome_buffer = StringIO()
    for i, line in enumerate(bacterial_genome_file.readlines()):
        if i > 1000:
            break
        else:
            bacterial_genome_buffer.write(line.strip())
bacterial_genome = bacterial_genome_buffer.getvalue()
print(bacterial_genome[:100])


# In[13]:


bacterial_training_data, bacterial_training_labels = generate_training_data(
    0.0, genome=bacterial_genome, read_count=input_vector_count, read_length=input_vector_length)
print('bacterial training data shape        : {}'.format(bacterial_training_data.shape))
print('bacterial training data labels shape : {}'.format(bacterial_training_labels.shape))
bacterial_validation_data, bacterial_validation_labels = generate_training_data(
    0.0, genome=bacterial_genome, read_count=input_vector_count, read_length=input_vector_length)
print('bacterial validation data shape        : {}'.format(bacterial_validation_data.shape))
print('bacterial validation data labels shape : {}'.format(bacterial_validation_labels.shape))


# In[14]:


max_features = 4+1
maxlen = input_vector_length
batch_size = mini_batch_size
embedding_dims = 50
filters = 250
kernel_size = 3
hidden_dims = 250
epochs = 2

model = Sequential()

# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
model.add(Embedding(max_features,
                    embedding_dims,
                    input_length=maxlen))
model.add(Dropout(0.2))

# we add a Convolution1D, which will learn filters
# word group filters of size filter_length:
model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))
# we use max pooling:
model.add(GlobalMaxPooling1D())

# We add a vanilla hidden layer:
model.add(Dense(hidden_dims))
model.add(Dropout(0.2))
model.add(Activation('relu'))

# We project onto a single unit output layer, and squash it with a sigmoid:
model.add(Dense(1))
model.add(Activation('sigmoid'))


# In[15]:


# For a binary classification problem
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# In[16]:


training_data = np.vstack((viral_training_data, bacterial_training_data))
print(training_data.shape)
training_labels = np.vstack((viral_training_labels, bacterial_training_labels))
print(training_labels.shape)

validation_data = np.vstack((viral_validation_data, bacterial_validation_data))
print(validation_data.shape)
validation_labels = np.vstack((viral_validation_labels, bacterial_validation_labels))
print(validation_labels.shape)


# In[ ]:


model.fit(training_data, training_labels,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(validation_data, validation_labels))

