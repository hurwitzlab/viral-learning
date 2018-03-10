from keras.callbacks import Callback
from keras.layers import Dense, Lambda
from keras.models import Sequential

import h5py
import matplotlib.pyplot as plt
import pandas as pd

from vl.data import load_kmer_range_batches_h5


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.history = {
            'acc': [],
            'loss': [],
            'val_acc': [],
            'val_loss': []
        }

    def on_batch_end(self, batch, logs={}):
        self.history['acc'].append(logs.get('acc'))
        self.history['loss'].append(logs.get('loss'))

    def on_epoch_end(self, epoch, logs=None):
        self.history['val_acc'].append(logs.get('val_acc'))
        self.history['val_loss'].append(logs.get('val_loss'))



def build_model(input_dim):
    model = Sequential()

    model.add(Dense(64, activation='relu', input_dim=input_dim))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def plot_training_performance(training_performance_df):
    plt.figure()
    plt.plot(training_performance_df.index, training_performance_df.loss, training_performance_df.val_loss)
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['training', 'validation'])

    plt.figure()
    plt.plot(training_performance_df.index, training_performance_df.acc, training_performance_df.val_acc)
    plt.title('Training and Validation Accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['training', 'validation'])


def main():
    # we have 1,000,000 training samples
    # training    : 8/10 * 1,000,000 = 800,000
    # development : 1/10 * 1,000,000 = 100,000
    # test        : 1/10 * 1,000,000 = 100,000

    training_samples = 800000
    validation_samples = 100000

    training_end = 800000 // 2
    validation_end = 900000 // 2
    test_end = 1000000 // 2

    train_test_fp = '../data/perm_training_testing.h5'
    batch_size = 100

    with h5py.File(train_test_fp, 'r', libver='latest', swmr=True) as train_test_file:
        bacteria_dset = train_test_file['/clean-bact/training1/extract/kmers']
        virus_dset = train_test_file['/clean-vir/training1/extract/kmers']

        model = build_model(input_dim=bacteria_dset.shape[1])

        training_batches_per_epoch = training_samples // batch_size
        development_batch_count = validation_samples // batch_size
        print('batch size is {}'.format(batch_size))
        print('{} training samples'.format(training_samples))
        print('{} batches of training data'.format(training_batches_per_epoch))
        print('{} validation samples'.format(validation_samples))
        print('{} batches of validation data'.format(development_batch_count))

        epochs = 2
        print('{} epochs'.format(epochs))

        history = LossHistory()
        model.fit_generator(
            generator=load_kmer_range_batches_h5(
                name='training',
                bacteria_dset=bacteria_dset,
                bacteria_range=(0, training_end),
                virus_dset=virus_dset,
                virus_range=(0, training_end),
                half_batch_size=batch_size // 2
            ),
            validation_data=load_kmer_range_batches_h5(
                name='validation',
                bacteria_dset=bacteria_dset,
                bacteria_range=(training_end, validation_end),
                virus_dset=virus_dset,
                virus_range=(training_end, validation_end),
                half_batch_size=batch_size // 2
            ),
            epochs=epochs,
            steps_per_epoch=(training_samples // batch_size),
            validation_steps=(validation_samples // batch_size),
            workers=2,
            callbacks=[history]
        )

        training_performance_df = pd.DataFrame(data=history.history, index=range(1, 1 + epochs * (training_samples // batch_size)))
        training_performance_df.index.name = 'epoch'
        plot_training_performance(training_performance_df)


if __name__ == '__main__':
    main()