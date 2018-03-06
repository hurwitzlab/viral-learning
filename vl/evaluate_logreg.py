import sys
import warnings
warnings.simplefilter('ignore', UserWarning)

from keras.models import Sequential
from keras.layers import Dense

import h5py
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
import pandas as pd

from vl.data import load_kmer_random_batches_h5, load_kmer_range_batches_h5


def build_model(input_dim):
    model = Sequential()
    model.add(Dense(1, activation='sigmoid', input_dim=input_dim))
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy'])
    return model


def plot_training_performance(training_performance_df, title, pdfpages):
    plt.figure()
    plt.plot(training_performance_df.index, training_performance_df.loss, training_performance_df.val_loss)
    plt.title('{}\nLogistic Regression\nTraining and Validation Loss'.format(title))
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['training', 'validation'])
    pdfpages.savefig()  # saves the current figure into a pdf page

    plt.figure()
    plt.plot(training_performance_df.index, training_performance_df.acc, training_performance_df.val_acc)
    plt.title('{}\nLogistic Regression\nTraining and Validation Accuracy'.format(title))
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['training', 'validation'])
    pdfpages.savefig()  # saves the current figure into a pdf page


def main():
    # go!
    #train_test_fp = '../data/short_training_testing.h5'
    train_test_fp = sys.argv[1]
    epochs = int(sys.argv[2])
    print('reading file "{}"'.format(train_test_fp))
    # epochs = 2
    print('{} epochs'.format(epochs))

    batch_size = 50

    with h5py.File(train_test_fp, 'r', libver='latest', swmr=True) as train_test_file, PdfPages('evaluate_logreg_plots.pdf') as pdf:
        bacteria_dset = train_test_file['/clean-bact/training1/extract/kmers/kmer_file1']
        virus_dset = train_test_file['/clean-vir/training1/extract/kmers/kmer_file1']

        model = build_model(input_dim=bacteria_dset.shape[1])

        # use 7/8 of the data for training, 1/8 for validation
        training_sample_count = (7 * bacteria_dset.shape[0] // 8) + (7 * virus_dset.shape[0] // 8)
        training_batches_per_epoch = training_sample_count // batch_size
        validation_sample_count = (bacteria_dset.shape[0] // 8) + (virus_dset.shape[0] // 8)
        validation_batch_count = validation_sample_count // batch_size
        print('batch size is {}'.format(batch_size))
        print('{} training samples'.format(training_sample_count))
        print('{} batches of training data'.format(training_batches_per_epoch))
        print('{} validation samples'.format(validation_sample_count))
        print('{} batches of validation data'.format(validation_batch_count))

        #
        # train and validate on un-shuffled data
        #
        history = model.fit_generator(
            generator=load_kmer_range_batches_h5(
                name='training',
                bacteria_dset=bacteria_dset,
                bacteria_range=(0, 7 * bacteria_dset.shape[0] // 8),
                virus_dset=virus_dset,
                virus_range=(0, 7 * virus_dset.shape[0] // 8),
                half_batch_size=batch_size // 2
            ),
            validation_data=load_kmer_range_batches_h5(
                name='validation',
                bacteria_dset=bacteria_dset,
                bacteria_range=(7 * bacteria_dset.shape[0] // 8, bacteria_dset.shape[0]),
                virus_dset=virus_dset,
                virus_range=(7 * virus_dset.shape[0] // 8, virus_dset.shape[0]),
                half_batch_size=batch_size // 2
            ),
            epochs=epochs,
            steps_per_epoch=100,
            validation_steps=validation_batch_count,
            workers=2
        )

        training_performance_df = pd.DataFrame(data=history.history, index=range(1, epochs + 1))
        training_performance_df.index.name = 'epoch'
        plot_training_performance(training_performance_df, title='Un-Shuffled Data', pdfpages=pdf)

        #
        # train and validate on shuffled data
        #

        bacteria_permuted_index = np.random.permutation(bacteria_dset.shape[0])
        bacteria_seven_eighths = 7 * bacteria_dset.shape[0] // 8
        bacteria_training_subsample = bacteria_permuted_index[:bacteria_seven_eighths]
        bacteria_validation_subsample = bacteria_permuted_index[bacteria_seven_eighths:]

        virus_permuted_index = np.random.permutation(virus_dset.shape[0])
        virus_seven_eighths = 7 * virus_dset.shape[0] // 8
        virus_training_subsample = virus_permuted_index[:virus_seven_eighths]
        virus_validation_subsample = virus_permuted_index[virus_seven_eighths:]

        history = model.fit_generator(
            generator=load_kmer_random_batches_h5(
                name='training',
                bacteria_dset=bacteria_dset,
                bacteria_subsample=bacteria_training_subsample,
                virus_dset=virus_dset,
                virus_subsample=virus_training_subsample,
                half_batch_size=batch_size // 2
            ),
            validation_data=load_kmer_random_batches_h5(
                name='validation',
                bacteria_dset=bacteria_dset,
                bacteria_subsample=bacteria_validation_subsample,
                virus_dset=virus_dset,
                virus_subsample=virus_validation_subsample,
                half_batch_size=batch_size // 2
            ),
            epochs=epochs,
            steps_per_epoch=100,
            validation_steps=validation_batch_count,
            workers=2
        )

        training_performance_df = pd.DataFrame(data=history.history, index=range(1, epochs + 1))
        training_performance_df.index.name = 'epoch'
        plot_training_performance(training_performance_df, title='Shuffled Data', pdfpages=pdf)


if __name__ == '__main__':
    main()