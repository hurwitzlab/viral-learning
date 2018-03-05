from keras.models import Sequential
from keras.layers import Dense, Lambda
import h5py
import matplotlib.pyplot as plt
import pandas as pd

from vl.data import load_kmer_range_batches_h5


def build_model(input_dim, mean, variance):
    model = Sequential()
    #variance[variance == 0.0] = 1.0
    model.add(Lambda(function=lambda x: print(x.shape), input_shape=(input_dim,)))
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
    # go!
    train_test_fp = '../data/training_testing.h5'
    batch_size = 50

    with h5py.File(train_test_fp, 'r', libver='latest', swmr=True) as train_test_file:
        mean_dset = train_test_file['/clean-bact-vir/training1/extract/kmers/kmer_file1/mean']
        variance_dset = train_test_file['/clean-bact-vir/training1/extract/kmers/kmer_file1/variance']

        bacteria_dset = train_test_file['/clean-bact/training1/extract/kmers/kmer_file1']
        virus_dset = train_test_file['/clean-vir/training1/extract/kmers/kmer_file1']

        model = build_model(input_dim=bacteria_dset.shape[1], mean=mean_dset[:, :], variance=variance_dset[:, :])

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

        epochs = 100
        print('{} epochs'.format(epochs))

        history = model.fit_generator(
            generator=load_kmer_range_batches_h5(
                name='training',
                bacteria_dset=bacteria_dset,
                bacteria_range=(0, 7 * bacteria_dset.shape[0] // 8),
                virus_dset=virus_dset,
                virus_range=(0, 7 * virus_dset.shape[0] // 8),
                half_batch_size=batch_size // 2
            ),
            # there is no advantage to permuting the validation samples
            # and there may be a speed advantage to reading them in order
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
        plot_training_performance(training_performance_df)


if __name__ == '__main__':
    main()