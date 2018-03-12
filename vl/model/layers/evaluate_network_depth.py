import argparse
import concurrent.futures
import io
import os.path
import time

from keras.callbacks import Callback
from keras.layers import Dense, Dropout
from keras.models import Sequential

import h5py
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
plt.switch_backend('agg')
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


def get_args():
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('-i', '--input-fp', required=True, help='Input file path')
    arg_parser.add_argument('-o', '--output-fp', required=True, help='Output file path')
    arg_parser.add_argument('-e', '--epoch-count', type=int, required=True, help='Number of epochs')
    arg_parser.add_argument('-n', '--process-count', type=int, required=True, help='Number of concurrent processes')

    args = arg_parser.parse_args()

    return args


def build_model(input_dim, layers):
    """
    Build and return a Sequential model with Dense layers given by the layers argument.

    Arguments
        input_dim  (int) dimension of input
        layers     (tuple) sequence of 2-ples, one per layer, such as ((64, 'relu'), (64, 'relu'), (1, 'sigmoid'))

    Return
        model_name (str) a name for the model
        model      (Model) a compiled model
    """
    model_name = io.StringIO()
    model = Sequential()
    layer_type, kwargs = layers[0]
    model_name.write('dns_{}'.format(kwargs['units']))
    model.add(Dense(**kwargs, input_dim=input_dim))
    for layer_type, kwargs in layers[1:]:
        if layer_type == 'Dense':
            model_name.write('_dns_{}'.format(kwargs['units']))
            model.add(Dense(**kwargs))
        elif layer_type == 'Dropout':
            model_name.write('_drp_{:2.1f}'.format(kwargs['rate']))
            model.add(Dropout(**kwargs))

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model_name.getvalue(), model


def plot_training_performance_combined(training_history_list, batches_per_epoch, epoch_count, output_fp):
    # generate network-specific accuracy and loss keys
    training_accuracy_loss = {}
    validation_accuracy_loss = {}
    for model_name, layers, history, t in training_history_list:
        training_accuracy_loss['acc ' + model_name] = history['acc']
        training_accuracy_loss['loss ' + model_name] = history['loss']
        validation_accuracy_loss['val_acc ' + model_name] = history['val_acc']
        validation_accuracy_loss['val_loss ' + model_name] = history['val_loss']

    training_df = pd.DataFrame(data=training_accuracy_loss, index=[b+1 for b in range(epoch_count*batches_per_epoch)])
    training_df.index.name = 'batch'

    training_df.to_csv(path_or_buf='training_accuracy_loss.tab', sep='\t')

    validation_df = pd.DataFrame(data=validation_accuracy_loss, index=[(e+1) * batches_per_epoch for e in range(epoch_count)])
    validation_df.index.name = 'batch'

    validation_df.to_csv(path_or_buf='validation_accuracy_loss.tab', sep='\t')

    output_dp, output_filename = os.path.split(output_fp)
    output_basename, output_ext = os.path.splitext(output_filename)

    combined_plots_fp = os.path.join(output_dp, output_basename + '_combined' + output_ext)
    with PdfPages(combined_plots_fp) as pdfpages:
        plt.figure()
        legend = []
        for loss_column in [column for column in training_df.columns if 'loss' in column]:
            print('training loss column: {}'.format(loss_column))
            plt.plot(training_df.index, training_df.loc[:, loss_column])
            legend.append(loss_column)
        for loss_column in [column for column in validation_df.columns if 'loss' in column]:
            print('validation loss column: {}'.format(loss_column))
            plt.plot(validation_df.index, validation_df.loc[:, loss_column])
            legend.append(loss_column)
        plt.title('Training and Validation Loss')
        plt.xlabel('batch')
        plt.ylabel('loss')
        plt.legend(legend)
        pdfpages.savefig()

        plt.figure()
        legend = []
        for acc_column in [column for column in training_df.columns if 'acc' in column]:
            print('training acc column: {}'.format(acc_column))
            plt.plot(training_df.index, training_df.loc[:, acc_column])
            legend.append(acc_column)
        for acc_column in [column for column in validation_df.columns if 'acc' in column]:
            print('validation acc column: {}'.format(acc_column))
            plt.plot(validation_df.index, validation_df.loc[:, acc_column])
            legend.append(acc_column)
        plt.title('Training and Validation Accuracy')
        plt.xlabel('batch')
        plt.ylabel('accuracy')
        plt.legend(legend)
        pdfpages.savefig()


def plot_training_performance_separate(training_history_list, batches_per_epoch, epoch_count, output_fp):
    # generate network-specific accuracy and loss keys
    output_dp, output_filename = os.path.split(output_fp)
    output_basename, output_ext = os.path.splitext(output_filename)

    separate_plots_fp = os.path.join(output_dp, output_basename + '_separate' + output_ext)

    with PdfPages(separate_plots_fp) as pdfpages:
        for model_name, layers, history, t in training_history_list:
            training_accuracy_loss = {}
            validation_accuracy_loss = {}

            training_accuracy_loss['acc ' + model_name] = history['acc']
            training_accuracy_loss['loss ' + model_name] = history['loss']
            validation_accuracy_loss['val_acc ' + model_name] = history['val_acc']
            validation_accuracy_loss['val_loss ' + model_name] = history['val_loss']

            training_df = pd.DataFrame(
                data=training_accuracy_loss,
                index=[b + 1 for b in range(epoch_count * batches_per_epoch)])
            training_df.index.name = 'batch'

            #training_df.to_csv(path_or_buf='training_accuracy_loss.tab', sep='\t')

            validation_df = pd.DataFrame(
                data=validation_accuracy_loss,
                index=[(e + 1) * batches_per_epoch for e in range(epoch_count)])
            validation_df.index.name = 'batch'

            #validation_df.to_csv(path_or_buf='validation_accuracy_loss.tab', sep='\t')

            plt.figure()
            legend = []
            for loss_column in [column for column in training_df.columns if 'loss' in column]:
                print('training loss column: {}'.format(loss_column))
                plt.plot(training_df.index, training_df.loc[:, loss_column])
                legend.append(loss_column)
            for loss_column in [column for column in validation_df.columns if 'loss' in column]:
                print('validation loss column: {}'.format(loss_column))
                plt.plot(validation_df.index, validation_df.loc[:, loss_column])
                legend.append(loss_column)
            plt.title('Training and Validation Loss\n{}'.format(model_name))
            plt.xlabel('batch')
            plt.ylabel('loss')
            plt.legend(legend)
            pdfpages.savefig()

            plt.figure()
            legend = []
            for acc_column in [column for column in training_df.columns if 'acc' in column]:
                print('training acc column: {}'.format(acc_column))
                plt.plot(training_df.index, training_df.loc[:, acc_column])
                legend.append(acc_column)
            for acc_column in [column for column in validation_df.columns if 'acc' in column]:
                print('validation acc column: {}'.format(acc_column))
                plt.plot(validation_df.index, validation_df.loc[:, acc_column])
                legend.append(acc_column)
            plt.title('Training and Validation Accuracy\n{}'.format(model_name))
            plt.xlabel('batch')
            plt.ylabel('accuracy')
            plt.legend(legend)
            pdfpages.savefig()


def train(train_test_fp, layers, parameters):
    # we have 1,000,000 training samples
    # training    : 8/10 * 1,000,000 = 800,000
    # development : 1/10 * 1,000,000 = 100,000
    # test        : 1/10 * 1,000,000 = 100,000


    with h5py.File(train_test_fp, 'r', libver='latest', swmr=True) as train_test_file:
        bacteria_dset = train_test_file['/clean-bact/training1/extract/kmers']
        virus_dset = train_test_file['/clean-vir/training1/extract/kmers']

        model_name, model = build_model(layers=layers, input_dim=bacteria_dset.shape[1])

        epochs = parameters['epochs']
        batch_size = parameters['batch_size']
        training_samples = parameters['training_samples']
        validation_samples = parameters['validation_samples']

        training_end = parameters['training_end']
        validation_end = parameters['validation_end']

        training_batches_per_epoch = training_samples // batch_size
        validation_batch_count = validation_samples // batch_size
        print('batch size is {}'.format(batch_size))
        print('{} training samples'.format(training_samples))
        print('{} batches of training data'.format(training_batches_per_epoch))
        print('{} validation samples'.format(validation_samples))
        print('{} batches of validation data'.format(validation_batch_count))

        print('{} epochs'.format(epochs))

        t0 = time.time()
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
            callbacks=[history],
            verbose=0
        )
        return model_name, history.history, time.time()-t0


def main():
    args = get_args()

    network_depths = (
        (('Dense', {'units': 64, 'activation': 'relu'}), ('Dropout', {'rate': 0.4}), ('Dense', {'units': 64, 'activation': 'relu'}), ('Dense', {'units': 1, 'activation': 'sigmoid'})),
        (('Dense', {'units': 64, 'activation': 'relu'}), ('Dropout', {'rate': 0.5}), ('Dense', {'units': 64, 'activation': 'relu'}), ('Dense', {'units': 1, 'activation': 'sigmoid'})),
        (('Dense', {'units': 64, 'activation': 'relu'}), ('Dropout', {'rate': 0.6}), ('Dense', {'units': 64, 'activation': 'relu'}), ('Dense', {'units': 1, 'activation': 'sigmoid'})),
        (('Dense', {'units': 64, 'activation': 'relu'}), ('Dropout', {'rate': 0.7}), ('Dense', {'units': 64, 'activation': 'relu'}), ('Dense', {'units': 1, 'activation': 'sigmoid'})),
        (('Dense', {'units': 64, 'activation': 'relu'}), ('Dropout', {'rate': 0.4}), ('Dense', {'units': 64, 'activation': 'relu'}), ('Dropout', {'rate': 0.4}), ('Dense', {'units': 1, 'activation': 'sigmoid'})),
        (('Dense', {'units': 64, 'activation': 'relu'}), ('Dropout', {'rate': 0.5}), ('Dense', {'units': 64, 'activation': 'relu'}), ('Dropout', {'rate': 0.5}), ('Dense', {'units': 1, 'activation': 'sigmoid'})),
        (('Dense', {'units': 64, 'activation': 'relu'}), ('Dropout', {'rate': 0.6}), ('Dense', {'units': 64, 'activation': 'relu'}), ('Dropout', {'rate': 0.6}), ('Dense', {'units': 1, 'activation': 'sigmoid'})),
        (('Dense', {'units': 64, 'activation': 'relu'}), ('Dropout', {'rate': 0.7}), ('Dense', {'units': 64, 'activation': 'relu'}), ('Dropout', {'rate': 0.7}), ('Dense', {'units': 1, 'activation': 'sigmoid'})),
        (('Dense', {'units': 64, 'activation': 'relu'}), ('Dropout', {'rate': 0.4}),
         ('Dense', {'units': 64, 'activation': 'relu'}), ('Dropout', {'rate': 0.3}),
         ('Dense', {'units': 1, 'activation': 'sigmoid'})),
        (('Dense', {'units': 64, 'activation': 'relu'}), ('Dropout', {'rate': 0.5}),
         ('Dense', {'units': 64, 'activation': 'relu'}), ('Dropout', {'rate': 0.4}),
         ('Dense', {'units': 1, 'activation': 'sigmoid'})),
        (('Dense', {'units': 64, 'activation': 'relu'}), ('Dropout', {'rate': 0.6}),
         ('Dense', {'units': 64, 'activation': 'relu'}), ('Dropout', {'rate': 0.5}),
         ('Dense', {'units': 1, 'activation': 'sigmoid'})),
        (('Dense', {'units': 64, 'activation': 'relu'}), ('Dropout', {'rate': 0.7}),
         ('Dense', {'units': 64, 'activation': 'relu'}), ('Dropout', {'rate': 0.6}),
         ('Dense', {'units': 1, 'activation': 'sigmoid'}))
    )

    # quick
    #training_samples = 800
    #validation_samples = 100
    #test_samples = 100

    # for reals
    training_samples = 800000
    validation_samples = 100000
    test_samples = 100000

    batch_size = 100

    parameters = {
        'training_samples': training_samples,
        'validation_samples': validation_samples,
        'test_samples': test_samples,
        'training_end': training_samples // 2,
        'validation_end': (training_samples + validation_samples) // 2,
        'test_end': (training_samples + validation_samples + test_samples) // 2,
        'epochs': args.epoch_count,
        'batch_size': batch_size,
        'batches_per_epoch': training_samples // batch_size
    }

    training_history_list = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.process_count) as executor:
        future_to_layers = {
            executor.submit(train, args.input_fp, layers, parameters): layers
            for layers
            in network_depths}

        for future in concurrent.futures.as_completed(future_to_layers):
            layers = future_to_layers[future]
            try:
                model_name, training_history, t = future.result()
                print('finished network {} in {:5.2f}s'.format(layers, t))
            except Exception as exc:
                print(exc)
            else:
                training_history_list.append((model_name, layers, training_history, t))

        plot_training_performance_combined(
            training_history_list=training_history_list,
            batches_per_epoch=parameters['batches_per_epoch'],
            epoch_count=args.epoch_count,
            output_fp=args.output_fp)

        plot_training_performance_separate(
            training_history_list=training_history_list,
            batches_per_epoch=parameters['batches_per_epoch'],
            epoch_count=args.epoch_count,
            output_fp=args.output_fp)


if __name__ == '__main__':
    main()