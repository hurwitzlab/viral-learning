import h5py
import numpy as np
import sklearn.utils


class CNNSyntheticData:
    def __init__(self, seed, bp, batch_size, phage_dset_name, phage_training_range, proka_dset_name, proka_training_range, dev_parameters):
        self.name = 'CNNSyntheticData'

        self.seed = seed
        self.bp = bp
        self.requested_batch_size = batch_size

        self.phage_dset_name = phage_dset_name
        self.phage_training_range = phage_training_range

        self.proka_dset_name = proka_dset_name
        self.proka_training_range = proka_training_range

        # given the batch size let's figure out how many phage samples and how
        # many prokaryote samples to put in
        # eg
        # 100000 phage samples
        #  30000 prokaryote samples
        # batch size 64
        # 100000 / (100000 + 30000) * 64 = 49.2
        #  30000 / (100000 + 30000) * 64 = 14.7

        self.phage_training_sample_count = phage_training_range[1] - phage_training_range[0]
        self.proka_training_sample_count = proka_training_range[1] - proka_training_range[0]
        self.total_training_sample_count = self.phage_training_sample_count + self.proka_training_sample_count

        self.phage_sample_count_batch = int(np.round((self.phage_training_sample_count / self.total_training_sample_count) * self.requested_batch_size))
        self.proka_sample_count_batch = int(np.round((self.proka_training_sample_count / self.total_training_sample_count) * self.requested_batch_size))
        self.actual_batch_size = self.phage_sample_count_batch + self.proka_sample_count_batch

        self.phage_batch_count = int(np.round(self.phage_training_sample_count / self.phage_sample_count_batch))
        self.proka_batch_count = int(np.round(self.proka_training_sample_count / self.proka_sample_count_batch))
        self.actual_batch_count = np.min((self.phage_batch_count, self.proka_batch_count))

        print('actual minibatch size               : {}'.format(self.actual_batch_size))
        print('  phage samples per minibatch       : {}'.format(self.phage_sample_count_batch))
        print('  prokaryote samples per minibatch  : {}'.format(self.proka_sample_count_batch))

        print('actual batch count                  : {}'.format(self.actual_batch_count))
        print('  phage batch count                 : {}'.format(self.phage_batch_count))
        print('  prokaryote batch count            : {}'.format(self.proka_batch_count))

        self.phage_dev_range = dev_parameters['phage_range']
        self.proka_dev_range = dev_parameters['proka_range']

        self.all_dev_sets = (
            ('dev', self.proka_dset_name, self.proka_dev_range, self.phage_dset_name, self.phage_dev_range),
        )

        #with h5py.File(name=self.fp) as f:
        #    self.phage_dset_shape = f[self.phage_dset_name].shape
        #    print(self.phage_dset_shape)
        #    self.proka_dset_shape = f[self.proka_dset_name].shape
        #    print(self.proka_dset_shape)

        #self.input_shape = self.phage_dset_shape[1:]
        self.input_height = int(np.floor(np.sqrt(self.bp)))
        self.input_width = int(np.floor(self.bp / self.input_height))
        self.input_channel_count = 4
        self.input_shape = (self.input_height, self.input_width, self.input_channel_count)
        print('input shape: {}'.format(self.input_shape))

    def get_input_shape(self):
        return self.input_shape

    def get_training_mini_batches_per_epoch(self):
        return self.actual_batch_count

    def get_dev_set_names(self):
        return tuple([dev_set_name for dev_set_name, *_ in self.all_dev_sets])

    def get_training_data_generator(self):
        """
        Return batches of input and labels from a combined H5 file.

        The returned data is shuffled. This is very important. If the
        batches are returned with the first half bacteria data and
        the second half virus data the models train 'almost perfectly'
        and evaluate 'perfectly'.

        Arguments:
            shuffle_batch    (True or False) shuffle the samples in each batch
            yield_state      (True of False) yield the current step and epoch with the current mini batch

        Yield:
            (batch, labels, step, epoch) tuple of (half_batch_size*2, features) and (half_batch_size*2, 1) labels
        """

        print('{} batches of {} samples will be yielded in each epoch'.format(self.actual_batch_count, self.actual_batch_size))

        def training_data_generator(shuffle_batch=True, yield_state=False):
            if shuffle_batch:
                print('batches and labels will be shuffled')
            else:
                print('batches and labels will NOT be shuffled')

            # bacteria label is 0
            # virus label is 1
            labels = np.vstack((
                np.zeros((self.proka_sample_count_batch, 1)),
                np.ones((self.phage_sample_count_batch, 1))))

            # this is a never ending generator
            epoch = 0
            while True:
                epoch += 1
                np.random.seed(seed=self.seed)
                step = 0
                for b in range(self.actual_batch_count):
                    step += 1
                    #for s in range(self.actual_batch_size):

                    #bacteria_m = bacteria_n + self.proka_sample_count_batch
                    #virus_m = virus_n + self.phage_sample_count_batch

                    # proka will be only C,G
                    proka = np.zeros((
                        self.actual_batch_size // 2,
                        self.input_height,
                        self.input_width,
                        self.input_channel_count))

                    proka[:, :, :, 1] = np.random.choice(
                        a=(0, 1),
                        size=(self.actual_batch_size // 2, self.input_height, self.input_width))
                    proka[:, :, :, 2] = np.abs(1.0 - proka[:, :, :, 1])

                    # phage will be only A,T
                    phage = np.zeros((
                        self.actual_batch_size // 2,
                        self.input_height,
                        self.input_width,
                        self.input_channel_count))
                    phage[:, :, :, 0] = np.random.choice(
                        a=(0, 1),
                        size=(self.actual_batch_size // 2, self.input_height, self.input_width))
                    phage[:, :, :, 3] = np.abs(1.0 - phage[:, :, :, 0])

                    batch = np.vstack((proka, phage))

                    return_tuple = (batch, labels)
                    if shuffle_batch:
                        # yield shuffled views
                        # the source arrays are not modified
                        return_tuple = sklearn.utils.shuffle(*return_tuple)

                    if yield_state:
                        return_tuple = (*return_tuple, step, epoch)

                    yield return_tuple

                print('generator "{}" epoch {} has ended'.format(self.name, epoch))

        return training_data_generator

    def get_dev_generators(self):
        """
        Yield development set generators.

        """

        def dev_data_generator():
            # bacteria label is 0
            # virus label is 1
            labels = np.vstack((
                np.zeros((self.proka_dev_range[1], 1)),
                np.ones((self.phage_dev_range[1], 1))))

            np.random.seed(seed=self.seed)
            # proka will be only C,G
            proka = np.zeros((
                self.proka_dev_range[1],
                self.input_height,
                self.input_width,
                self.input_channel_count))

            proka[:, :, :, 1] = np.random.choice(
                a=(0, 1),
                size=(self.proka_dev_range[1], self.input_height, self.input_width))
            proka[:, :, :, 2] = np.abs(1.0 - proka[:, :, :, 1])

            # phage will be only A,T
            phage = np.zeros((
                self.phage_dev_range[1],
                self.input_height,
                self.input_width,
                self.input_channel_count))
            phage[:, :, :, 0] = np.random.choice(
                a=(0, 1),
                size=(self.phage_dev_range[1], self.input_height, self.input_width))
            phage[:, :, :, 3] = np.abs(1.0 - phage[:, :, :, 0])

            batch = np.vstack((proka, phage))

            return_tuple = (batch, labels)

            yield return_tuple

        for dev_set_name, proka_dset_name, proka_dev_range, phage_dset_name, phage_dev_range in self.all_dev_sets:
            yield 1, dev_set_name, dev_data_generator()
