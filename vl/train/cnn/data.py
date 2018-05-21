import h5py
import numpy as np
import sklearn.utils


class CNNData:
    def __init__(self, input_fp, batch_size, phage_dset_name, phage_training_range, prok_dset_name, prok_training_range):
        self.fp = input_fp
        self.requested_batch_size = batch_size

        self.phage_dset_name = phage_dset_name
        self.phage_training_range = phage_training_range

        self.prok_dset_name = prok_dset_name
        self.prok_training_range = prok_training_range

        # given the batch size let's figure out how many phage samples and how
        # many prokaryote samples to put in
        # eg
        # 100000 phage samples
        #  30000 prokaryote samples
        # batch size 64
        # 100000 / (100000 + 30000) * 64 = 49.2
        #  30000 / (100000 + 30000) * 64 = 14.7

        self.phage_training_sample_count = phage_training_range[1] - phage_training_range[0]
        self.prok_training_sample_count = prok_training_range[1] - prok_training_range[0]
        self.total_training_sample_count = self.phage_training_sample_count + self.prok_training_sample_count

        self.phage_sample_count_batch = int(np.round((self.phage_training_sample_count / self.total_training_sample_count) * self.requested_batch_size))
        self.prok_sample_count_batch = int(np.round((self.prok_training_sample_count / self.total_training_sample_count) * self.requested_batch_size))
        self.actual_batch_size = self.phage_sample_count_batch + self.prok_sample_count_batch

        self.phage_batch_count = int(np.round(self.phage_training_sample_count / self.phage_sample_count_batch))
        self.prok_batch_count = int(np.round(self.prok_training_sample_count / self.prok_sample_count_batch))
        self.actual_batch_count = np.min((self.phage_batch_count, self.prok_batch_count))

        print('actual minibatch size               : {}'.format(self.actual_batch_size))
        print('  phage samples per minibatch       : {}'.format(self.phage_sample_count_batch))
        print('  prokaryote samples per minibatch  : {}'.format(self.prok_sample_count_batch))

        print('actual batch count                  : {}'.format(self.actual_batch_count))
        print('  phage batch count                 : {}'.format(self.phage_batch_count))
        print('  prokaryote batch count            : {}'.format(self.prok_batch_count))

        with h5py.File(name=self.fp) as f:
            self.phage_dset_shape = f[self.phage_dset_name].shape
            print(self.phage_dset_shape)
            self.prok_dset_shape = f[self.prok_dset_name].shape
            print(self.prok_dset_shape)

        self.input_shape = self.phage_dset_shape[1:]
        print('input shape: {}'.format(self.input_shape))

    def get_input_shape(self):
        return self.input_shape

    def get_training_mini_batches_per_epoch(self):
        return self.actual_batch_count

    def get_dev_set_names(self):
        return ('dev set', )

    #def get_training_mini_batches(train_test_file, yield_state=True):

    def get_training_mini_batches(self, data_file, shuffle_batch=True, yield_state=False):
        """
        Return batches of input and labels from a combined H5 file.

        The returned data is shuffled. This is very important. If the
        batches are returned with the first half bacteria data and
        the second half virus data the models train 'almost perfectly'
        and evaluate 'perfectly'.

        Arguments:
            shuffle_batch        (True or False) shuffle the samples in each batch
            yield_state          (True of False) yield the current step and epoch with the current mini batch

        Yield:
            (batch, labels, step, epoch) tuple of (half_batch_size*2, features) and (half_batch_size*2, 1) labels
        """
        bacteria_dset = data_file[self.prok_dset_name]
        virus_dset = data_file[self.phage_dset_name]

        print('reading bacteria dataset "{}" with shape {}'.format(bacteria_dset.name, bacteria_dset.shape))
        print('reading virus dataset "{}" with shape {}'.format(virus_dset.name, virus_dset.shape))

        print('{} batches of {} samples will be yielded in each epoch'.format(self.actual_batch_count, self.actual_batch_size))

        if shuffle_batch:
            print('batches and labels will be shuffled')
        else:
            print('batches and labels will NOT be shuffled')

        # bacteria label is 0
        # virus label is 1
        labels = np.vstack((np.zeros((self.prok_sample_count_batch, 1)), np.ones((self.phage_sample_count_batch, 1))))

        # this is a never ending generator
        epoch = 0
        while True:
            epoch += 1

            step = 0
            for bacteria_n, virus_n in zip(range(*self.prok_training_range, self.prok_batch_count),
                                           range(*self.phage_training_range, self.phage_batch_count)):
                step += 1

                bacteria_m = bacteria_n + self.prok_sample_count_batch
                virus_m = virus_n + self.phage_sample_count_batch

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

            print('generator "{}" epoch {} has ended'.format(self.name, epoch))

    def get_dev_generators(self):
        return None
