import h5py
import numpy as np
import sklearn.utils


class CNNData:
    def __init__(self, input_fp, batch_size, phage_dset_name, phage_training_range, prok_dset_name, prok_training_range, dev_parameters):
        self.name = 'CNNData'

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

        self.phage_dev_range = dev_parameters['phage_range']
        self.prok_dev_range = dev_parameters['prok_range']

        self.all_dev_sets = (
            ('dev', self.prok_dset_name, self.prok_dev_range, self.phage_dset_name, self.phage_dev_range),
        )

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
        return tuple([dev_set_name for dev_set_name, *_ in self.all_dev_sets])

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
        labels = np.vstack((
            np.zeros((self.prok_sample_count_batch, 1)),
            np.ones((self.phage_sample_count_batch, 1))))

        # this is a never ending generator
        epoch = 0
        while True:
            epoch += 1

            step = 0
            for bacteria_n, virus_n in zip(range(*self.prok_training_range, self.prok_sample_count_batch),
                                           range(*self.phage_training_range, self.phage_sample_count_batch)):
                step += 1

                bacteria_m = bacteria_n + self.prok_sample_count_batch
                virus_m = virus_n + self.phage_sample_count_batch

                batch = np.vstack((
                    bacteria_dset[bacteria_n:bacteria_m, :],
                    virus_dset[virus_n:virus_m, :]))

                return_tuple = (batch, labels)
                if shuffle_batch:
                    # yield shuffled views
                    # the source arrays are not modified
                    return_tuple = sklearn.utils.shuffle(*return_tuple)

                if yield_state:
                    return_tuple = (*return_tuple, step, epoch)

                yield return_tuple

            print('generator "{}" epoch {} has ended'.format(self.name, epoch))

    def get_dev_generators(self, data_file):
        """
        Yield development set generators.

        """

        def dev_gen(bdset, vdset, bnm, vnm):
            labels = np.vstack((
                np.zeros((bnm[0][1] - bnm[0][0], 1)),
                np.ones((vnm[0][1] - vnm[0][0], 1))
            ))

            for (bn, bm), (vn, vm) in zip(bnm, vir_start_stop):
                # print('loading validation data slice {}:{}'.format(n, m))

                batch = np.vstack((
                    bdset[bn:bm, :],
                    vdset[vn:vm, :]
                ))

                yield (batch, labels)

        for dev_set_name, bact_dset_name, bact_sample_interval, vir_dset_name, vir_sample_interval in self.all_dev_sets:
            bacteria_dset = data_file[bact_dset_name]
            virus_dset = data_file[vir_dset_name]

            print('loading validation data "{}" with shape {} interval {}'.format(bacteria_dset.name, bacteria_dset.shape, bact_sample_interval))
            print('loading validation data "{}" with shape {} interval {}'.format(virus_dset.name, virus_dset.shape, vir_sample_interval))

            # how many mini batches will there be?
            # for example, given:
            #   b = bact_sample_interval = (0, 10000)
            #   v = vir_sample_interval = (0, 10000)
            #   self.half_batch_size = 100
            # construct a list of all mini batch start indices:
            #   bact_start = range(b[0], b[1], self.half_batch_size) = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]
            #   vir_start = range(v[0], v[1], self.half_batch_size) = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]
            # construct a list of mini batch (start, stop) indices
            #   bact_start_stop = zip(bact_start, [start + self.half_batch_size for start in bact_start])
            #                   = [(0, 100), (100, 200), ..., (900, 1000)]
            #   vir_start_stop = zip(vir_start, [start + self.half_batch_size for start in vir_start])
            #                  = [(0, 100), (100, 200), ..., (900, 1000)]

            half_batch_size = self.actual_batch_size // 2

            bact_start = range(bact_sample_interval[0], bact_sample_interval[1], half_batch_size)
            vir_start = range(vir_sample_interval[0], vir_sample_interval[1], half_batch_size)

            bact_start_stop = tuple(zip(bact_start, [start + half_batch_size for start in bact_start]))
            vir_start_stop = tuple(zip(bact_start, [start + half_batch_size for start in vir_start]))

            steps = min(len(bact_start_stop), len(vir_start_stop))

            yield steps, dev_set_name, dev_gen(
                bdset=bacteria_dset,
                vdset=virus_dset,
                bnm=bact_start_stop,
                vnm=vir_start_stop)
