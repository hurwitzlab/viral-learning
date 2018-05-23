"""
This class partitions the data in file /extra/jklynch/viral-learning/perm_training_testing.h5 into training
and development sets.

[jklynch@login2 data]$ ls -l /extra/jklynch/viral-learning/
total 5403360
-rw-r--r-- 1 jklynch bhurwitz 2514485483 Mar  9 19:14 perm_training_testing.h5
-rw-r--r-- 1 jklynch bhurwitz 3018545506 Mar  6 13:10 training_testing.h5

perm_training_testing.h5 looks like this:

[jklynch@login2 data]$ h5ls -r /extra/jklynch/viral-learning/perm_training_testing.h5
/                        Group
/bact_marinePatric       Group
/bact_marinePatric/extract_bact_100 Group
/bact_marinePatric/extract_bact_100/kmers Dataset {5000, 32768}
/bact_marinePatric/extract_bact_1000 Group
/bact_marinePatric/extract_bact_1000/kmers Dataset {5000, 32768}
/bact_marinePatric/extract_bact_10000 Group
/bact_marinePatric/extract_bact_10000/kmers Dataset {5000, 32768}
/bact_marinePatric/extract_bact_200 Group
/bact_marinePatric/extract_bact_200/kmers Dataset {5000, 32768}
/bact_marinePatric/extract_bact_500 Group
/bact_marinePatric/extract_bact_500/kmers Dataset {5000, 32768}
/bact_marinePatric/extract_bact_5000 Group
/bact_marinePatric/extract_bact_5000/kmers Dataset {5000, 32768}
/clean-bact              Group
/clean-bact/training1    Group
/clean-bact/training1/extract Group
/clean-bact/training1/extract/kmers Dataset {500000, 32768}
/clean-vir               Group
/clean-vir/training1     Group
/clean-vir/training1/extract Group
/clean-vir/training1/extract/kmers Dataset {500000, 32768}
/vir_marinePatric        Group
/vir_marinePatric/extract_vir_100 Group
/vir_marinePatric/extract_vir_100/kmers Dataset {5000, 32768}
/vir_marinePatric/extract_vir_1000 Group
/vir_marinePatric/extract_vir_1000/kmers Dataset {5000, 32768}
/vir_marinePatric/extract_vir_10000 Group
/vir_marinePatric/extract_vir_10000/kmers Dataset {4900, 32768}
/vir_marinePatric/extract_vir_200 Group
/vir_marinePatric/extract_vir_200/kmers Dataset {5000, 32768}
/vir_marinePatric/extract_vir_500 Group
/vir_marinePatric/extract_vir_500/kmers Dataset {5000, 32768}
/vir_marinePatric/extract_vir_5000 Group
/vir_marinePatric/extract_vir_5000/kmers Dataset {4900, 32768}

990,000 training samples and 10,000 development samples are taken from datasets

/clean-bact/training1/extract/kmers
/clean-vir/training1/extract/kmers

In addition development sets are taken from the first half of each 'marinePatric' set.

"""
import h5py
import numpy as np
from sklearn.utils import shuffle


class BacteriaAndVirusKMers:

    def __init__(self, fp, half_batch_size, training_sample_count=990000, development_sample_count=10000):
        self.name = 'Bacteria and Virus KMers'
        self.fp = fp
        self.half_batch_size = half_batch_size
        self.batch_size = 2 * half_batch_size

        self.training_sample_count = training_sample_count
        self.development_sample_count = development_sample_count
        if self.training_sample_count + self.development_sample_count > 1000000:
            raise Exception('training sample count and development sample count exceed total number of samples')

        self.bacteria_training_dset_name = '/clean-bact/training1/extract/kmers'
        self.virus_training_dset_name = '/clean-vir/training1/extract/kmers'

        self.bacteria_mg_sample_count = 500000
        self.virus_mg_sample_count = 500000

        self.bacteria_mg_training_interval = (0, training_sample_count // 2)
        self.virus_mg_training_interval = (0, training_sample_count // 2)

        self.training_batch_count = min(
            (self.bacteria_mg_training_interval[1] - self.bacteria_mg_training_interval[0]) // self.half_batch_size,
            (self.virus_mg_training_interval[1] - self.virus_mg_training_interval[0]) // self.half_batch_size)

        self.bacteria_mg_development_interval = (
            self.bacteria_mg_sample_count - (self.development_sample_count // 2),
            self.bacteria_mg_sample_count)
        self.virus_mg_development_interval = (
            self.virus_mg_sample_count - (self.development_sample_count // 2),
            self.virus_mg_sample_count)

        self.marine_dev_set_list = tuple([
            (
                'dev rl: {}'.format(r),
                '/bact_marinePatric/extract_bact_{}/kmers'.format(r), (0, 5000 // 2),
                '/vir_marinePatric/extract_vir_{}/kmers'.format(r), (0, 5000 // 2)
            )
            for r
            in (100, 200, 500, 1000, 5000, 10000)])

        self.all_dev_sets = (
            (
                'dev mg',
                '/clean-bact/training1/extract/kmers', self.bacteria_mg_development_interval,
                '/clean-vir/training1/extract/kmers', self.virus_mg_development_interval
            ),
            *self.marine_dev_set_list)

    def get_training_mini_batches_per_epoch(self):
        return self.training_sample_count // self.batch_size

    def get_input_dim(self):
        with h5py.File(self.fp, 'r', libver='latest', swmr=True) as train_test_file:
            bacteria_dset = train_test_file[self.bacteria_training_dset_name]
            return bacteria_dset.shape[1]

    def get_training_mini_batches(self, data_file, shuffle_batch=True, yield_state=False):
        """
        Return batches of input and labels from a combined H5 file.

        The returned data is shuffled. This is very important. If the
        batches are returned with the first half bacteria data and
        the second half virus data the models train 'almost perfectly'
        and evaluate 'perfectly'.

        Arguments:
            data_file            H5 file object
            shuffle_batch        (True or False) shuffle the samples in each batch
            yield_state          (True of False) yield the current step and epoch with the current mini batch

        Yield:
            (batch, labels, step, epoch) tuple of (half_batch_size*2, features) and (half_batch_size*2, 1) labels
        """
        bacteria_dset = data_file[self.bacteria_training_dset_name]
        virus_dset = data_file[self.virus_training_dset_name]

        print('reading bacteria dataset "{}" with shape {}'.format(bacteria_dset.name, bacteria_dset.shape))
        print('reading virus dataset "{}" with shape {}'.format(virus_dset.name, virus_dset.shape))

        print('{} batches of {} samples will be yielded in each epoch'.format(self.training_batch_count, self.batch_size))

        if shuffle_batch:
            print('batches and labels will be shuffled')
        else:
            print('batches and labels will NOT be shuffled')

        # bacteria label is 0
        # virus label is 1
        labels = np.vstack((np.zeros((self.half_batch_size, 1)), np.ones((self.half_batch_size, 1))))

        # this is a never ending generator
        epoch = 0
        while True:
            epoch += 1

            step = 0
            for bacteria_n, virus_n in zip(range(*self.bacteria_mg_training_interval, self.half_batch_size),
                                           range(*self.virus_mg_training_interval, self.half_batch_size)):
                step += 1

                bacteria_m = bacteria_n + self.half_batch_size
                virus_m = virus_n + self.half_batch_size

                batch = np.vstack((
                    bacteria_dset[bacteria_n:bacteria_m, :],
                    virus_dset[virus_n:virus_m, :]
                ))

                return_tuple = (batch, labels)
                if shuffle_batch:
                    # yield shuffled views
                    # the source arrays are not modified
                    return_tuple = shuffle(*return_tuple)

                if yield_state:
                    return_tuple = (*return_tuple, step, epoch)

                yield return_tuple

            print('generator "{}" epoch {} has ended'.format(self.name, epoch))

    def get_dev_set_names(self):
        return tuple([dev_set_name for dev_set_name, *_ in self.all_dev_sets])

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

            bact_start = range(bact_sample_interval[0], bact_sample_interval[1], self.half_batch_size)
            vir_start = range(vir_sample_interval[0], vir_sample_interval[1], self.half_batch_size)

            bact_start_stop = tuple(zip(bact_start, [start + self.half_batch_size for start in bact_start]))
            vir_start_stop = tuple(zip(bact_start, [start + self.half_batch_size for start in vir_start]))

            steps = min(len(bact_start_stop), len(vir_start_stop))

            yield steps, dev_set_name, dev_gen(
                bdset=bacteria_dset,
                vdset=virus_dset,
                bnm=bact_start_stop,
                vnm=vir_start_stop)
