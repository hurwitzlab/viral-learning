"""
This class handles partitioning Alise's kmer data derived from bacterial and viral genomes into training
and development sets. Her data files are in

/rsgrps/bhurwitz/alise/my_data/Riveal_exp/Models/RefSeq_based_models/Prok_Phages_models

I have made a copy of the data in HDF5 files here:

/extra/jklynch/viral-learning/

[jklynch@login2 genome]$ ls -l /extra/jklynch/viral-learning/
total 5828736
-rw-r--r-- 1 jklynch bhurwitz 2514485483 Mar  9 19:14 perm_training_testing.h5
-rw-r--r-- 1 jklynch bhurwitz   65383042 Apr 24 19:08 riveal_refseq_prok_phage_1000pb_kmers4.h5
-rw-r--r-- 1 jklynch bhurwitz  222114217 Apr 24 19:11 riveal_refseq_prok_phage_1000pb_kmers6.h5
-rw-r--r-- 1 jklynch bhurwitz  557841124 Apr 24 19:44 riveal_refseq_prok_phage_1000pb_kmers8.h5
-rw-r--r-- 1 jklynch bhurwitz  110068952 Apr 24 19:45 riveal_refseq_prok_phage_5000pb_kmers4.h5
-rw-r--r-- 1 jklynch bhurwitz  396763793 Apr 24 19:49 riveal_refseq_prok_phage_5000pb_kmers6.h5
-rw-r--r-- 1 jklynch bhurwitz 1531270634 Apr 24 20:24 riveal_refseq_prok_phage_5000pb_kmers8.h5
-rw-r--r-- 1 jklynch staff      57170906 Apr 24 18:31 riveal_refseq_prok_phage_500pb_kmers4.h5
-rw-r--r-- 1 jklynch staff     170264436 Apr 24 18:34 riveal_refseq_prok_phage_500pb_kmers6.h5
-rw-r--r-- 1 jklynch staff     343214490 Apr 24 19:08 riveal_refseq_prok_phage_500pb_kmers8.h5
-rw-r--r-- 1 jklynch bhurwitz        800 Apr 24 11:54 training_testing.h5

The HDF5 files look like this:

[jklynch@login2 genome]$ h5ls -r /extra/jklynch/viral-learning/riveal_refseq_prok_phage_500pb_kmers4.h5
/                        Group
/Phage                   Group
/Phage/500pb             Group
/Phage/500pb/kmers4      Dataset {100000/Inf, 128}
/Proc                    Group
/Proc/500pb              Group
/Proc/500pb/kmers4       Dataset {100000/Inf, 128}
[jklynch@login2 genome]$ h5ls -r /extra/jklynch/viral-learning/riveal_refseq_prok_phage_500pb_kmers6.h5
/                        Group
/Phage                   Group
/Phage/500pb             Group
/Phage/500pb/kmers6      Dataset {100000/Inf, 2048}
/Proc                    Group
/Proc/500pb              Group
/Proc/500pb/kmers6       Dataset {100000/Inf, 2048}
[jklynch@login2 genome]$ h5ls -r /extra/jklynch/viral-learning/riveal_refseq_prok_phage_500pb_kmers8.h5
/                        Group
/Phage                   Group
/Phage/500pb             Group
/Phage/500pb/kmers8      Dataset {100000/Inf, 32768}
/Proc                    Group
/Proc/500pb              Group
/Proc/500pb/kmers8       Dataset {100000/Inf, 32768}

90,000 training samples and 10,000 development samples are taken from each dataset.

"""
import h5py
import numpy as np
from sklearn.utils import shuffle


class BacteriaAndVirusKMers:

    def __init__(self, fp, pb, k, half_batch_size, training_sample_count=90000, development_sample_count=10000):
        self.name = 'Bacteria and Virus KMers'
        self.fp = fp
        if pb in (500, 1000, 5000):
            pass
        else:
            raise Exception('pb must be 500, 1000, or 5000')

        if k in (4, 6, 8):
            pass
        else:
            raise Exception('k must be 4, 6, or 8')

        self.half_batch_size = half_batch_size
        self.batch_size = 2 * half_batch_size

        self.training_sample_count = training_sample_count
        self.development_sample_count = development_sample_count
        if self.training_sample_count + self.development_sample_count > 100000:
            raise Exception('training sample count and development sample count exceed total number of samples')

        self.bacteria_dset_name = '/Proc/{}pb/kmers{}'.format(pb, k)
        self.virus_dset_name = '/Phage/{}pb/kmers{}'.format(pb, k)

        self.bacteria_sample_count = 100000
        self.virus_sample_count = 100000

        self.bacteria_training_interval = (0, training_sample_count)
        self.virus_training_interval = (0, training_sample_count)

        self.training_batch_count = min(
            (self.bacteria_training_interval[1] - self.bacteria_training_interval[0]) // self.half_batch_size,
            (self.virus_training_interval[1] - self.virus_training_interval[0]) // self.half_batch_size)

        self.bacteria_development_interval = (self.training_sample_count, self.bacteria_sample_count)
        self.virus_development_interval = (self.training_sample_count, self.virus_sample_count)

    def get_training_mini_batches_per_epoch(self):
        return self.training_sample_count // self.batch_size

    def get_input_dim(self):
        with h5py.File(self.fp, 'r', libver='latest', swmr=True) as train_test_file:
            bacteria_dset = train_test_file[self.bacteria_dset_name]
            return bacteria_dset.shape[1]

    def get_training_mini_batches(self, data_file, shuffle_batch=True, yield_state=False):
        """
        Return batches of input and labels from a combined H5 file.

        By default the returned data is shuffled. This is very important. If the
        batches are returned with the first half bacteria data and the second half
        virus data the models train 'almost perfectly' and evaluate 'perfectly'
        but they are not perfect.

        Arguments:
            data_file            H5 file object
            shuffle_batch        (True or False) shuffle the samples in each batch
            yield_state          (True of False) yield the current step and epoch with the current mini batch

        Yield:
            (batch, labels, step, epoch) tuple of (half_batch_size*2, features) and (half_batch_size*2, 1) labels
        """
        bacteria_dset = data_file[self.bacteria_dset_name]
        virus_dset = data_file[self.virus_dset_name]

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
            for bacteria_n, virus_n in zip(range(*self.bacteria_training_interval, self.half_batch_size),
                                           range(*self.virus_training_interval, self.half_batch_size)):
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
