import h5py
import numpy as np

from vl.data import load_kmer_random_batches_h5, load_kmer_range_batches_h5


def test_load_kmer_range_batches_h5():
    with h5py.File('unit_test.h5', 'w') as test_data:
        dset1_shape = (12, 2)
        dset1 = test_data.create_dataset('/test/data1', dset1_shape, dtype=np.float64)
        dset1_array = np.arange(np.product(dset1_shape)).reshape(dset1_shape)
        dset1[:, :] = dset1_array

        # dset1 looks like this:
        #  [[0.   1.]
        #   [2.   3.]
        #   [4.   5.]
        #   [6.   7.]
        #   [8.   9.]
        #   [10.  11.]
        #   [12.  13.]
        #   [14.  15.]
        #   [16.  17.]
        #   [18.  19.]
        #   [20.  21.]
        #   [22.  23.]]

        dset2_shape = (10, 2)
        dset2 = test_data.create_dataset('/test/data2', dset2_shape, dtype=np.float64)
        dset2_array = np.arange(np.product(dset2_shape)).reshape(dset2_shape) + np.product(dset2_shape)
        dset2[:, :] = dset2_array

        # dset2 looks like this:
        #  [[20.  21.]
        #   [22.  23.]
        #   [24.  25.]
        #   [26.  27.]
        #   [28.  29.]
        #   [30.  31.]
        #   [32.  33.]
        #   [34.  35.]
        #   [36.  37.]
        #   [38.  39.]]

    with h5py.File('unit_test.h5', 'r') as test_data:
        dset1 = test_data['/test/data1']
        dset2 = test_data['/test/data2']

        gen1 = load_kmer_range_batches_h5(
            name='gen1',
            bacteria_dset=dset1,
            virus_dset=dset2,
            bacteria_range=(0, dset1_shape[0]),
            virus_range=(0, dset2.shape[0]),
            half_batch_size=5,
            shuffle_batch=False)

        batch1, labels1 = gen1.__next__()
        batch2, labels2 = gen1.__next__()
        batch3, labels3 = gen1.__next__()

        assert np.all(batch1 == np.vstack((dset1_array[:5, :], dset2_array[:5, :])))
        assert np.all(labels1 == [[0], [0], [0], [0], [0], [1], [1], [1], [1], [1]])

        assert np.all(batch2 == np.vstack((dset1_array[5:10, :], dset2_array[5:10, :])))
        assert np.all(labels2 == [[0], [0], [0], [0], [0], [1], [1], [1], [1], [1]])

        assert np.all(batch3 == np.vstack((dset1_array[:5, :], dset2_array[:5, :])))
        assert np.all(labels3 == [[0], [0], [0], [0], [0], [1], [1], [1], [1], [1]])

        np.random.seed(1)

        gen2 = load_kmer_range_batches_h5(
            name='gen2',
            bacteria_dset=dset1,
            virus_dset=dset2,
            bacteria_range=(0, dset1_shape[0]),
            virus_range=(0, dset2.shape[0]),
            half_batch_size=5,
            shuffle_batch=True)

        batch1, labels1 = gen2.__next__()
        batch2, labels2 = gen2.__next__()
        batch3, labels3 = gen2.__next__()

        assert batch1.shape == (10, 2)
        assert np.all(batch1 == np.vstack((
            dset1_array[2, :],
            dset2_array[4, :],
            dset2_array[1, :],
            dset1_array[4, :],
            dset1_array[0, :],
            dset1_array[3, :],
            dset1_array[1, :],
            dset2_array[2, :],
            dset2_array[3, :],
            dset2_array[0, :])))
        assert labels1.shape == (10, 1)
        assert np.all(labels1 == [[0], [1], [1], [0], [0], [0], [0], [1] ,[1] ,[1]])

        assert batch2.shape == (10, 2)
        assert np.all(batch2 == np.vstack((
            dset2_array[9, :],
            dset2_array[5, :],
            dset1_array[8, :],
            dset1_array[5, :],
            dset2_array[8, :],
            dset1_array[9, :],
            dset1_array[7, :],
            dset1_array[6, :],
            dset2_array[6, :],
            dset2_array[7, :])))
        assert labels2.shape == (10, 1)
        assert np.all(labels2 == [[1], [1], [0], [0], [1], [0], [0], [0] ,[1] ,[1]])

        assert batch3.shape == (10, 2)
        assert np.all(batch3 == np.vstack((
            dset2_array[3, :],
            dset1_array[3, :],
            dset2_array[0, :],
            dset2_array[4, :],
            dset1_array[0, :],
            dset2_array[1, :],
            dset1_array[1, :],
            dset2_array[2, :],
            dset1_array[4, :],
            dset1_array[2, :])))
        assert labels3.shape == (10, 1)
        assert np.all(labels3 == [[1], [0], [1], [1], [0], [1], [0], [1] ,[0] ,[0]])


def test_load_kmer_random_batches_h5():
    with h5py.File('unit_test.h5', 'w') as test_data:
        dset1_shape = (12, 2)
        dset1 = test_data.create_dataset('/test/data1', dset1_shape, dtype=np.float64)
        dset1_array = np.arange(np.product(dset1_shape)).reshape(dset1_shape)
        dset1[:, :] = dset1_array

        # dset1 looks like this:
        #  [[0.   1.]
        #   [2.   3.]
        #   [4.   5.]
        #   [6.   7.]
        #   [8.   9.]
        #   [10.  11.]
        #   [12.  13.]
        #   [14.  15.]
        #   [16.  17.]
        #   [18.  19.]
        #   [20.  21.]
        #   [22.  23.]]

        dset2_shape = (10, 2)
        dset2 = test_data.create_dataset('/test/data2', dset2_shape, dtype=np.float64)
        dset2_array = np.arange(np.product(dset2_shape)).reshape(dset2_shape) + np.product(dset2_shape)
        dset2[:, :] = dset2_array

        # dset2 looks like this:
        #  [[20.  21.]
        #   [22.  23.]
        #   [24.  25.]
        #   [26.  27.]
        #   [28.  29.]
        #   [30.  31.]
        #   [32.  33.]
        #   [34.  35.]
        #   [36.  37.]
        #   [38.  39.]]

    with h5py.File('unit_test.h5', 'r') as test_data:
        dset1 = test_data['/test/data1']
        dset2 = test_data['/test/data2']

        gen1 = load_kmer_random_batches_h5(
            name='gen1',
            bacteria_dset=dset1,
            virus_dset=dset2,
            bacteria_subsample=np.arange(dset1.shape[0]),
            virus_subsample=np.arange(dset2.shape[0]),
            half_batch_size=5,
            shuffle_batch=False)

        batch1, labels1 = gen1.__next__()
        batch2, labels2 = gen1.__next__()
        batch3, labels3 = gen1.__next__()

        print(batch2)

        assert np.all(batch1 == np.vstack((dset1_array[:5, :], dset2_array[:5, :])))
        assert np.all(labels1 == [[0], [0], [0], [0], [0], [1], [1], [1], [1], [1]])

        assert np.all(batch2 == np.vstack((dset1_array[5:10, :], dset2_array[5:10, :])))
        assert np.all(labels2 == [[0], [0], [0], [0], [0], [1], [1], [1], [1], [1]])

        assert np.all(batch3 == np.vstack((dset1_array[:5, :], dset2_array[:5, :])))
        assert np.all(labels3 == [[0], [0], [0], [0], [0], [1], [1], [1], [1], [1]])

        np.random.seed(1)

        gen2 = load_kmer_random_batches_h5(
            name='gen2',
            bacteria_dset=dset1,
            virus_dset=dset2,
            bacteria_subsample=np.arange(dset1_shape[0]),
            virus_subsample=np.arange(dset2.shape[0]),
            half_batch_size=5,
            shuffle_batch=True)

        batch1, labels1 = gen2.__next__()
        batch2, labels2 = gen2.__next__()
        batch3, labels3 = gen2.__next__()

        assert batch1.shape == (10, 2)
        assert np.all(batch1 == np.vstack((
            dset1_array[2, :],
            dset2_array[4, :],
            dset2_array[1, :],
            dset1_array[4, :],
            dset1_array[0, :],
            dset1_array[3, :],
            dset1_array[1, :],
            dset2_array[2, :],
            dset2_array[3, :],
            dset2_array[0, :])))
        assert labels1.shape == (10, 1)
        assert np.all(labels1 == [[0], [1], [1], [0], [0], [0], [0], [1] ,[1] ,[1]])

        assert batch2.shape == (10, 2)
        assert np.all(batch2 == np.vstack((
            dset2_array[9, :],
            dset2_array[5, :],
            dset1_array[8, :],
            dset1_array[5, :],
            dset2_array[8, :],
            dset1_array[9, :],
            dset1_array[7, :],
            dset1_array[6, :],
            dset2_array[6, :],
            dset2_array[7, :])))
        assert labels2.shape == (10, 1)
        assert np.all(labels2 == [[1], [1], [0], [0], [1], [0], [0], [0] ,[1] ,[1]])

        assert batch3.shape == (10, 2)
        assert np.all(batch3 == np.vstack((
            dset2_array[3, :],
            dset1_array[3, :],
            dset2_array[0, :],
            dset2_array[4, :],
            dset1_array[0, :],
            dset2_array[1, :],
            dset1_array[1, :],
            dset2_array[2, :],
            dset1_array[4, :],
            dset1_array[2, :])))
        assert labels3.shape == (10, 1)
        assert np.all(labels3 == [[1], [0], [1], [1], [0], [1], [0], [1] ,[0] ,[0]])
