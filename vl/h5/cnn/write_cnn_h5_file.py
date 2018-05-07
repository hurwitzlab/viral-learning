import argparse
import itertools
import os.path
import time

from Bio import SeqIO
import h5py
import numpy as np


def cli():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--phage-fp', required=True,
                            help='path to phage FASTA file')
    arg_parser.add_argument('--prok-fp', required=True,
                            help='path to prokaryote FASTA file')
    arg_parser.add_argument('--output-h5-fp', required=True,
                            help='path to output H5 file')
    arg_parser.add_argument('--image-width', type=int, required=True,
                            help='sequence image width in nucleotides')
    arg_parser.add_argument('--image-limit', type=int, required=False, default=None,
                            help='optional limit on output images')

    args = arg_parser.parse_args()
    print('command line arguments:\n{}'.format(args))

    write_phage_prok_cnn_training_file(**vars(args))


def count_fasta_sequences(fasta_fp):
    seq_count = 0
    with open(fasta_fp, 'rt') as phage_file:
        for line in phage_file:
            if line.startswith('>'):
                seq_count += 1
            else:
                pass
    return seq_count


def first_fasta_sequence_length(fasta_fp):
    for record in SeqIO.parse(fasta_fp, "fasta"):
        return len(record.seq)


def create_dataset(h5_file, dset_name, sample_count, sequence_length, im_width):
    return h5_file.create_dataset(
        name=dset_name,
        shape=(sample_count, sequence_length - im_width + 1, im_width, 5),
        maxshape=(None, sequence_length - im_width + 1, im_width, 5),
        dtype=np.float64,
        chunks=(1, sequence_length - im_width + 1, im_width, 5),
        compression='gzip',
        compression_opts=9)


def get_start_stop(N, M):
    # N = 10
    # M = 5
    start_stop = np.zeros((N - M + 1, 2), dtype=np.int64)
    start_stop[:, 1] = M
    start_stop = start_stop + np.arange(N - M + 1, dtype=np.int64).reshape((N - M + 1, 1))
    return start_stop


def get_2D_sequence(seq, start_stop_indices):
    seq_2d = []
    for start, stop in start_stop_indices:  # get_start_stop(N=len(seq), M=M):
        seq_2d.append(seq[start:stop])
    return seq_2d


nucleotide_to_channels = {
    'A':[1.00, 0.00, 0.00, 0.00, 0.00],
    'C':[0.00, 1.00, 0.00, 0.00, 0.00],
    'G':[0.00, 0.00, 1.00, 0.00, 0.00],
    'T':[0.00, 0.00, 0.00, 1.00, 0.00],
    'U':[0.00, 0.00, 0.00, 0.00, 0.00],
    'N':[0.20, 0.20, 0.20, 0.20, 0.20],
    'R':[0.50, 0.00, 0.50, 0.00, 0.00],
    'M':[0.50, 0.50, 0.00, 0.00, 0.00], # A or C
    'S':[0.00, 0.50, 0.50, 0.00, 0.00], # C or G
    'K':[0.00, 0.00, 0.333, 0.333, 0.333], # G, T, or U
    'W':[0.333, 0.00, 0.00, 0.333, 0.333], # A, T, or U
    'Y':[0.00, 0.333, 0.00, 0.333, 0.333]} # C, T, ur U


def translate_seq_to_training_input(seq, M, start_stop_indices, verbose=False):
    ##S = 1
    N = len(seq)
    ##M = 100
    training_data = np.zeros((N-M+1, M, 5))
    for start, partial_seq in enumerate(get_2D_sequence(seq, start_stop_indices=start_stop_indices)):
        #print(partial_seq)
        for n, nucleotide in enumerate(partial_seq):
            training_data[start, n, :] = nucleotide_to_channels[nucleotide]
        if verbose:
            print(partial_seq)
            print(training_data[start, :, :])

    return training_data


def write_images_to_dataset(dset, fasta_fp, im_limit):
    max_samples, im_height, im_width, n_channels = dset.shape
    seq_length = im_height + im_width - 1
    print('max_samples     : {}'.format(max_samples))
    print('image height    : {}'.format(im_height))
    print('image width     : {}'.format(im_width))
    print('channels        : {}'.format(n_channels))
    print('sequence length : {}'.format(seq_length))

    start_stop_indices = get_start_stop(seq_length, im_width)

    # i is the current output row index
    # r is the current input row index
    # they may not be equal
    i = 0
    t0 = time.time()
    for r, record in enumerate(itertools.islice(SeqIO.parse(fasta_fp, "fasta"), im_limit)):
        if len(record.seq) != seq_length:
            print('{} record.seq length: {} != {}'.format(r, len(record.seq), seq_length))
        else:
            dset[i, :, :, :] = translate_seq_to_training_input(
                seq=str(record.seq),
                start_stop_indices=start_stop_indices,
                M=im_width)
            i += 1

        if (i + 1) % 100 == 0:
            print('finished 100 records in {:5.2f}s'.format(i, time.time() - t0))
            t0 = time.time()

    # return the number of images written to dset
    return i + 1


def write_phage_prok_cnn_training_file(phage_fp, prok_fp, output_h5_fp, image_width, image_limit=None):
    phage_seq_count = count_fasta_sequences(fasta_fp=phage_fp)
    print('{} sequences in file "{}"'.format(phage_seq_count, phage_fp))

    prok_seq_count = count_fasta_sequences(fasta_fp=prok_fp)
    print('{} sequences in file "{}"'.format(prok_seq_count, prok_fp))

    # phage_prok_pair_count = min(phage_seq_count, prok_seq_count)
    # print('allocating space for {} phage-prokaryote pairs'.format(phage_prok_pair_count))

    phage_seq_length = first_fasta_sequence_length(fasta_fp=phage_fp)
    prok_seq_length = first_fasta_sequence_length(fasta_fp=prok_fp)

    if phage_seq_length == prok_seq_length:
        seq_length = phage_seq_length
        print('phage and prokaryote sequence length is {}'.format(seq_length))
        print('image height : {}'.format(seq_length - image_width + 1))
        print('image width  : {}'.format(image_width))
    else:
        raise Exception('phage and prokaryote sequence lengths are different')

    with h5py.File(output_h5_fp, 'w') as h5_file:
        phage_dset = create_dataset(
            h5_file=h5_file,
            dset_name=os.path.basename(phage_fp),
            sample_count=phage_seq_count,
            sequence_length=seq_length,
            im_width=image_width)

        i = write_images_to_dataset(
            dset=phage_dset,
            fasta_fp=phage_fp,
            im_limit=image_limit)

        # resize the data set
        (s, m, n, c) = phage_dset.shape
        phage_dset.resize((i, m, n, c))

        prok_dset = create_dataset(
            h5_file=h5_file,
            dset_name=os.path.basename(prok_fp),
            sample_count=prok_seq_count,
            sequence_length=seq_length,
            im_width=image_width)

        i = write_images_to_dataset(
            dset=prok_dset,
            fasta_fp=prok_fp,
            im_limit=image_limit)

        # resize the data set
        (s, m, n, c) = prok_dset.shape
        phage_dset.resize((i, m, n, c))


if __name__ == '__main__':
    cli()
