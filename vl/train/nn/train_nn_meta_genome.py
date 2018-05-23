"""

usage:

python train_nn_meta_genome.py \
    --input-fp /home/jklynch/host/project/viral-learning/data/perm_training_testing.h5 \
    --training-samples 1000 \
    --dev-samples 2000 \
    --half-batch-size 50 \
    --epoch-count 5

python train_nn_meta_genome.py \
    --input-fp /extra/jklynch/perm_training_testing.h5 \
    --training-samples 100000 \
    --dev-samples 10000 \
    --half-batch-size 50 \
    --epoch-count 5

"""


import argparse
import sys

from pprint import pprint

from keras import Sequential

from vl.model.training import train_and_evaluate, build_model
from vl.data.kmers import BacteriaAndVirusKMers


def get_args(argv):
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--input-fp', required=True,
                            help='path to training file')
    arg_parser.add_argument('--training-sample-count', type=int, required=True,
                            help='number of samples for training')
    arg_parser.add_argument('--dev-sample-count', type=int, required=True,
                            help='number of samples for development')
    #arg_parser.add_argument('--test', required=True,
    #                        help='test set dataset names and ranges, e.g. phage:90000:,prok:90000:')
    arg_parser.add_argument('--half-batch-count', required=True, type=int,
                            help='one half the number of samples per batch')
    arg_parser.add_argument('--epoch-count', required=True, type=int,
                            help='number of training epochs')

    args = arg_parser.parse_args(args=argv)
    print(args)

    return args


def main():
    args = get_args(sys.argv[1:])

    first_stage_network_depths = (
        (('Dense', {'units': 128, 'activation': 'relu'}),
         ('Dropout', {'rate': 0.2}),
         ('Dense', {'units': 64, 'activation': 'relu'}),
         ('Dropout', {'rate': 0.2}),
         ('Dense', {'units': 32, 'activation': 'relu'}),
         ('Dense', {'units': 1, 'activation': 'sigmoid'})),
    )

    first_stage_data = BacteriaAndVirusKMers(
        fp=args.input_fp,  # '/home/jklynch/host/project/viral-learning/data/perm_training_testing.h5',
        training_sample_count=args.training_sample_count,  # 1000,
        development_sample_count=args.dev_sample_count,  # 2000,
        half_batch_size=args.half_batch_count)  # 50

    first_stage_model_name, first_stage_model = build_model(
        model=Sequential(),
        input_dim=first_stage_data.get_input_dim(),
        layers=first_stage_network_depths[0])

    training_metrics_df, dev_metrics_df = train_and_evaluate(
        model=first_stage_model,
        model_name=first_stage_model_name,
        training_epochs=args.epoch_count,
        the_data=first_stage_data
    )

    pprint(first_stage_model.get_config())

    # store the model
    with open(first_stage_model_name + '.json', 'wt') as model_json:
        model_json.write(first_stage_model.to_json())

    first_stage_model.save_weights(filepath=first_stage_model_name + '.h5', overwrite=True)


if __name__ == '__main__':
    main()
