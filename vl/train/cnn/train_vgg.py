"""
Train VGG-like network on sequence images.

usage:

    python train_vgg.py \
        --input-fp /extra/jklynch/phage_proc_500.h5 \
        --train phage:0:80000,prok:0:80000 \
        --dev phage:80000:90000,prok:80000:90000 \
        --test phage:90000:,prok:90000: \
        --epochs 10



"""
import argparse
import sys

from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Model

from vl.model import training
from vl.train.cnn.data import CNNData


def get_args(argv):
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--input-fp', required=True,
                            help='path to training file')
    arg_parser.add_argument('--train', required=True,
                            help='training set dataset names and ranges, e.g. phage:0:80000,prok:0:80000')
    arg_parser.add_argument('--dev', required=True,
                            help='dev set dataset names and ranges, e.g. phage:80000:90000,prok:80000:9000')
    arg_parser.add_argument('--test', required=True,
                            help='test set dataset names and ranges, e.g. phage:90000:,prok:90000:')
    arg_parser.add_argument('--epochs', required=True, type=int,
                            help='number of training epochs')

    args = arg_parser.parse_args(args=argv)
    print(args)

    return args


def parse_command_line_name_and_range(name_and_range_arg):
    """
    Parse command line argument that looks like "phage:0:80000,prok:0:70000" into
    {
      phage_dset_name: "phage",
      phage_range: (0,80000),
      prok_dset_name: "prok",
      prok_range: (0,70000)
    }

    :param name_and_range: (str) "phage:0:80000,prok:0:70000"
    :return: (dict)
    """
    phage_arg, prok_arg = name_and_range_arg.split(',')
    phage_dset_name, phage_start, phage_stop = phage_arg.split(':')
    prok_dset_name, prok_start, prok_stop = prok_arg.split(':')

    names_and_ranges = {
        'phage_dset_name': phage_dset_name,
        'phage_range': (int(phage_start), int(phage_stop)),
        'prok_dset_name': prok_dset_name,
        'prok_range': (int(prok_start), int(prok_stop))
    }

    return names_and_ranges


def get_vgg_16_model(input_shape):
    """

    :return:
    """

    img_input = Input(shape=input_shape)

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(1, activation='sigmoid', name='predictions')(x)

    vgg16 = Model(img_input, x, name='vgg16')

    vgg16.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy'])

    return vgg16


def train_vgg(input_fp, train, dev, test, epochs):
    names_and_ranges = parse_command_line_name_and_range(train)

    data = CNNData(
        input_fp=input_fp,
        batch_size=64,
        phage_dset_name=names_and_ranges['phage_dset_name'],
        phage_training_range=names_and_ranges['phage_range'],
        prok_dset_name=names_and_ranges['prok_dset_name'],
        prok_training_range=names_and_ranges['prok_range'])

    model = get_vgg_16_model(input_shape=data.get_input_shape())

    # train the nn
    training.train_and_evaluate(
        model=model,
        model_name='VGG_almost',
        training_epochs=epochs,
        the_data=data)


def main(argv):
    args = get_args(argv)

    train_vgg(**vars(args))


def cli():
    main(sys.argv[1:])


if __name__ == '__main__':
    cli()
