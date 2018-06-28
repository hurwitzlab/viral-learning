"""
Train VGG-like network on SYNTHETIC sequence images.

'Phage' images will have only A,T.
'Proka' images will have only C,G.

usage:

    python train_vgg.py \
        --seed 1 \
        --bp 500 \
        --train phage:1000,proka:1000 \
        --dev phage:1000,proka:1000 \
        --test phage:1000,proka:1000: \
        --epochs 2



"""
import argparse
import sys

from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Model

from vl.model import training
from vl.train.cnn.synthetic.data import CNNSyntheticData


def get_args(argv):
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--seed', type=int, required=True,
                            help='random seed')
    arg_parser.add_argument('--bp', type=int, required=True,
                            help='number of base pairs per sequence')
    arg_parser.add_argument('--batch-size', type=int, required=True,
                            help='number of samples per training batch')
    arg_parser.add_argument('--train', required=True,
                            help='training set dataset names and ranges, e.g. phage:1000,proka:1000')
    arg_parser.add_argument('--dev', required=True,
                            help='dev set dataset names and ranges, e.g. phage:1000,proka:1000')
    arg_parser.add_argument('--test', required=True,
                            help='test set dataset names and ranges, e.g. phage:1000,proka:1000')
    arg_parser.add_argument('--epochs', required=True, type=int,
                            help='number of training epochs')

    args = arg_parser.parse_args(args=argv)
    print(args)

    return args


def parse_command_line_name_and_range(name_and_count_arg):
    """
    Parse command line argument that looks like "phage:1000,proka:1000" into
    {
      phage_dset_name: "phage",
      phage_count: (0, 1000),
      prok_dset_name: "proka",
      proka_count: (0, 1000)
    }

    :param name_and_count: (str) "phage:1000,proka:1000"
    :return: (dict)
    """
    phage_arg, prok_arg = name_and_count_arg.split(',')
    phage_dset_name, phage_stop = phage_arg.split(':')
    proka_dset_name, proka_stop = prok_arg.split(':')

    names_and_counts = {
        'phage_dset_name': phage_dset_name,
        'phage_range': (0, int(phage_stop)),
        'proka_dset_name': proka_dset_name,
        'proka_range': (0, int(proka_stop))
    }

    return names_and_counts


def get_lenet_model(input_shape):
    """

    :param input_shape:
    :return:
    """

    img_input = Input(shape=input_shape)

    # first set of CONV => RELU => POOL
    x = Conv2D(20, (5, 5), activation='relu', padding="same", name='first_conv')(img_input)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='first_max_pool')(x)


    # second set of CONV => RELU => POOL
    x = Conv2D(50, (5, 5), activation='relu', padding="same", name='second_conv')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='second_max_pool')(x)

    # set of FC => RELU layers
    x = Flatten(name='flatten')(x)
    x = Dense(500, activation='relu', name='fc1')(x)

    x = Dense(1, activation='sigmoid', name='predictions')(x)

    lenet = Model(img_input, x, name='lenet')

    lenet.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy'])

    lenet.summary()

    return lenet


def get_vgg_16_model(input_shape):
    """
    This model started to work with 10000 training samples.
    With 1000 training samples it did not work.
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

    vgg16.summary()

    return vgg16


def train_vgg(seed, bp, batch_size, train, dev, test, epochs):
    training_names_and_ranges = parse_command_line_name_and_range(train)
    dev_parameters = parse_command_line_name_and_range(dev)

    data = CNNSyntheticData(
        seed=seed,
        bp=bp,
        batch_size=batch_size,
        phage_dset_name=training_names_and_ranges['phage_dset_name'],
        phage_training_range=training_names_and_ranges['phage_range'],
        proka_dset_name=training_names_and_ranges['proka_dset_name'],
        proka_training_range=training_names_and_ranges['proka_range'],
        dev_parameters=dev_parameters)

    #model = get_vgg_16_model(input_shape=data.get_input_shape())
    model = get_lenet_model(input_shape=data.get_input_shape())

    # train the nn
    training.train_and_evaluate(
        model=model,
        model_name='VGG_synthetic_data',
        training_epochs=epochs,
        the_data=data)


def main(argv):
    args = get_args(argv)

    train_vgg(**vars(args))


def cli():
    main(sys.argv[1:])


if __name__ == '__main__':
    cli()
