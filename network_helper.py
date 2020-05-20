#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 20:30, 20/05/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
# -------------------------------------------------------------------------------------------------------%
"""
    Utility used by the Network class to actually train.
    Based on: https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py
"""
from keras.datasets import mnist, cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
early_stopper = EarlyStopping(patience=5)               # Helper: Early stopping.


def get_dataset(dataset="mnist"):
    nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test = None, None, None, None, None, None, None

    if dataset == "cifar10":    # Retrieve the CIFAR dataset and process the data.
        # Set defaults.
        nb_classes = 10
        batch_size = 64
        input_shape = (3072,)

        # Get the data.
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = x_train.reshape(50000, 3072)
        x_test = x_test.reshape(10000, 3072)
        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255

        # convert class vectors to binary class matrices
        y_train = to_categorical(y_train, nb_classes)
        y_test = to_categorical(y_test, nb_classes)

    elif dataset == "mnist":
        """Retrieve the MNIST dataset and process the data."""
        # Set defaults.
        nb_classes = 10
        batch_size = 128
        input_shape = (784,)

        # Get the data.
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.reshape(60000, 784)
        x_test = x_test.reshape(10000, 784)
        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255

        # convert class vectors to binary class matrices
        y_train = to_categorical(y_train, nb_classes)
        y_test = to_categorical(y_test, nb_classes)

    else:
        print("This project not support: {} dataset".format(dataset))
        exit(0)

    return nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test


def compile_model(paras, nb_classes, input_shape):
    """Compile a sequential model.
    Args:
        input_shape (tuple):
        nb_classes (int):
        paras (dict): the parameters of the network
    Returns:
        a compiled network.
    """
    # Get our network parameters.
    nb_layers = paras['nb_layers']
    nb_neurons = paras['nb_neurons']
    activation = paras['activation']
    optimizer = paras['optimizer']
    dropout = paras['dropout']

    model = Sequential()
    model.add(Dense(nb_neurons, activation=activation, input_shape=input_shape))            # input layers

    # Add each layer.                                                                       # Hidden layers
    for i in range(nb_layers):
        model.add(Dense(nb_neurons, activation=activation))
        model.add(Dropout(dropout))  # hard-coded dropout

    model.add(Dense(nb_classes, activation='softmax'))                                      # Output layer.
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def train_and_score(paras, dataset):
    """Train the model, return test loss.
    Args:
        paras (dict): the parameters of the network
        dataset (str): Dataset to use for training/evaluating
    """
    nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test = get_dataset(dataset)

    model = compile_model(paras, nb_classes, input_shape)

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=1000,  # using early stopping, so no real limit
              verbose=2,
              validation_data=(x_test, y_test),
              callbacks=[early_stopper])

    score = model.evaluate(x_test, y_test, verbose=0)

    return score[1]  # 1 is accuracy. 0 is loss.
