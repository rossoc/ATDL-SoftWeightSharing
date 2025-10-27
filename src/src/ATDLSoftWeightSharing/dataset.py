"""
Loading the MNIST in the right format.

Implementation is close to Keras Tutorial on CNNs.

Karen Ullrich, Jan 2017
"""

import numpy as np
from tensorflow import keras


def mnist():
    img_rows, img_cols = 28, 28
    nb_classes = 10
    # the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

    # Reshape for TensorFlow format
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    print("Successfully loaded %d train samples and %d test samples." % (X_train.shape[0], X_test.shape[0]))

    # convert class vectors to binary class matrices
    Y_train = keras.utils.to_categorical(y_train, nb_classes)
    Y_test = keras.utils.to_categorical(y_test, nb_classes)

    return [X_train, X_test], [Y_train, Y_test], [img_rows, img_cols], nb_classes