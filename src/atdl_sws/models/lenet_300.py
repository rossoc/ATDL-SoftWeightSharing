from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.regularizers import L1


def lenet_300_100(num_classes=10, regularizer=0):
    model = Sequential(
        [
            Input(shape=(28, 28)),
            Flatten(),
            Dense(300, activation="relu", kernel_regularizer=L1(regularizer)),
            Dense(100, activation="relu", kernel_regularizer=L1(regularizer)),
            Dense(num_classes, activation="softmax"),
        ]
    )
    return model
