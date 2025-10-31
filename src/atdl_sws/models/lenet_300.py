from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.regularizers import L1
from tensorflow.keras.initializers import RandomNormal


def lenet_300_100(num_classes=10, regularizer=1e-1, std_init=0.01):
    gaussian_init = RandomNormal(mean=0.0, stddev=std_init)
    model = Sequential(
        [
            Input(shape=(28, 28)),
            Flatten(),
            Dense(
                300,
                activation="relu",
                kernel_regularizer=L1(regularizer),
                kernel_initializer=gaussian_init,
            ),
            Dense(
                100,
                activation="relu",
                kernel_regularizer=L1(regularizer),
                kernel_initializer=gaussian_init,
            ),
            Dense(num_classes, activation="softmax", kernel_initializer=gaussian_init),
        ]
    )
    return model
