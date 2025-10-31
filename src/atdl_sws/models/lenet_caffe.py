from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input, Conv2D, MaxPooling2D
from tensorflow.keras.regularizers import L1


def lenet_caffe(num_classes=10, regularizer=0):
    model = Sequential(
        [
            Input(shape=(28, 28, 1)),
            Conv2D(
                20, kernel_size=5, activation="relu", kernel_regularizer=L1(regularizer)
            ),
            MaxPooling2D(pool_size=2),
            Conv2D(
                50, kernel_size=5, activation="relu", kernel_regularizer=L1(regularizer)
            ),
            MaxPooling2D(pool_size=2),
            Flatten(),
            Dense(500, activation="relu", kernel_regularizer=L1(regularizer)),
            Dense(num_classes, activation="softmax"),
        ]
    )
    return model
