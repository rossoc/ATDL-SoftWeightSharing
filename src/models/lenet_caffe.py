from keras.models import Sequential
from keras.layers import Dense, Flatten, Input, Conv2D, MaxPooling2D


def lenet_caffe(num_classes=10):
    model = Sequential(
        [
            Input(shape=(28, 28, 1)),
            Conv2D(20, kernel_size=5, activation="relu"),
            MaxPooling2D(pool_size=2),
            Conv2D(50, kernel_size=5, activation="relu"),
            MaxPooling2D(pool_size=2),
            Flatten(),
            Dense(500, activation="relu"),
            Dense(num_classes, activation="softmax"),
        ]
    )
    return model
