from keras.models import Sequential
from keras.layers import Dense, Flatten, Input


def lenet_300_100(num_classes=10):
    model = Sequential(
        [
            Input(shape=(28, 28)),
            Flatten(),
            Dense(300, activation="relu"),
            Dense(100, activation="relu"),
            Dense(num_classes, activation="softmax"),
        ]
    )
    return model
