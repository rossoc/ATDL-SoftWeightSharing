from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input, Conv2D, MaxPooling2D
from tensorflow.keras.regularizers import L1
from tensorflow.keras.initializers import RandomNormal


def lenet_caffe(num_classes=10, regularizer=1e-1, std_init=0.01):
    gaussian_init = RandomNormal(mean=0.0, stddev=std_init)
    model = Sequential(
        [
            Input(shape=(28, 28, 1)),
            Conv2D(
                20,
                kernel_size=5,
                activation="relu",
                kernel_regularizer=L1(regularizer),
                kernel_initializer=gaussian_init,
            ),
            MaxPooling2D(pool_size=2),
            Conv2D(
                50, 
                kernel_size=5, 
                activation="relu", 
                kernel_regularizer=L1(regularizer),
                kernel_initializer=gaussian_init
            ),
            MaxPooling2D(pool_size=2),
            Flatten(),
            Dense(
                500, 
                activation="relu", 
                kernel_regularizer=L1(regularizer),
                kernel_initializer=gaussian_init
            ),
            Dense(
                num_classes, 
                activation="softmax",
                kernel_initializer=gaussian_init
            ),
        ]
    )
    return model
