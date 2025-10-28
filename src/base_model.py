from .data import get_mnist_data, get_cifar10_data, get_cifar100_data
import keras


def pretraining(model, epochs, learning_rate, dataset, batch_size=64, save_dir=None):
    if dataset == "mnist":
        (x_train, y_train), (x_test, y_test) = get_mnist_data()
    elif dataset == "cifar10":
        (x_train, y_train), (x_test, y_test) = get_cifar10_data()
    elif dataset == "cifar100":
        (x_train, y_train), (x_test, y_test) = get_cifar100_data()

    epochs = 20

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    _ = model.fit(
        x_train,  # type: ignore
        y_train,  # type: ignore
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,  # type: ignore
        # validation_data=(x_test, y_test),  # type: ignore
    )

    if save_dir:
        model.save(filepath=save_dir)
