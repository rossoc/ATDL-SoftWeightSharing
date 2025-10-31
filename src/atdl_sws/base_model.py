from .data import get_mnist_data, get_cifar10_data, get_cifar100_data
from .csv_logger import CSVLogger, format_seconds
import tensorflow as tf
import time
from tqdm import tqdm


def pretraining(
    model,
    epochs,
    learning_rate,
    dataset,
    batch_size=64,
    save_dir=None,
    logger=None,  # optional CSVLogger
    eval_every=1,  # how often to evaluate
):
    """
    Pretrain a model with standard training.

    Args:
        model: The Keras model to pretrain
        epochs: Number of training epochs
        learning_rate: Learning rate for training
        dataset: Dataset name ("mnist", "cifar10", or "cifar100")
        batch_size: Batch size for training
        save_dir: Directory to save the pretrained model
        logger: Optional CSVLogger to log metrics
        eval_every: How often to evaluate the model (default: every epoch)
    """
    if dataset == "mnist":
        (x_train, y_train), (x_test, y_test) = get_mnist_data()
    elif dataset == "cifar10":
        (x_train, y_train), (x_test, y_test) = get_cifar10_data()
    elif dataset == "cifar100":
        (x_train, y_train), (x_test, y_test) = get_cifar100_data()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    t0 = time.time()

    # Create a custom training loop to support logging and visualization
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Set up the optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Define the training step
    @tf.function(jit_compile=True)
    def train_step(x_batch, y_batch):
        with tf.GradientTape() as tape:
            predictions = model(x_batch, training=True)
            loss = tf.keras.losses.categorical_crossentropy(y_batch, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return tf.reduce_mean(loss)

    # Training loop
    for epoch in range(epochs):
        epoch_loss = 0
        num_batches = 0

        pbar = tqdm(train_dataset, desc="Pretraining 0/0", leave=True)
        for x_batch, y_batch in pbar:
            loss = train_step(x_batch, y_batch)
            epoch_loss += loss
            num_batches += 1

        avg_loss = epoch_loss / num_batches

        # Evaluate model if needed
        test_acc = None
        if eval_every > 0 and (epoch + 1) % eval_every == 0:
            test_acc = evaluate(model, (x_test, y_test))

        # Visualization hook

        # Log metrics if logger provided
        if logger is not None:
            logger.log(
                {
                    "phase": "pretrain",
                    "epoch": epoch + 1,
                    "train_ce": float(avg_loss.numpy()),
                    "complexity": "",  # Not applicable for pretraining
                    "total_loss": float(avg_loss.numpy()),
                    "tau": "",  # Not applicable for pretraining
                    "test_acc": ("" if test_acc is None else f"{test_acc:.4f}"),
                    "elapsed": format_seconds(time.time() - t0),
                }
            )

        pbar.set_postfix(
            {
                "loss": f"{avg_loss:.4f}",
                "test_acc": ("" if test_acc is None else f"{test_acc:.4f}"),
            }
        )

    if save_dir:
        model.save(filepath=save_dir + "/pretrained.keras")

    return model


def evaluate(model, test_data):
    """
    Evaluate the model on test data.
    Args:
        model: The Keras model to evaluate
        test_data: Tuple of (x_test, y_test)
    Returns:
        test_accuracy: The accuracy of the model on test data
    """
    x_test, y_test = test_data
    _, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    return test_accuracy
