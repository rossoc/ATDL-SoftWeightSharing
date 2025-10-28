"""
Keras implementation of retraining with mixture prior for soft weight sharing.

This module provides functionality for retraining models with mixture prior regularization.
"""

import tensorflow as tf
from prior import init_mixture
from data import get_mnist_data, get_cifar10_data, get_cifar100_data
from tqdm import trange


def retraining(
    model,
    epochs,
    learning_rate,
    dataset,
    prior_J=16,
    prior_pi0=0.99,
    prior_init_sigma=0.25,
    tau=0.005,
    lr_w=None,  # Learning rate for model weights
    lr_mu=None,  # Learning rate for means
    lr_sigma=None,  # Learning rate for log_sigma2
    lr_pi=None,  # Learning rate for pi_logits
    batch_size=64,
    save_dir=None,
    **prior_kwargs,
):
    """
    Retrain a model with mixture prior regularization.

    Args:
        model: The Keras model to retrain
        epochs: Number of training epochs
        learning_rate: Base learning rate (used for model weights if lr_w not specified)
        dataset: Dataset name ("mnist", "cifar10", or "cifar100")
        prior_J: Number of mixture components
        prior_pi0: Probability of zero component in the prior
        prior_init_sigma: Initial standard deviation for mixture components
        tau: Regularization coefficient for the prior
        lr_w: Learning rate for model weights (default: learning_rate)
        lr_mu: Learning rate for mixture means (default: learning_rate*10)
        lr_sigma: Learning rate for log_sigma2 (default: learning_rate*100)
        lr_pi: Learning rate for pi_logits (default: learning_rate*100)
        batch_size: Batch size for training
        save_dir: Directory to save the retrained model
        **prior_kwargs: Additional arguments to pass to MixturePrior
    """
    # Load data
    if dataset == "mnist":
        (x_train, y_train), (x_test, y_test) = get_mnist_data()
    elif dataset == "cifar10":
        (x_train, y_train), (x_test, y_test) = get_cifar10_data()
    elif dataset == "cifar100":
        (x_train, y_train), (x_test, y_test) = get_cifar100_data()
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    # Initialize the mixture prior based on the model's current weights
    prior = init_mixture(
        model, J=prior_J, pi0=prior_pi0, init_sigma=prior_init_sigma, **prior_kwargs
    )

    # Set up separate optimizers for different parameters
    lr_w = lr_w or learning_rate
    lr_mu = lr_mu or learning_rate * 10
    lr_sigma = lr_sigma or learning_rate * 100
    lr_pi = lr_pi or learning_rate * 100

    model_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_w)
    mu_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_mu)
    sigma_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_sigma)
    pi_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_pi)

    # Define training step with custom gradients
    @tf.function
    def train_step(x_batch, y_batch):
        with tf.GradientTape() as tape:
            # Compute model predictions
            predictions = model(x_batch, training=True)
            err_loss = tf.keras.losses.categorical_crossentropy(y_batch, predictions)

            # Get model weights for the prior
            model_weights = []
            for layer in model.layers:
                if hasattr(layer, "kernel"):  # Dense, Conv2D layers
                    model_weights.append(layer.kernel)
                if hasattr(layer, "bias") and layer.bias is not None:
                    model_weights.append(layer.bias)

            complex_loss = prior.complexity_loss(model_weights)
            total_loss = tf.reduce_mean(err_loss) + tau * complex_loss

        model_grads = tape.gradient(total_loss, model.trainable_variables)
        prior_grads = tape.gradient(total_loss, prior.trainable_variables)

        model_optimizer.apply_gradients(zip(model_grads, model.trainable_variables))
        mu_optimizer.apply_gradients([(prior_grads[0], prior.trainable_variables[0])])
        sigma_optimizer.apply_gradients(
            [(prior_grads[1], prior.trainable_variables[1])]
        )
        pi_optimizer.apply_gradients([(prior_grads[2], prior.trainable_variables[2])])

        return total_loss, err_loss, complex_loss

    # Create dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Training loop with tqdm
    pbar = trange(epochs, desc="Retraining 0/0", leave=True)

    for epoch in pbar:
        epoch_total_loss = 0
        epoch_err_loss = 0
        epoch_complex_loss = 0
        num_batches = 0

        for x_batch, y_batch in train_dataset:
            total_loss, err_loss, complex_loss = train_step(x_batch, y_batch)
            epoch_total_loss += total_loss
            epoch_err_loss += tf.reduce_mean(err_loss)
            epoch_complex_loss += complex_loss
            num_batches += 1

        avg_total_loss = epoch_total_loss / num_batches
        avg_err_loss = epoch_err_loss / num_batches
        avg_complex_loss = epoch_complex_loss / num_batches

        # Update progress bar with losses
        pbar.set_description(f"Retraining {epoch + 1}/{epochs}")
        pbar.set_postfix(
            {
                "Total Loss": f"{avg_total_loss:.4f}",
                "Err Loss": f"{avg_err_loss:.4f}",
                "Complex Loss": f"{avg_complex_loss:.4f}",
            }
        )

    # Compile the model with standard optimizer after training for evaluation
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Evaluate the retrained model
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy after retraining with prior: {test_accuracy:.4f}")

    # Save the retrained model
    if save_dir:
        model.save(filepath=save_dir)
        print(f"Retrained model saved to {save_dir}")

    return model
