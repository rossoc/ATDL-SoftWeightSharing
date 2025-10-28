"""
Keras implementation of retraining with mixture prior for soft weight sharing.

This module provides functionality for retraining models with mixture prior regularization.
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tqdm import tqdm
from .csv_logger import format_seconds
from .prior import flatten_weights
import time


def retraining(
    model,
    prior,
    epochs,
    dataset,
    lr_w=1e-4,  # Learning rate for model weights
    lr_mu=1e-4,  # Learning rate for means
    lr_sigma=3e-3,  # Learning rate for log_sigma2
    lr_pi=3e-3,  # Learning rate for pi_logits
    save_dir=None,
    tau=0.005,
    batch_size=64,
    viz=None,  # optional TrainingGifVisualizer
    logger=None,  # optional CSVLogger
    eval_every=1,  # how often to evaluate
    eval=None,
    verbose=1,
):
    """
    Retrain a model with mixture prior regularization.

    Args:
        model: The Keras model to retrain
        prior: The MixtureGaussian prior
        epochs: Number of training epochs
        dataset: Dataset name ("mnist", "cifar10", or "cifar100")
        tau: Regularization coefficient for the prior
        lr_w: Learning rate for model weights
        lr_mu: Learning rate for mixture means
        lr_sigma: Learning rate for log_sigma2
        lr_pi: Learning rate for pi_logits
        batch_size: Batch size for training
        save_dir: Directory to save the retrained model
        viz: Optional TrainingGifVisualizer to create animated visualization
        logger: Optional CSVLogger to log metrics
        eval_every: How often to evaluate the model (default: every epoch)
        **prior_kwargs: Additional arguments to pass to MixturePrior
    """
    x_train, y_train = dataset

    if eval:
        x_test, y_test = eval
    # Build the prior layer to initialize its variables
    # We need to build the layer to create the trainable variables before using them

    # Set up separate optimizers for different parameters
    lr_w = lr_w
    lr_mu = lr_mu
    lr_sigma = lr_sigma
    lr_pi = lr_pi

    model_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_w)
    mu_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_mu)
    sigma_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_sigma)
    pi_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_pi)

    # Define training step with custom gradients
    @tf.function(jit_compile=True)
    def train_step(x_batch, y_batch):
        with tf.GradientTape(persistent=True) as tape:
            # Compute model predictions
            predictions = model(x_batch, training=True)
            err_loss = tf.keras.losses.categorical_crossentropy(y_batch, predictions)

            # Get model weights for the prior
            model_weights = [
                layer.kernel for layer in model.layers if hasattr(layer, "kernel")
            ]

            w_flatten = flatten_weights(model_weights)

            complex_loss = prior.complexity_loss(w_flatten)
            total_loss = tf.reduce_mean(err_loss) + tau * complex_loss

        model_grads = tape.gradient(total_loss, model.trainable_variables)
        prior_grads = tape.gradient(total_loss, prior.trainable_variables)

        model_optimizer.apply_gradients(zip(model_grads, model.trainable_variables))
        mu_optimizer.apply_gradients([(prior_grads[0], prior.trainable_variables[0])])
        sigma_optimizer.apply_gradients(
            [(prior_grads[1], prior.trainable_variables[1])]
        )
        pi_optimizer.apply_gradients([(prior_grads[2], prior.trainable_variables[2])])

        # Delete the persistent tape to avoid memory leaks
        del tape

        return total_loss, err_loss, complex_loss

    # Create dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # --- Optional: epoch 0 visual frame & baseline accuracy
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_w),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    if eval:
        _, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

        if viz is not None:
            viz.on_train_begin(model, prior, total_epochs=epochs)
            viz.on_epoch_end(0, model, prior, test_acc=test_accuracy)

    # Training loop with tqdm
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        epoch_total_loss = 0
        epoch_err_loss = 0
        epoch_complex_loss = 0
        num_batches = 0

        pbar = (
            tqdm(
                train_dataset,
                desc=f"Retraining {epoch}/{epochs}",
                leave=True,
            )
            if verbose > 0
            else train_dataset
        )

        for x_batch, y_batch in pbar:
            total_loss, err_loss, complex_loss = train_step(x_batch, y_batch)
            epoch_total_loss += total_loss
            epoch_err_loss += tf.reduce_mean(err_loss)
            epoch_complex_loss += complex_loss
            num_batches += 1

            if verbose > 0:
                pbar.set_postfix(
                    {
                        "Total Loss": f"{float(tf.reduce_mean(total_loss).numpy()):.4f}",
                        "Err Loss": f"{float(tf.reduce_mean(err_loss).numpy()):.4f}",
                        "Complex Loss": f"{float(tf.reduce_mean(complex_loss).numpy()):.4f}",
                    }
                )

        avg_total_loss = tf.reduce_mean(epoch_total_loss / num_batches)
        avg_err_loss = tf.reduce_mean(epoch_err_loss / num_batches)
        avg_complex_loss = tf.reduce_mean(epoch_complex_loss / num_batches)

        # Evaluate model if needed
        test_acc = None
        if eval and eval_every > 0 and (epoch + 1) % eval_every == 0:
            _, test_acc = model.evaluate(x_test, y_test, verbose=0)

        if viz:
            viz.on_epoch_end(epoch + 1, model, prior, test_acc=test_acc)

        # Log metrics if logger provided
        if logger is not None:
            logger.log(
                {
                    "phase": "retrain",
                    "epoch": epoch + 1,
                    "train_ce": float(avg_err_loss.numpy()),
                    "complexity": float(avg_complex_loss.numpy()),
                    "total_loss": float(avg_total_loss.numpy()),
                    "test_acc": ("" if test_acc is None else f"{test_acc:.4f}"),
                    "elapsed": format_seconds(time.time() - t0),
                }
            )

    # finalize GIF
    if viz is not None and verbose > 0:
        gif_path = viz.on_train_end()
        print(f"[viz] wrote GIF to: {gif_path}")

    # Evaluate the retrained model
    if eval and verbose > 0:
        _, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
        print(f"Test accuracy after retraining with prior: {test_accuracy:.4f}")

    # Save the retrained model
    if save_dir:
        model.save(filepath=save_dir + "/retrained.keras")
        print(f"Retrained model saved to {save_dir}")

    return model
