#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script: LeNet-5 Caffe on MNIST with soft weight sharing and GIF visualization
"""

import sys
import os
import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from PIL import Image
import imageio
import io
from tqdm import tqdm

# Import our new modules
from src.ATDLSoftWeightSharing.models import LeNet5Caffe
from src.ATDLSoftWeightSharing.data import get_mnist_data


def logsumexp(x, axis=-1):
    """
    TensorFlow implementation of logsumexp for numerical stability.
    """
    m = tf.reduce_max(x, axis=axis, keepdims=True)
    return tf.squeeze(m, axis=axis) + tf.math.log(
        tf.reduce_sum(tf.exp(x - m), axis=axis)
    )


class SimpleMixturePrior:
    """
    A simplified mixture prior implementation for the tutorial.
    """

    def __init__(self, J=16, pi0=0.99, init_sigma=0.25, mu_init=None):
        self.J = J
        self.pi0 = pi0
        self.init_sigma = init_sigma
        self.eps = 1e-8

        # Initialize parameters as tf.Variables
        if mu_init is None:
            self.mu = tf.Variable(
                tf.linspace(-0.6, 0.6, J - 1), trainable=True, name="mu"
            )
        else:
            self.mu = tf.Variable(mu_init, trainable=True, name="mu")
        self.log_sigma2 = tf.Variable(
            tf.fill([J - 1], tf.math.log(init_sigma**2)),
            trainable=True,
            name="log_sigma2",
        )
        self.pi_logits = tf.Variable(tf.zeros(J - 1), trainable=True, name="pi_logits")
        self.log_sigma2_0 = tf.Variable(
            tf.constant(tf.math.log(init_sigma**2)), trainable=True, name="log_sigma2_0"
        )

    def get_mixture_params(self):
        """Get the current mixture parameters: (means, variances, mixing proportions)."""
        # Calculate non-zero component mixing proportions (softmax over logits)
        pi_nonzero = tf.nn.softmax(self.pi_logits)

        # Handle pi0 (fixed)
        pi = tf.concat(
            [
                tf.expand_dims(tf.constant(self.pi0, dtype=tf.float32), 0),
                (1.0 - self.pi0) * pi_nonzero,
            ],
            axis=0,
        )

        # Means: zero component + learned means
        mu = tf.concat([tf.constant([0.0], dtype=tf.float32), self.mu], axis=0)

        # Variances: zero component + learned variances
        sigma2 = tf.concat(
            [tf.expand_dims(tf.exp(self.log_sigma2_0), 0), tf.exp(self.log_sigma2)],
            axis=0,
        )

        # Clamp variances to prevent numerical issues
        sigma2 = tf.clip_by_value(
            sigma2, clip_value_min=1e-8, clip_value_max=float("inf")
        )

        return mu, sigma2, pi

    def log_prob_w(self, w_flat):
        """Compute log probability of flattened weights under the mixture model."""
        mu, sigma2, pi = self.get_mixture_params()

        # Expand dimensions for broadcasting
        w = tf.expand_dims(w_flat, axis=1)  # [N, 1]
        mu_expanded = tf.expand_dims(mu, axis=0)  # [1, J]
        sigma2_expanded = tf.expand_dims(sigma2, axis=0)  # [1, J]
        pi_expanded = tf.expand_dims(pi, axis=0)  # [1, J]

        # Calculate log probabilities for each component
        log_pi = tf.math.log(pi_expanded + self.eps)
        log_norm = -0.5 * (
            tf.math.log(2 * math.pi * sigma2_expanded)
            + tf.square(w - mu_expanded) / sigma2_expanded
        )

        # Sum over mixture components using logsumexp for numerical stability
        return logsumexp(log_pi + log_norm, axis=1)

    def complexity_loss(self, model_weights):
        """
        Compute the complexity loss (negative log prior probability) for model weights.
        """
        # Collect flattened weights
        flattened_weights = []
        for W in model_weights:
            if len(W.shape) > 0:  # Skip scalar parameters if any
                flattened_weights.append(tf.reshape(W, [-1]))

        if not flattened_weights:
            return tf.constant(0.0)

        w_flat = tf.concat(flattened_weights, axis=0)

        # Compute negative log probability (the main complexity loss term)
        log_prob_w = self.log_prob_w(w_flat)
        total_loss = -tf.reduce_sum(log_prob_w)

        # Add hyperpriors on precisions (inverse variances)
        mu, sigma2, pi = self.get_mixture_params()
        inv_sigma2 = 1.0 / sigma2  # Precision

        # Zero component hyperprior
        gamma_alpha0, gamma_beta0 = 5000.0, 2.0
        inv_sigma2_0 = inv_sigma2[0]
        gamma_term_0 = (
            gamma_alpha0 * tf.math.log(gamma_beta0)
            - tf.math.lgamma(gamma_alpha0)
            + (gamma_alpha0 - 1.0) * tf.math.log(inv_sigma2_0)
            - gamma_beta0 * inv_sigma2_0
        )
        total_loss = total_loss - gamma_term_0

        # Non-zero components hyperprior
        gamma_alpha, gamma_beta = 250.0, 0.1
        inv_sigma2_nonzero = inv_sigma2[1:]  # Exclude zero component
        gamma_terms_nonzero = (
            gamma_alpha * tf.math.log(gamma_beta) - tf.math.lgamma(gamma_alpha)
        ) + (
            (gamma_alpha - 1.0) * tf.math.log(inv_sigma2_nonzero)
            - gamma_beta * inv_sigma2_nonzero
        )
        total_loss = total_loss - tf.reduce_sum(gamma_terms_nonzero)

        return total_loss

    def quantize_model(self, model, skip_last_matrix=False, assign_mode="ml"):
        """
        Hard-quantize model weights to mixture means.
        """
        mu, sigma2, pi = self.get_mixture_params()

        # Convert tensors to numpy
        mu_np = mu.numpy()
        sigma2_np = sigma2.numpy()
        pi_np = pi.numpy()

        # Compute log probabilities and constants
        log_pi = np.log(pi_np + self.eps)
        const = -0.5 * np.log(2 * math.pi * sigma2_np)
        inv_s2 = 1.0 / sigma2_np

        # Get all trainable weights that are from dense/conv layers
        weight_layers = []
        for layer in model.layers:
            if hasattr(layer, "kernel"):  # Dense, Conv2D layers
                weight_layers.append((layer, "kernel"))
            elif hasattr(layer, "weights") and len(layer.weights) > 0:
                for i, w in enumerate(layer.weights):
                    if "kernel" in w.name or "weight" in w.name:
                        weight_layers.append((layer, i))

        # Last 2D weight index (for potentially skipping)
        last_2d_idx = -1
        for idx, (layer, param_name) in enumerate(weight_layers):
            if hasattr(layer, "kernel"):
                if len(layer.kernel.shape) >= 2:
                    last_2d_idx = idx

        # Quantize weights
        for idx, (layer, param_name) in enumerate(weight_layers):
            if skip_last_matrix and idx == last_2d_idx:
                continue

            # Get the weight tensor
            if isinstance(param_name, str):
                weight_tensor = getattr(layer, param_name)
            else:
                weight_tensor = layer.weights[param_name]

            # Get weight values as numpy
            weight_values = weight_tensor.numpy()
            w_flat = weight_values.flatten()

            # Compute assignment scores based on assign_mode
            w_expanded = np.expand_dims(w_flat, axis=1)  # [N, 1]
            mu_expanded = np.expand_dims(mu_np, axis=0)  # [1, J]

            if assign_mode == "map":
                # MAP: includes mixing proportions (π)
                log_pi_expanded = np.expand_dims(log_pi, axis=0)  # [1, J]
                const_expanded = np.expand_dims(const, axis=0)  # [1, J]
                inv_s2_expanded = np.expand_dims(inv_s2, axis=0)  # [1, J]

                scores = (
                    log_pi_expanded
                    + const_expanded
                    - 0.5 * ((w_expanded - mu_expanded) ** 2 * inv_s2_expanded)
                )
            elif assign_mode == "ml":
                # ML: ignore π, keep per-component likelihood
                const_expanded = np.expand_dims(const, axis=0)  # [1, J]
                inv_s2_expanded = np.expand_dims(inv_s2, axis=0)  # [1, J]

                scores = const_expanded - 0.5 * (
                    (w_expanded - mu_expanded) ** 2 * inv_s2_expanded
                )
            else:
                raise ValueError(f"Unknown assign mode: {assign_mode}")

            # Find the best component for each weight
            best_component_idx = np.argmax(scores, axis=1)

            # Create quantized weight values
            quantized_flat = mu_np[best_component_idx]

            # Set zero for component 0 (exact zeros for numerical hygiene)
            zero_mask = best_component_idx == 0
            quantized_flat[zero_mask] = 0.0

            # Reshape back to original shape
            quantized_values = quantized_flat.reshape(weight_values.shape)

            # Assign back to the layer
            weight_tensor.assign(quantized_values)


class GIFVisualizer:
    """
    Create GIF visualization of weight evolution during training
    """

    def __init__(self, model, prior, save_path="weights_evolution.gif"):
        self.model = model
        self.prior = prior
        self.save_path = save_path
        self.frames = []
        self.initial_weights = None

    def capture_weights(self, epoch, test_accuracy=None):
        """Capture current weights and create a visualization frame"""
        # Collect all weights from the model
        all_weights = []
        for layer in self.model.layers:
            if hasattr(layer, "kernel"):
                all_weights.append(layer.kernel.numpy().flatten())

        if len(all_weights) == 0:
            return  # No weights to visualize

        current_weights = np.concatenate(all_weights)

        # If this is the first call, store initial weights
        if self.initial_weights is None:
            self.initial_weights = current_weights.copy()

        # Create the visualization
        fig, ax = plt.subplots(figsize=(10, 6))

        # Scatter plot of current vs initial weights
        ax.scatter(self.initial_weights, current_weights, alpha=0.5, s=1, c="g")
        ax.set_xlabel("Initial Weights")
        ax.set_ylabel("Current Weights")
        ax.set_title(
            f"Epoch: {epoch}, Test Acc: {test_accuracy:.4f}"
            if test_accuracy
            else f"Epoch: {epoch}"
        )
        ax.set_xlim(-0.8, 0.8)
        ax.set_ylim(-0.8, 0.8)
        ax.grid(True, alpha=0.3)

        # Add mixture means and variance bands
        mu, sigma2, pi = self.prior.get_mixture_params()
        mu_np = mu.numpy()
        sigma2_np = sigma2.numpy()

        # Plot mixture means as vertical lines
        for i, mean_val in enumerate(mu_np[1:11]):  # Plot first 10 means
            ax.axhline(y=mean_val, color="r", linestyle="--", alpha=0.5, linewidth=1)

        # Save to bytes and add to frames
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close()
        buf.seek(0)
        frame = Image.open(buf)
        self.frames.append(np.array(frame))

    def save_gif(self, duration=0.5):
        """Save all captured frames as a GIF"""
        if self.frames:
            imageio.mimsave(self.save_path, self.frames, duration=duration)


def main():
    print(
        "=== LeNet-5 Caffe on MNIST with soft weight sharing and GIF visualization ==="
    )

    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)

    # Load data
    print("Loading MNIST data...")
    (x_train, y_train), (x_test, y_test) = get_mnist_data()

    # Create model
    print("Creating LeNet-5 Caffe model...")
    model = LeNet5Caffe(num_classes=10)

    # Pretraining for a few epochs - create the model with a dummy input to build it
    print("Building model...")
    # Build the model by calling it once
    _ = model(tf.zeros((1, 28, 28, 1)))  # This builds the model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Initialize the mixture prior
    print("Initializing mixture prior...")
    J = 16  # Number of mixture components
    pi0 = 0.99  # Probability of zero component

    # Get the model's weights to initialize the prior means
    weight_values = []
    for layer in model.layers:
        if hasattr(layer, "kernel"):  # Dense, Conv2D layers
            weight_values.append(layer.kernel.numpy())
        elif hasattr(layer, "weights"):
            for w in layer.weights:
                if "kernel" in w.name or "weight" in w.name:
                    weight_values.append(w.numpy())

    if weight_values:
        all_weights = np.concatenate([w.flatten() for w in weight_values])
        wmin, wmax = np.min(all_weights), np.max(all_weights)
        if wmin == wmax:
            wmin, wmax = -0.6, 0.6
        means_init = np.linspace(wmin, wmax, J - 1)
    else:
        means_init = np.linspace(-0.6, 0.6, J - 1)

    # Create a simple prior with the initialized means
    prior = SimpleMixturePrior(J=J, pi0=pi0, init_sigma=0.25, mu_init=means_init)

    # Pretraining with higher learning rate for faster convergence
    print("Pretraining model...")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    
    # Train for a few epochs without prior (pretraining)
    pretrain_epochs = 5  # Reduced to make it more efficient
    history = model.fit(
        x_train,
        y_train,
        epochs=pretrain_epochs,
        batch_size=256,
        verbose=1,
        validation_data=(x_test, y_test),
    )
    
    print(f"Pretraining completed. Final pretraining accuracy: {max(history.history['val_accuracy']):.4f}")

    # Create visualizer
    visualizer = GIFVisualizer(
        model, prior, save_path="lenet5caffe_weights_evolution.gif"
    )

    # Create optimizers outside tf.function
    model_optimizer = keras.optimizers.Adam(learning_rate=1e-3)  # Increased from 1e-4
    prior_optimizer = keras.optimizers.Adam(learning_rate=1e-2)  # Increased from 1e-3

    # Define a custom training step that includes the prior loss
    @tf.function
    def train_step(x_batch, y_batch):
        with tf.GradientTape() as tape:
            # Watch the prior variables for gradients
            tape.watch(
                [prior.mu, prior.log_sigma2, prior.pi_logits, prior.log_sigma2_0]
            )

            predictions = model(x_batch, training=True)
            classification_loss = tf.keras.losses.categorical_crossentropy(
                y_batch, predictions
            )

            # Get model weights for the prior
            model_weights = []
            for layer in model.layers:
                if hasattr(layer, "kernel"):  # Dense, Conv2D layers
                    model_weights.append(layer.kernel)
                if hasattr(layer, "bias") and layer.bias is not None:
                    model_weights.append(layer.bias)

            # Calculate the prior loss (complexity penalty)
            prior_loss = prior.complexity_loss(model_weights)

            # Total loss
            total_loss = (
                tf.reduce_mean(classification_loss)
                + 0.003 * prior_loss  # Removed batch size normalization to strengthen prior
            )

        # Compute gradients for both model and prior parameters
        trainable_vars = model.trainable_variables + [
            prior.mu,
            prior.log_sigma2,
            prior.pi_logits,
            prior.log_sigma2_0,
        ]
        gradients = tape.gradient(total_loss, trainable_vars)

        # Apply gradients
        model_vars = model.trainable_variables
        prior_vars = [prior.mu, prior.log_sigma2, prior.pi_logits, prior.log_sigma2_0]

        model_grads = gradients[: len(model_vars)]
        prior_grads = gradients[len(model_vars) :]

        # Apply gradients
        model_optimizer.apply_gradients(zip(model_grads, model_vars))
        prior_optimizer.apply_gradients(zip(prior_grads, prior_vars))

        return total_loss, classification_loss

    # Training loop with visualization
    print("Training with mixture prior for 5 epochs (with visualization)...")
    epochs = 30
    batch_size = 256

    # Create dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    pbar = tqdm(range(epochs), desc="Training with visualization")
    for epoch in pbar:
        epoch_loss = 0
        num_batches = 0

        for x_batch, y_batch in train_dataset:
            total_loss, classification_loss = train_step(x_batch, y_batch)
            epoch_loss += total_loss
            num_batches += 1

        avg_loss = epoch_loss / num_batches

        # Evaluate accuracy for this epoch
        test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

        # Capture visualization for this epoch
        visualizer.capture_weights(epoch + pretrain_epochs + 1, test_accuracy)

        pbar.set_postfix_str(
            {
                "Avg loss": f"{avg_loss:.4f}",
                "Test Acc": f"{test_accuracy:.4f}",
            }
        )

    # Save the GIF
    visualizer.save_gif(duration=1.0)
    print(f"Weight evolution GIF saved as lenet5caffe_weights_evolution.gif")

    # Final evaluation
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Final test accuracy: {test_accuracy:.4f}")

    # Save the model
    model.save("./scripts/lenet5caffe_model.keras")
    print("Model saved as lenet5caffe_model.keras")

    print("LeNet-5 Caffe training with visualization completed successfully!")


if __name__ == "__main__":
    main()
