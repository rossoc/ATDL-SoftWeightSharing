#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script 3: Post-processing and compression
This script tests the quantization and compression functionality of the mixture model
"""

import sys
import os
import math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Import our new modules
from src.ATDLSoftWeightSharing.models import create_tutorial_model
from src.ATDLSoftWeightSharing.data import get_mnist_data
from src.ATDLSoftWeightSharing.helpers import special_flatten


def logsumexp(x, axis=-1):
    """
    TensorFlow implementation of logsumexp for numerical stability.
    """
    m = tf.reduce_max(x, axis=axis, keepdims=True)
    return tf.squeeze(m, axis=axis) + tf.math.log(tf.reduce_sum(tf.exp(x - m), axis=axis))


class SimpleMixturePrior:
    """
    A simplified mixture prior implementation for the tutorial.
    This doesn't inherit from Layer to avoid complexity with custom training loops.
    """
    
    def __init__(self, J=16, pi0=0.99, init_sigma=0.25, mu_init=None):
        self.J = J
        self.pi0 = pi0
        self.init_sigma = init_sigma
        self.eps = 1e-8
        
        # Initialize parameters as tf.Variables
        if mu_init is None:
            self.mu = tf.Variable(tf.linspace(-0.6, 0.6, J - 1), trainable=True, name='mu')
        else:
            self.mu = tf.Variable(mu_init, trainable=True, name='mu')
        self.log_sigma2 = tf.Variable(tf.fill([J - 1], tf.math.log(init_sigma**2)), trainable=True, name='log_sigma2')
        self.pi_logits = tf.Variable(tf.zeros(J - 1), trainable=True, name='pi_logits')
        self.log_sigma2_0 = tf.Variable(tf.constant(tf.math.log(init_sigma**2)), trainable=True, name='log_sigma2_0')
        
    def get_mixture_params(self):
        """Get the current mixture parameters: (means, variances, mixing proportions)."""
        # Calculate non-zero component mixing proportions (softmax over logits)
        pi_nonzero = tf.nn.softmax(self.pi_logits)
        
        # Handle pi0 (fixed)
        pi = tf.concat([
            tf.expand_dims(tf.constant(self.pi0, dtype=tf.float32), 0),
            (1.0 - self.pi0) * pi_nonzero
        ], axis=0)
        
        # Means: zero component + learned means
        mu = tf.concat([
            tf.constant([0.0], dtype=tf.float32),
            self.mu
        ], axis=0)
        
        # Variances: zero component + learned variances
        sigma2 = tf.concat([
            tf.expand_dims(tf.exp(self.log_sigma2_0), 0),
            tf.exp(self.log_sigma2)
        ], axis=0)
        
        # Clamp variances to prevent numerical issues
        sigma2 = tf.clip_by_value(sigma2, clip_value_min=1e-8, clip_value_max=float('inf'))
        
        return mu, sigma2, pi

    def quantize_model(self, model, skip_last_matrix=False, assign_mode="ml"):
        """
        Hard-quantize model weights to mixture means.
        
        Args:
            model: The Keras model to quantize
            skip_last_matrix: Whether to skip quantizing the last weight matrix
            assign_mode: "map" for MAP assignment (includes mixing), "ml" for ML assignment
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
            if hasattr(layer, 'kernel'):  # Dense, Conv2D layers
                weight_layers.append((layer, 'kernel'))
            elif hasattr(layer, 'weights') and len(layer.weights) > 0:
                # Other layers that might have weights - but let's focus on kernel weights
                for i, w in enumerate(layer.weights):
                    if 'kernel' in w.name or 'weight' in w.name:
                        weight_layers.append((layer, i))
        
        # Last 2D weight index (for potentially skipping)
        last_2d_idx = -1
        for idx, (layer, param_name) in enumerate(weight_layers):
            if hasattr(layer, 'kernel'):
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
                const_expanded = np.expand_dims(const, axis=0)   # [1, J]
                inv_s2_expanded = np.expand_dims(inv_s2, axis=0)  # [1, J]
                
                scores = (
                    log_pi_expanded +
                    const_expanded -
                    0.5 * ((w_expanded - mu_expanded)**2 * inv_s2_expanded)
                )
            elif assign_mode == "ml":
                # ML: ignore π, keep per-component likelihood
                const_expanded = np.expand_dims(const, axis=0)   # [1, J]
                inv_s2_expanded = np.expand_dims(inv_s2, axis=0)  # [1, J]
                
                scores = (
                    const_expanded -
                    0.5 * ((w_expanded - mu_expanded)**2 * inv_s2_expanded)
                )
            else:
                raise ValueError(f"Unknown assign mode: {assign_mode}")
            
            # Find the best component for each weight
            best_component_idx = np.argmax(scores, axis=1)
            
            # Create quantized weight values
            quantized_flat = mu_np[best_component_idx]
            
            # Set zero for component 0 (exact zeros for numerical hygiene)
            zero_mask = (best_component_idx == 0)
            quantized_flat[zero_mask] = 0.0
            
            # Reshape back to original shape
            quantized_values = quantized_flat.reshape(weight_values.shape)
            
            # Assign back to the layer
            weight_tensor.assign(quantized_values)


def main():
    print("=== Test Script 3: Post-processing and Compression ===")
    print("Testing quantization and compression functionality...")
    
    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
    
    # Load data
    print("Loading MNIST data...")
    (x_train, y_train), (x_test, y_test) = get_mnist_data()
    
    # Load pretrained model (try the retrained one if available, otherwise use basic pretrained)
    print("Loading model for compression testing...")
    try:
        model = keras.models.load_model("./scripts/retrained_with_prior_model.keras")
        print("Loaded retrained model with prior")
    except:
        try:
            model = keras.models.load_model("./scripts/pretrained_model.keras")
            print("Loaded basic pretrained model")
        except:
            # If no models exist, create and train a quick one
            print("No pretrained models found, creating and training a quick model...")
            model = create_tutorial_model(input_shape=(28, 28, 1), num_classes=10)
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            # Train for just 2 epochs to get some weights
            model.fit(
                x_train, y_train,
                epochs=2,
                batch_size=256,
                verbose=0
            )
            print("Created and trained temporary model")
    
    # Initialize a mixture prior for the model
    print("Initializing mixture prior for compression...")
    J = 16  # Number of mixture components
    pi0 = 0.99  # Probability of zero component
    
    # Get the model's weights to initialize the prior means
    weight_values = []
    for layer in model.layers:
        if hasattr(layer, 'kernel'):  # Dense, Conv2D layers
            weight_values.append(layer.kernel.numpy())
        elif hasattr(layer, 'weights'):
            for w in layer.weights:
                if 'kernel' in w.name or 'weight' in w.name:
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
    
    # Print initial stats
    print("Before compression:")
    initial_weights = model.get_weights()
    initial_flat = special_flatten(initial_weights)
    initial_flat = initial_flat.flatten()
    
    nonzero_count = np.count_nonzero(initial_flat)
    total_count = len(initial_flat)
    initial_sparsity = 100.0 * (total_count - nonzero_count) / total_count
    print(f"  Total weights: {total_count}")
    print(f"  Non-zero weights: {nonzero_count}")
    print(f"  Sparsity: {initial_sparsity:.2f}%")
    
    # Evaluate initial model performance
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    initial_test_loss, initial_test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"  Initial test accuracy: {initial_test_accuracy:.4f}")
    
    # Apply quantization/compression to the model
    print("Applying quantization to model weights...")
    prior.quantize_model(model, skip_last_matrix=False, assign_mode="ml")
    
    # Print stats after compression
    print("After compression:")
    compressed_weights = model.get_weights()
    compressed_flat = special_flatten(compressed_weights)
    compressed_flat = compressed_flat.flatten()
    
    nonzero_count = np.count_nonzero(compressed_flat)
    total_count = len(compressed_flat)
    compressed_sparsity = 100.0 * (total_count - nonzero_count) / total_count
    print(f"  Total weights: {total_count}")
    print(f"  Non-zero weights: {nonzero_count}")
    print(f"  Sparsity: {compressed_sparsity:.2f}%")
    
    # Evaluate compressed model performance
    compressed_test_loss, compressed_test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"  Compressed test accuracy: {compressed_test_accuracy:.4f}")
    print(f"  Accuracy drop: {initial_test_accuracy - compressed_test_accuracy:.4f}")
    
    # Save the compressed model
    model.save("./scripts/compressed_model.keras")
    print("Compressed model saved as compressed_model.keras")
    
    # Additional test: Get the mixture parameters
    print("\nMixture model parameters:")
    mu, sigma2, pi = prior.get_mixture_params()
    print(f"  Means: {mu.numpy()[:5]}...")  # Show first 5
    print(f"  Variances: {sigma2.numpy()[:5]}...")  # Show first 5
    print(f"  Mixing proportions: {pi.numpy()[:5]}...")  # Show first 5
    
    print("\nPost-processing and compression test completed successfully!")
    
    return model


if __name__ == "__main__":
    main()