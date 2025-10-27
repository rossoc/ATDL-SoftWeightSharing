#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script 2: Training with mixture prior
This script tests the prior implementation with a pretrained model
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
    
    def __init__(self, J=16, pi0=0.99, init_sigma=0.25):
        self.J = J
        self.pi0 = pi0
        self.init_sigma = init_sigma
        self.eps = 1e-8
        
        # Initialize parameters as tf.Variables
        self.mu = tf.Variable(tf.linspace(-0.6, 0.6, J - 1), trainable=True, name='mu')
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
            tf.math.log(2 * math.pi * sigma2_expanded) +
            tf.square(w - mu_expanded) / sigma2_expanded
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
            gamma_alpha0 * tf.math.log(gamma_beta0) - tf.math.lgamma(gamma_alpha0) +
            (gamma_alpha0 - 1.0) * tf.math.log(inv_sigma2_0) - gamma_beta0 * inv_sigma2_0
        )
        total_loss = total_loss - gamma_term_0
        
        # Non-zero components hyperprior
        gamma_alpha, gamma_beta = 250.0, 0.1
        inv_sigma2_nonzero = inv_sigma2[1:]  # Exclude zero component
        gamma_terms_nonzero = (
            (gamma_alpha * tf.math.log(gamma_beta) - tf.math.lgamma(gamma_alpha)) +
            ((gamma_alpha - 1.0) * tf.math.log(inv_sigma2_nonzero) - gamma_beta * inv_sigma2_nonzero)
        )
        total_loss = total_loss - tf.reduce_sum(gamma_terms_nonzero)
        
        return total_loss


def main():
    print("=== Test Script 2: Training with Mixture Prior ===")
    print("Testing Keras prior implementation...")
    
    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
    
    # Load data
    print("Loading MNIST data...")
    (x_train, y_train), (x_test, y_test) = get_mnist_data()
    
    # Load pretrained model for testing
    print("Loading pretrained model...")
    try:
        pretrained_model = keras.models.load_model("./scripts/pretrained_model.keras")
        print("Loaded pretrained model from file")
    except:
        # If no pretrained model exists, create a new one and train briefly
        print("No pretrained model found, creating and training a quick model...")
        pretrained_model = create_tutorial_model(input_shape=(28, 28, 1), num_classes=10)
        pretrained_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        # Train for just 2 epochs to get some weights
        pretrained_model.fit(
            x_train, y_train,
            epochs=2,
            batch_size=256,
            verbose=0
        )
        print("Created and trained temporary pretrained model")
    
    # Initialize the mixture prior
    print("Initializing mixture prior...")
    J = 16  # Number of mixture components
    pi0 = 0.99  # Probability of zero component
    
    prior = SimpleMixturePrior(J=J, pi0=pi0, init_sigma=0.25)
    
    # Create a new model for retraining with the same architecture
    print("Creating retraining model with same architecture...")
    retrain_model = create_tutorial_model(input_shape=(28, 28, 1), num_classes=10)
    retrain_model.set_weights(pretrained_model.get_weights())
    
    # Create optimizers outside tf.function
    model_optimizer = keras.optimizers.Adam(learning_rate=1e-4)
    prior_optimizer = keras.optimizers.Adam(learning_rate=1e-3)
    
    # Define a custom training step that includes the prior loss
    @tf.function
    def train_step(x_batch, y_batch):
        with tf.GradientTape() as tape:
            # Watch the prior variables for gradients
            tape.watch([prior.mu, prior.log_sigma2, prior.pi_logits, prior.log_sigma2_0])
            
            predictions = retrain_model(x_batch, training=True)
            classification_loss = tf.keras.losses.categorical_crossentropy(y_batch, predictions)
            
            # Get model weights for the prior
            model_weights = []
            for layer in retrain_model.layers:
                if hasattr(layer, 'kernel'):  # Dense, Conv2D layers
                    model_weights.append(layer.kernel)
                if hasattr(layer, 'bias') and layer.bias is not None:
                    model_weights.append(layer.bias)
            
            # Calculate the prior loss (complexity penalty)
            prior_loss = prior.complexity_loss(model_weights)
            
            # Total loss
            total_loss = tf.reduce_mean(classification_loss) + (0.003 / tf.cast(tf.shape(x_batch)[0], tf.float32)) * prior_loss
    
        # Compute gradients for both model and prior parameters
        trainable_vars = retrain_model.trainable_variables + [prior.mu, prior.log_sigma2, prior.pi_logits, prior.log_sigma2_0]
        gradients = tape.gradient(total_loss, trainable_vars)
        
        # Apply gradients
        model_vars = retrain_model.trainable_variables
        prior_vars = [prior.mu, prior.log_sigma2, prior.pi_logits, prior.log_sigma2_0]
        
        model_grads = gradients[:len(model_vars)]
        prior_grads = gradients[len(model_vars):]
        
        # Apply gradients
        model_optimizer.apply_gradients(zip(model_grads, model_vars))
        prior_optimizer.apply_gradients(zip(prior_grads, prior_vars))
        
        return total_loss, classification_loss

    # Training loop
    print("Training with mixture prior for 5 epochs...")
    epochs = 5
    batch_size = 256
    
    # Create dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    for epoch in range(epochs):
        epoch_loss = 0
        num_batches = 0
        
        for x_batch, y_batch in train_dataset:
            total_loss, classification_loss = train_step(x_batch, y_batch)
            epoch_loss += total_loss
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")
    
    # Compile the model before evaluation
    retrain_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Evaluate the retrained model
    test_loss, test_accuracy = retrain_model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy after retraining with prior: {test_accuracy:.4f}")
    
    # Save the retrained model
    retrain_model.save("./scripts/retrained_with_prior_model.keras")
    print("Retrained model saved as retrained_with_prior_model.keras")
    
    print("Training with prior test completed successfully!")
    return retrain_model


if __name__ == "__main__":
    main()