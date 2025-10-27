"""
Keras implementation of mixture prior for soft weight sharing.

This module provides Keras equivalents of PyTorch prior utilities from the ATDL2/sws/prior.py file.
"""

import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import List, Tuple, Optional, Dict
from .helpers import special_flatten


def logsumexp(x, axis=-1):
    """
    TensorFlow implementation of logsumexp for numerical stability.
    """
    m = tf.reduce_max(x, axis=axis, keepdims=True)
    return tf.squeeze(m, axis=axis) + tf.math.log(tf.reduce_sum(tf.exp(x - m), axis=axis))


class MixturePrior(layers.Layer):
    """
    Factorized Gaussian mixture prior over weights for Keras models:
        p(w) = Π_i Σ_{j=0}^{J-1} π_j N(w_i | μ_j, σ_j^2)

    j=0 is the pruning component with μ₀=0; π₀ is fixed close to 1.

    Defaults match the tutorial layer from the original authors:
      - init σ ≈ 0.25
      - Gamma priors on precisions (different for zero vs non-zero)
    """
    
    def __init__(
        self,
        J: int,
        pi0: float = 0.999,
        learn_pi0: bool = False,
        init_means: Optional[np.ndarray] = None,
        init_log_sigma2: float = math.log(0.25**2),
        # --- Hyper-priors ON by default (tutorial values) ---
        gamma_alpha: Optional[float] = 250.0,  # non-zero comps
        gamma_beta: Optional[float] = 0.1,
        gamma_alpha0: Optional[float] = 5000.0,  # zero comp
        gamma_beta0: Optional[float] = 2.0,
        # Beta prior on π0 (unused unless learn_pi0=True):
        beta_alpha: Optional[float] = None,
        beta_beta: Optional[float] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        assert J >= 2
        self.J = J
        self.learn_pi0 = learn_pi0
        self.pi0_init = float(pi0)
        
        # Store hyperparameters
        self.gamma_alpha = gamma_alpha
        self.gamma_beta = gamma_beta
        self.gamma_alpha0 = gamma_alpha0
        self.gamma_beta0 = gamma_beta0
        self.beta_alpha = beta_alpha
        self.beta_beta = beta_beta
        self.eps = 1e-8
        self.init_log_sigma2 = init_log_sigma2
        
        # Initialize means
        if init_means is None:
            self.init_means = tf.linspace(-0.6, 0.6, J - 1)
        else:
            self.init_means = init_means

    def build(self, input_shape):
        """Build the layer and create trainable variables."""
        # Create trainable parameters
        self.mu = self.add_weight(
            name='mu',
            shape=(self.J - 1,),
            initializer=keras.initializers.Constant(self.init_means),
            trainable=True,
            dtype=tf.float32
        )
        
        self.log_sigma2 = self.add_weight(
            name='log_sigma2',
            shape=(self.J - 1,),
            initializer=keras.initializers.Constant(self.init_log_sigma2),
            trainable=True,
            dtype=tf.float32
        )
        
        self.pi_logits = self.add_weight(
            name='pi_logits',
            shape=(self.J - 1,),
            initializer='zeros',
            trainable=True,
            dtype=tf.float32
        )
        
        # Parameter for the zero component (log variance)
        self.log_sigma2_0 = self.add_weight(
            name='log_sigma2_0',
            shape=(),
            initializer=keras.initializers.Constant(self.init_log_sigma2),
            trainable=True,
            dtype=tf.float32
        )
        
        super().build(input_shape)
    
    def get_mixture_params(self, mu=None, log_sigma2=None, pi_logits=None, log_sigma2_0=None) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Get the current mixture parameters: (means, variances, mixing proportions). Can accept parameters directly or use layer's parameters."""
        # Use either provided parameters or layer's parameters
        if mu is None:
            mu_param = self.mu
            log_sigma2_param = self.log_sigma2
            pi_logits_param = self.pi_logits
            log_sigma2_0_param = self.log_sigma2_0
        else:
            mu_param = mu
            log_sigma2_param = log_sigma2
            pi_logits_param = pi_logits
            log_sigma2_0_param = log_sigma2_0
        
        # Calculate non-zero component mixing proportions (softmax over logits)
        pi_nonzero = tf.nn.softmax(pi_logits_param)
        
        # Handle pi0 (either fixed or learned)
        if self.learn_pi0:
            pi0 = tf.clip_by_value(self.pi0_init, 0.5, 0.9999)
            pi = tf.concat([
                tf.expand_dims(pi0, 0), 
                (1.0 - pi0) * pi_nonzero
            ], axis=0)
        else:
            pi = tf.concat([
                tf.expand_dims(tf.constant(self.pi0_init, dtype=tf.float32), 0),
                (1.0 - self.pi0_init) * pi_nonzero
            ], axis=0)
        
        # Means: zero component + learned means
        mu_final = tf.concat([
            tf.constant([0.0], dtype=tf.float32),
            mu_param
        ], axis=0)
        
        # Variances: zero component + learned variances
        sigma2 = tf.concat([
            tf.expand_dims(tf.exp(log_sigma2_0_param), 0),
            tf.exp(log_sigma2_param)
        ], axis=0)
        
        # Clamp variances to prevent numerical issues
        sigma2 = tf.clip_by_value(sigma2, clip_value_min=1e-8, clip_value_max=float('inf'))
        
        return mu_final, sigma2, pi

    def mixture_params(self) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Get the current mixture parameters using the layer's parameters."""
        return self.get_mixture_params()

    def log_prob_w(self, w_flat: tf.Tensor, mu=None, log_sigma2=None, pi_logits=None, log_sigma2_0=None) -> tf.Tensor:
        """Compute log probability of flattened weights under the mixture model."""
        mu, sigma2, pi = self.get_mixture_params(mu, log_sigma2, pi_logits, log_sigma2_0)
        
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

    def complexity_loss(self, model_weights: List[tf.Variable]) -> tf.Tensor:
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
        if self.gamma_alpha0 is not None and self.gamma_beta0 is not None:
            a0, b0 = self.gamma_alpha0, self.gamma_beta0
            inv_sigma2_0 = inv_sigma2[0]
            gamma_term_0 = (
                a0 * math.log(b0) - math.lgamma(a0) +
                (a0 - 1.0) * tf.math.log(inv_sigma2_0) - b0 * inv_sigma2_0
            )
            total_loss = total_loss - gamma_term_0
        
        # Non-zero components hyperprior
        if self.gamma_alpha is not None and self.gamma_beta is not None:
            a, b = self.gamma_alpha, self.gamma_beta
            inv_sigma2_nonzero = inv_sigma2[1:]  # Exclude zero component
            gamma_terms_nonzero = (
                (a * math.log(b) - math.lgamma(a)) +
                ((a - 1.0) * tf.math.log(inv_sigma2_nonzero) - b * inv_sigma2_nonzero)
            )
            total_loss = total_loss - tf.reduce_sum(gamma_terms_nonzero)
        
        # Beta prior on pi0 (optional)
        if (self.learn_pi0 and 
            self.beta_alpha is not None and 
            self.beta_beta is not None):
            a, b = self.beta_alpha, self.beta_beta
            pi0 = pi[0]
            beta_term = (
                math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) +
                (a - 1.0) * tf.math.log(pi0 + self.eps) +
                (b - 1.0) * tf.math.log(1.0 - pi0 + self.eps)
            )
            total_loss = total_loss - beta_term
        
        return total_loss
    
    def call(self, inputs):
        """
        Computes the complexity loss and adds it to the layer's losses.
        """
        # Get the model weights from the model this layer is part of
        # The model weights are passed as inputs to this layer
        if isinstance(inputs, (list, tuple)):
            # If inputs is a list of weight tensors
            model_weights = inputs
        else:
            # If inputs is a single tensor, treat as the weights
            model_weights = [inputs]
        
        complexity_loss = self.complexity_loss(model_weights)
        
        # Add to layer's losses for automatic inclusion in training
        self.add_loss(lambda: complexity_loss)
        
        # Return inputs unchanged (identity operation)
        return inputs if not isinstance(inputs, (list, tuple)) else inputs[0]

    def get_config(self):
        """Get configuration for serialization."""
        config = super().get_config()
        config.update({
            'J': self.J,
            'pi0': self.pi0_init,
            'learn_pi0': self.learn_pi0,
            'init_log_sigma2': self.init_log_sigma2,
            'gamma_alpha': self.gamma_alpha,
            'gamma_beta': self.gamma_beta,
            'gamma_alpha0': self.gamma_alpha0,
            'gamma_beta0': self.gamma_beta0,
            'beta_alpha': self.beta_alpha,
            'beta_beta': self.beta_beta,
        })
        return config

    def merge_components(self, weights, kl_threshold: float = 1e-10, max_iter: int = 200):
        """
        Merge similar components in the mixture model.
        This is a simplified version for Keras implementation.
        """
        # Get current mixture parameters
        mu, sigma2, pi = self.mixture_params()
        
        # Convert to numpy for processing
        mu_np = tf.keras.utils.to_numpy(mu)
        sigma2_np = tf.keras.utils.to_numpy(sigma2)
        pi_np = tf.keras.utils.to_numpy(pi)
        
        it = 0
        while it < max_iter:
            it += 1
            merged = False
            for i in range(1, len(mu_np)):  # Skip zero component
                for j in range(i + 1, len(mu_np)):
                    # Calculate KL divergence between components i and j
                    kl_ij = self._kl_gauss(mu_np[i], sigma2_np[i], mu_np[j], sigma2_np[j])
                    kl_ji = self._kl_gauss(mu_np[j], sigma2_np[j], mu_np[i], sigma2_np[i])
                    d = 0.5 * (kl_ij + kl_ji)
                    
                    if d < kl_threshold:
                        # Merge components
                        pnew = pi_np[i] + pi_np[j]
                        if pnew <= 0:
                            continue
                            
                        mu_new = (pi_np[i] * mu_np[i] + pi_np[j] * mu_np[j]) / pnew
                        s2_new = (pi_np[i] * sigma2_np[i] + pi_np[j] * sigma2_np[j]) / pnew
                        
                        mu_np[i] = mu_new
                        sigma2_np[i] = s2_new
                        pi_np[i] = pnew
                        
                        # Remove j-th component
                        mu_np = np.concatenate([mu_np[:j], mu_np[j+1:]])
                        sigma2_np = np.concatenate([sigma2_np[:j], sigma2_np[j+1:]])
                        pi_np = np.concatenate([pi_np[:j], pi_np[j+1:]])
                        
                        merged = True
                        break
                if merged:
                    break
            if not merged:
                break
                
        # Update parameters
        if len(mu_np) > 1:
            # Update the trainable variables
            self.mu.assign(mu_np[1:])  # Exclude zero component
            self.log_sigma2.assign(np.log(sigma2_np[1:]))  # Exclude zero component
            pi_nonzero = pi_np[1:]  # Exclude zero component
            pi_nonzero = pi_nonzero / (np.sum(pi_nonzero) + 1e-12)  # Normalize
            self.pi_logits.assign(np.log(pi_nonzero + 1e-12))  # Convert to logits

    def _kl_gauss(self, mu0, s20, mu1, s21) -> float:
        """Compute KL divergence between two Gaussian distributions."""
        return 0.5 * (np.log(s21 / s20) + (s20 + (mu0 - mu1)**2) / s21 - 1.0).item()

    def quantize_model(self, model, skip_last_matrix: bool = True, assign_mode: str = "ml"):
        """
        Hard-quantize model weights to mixture means.
        
        Args:
            model: The Keras model to quantize
            skip_last_matrix: Whether to skip quantizing the last weight matrix
            assign_mode: "map" for MAP assignment (includes mixing), "ml" for ML assignment
        """
        mu, sigma2, pi = self.mixture_params()
        
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
                # Other layers that might have weights
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
            weight_values = tf.keras.utils.to_numpy(weight_tensor)
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


def init_mixture(model, J: int, pi0: float, init_means_mode: str = "from_weights", 
                 init_range_min: float = -0.6, init_range_max: float = 0.6,
                 init_sigma: float = 0.25):
    """
    Initialize a mixture prior based on the model's weights.
    """
    # Get all trainable weights from dense and conv layers
    weight_values = []
    for layer in model.layers:
        if hasattr(layer, 'kernel'):  # Dense, Conv2D layers
            weight_values.append(layer.kernel.numpy())
        elif hasattr(layer, 'weights'):
            for w in layer.weights:
                if 'kernel' in w.name or 'weight' in w.name:
                    weight_values.append(w.numpy())
    
    # Initialize means based on the mode
    if init_means_mode == "from_weights" and weight_values:
        all_weights = np.concatenate([w.flatten() for w in weight_values])
        wmin, wmax = np.min(all_weights), np.max(all_weights)
        if wmin == wmax:
            wmin, wmax = -0.6, 0.6
        means = np.linspace(wmin, wmax, J - 1)
    else:
        means = np.linspace(init_range_min, init_range_max, J - 1)
    
    # Create and return the mixture prior
    prior = MixturePrior(
        J=J,
        pi0=pi0,
        learn_pi0=False,
        init_means=means,
        init_log_sigma2=math.log(init_sigma**2)
    )
    
    return prior