"""
Keras implementation of mixture prior for soft weight sharing.

This module provides Keras equivalents of PyTorch prior utilities from the ATDL2/sws/prior.py file.
"""

import math
import numpy as np
import tensorflow as tf
from keras import layers
from typing import List, Tuple, Optional


def logsumexp(x, axis=-1):
    """
    TensorFlow implementation of logsumexp for numerical stability.
    """
    m = tf.reduce_max(x, axis=axis, keepdims=True)
    return tf.squeeze(m, axis=axis) + tf.math.log(
        tf.reduce_sum(tf.exp(x - m), axis=axis)
    )


def flatten_weights(model_weights):
    flattened_weights = []
    for W in model_weights:
        if len(W.shape) > 0:
            flattened_weights = tf.concat(
                [flattened_weights, tf.reshape(W, [-1])], axis=0
            )

    return flattened_weights


def hyperprior(alpha, beta, lam):
    return (
        alpha * tf.math.log(beta)
        - tf.math.lgamma(alpha)
        + (alpha - 1) * tf.math.log(lam)
        - beta * lam
    )


def beta_hyperprior(alpha, beta, pi, eps=1e-8):
    return (
        math.lgamma(alpha + beta)
        - math.lgamma(alpha)
        - math.lgamma(beta)
        + (alpha - 1.0) * tf.math.log(pi + eps)
        + (beta - 1.0) * tf.math.log(1.0 - pi + eps)
    )


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
        model,
        J: int,
        pi0: float = 0.999,
        init_log_sigma2: float = math.log(0.25**2),
        # --- Hyper-priors ON by default (tutorial values) ---
        gamma_alpha: float = 250.0,  # non-zero comps
        gamma_beta: float = 0.1,
        gamma_alpha0: float = 5000.0,  # zero comp
        gamma_beta0: float = 2.0,
        # beta_alpha: float = 500,
        # beta_beta: float = 6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert J >= 2
        self.J = J
        self.pi0 = float(pi0)

        self.gamma_alpha = tf.cast(gamma_alpha, tf.float32)
        self.gamma_beta = gamma_beta
        self.gamma_alpha0 = tf.cast(gamma_alpha0, tf.float32)
        self.gamma_beta0 = gamma_beta0
        # self.beta_alpha = tf.cast(beta_alpha, tf.float32)
        # self.beta_beta = tf.cast(beta_beta, tf.float32)
        self.eps = 1e-8
        self.init_log_sigma2 = init_log_sigma2

        weights = [layer.kernel for layer in model.layers if hasattr(layer, "kernel")]

        flattened_weights = np.array(flatten_weights(weights))

        wmin, wmax = np.min(flattened_weights), np.max(flattened_weights)
        self.init_means = np.linspace(wmin, wmax, J - 1)

        _ = self(weights)

    def build(self, input_shape):
        """Build the layer and create trainable variables."""
        self.mu = self.add_weight(
            name="mu",
            shape=(self.J - 1,),
            initializer=np.ones(self.J - 1) * self.init_means,
            trainable=True,
            dtype=tf.float32,
        )

        self.log_sigma2 = self.add_weight(
            name="log_sigma2",
            shape=(self.J,),
            initializer=np.ones(self.J) * self.init_log_sigma2,
            trainable=True,
            dtype=tf.float32,
        )

        self.pi_logits = self.add_weight(
            name="pi_logits",
            shape=(self.J - 1,),
            initializer="zeros",
            trainable=True,
            dtype=tf.float32,
        )

        super().build(input_shape)

    def mixture_params(
        self,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Get the current mixture parameters: (means, variances, mixing proportions). Can accept parameters directly or use layer's parameters."""
        mu_param = self.mu
        log_sigma2_param = self.log_sigma2
        pi_logits_param = self.pi_logits

        pi_nonzero = tf.nn.softmax(pi_logits_param)

        pi = tf.concat(
            [
                tf.constant([self.pi0], dtype=tf.float32),
                (1.0 - self.pi0) * pi_nonzero,  # type: ignore
            ],
            axis=0,
        )

        mu_final = tf.concat([tf.constant([0.0], dtype=tf.float32), mu_param], axis=0)

        sigma2 = tf.exp(log_sigma2_param)

        sigma2 = tf.clip_by_value(
            sigma2, clip_value_min=1e-8, clip_value_max=float("inf")
        )

        return mu_final, sigma2, pi  # type: ignore

    def log_prob_w(
        self,
        w_flat: tf.Tensor,
    ) -> tf.Tensor:
        """Compute log probability of flattened weights under the mixture model."""
        mu, sigma2, pi = self.mixture_params()

        w = tf.expand_dims(w_flat, axis=1)  # [N, 1]
        mu = tf.expand_dims(mu, axis=0)  # [1, J]
        sigma2 = tf.expand_dims(sigma2, axis=0)  # [1, J]
        pi = tf.expand_dims(pi, axis=0)  # [1, J]

        log_pi = tf.math.log(pi + self.eps)
        log_norm = -0.5 * (
            tf.math.log(2 * math.pi * sigma2) + tf.square(w - mu) / sigma2
        )

        return logsumexp(log_pi + log_norm, axis=1)

    def complexity_loss(self, w_flat) -> tf.Tensor:
        """
        Compute the complexity loss (negative log prior probability) for model weights.
        """
        total_loss = -self.log_prob_w(w_flat)  # type: ignore

        _, sigma2, _ = self.mixture_params()
        lambdas = 1.0 / sigma2  # type: ignore

        # Zero component hyperprior
        gamma_0 = hyperprior(self.gamma_alpha0, self.gamma_beta0, lambdas[0])
        total_loss -= gamma_0

        # Non-zero components hyperprior
        gamma_nonzero = hyperprior(self.gamma_alpha, self.gamma_beta, lambdas[1:])
        total_loss -= tf.reduce_sum(gamma_nonzero)

        # beta_zero = beta_hyperprior(self.beta_alpha, self.beta_beta, self.pi0)
        # total_loss -= beta_zero

        return total_loss

    def call(self, inputs: List[tf.Variable]):
        """
        Computes the complexity loss and adds it to the layer's losses.
        """
        w_flat = flatten_weights(inputs)
        complexity_loss = self.complexity_loss(w_flat)
        self.add_loss(complexity_loss)
        return inputs[0]

    def get_config(self):
        """Get configuration for serialization."""
        config = super().get_config()
        config.update(
            {
                "J": self.J,
                "pi0": self.pi0,
                "init_log_sigma2": self.init_log_sigma2,
                "gamma_alpha": self.gamma_alpha,
                "gamma_beta": self.gamma_beta,
                "gamma_alpha0": self.gamma_alpha0,
                "gamma_beta0": self.gamma_beta0,
            }
        )
        return config

    def merge_components(self, kl_threshold: float = 1e-7, max_iter: int = 200):
        """
        Merge similar components in the mixture model.
        This is a simplified version for Keras implementation.
        """
        # Get current mixture parameters
        mu, sigma2, pi = self.mixture_params()

        # Convert to numpy for processing
        mu_np = mu.numpy()  # type: ignore
        sigma2_np = sigma2.numpy()  # type: ignore
        pi_np = pi.numpy()  # type: ignore

        for _ in range(max_iter):
            merged = False
            for i in range(1, len(mu_np)):
                for j in range(i + 1, len(mu_np)):
                    kl_ij = self._kl_gauss(
                        mu_np[i], sigma2_np[i], mu_np[j], sigma2_np[j]
                    )
                    kl_ji = self._kl_gauss(
                        mu_np[j], sigma2_np[j], mu_np[i], sigma2_np[i]
                    )
                    d = 0.5 * (kl_ij + kl_ji)

                    if d < kl_threshold:
                        pnew = pi_np[i] + pi_np[j]
                        if pnew <= 0:
                            continue

                        mu_new = (pi_np[i] * mu_np[i] + pi_np[j] * mu_np[j]) / pnew
                        s2_new = (
                            pi_np[i] * sigma2_np[i] + pi_np[j] * sigma2_np[j]
                        ) / pnew

                        mu_np[i] = mu_new
                        sigma2_np[i] = s2_new
                        pi_np[i] = pnew

                        mu_np = np.concatenate([mu_np[:j], mu_np[j + 1 :]])
                        sigma2_np = np.concatenate([sigma2_np[:j], sigma2_np[j + 1 :]])
                        pi_np = np.concatenate([pi_np[:j], pi_np[j + 1 :]])

                        merged = True
                        break
                if merged:
                    break
            if not merged:
                break

        # Update parameters
        if len(mu_np) > 1:
            self.mu = mu_np[1:]
            self.log_sigma2 = np.log(sigma2_np)
            pi_nonzero = pi_np[1:]
            pi_nonzero = pi_nonzero / (np.sum(pi_nonzero) + 1e-12)
            self.pi_logits = np.log(pi_nonzero + 1e-12)

    def _kl_gauss(self, mu0, s0, mu1, s1) -> float:
        """Compute KL divergence between two Gaussian distributions."""
        return 0.5 * (np.log(s1 / s0) + (s0 + (mu0 - mu1) ** 2) / s1 - 1.0).item()

    def quantize_model(self, model, skip_last_matrix: bool = True):
        """
        Hard-quantize model weights to mixture means.

        Args:
            model: The Keras model to quantize
            skip_last_matrix: Whether to skip quantizing the last weight matrix
        """
        mu, sigma2, _ = self.mixture_params()

        mu_np = mu.numpy()  # type: ignore
        sigma2_np = sigma2.numpy()  # type: ignore

        mu_expanded = np.expand_dims(mu_np, axis=0)  # [1, J]
        inv_s2 = np.expand_dims(1.0 / sigma2_np, axis=0)  # [1, J]

        weight_layers = []
        for layer in model.layers:
            if hasattr(layer, "kernel"):  # Dense, Conv2D layers
                weight_layers.append((layer, "kernel"))
            elif hasattr(layer, "weights") and len(layer.weights) > 0:
                for i, w in enumerate(layer.weights):
                    if "kernel" in w.name or "weight" in w.name:
                        weight_layers.append((layer, i))

        last_2d_idx = -1
        for idx, (layer, param_name) in enumerate(weight_layers):
            if hasattr(layer, "kernel"):
                if len(layer.kernel.shape) >= 2:
                    last_2d_idx = idx

        for idx, (layer, param_name) in enumerate(weight_layers):
            if skip_last_matrix and idx == last_2d_idx:
                continue

            if isinstance(param_name, str):
                weight_tensor = getattr(layer, param_name)
            else:
                weight_tensor = layer.weights[param_name]

            weight_values = weight_tensor.numpy()
            w_flat = weight_values.flatten()
            w_expanded = np.expand_dims(w_flat, axis=1)  # [N, 1]

            scores = -0.5 * ((w_expanded - mu_expanded) ** 2 * inv_s2)

            best_component_idx = np.argmax(scores, axis=1)
            quantized_flat = mu_np[best_component_idx]
            zero_mask = best_component_idx == 0
            quantized_flat[zero_mask] = 0.0
            quantized_values = quantized_flat.reshape(weight_values.shape)
            weight_tensor.assign(quantized_values)
            return model

    def initialize_weights_from_mixture(self, model):
        """
        Initialize model weights by sampling from the current mixture of Gaussians distribution.

        Args:
            model: The Keras model whose weights will be initialized
        """
        mu, sigma2, pi = self.mixture_params()

        mu_np = mu.numpy()  # type: ignore
        sigma2_np = sigma2.numpy()  # type: ignore
        pi_np = pi.numpy()  # type: ignore

        weight_layers = []
        for layer in model.layers:
            if hasattr(layer, "kernel"):  # Dense, Conv2D layers
                weight_layers.append((layer, "kernel"))
            elif hasattr(layer, "weights") and len(layer.weights) > 0:
                for i, w in enumerate(layer.weights):
                    if "kernel" in w.name or "weight" in w.name:
                        weight_layers.append((layer, i))

        for layer, param_name in weight_layers:
            if isinstance(param_name, str):
                weight_tensor = getattr(layer, param_name)
            else:
                weight_tensor = layer.weights[param_name]

            weight_shape = weight_tensor.shape
            n_weights = tf.reduce_prod(weight_shape)

            # Sample component assignments based on mixing proportions
            component_indices = np.random.choice(
                len(pi_np), size=int(n_weights), p=pi_np
            )

            # Sample from the selected components
            samples = np.zeros(int(n_weights))
            for i, comp_idx in enumerate(component_indices):
                samples[i] = np.random.normal(
                    loc=mu_np[comp_idx], scale=np.sqrt(sigma2_np[comp_idx])
                )

            # Reshape to original weight shape and assign
            sampled_weights = samples.reshape(weight_shape)
            weight_tensor.assign(sampled_weights)
