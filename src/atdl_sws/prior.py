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
        init_means: Optional[np.ndarray] = None,
        init_log_sigma2: float = math.log(0.25**2),
        # --- Hyper-priors ON by default (tutorial values) ---
        gamma_alpha: Optional[float] = 250.0,  # non-zero comps
        gamma_beta: Optional[float] = 0.1,
        gamma_alpha0: Optional[float] = 5000.0,  # zero comp
        gamma_beta0: Optional[float] = 2.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert J >= 2
        self.J = J
        self.pi0 = float(pi0)

        self.gamma_alpha = gamma_alpha
        self.gamma_beta = gamma_beta
        self.gamma_alpha0 = gamma_alpha0
        self.gamma_beta0 = gamma_beta0
        self.eps = 1e-8
        self.init_log_sigma2 = init_log_sigma2

        if init_means is None:
            self.init_means = np.linspace(-0.6, 0.6, J - 1)
        else:
            self.init_means = init_means

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
        mu=None,
        log_sigma2=None,
        pi_logits: Optional[tf.Variable] = None,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Get the current mixture parameters: (means, variances, mixing proportions). Can accept parameters directly or use layer's parameters."""
        mu_param = mu or self.mu
        log_sigma2_param = log_sigma2 or self.log_sigma2
        pi_logits_param = pi_logits or self.pi_logits

        pi_nonzero = tf.nn.softmax(pi_logits_param)

        pi = tf.concat(
            [
                tf.constant([self.pi0], dtype=tf.float32),
                (1.0 - self.pi0) * pi_nonzero,  # type: ignore
            ],
            axis=0,
        )

        mu_final = tf.concat([tf.constant([0.0], dtype=tf.float32), mu_param], axis=0)

        sigma2 = (tf.exp(log_sigma2_param),)

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

    def complexity_loss(self, model_weights: List[tf.Variable]) -> tf.Tensor:
        """
        Compute the complexity loss (negative log prior probability) for model weights.
        """
        flattened_weights = []
        for W in model_weights:
            if len(W.shape) > 0:
                flattened_weights.append(tf.reshape(W, [-1]))

        if not flattened_weights:
            raise Exception("Complexity loss not computed")

        w_flat = tf.concat(flattened_weights, axis=0)

        log_prob_w = self.log_prob_w(w_flat)  # type: ignore
        total_loss = -tf.reduce_sum(log_prob_w)

        _, sigma2, _ = self.mixture_params()
        lambdas = 1.0 / sigma2  # type: ignore

        # Zero component hyperprior
        if self.gamma_alpha0 is not None and self.gamma_beta0 is not None:
            a0, b0 = self.gamma_alpha0, self.gamma_beta0
            lambda0 = lambdas[0]
            gamma_term_0 = (
                a0 * tf.math.log(b0)
                - tf.math.lgamma(a0)
                + (a0 - 1.0) * tf.math.log(lambda0)
                - b0 * lambda0
            )
            total_loss -= gamma_term_0

        # Non-zero components hyperprior
        if self.gamma_alpha is not None and self.gamma_beta is not None:
            a, b = self.gamma_alpha, self.gamma_beta
            lambdas = lambdas[1:]
            gamma_terms_nonzero = (a * tf.math.log(b) - tf.math.lgamma(a)) + (
                (a - 1.0) * tf.math.log(lambdas) - b * lambdas
            )
            total_loss -= tf.reduce_sum(gamma_terms_nonzero)

        return total_loss

    def call(self, inputs: List[tf.Variable]):
        """
        Computes the complexity loss and adds it to the layer's losses.
        """
        complexity_loss = self.complexity_loss(inputs)
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

    def merge_components(self, kl_threshold: float = 1e-10, max_iter: int = 200):
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
            # Update the trainable variables
            self.mu.assign(mu_np[1:])  # Exclude zero component
            self.log_sigma2.assign(np.log(sigma2_np[1:]))  # Exclude zero component
            pi_nonzero = pi_np[1:]  # Exclude zero component
            pi_nonzero = pi_nonzero / (np.sum(pi_nonzero) + 1e-12)  # Normalize
            self.pi_logits.assign(np.log(pi_nonzero + 1e-12))  # Convert to logits

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
        inv_s2 = 1.0 / sigma2_np
        inv_s2 = np.expand_dims(inv_s2, axis=0)  # [1, J]

        weight_layers = []
        for layer in model.layers:
            if hasattr(layer, "kernel"):  # Dense, Conv2D layers
                weight_layers.append((layer, "kernel"))
            elif hasattr(layer, "weights") and len(layer.weights) > 0:
                # Other layers that might have weights
                for i, w in enumerate(layer.weights):
                    if "kernel" in w.name or "weight" in w.name:
                        weight_layers.append((layer, i))

        # Last 2D weight index (for potentially skipping)
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


def init_mixture(
    model,
    J: int,
    pi0: float,
    init_means_mode: str = "from_weights",
    init_range_min: float = -0.6,
    init_range_max: float = 0.6,
    init_sigma: float = 0.25,
):
    """
    Initialize a mixture prior based on the model's weights.
    """
    # Get all trainable weights from dense and conv layers
    weight_values = []
    for layer in model.layers:
        if hasattr(layer, "kernel"):  # Dense, Conv2D layers
            weight_values.append(layer.kernel.numpy())
        elif hasattr(layer, "weights"):
            for w in layer.weights:
                if "kernel" in w.name or "weight" in w.name:
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
        init_means=means,
        init_log_sigma2=math.log(init_sigma**2),
    )

    return prior
