"""
Module for storing and managing metrics and arguments for SWS experiments.
"""

import os
import json
import sys
import platform
import time
from typing import Dict, Any, Optional, List
import numpy as np
import tensorflow as tf

# Import plotting libraries
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns


class Metrics:
    """
    Metrics class to store various experiment data including retraining arguments,
    environment information, layer pruning statistics, summary metrics, and plotting equivalents.
    """

    def __init__(
        self,
        args,
        model,
        save_dir,
        pre_acc=None,
        post_acc=None,
        quantized_acc=None,
    ):
        self.retraining_args = args
        self.store_environment()
        self.store_environment()
        self.store_layer_pruning(model)

        if quantized_acc:
            self.store_summary_metrics(
                model, pre_acc, post_acc, acc_quantized=quantized_acc
            )

        self.save_dir = save_dir
        self.plot_data = {}

    def _create_figure(
        self, figsize=(8, 6), xlabel=None, ylabel=None, title=None, tight_layout=True
    ):
        """Create a matplotlib figure and axis with consistent formatting."""
        fig, ax = plt.subplots(figsize=figsize)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)
        if tight_layout:
            fig.tight_layout()
        return fig, ax

    def store_environment(self) -> Dict[str, Any]:
        """
        Store the current environment information.

        Returns:
            Dict containing environment information
        """
        self.environment_info = {
            "python": sys.version.replace("\n", " "),
            "platform": f"{platform.system()} {platform.release()}",
            "tensorflow_version": tf.__version__,
            "device": ("gpu" if tf.config.list_physical_devices("GPU") else "cpu"),
            "cuda_is_available": bool(tf.config.list_physical_devices("GPU")),
        }
        return self.environment_info

    def store_layer_pruning(self, model, **kwargs) -> Dict[str, Any]:
        """
        Store layer pruning statistics for the model.

        Args:
            model: The trained model
            **kwargs: Additional parameters for layer pruning

        Returns:
            Dict containing layer pruning statistics
        """
        stats = []
        for layer in model.layers:
            if hasattr(layer, "kernel"):
                W = layer.kernel
                W_np = W.numpy()
                total = W_np.size
                nnz = int((W_np != 0).sum())
                stats.append(
                    {
                        "layer": f"{layer.name}({layer.__class__.__name__})",
                        "shape": list(W_np.shape),
                        "total": total,
                        "nnz": nnz,
                        "sparsity_pct": 100.0 * (1 - nnz / max(1, total)),
                    }
                )

        self.layer_pruning_data = {"timestamp": time.time(), "layer_stats": stats}
        return self.layer_pruning_data

    def store_summary_metrics(
        self, model, pre_acc, post_acc, **kwargs
    ) -> Dict[str, Any]:
        """
        Store summary metrics for the experiment.

        Args:
            pre_acc: Pre-retraining accuracy
            post_acc: Post-retraining accuracy
            model: The trained model
            **kwargs: Additional metrics

        Returns:
            Dict containing summary metrics
        """
        # Calculate total parameters and non-zero parameters
        total_params = sum([tf.size(var).numpy() for var in model.trainable_variables])
        model_weights = [
            layer.kernel.numpy() for layer in model.layers if hasattr(layer, "kernel")
        ]
        nnz_total = sum([(w != 0).sum() for w in model_weights])
        nz_pct = 100.0 * nnz_total / max(1, total_params) if total_params > 0 else 0

        err_pre = 1.0 - float(pre_acc)
        err_post = 1.0 - float(post_acc)
        delta_err = (err_post - err_pre) * 100.0

        summary = {
            "top1_error_pre[%]": err_pre * 100.0,
            "top1_error_post[%]": err_post * 100.0,
            "Delta[%]": delta_err,
            "|W|": int(total_params),
            "|W_nonzero|/|W|[%]": float(nz_pct),
            "acc_pretrain": float(pre_acc),
            "acc_retrain": float(post_acc),
        }

        # Add any additional metrics passed in kwargs
        summary.update(kwargs)

        self.summary_metrics_data = summary
        return self.summary_metrics_data

    def plot_mixture_components(
        self, prior, checkpoint_weights=None, **kwargs
    ) -> Dict[str, Any]:
        """
        Equivalent to plot_mixture.py - plot mixture components and overlay weights if available.

        Args:
            prior: The mixture prior object
            checkpoint_weights: Optional weights to overlay on the mixture plot
            **kwargs: Additional parameters

        Returns:
            Dict containing mixture component data
        """
        # Extract mixture parameters from prior
        mu, sigma2, pi = prior.mixture_params()
        mu_np = mu.numpy() if hasattr(mu, "numpy") else np.array(mu)
        sigma2_np = sigma2.numpy() if hasattr(sigma2, "numpy") else np.array(sigma2)
        pi_np = pi.numpy() if hasattr(pi, "numpy") else np.array(pi)

        # Store mixture component data
        mixture_data = {
            "mu": mu_np.tolist(),
            "sigma2": sigma2_np.tolist(),
            "pi": pi_np.tolist(),
            "std": np.sqrt(sigma2_np).tolist(),
        }

        # Store weights data if provided
        if checkpoint_weights is not None and len(checkpoint_weights) > 0:
            mixture_data["weights_flat"] = checkpoint_weights.tolist()

        self.plot_data["mixture_components"] = mixture_data

        # Create the actual plot
        self.plot_data["mixture_components_plot"] = {
            "mu": mu_np,
            "sigma": np.sqrt(sigma2_np),
            "pi": pi_np,
            "weights_flat": checkpoint_weights,
        }

        return mixture_data

    def plot_weights_scatter(
        self, initial_weights, final_weights, mixture_params=None, **kwargs
    ) -> Dict[str, Any]:
        """
        Equivalent to plot_weights_scatter.py - plot scatter of weights from before to after training.

        Args:
            initial_weights: Initial weights
            final_weights: Final weights after training
            mixture_params: Optional mixture parameters for bands
            **kwargs: Additional parameters

        Returns:
            Dict containing scatter plot data
        """
        n = min(len(initial_weights), len(final_weights))
        if n == 0:
            return {}

        # Sample weights if too many
        sample_size = kwargs.get("sample_size", 20000)
        idx = np.random.choice(n, size=min(sample_size, n), replace=False)
        w0s = (
            initial_weights[idx]
            if hasattr(initial_weights, "shape")
            else np.array(initial_weights)[idx]
        )
        wTs = (
            final_weights[idx]
            if hasattr(final_weights, "shape")
            else np.array(final_weights)[idx]
        )

        scatter_data = {
            "initial_weights_sampled": w0s.tolist(),
            "final_weights_sampled": wTs.tolist(),
            "sample_size": len(w0s),
        }

        # Add mixture bands if provided
        if mixture_params is not None:
            scatter_data["mixture_bands"] = {
                "mu": mixture_params[0].numpy().tolist()
                if hasattr(mixture_params[0], "numpy")
                else mixture_params[0].tolist(),
                "sigma": np.sqrt(mixture_params[1].numpy()).tolist()
                if hasattr(mixture_params[1], "numpy")
                else np.sqrt(mixture_params[1]).tolist(),
                "pi": mixture_params[2].numpy().tolist()
                if hasattr(mixture_params[2], "numpy")
                else mixture_params[2].tolist(),
            }

        self.plot_data["weights_scatter"] = scatter_data

        # Store plot parameters for later plotting
        self.plot_data["weights_scatter_plot"] = {
            "initial_weights": w0s,
            "final_weights": wTs,
            "mixture_params": mixture_params,
        }

        return scatter_data

    def plot_curves(
        self,
        retrain_metrics: Optional[Dict] = None,
        pretrain_metrics: Optional[Dict] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Equivalent to plot_curves.py - plot training curves for loss, complexity, and test accuracy.

        Args:
            pretrain_metrics: Metrics from pretraining phase
            retrain_metrics: Metrics from retraining phase
            **kwargs: Additional parameters

        Returns:
            Dict containing curve plot data
        """
        curve_data = {}

        if pretrain_metrics:
            curve_data["pretrain"] = pretrain_metrics

        if retrain_metrics:
            curve_data["retrain"] = retrain_metrics

        self.plot_data["curves"] = curve_data
        self.plot_data["curves_plot"] = {
            "pretrain_metrics": pretrain_metrics,
            "retrain_metrics": retrain_metrics,
        }

        return curve_data

    def plot_mixture_dynamics(
        self, mixture_history: List[Dict], **kwargs
    ) -> Dict[str, Any]:
        """
        Equivalent to plot_mixture_dynamics.py - plot how mixture parameters change over time.

        Args:
            mixture_history: List of mixture parameters over epochs
            **kwargs: Additional parameters

        Returns:
            Dict containing mixture dynamics data
        """
        dynamics_data = {
            "mixture_history": mixture_history,
            "num_epochs": len(mixture_history),
        }

        self.plot_data["mixture_dynamics"] = dynamics_data
        self.plot_data["mixture_dynamics_plot"] = {"mixture_history": mixture_history}

        return dynamics_data

    def plot_filters(self, model, layer_idx: int = 0, **kwargs) -> Dict[str, Any]:
        """
        Equivalent to plot_filters.py - visualize filters from convolutional layers.

        Args:
            model: The trained model
            layer_idx: Index of the layer to visualize
            **kwargs: Additional parameters

        Returns:
            Dict containing filter visualization data
        """
        filters_data = {}

        # Find conv layers in the model
        conv_layers = []
        for i, layer in enumerate(model.layers):
            if hasattr(layer, "kernel") and len(layer.kernel.shape) == 4:  # Conv layer
                conv_layers.append((i, layer))

        if conv_layers and len(conv_layers) > layer_idx:
            layer_idx, conv_layer = conv_layers[layer_idx]
            kernel = conv_layer.kernel.numpy()

            filters_data = {
                "layer_index": layer_idx,
                "layer_name": conv_layer.name,
                "kernel_shape": kernel.shape.tolist(),
                "filters": kernel.tolist(),  # This could be quite large, consider sampling
            }

        self.plot_data["filters"] = filters_data
        self.plot_data["filters_plot"] = {
            "model": model,
            "layer_idx": layer_idx,
            "conv_layers": conv_layers,
        }

        return filters_data

    def plot_pareto(self, experiments_data: List[Dict], **kwargs) -> Dict[str, Any]:
        """
        Equivalent to plot_pareto.py - plot Pareto front showing trade-offs between accuracy and compression.

        Args:
            experiments_data: List of experiment results with accuracy and compression metrics
            **kwargs: Additional parameters

        Returns:
            Dict containing Pareto plot data
        """
        pareto_data = {
            "experiments": experiments_data,
            "num_experiments": len(experiments_data),
        }

        self.plot_data["pareto"] = pareto_data
        self.plot_data["pareto_plot"] = {"experiments_data": experiments_data}

        return pareto_data

    def _generate_mixture_components_plot(self, plot_dir: str):
        """Generate the mixture components plot."""
        if "mixture_components_plot" not in self.plot_data:
            return

        plot_data = self.plot_data["mixture_components_plot"]
        mu = plot_data["mu"]
        sigma = plot_data["sigma"]
        pi = plot_data["pi"]
        weights_flat = plot_data["weights_flat"]

        # Create the scatter plot
        fig, ax = self._create_figure(
            figsize=(6, 5),
            xlabel="component mean μ",
            ylabel="σ",
            title="Mixture components (size ∝ π)",
        )
        xs = mu
        stds = sigma
        pis = pi
        sizes = 300.0 * (pis / (pis.max() + 1e-12))
        ax.scatter(xs, stds, s=sizes, alpha=0.8)
        out1 = os.path.join(plot_dir, "plot_mixture_components.png")
        fig.savefig(out1, dpi=150)
        plt.close(fig)

        # Also create histogram of weights + mixture PDF overlay if weights are available
        if weights_flat is not None and len(weights_flat) > 0:
            fig, ax = self._create_figure(
                figsize=(8, 5),
                xlabel="w",
                ylabel="density",
                title="Weight histogram + mixture pdf",
            )
            ax.hist(weights_flat, bins=120, density=True, alpha=0.5, label="weights")
            grid = np.linspace(weights_flat.min(), weights_flat.max(), 1000)
            # Gaussian mixture PDF
            pdf = 0.0
            for j in range(len(xs)):
                pdf += (
                    pis[j]
                    * (1.0 / np.sqrt(2 * np.pi * sigma[j] ** 2))
                    * np.exp(-((grid - mu[j]) ** 2) / (2 * sigma[j] ** 2))
                )
            ax.plot(grid, pdf, linewidth=1.5, label="mixture pdf")
            ax.legend()
            out2 = os.path.join(plot_dir, "plot_weights_mixture.png")
            fig.savefig(out2, dpi=150)
            plt.close(fig)

    def _generate_weights_scatter_plot(self, plot_dir: str):
        """Generate the weights scatter plot."""
        if "weights_scatter_plot" not in self.plot_data:
            return

        plot_data = self.plot_data["weights_scatter_plot"]
        initial_weights = plot_data["initial_weights"]
        final_weights = plot_data["final_weights"]
        mixture_params = plot_data["mixture_params"]

        fig, ax = self._create_figure(
            figsize=(8, 6),
            xlabel="Initial w",
            ylabel="Final w",
            title="Weight movement w0 → wT",
        )
        ax.scatter(initial_weights, final_weights, s=4, alpha=0.4)

        # Add mixture bands if provided
        if mixture_params is not None:
            mu = (
                mixture_params[0].numpy()
                if hasattr(mixture_params[0], "numpy")
                else mixture_params[0]
            )
            sigma = np.sqrt(
                mixture_params[1].numpy()
                if hasattr(mixture_params[1], "numpy")
                else mixture_params[1]
            )
            pi = (
                mixture_params[2].numpy()
                if hasattr(mixture_params[2], "numpy")
                else mixture_params[2]
            )
            x0, x1 = initial_weights.min(), initial_weights.max()
            for m, s, p in zip(mu, sigma, pi):
                ax.fill_between([x0, x1], m - 2 * s, m + 2 * s, alpha=0.08)

        out = os.path.join(plot_dir, "plot_scatter_w0_wT.png")
        fig.savefig(out)
        plt.close(fig)

    def _generate_curves_plot(self, plot_dir: str):
        """Generate the training curves plot."""
        if "curves_plot" not in self.plot_data:
            return

        plot_data = self.plot_data["curves_plot"]
        pretrain_metrics = plot_data["pretrain_metrics"]
        retrain_metrics = plot_data["retrain_metrics"]

        if (
            pretrain_metrics
            and "epoch" in pretrain_metrics
            and "train_ce" in pretrain_metrics
        ):
            fig, ax = self._create_figure(
                figsize=(10, 6),
                xlabel="Epoch",
                ylabel="Cross-Entropy",
                title="Training Cross-Entropy",
            )
            ax.plot(
                pretrain_metrics["epoch"],
                pretrain_metrics["train_ce"],
                label="pretrain CE",
            )
            if retrain_metrics and "train_ce" in retrain_metrics:
                ax.plot(
                    retrain_metrics["epoch"],
                    retrain_metrics["train_ce"],
                    label="retrain CE",
                )
            ax.legend()
            out1 = os.path.join(plot_dir, "plot_train_ce.png")
            fig.savefig(out1)
            plt.close(fig)

        if (
            retrain_metrics
            and "epoch" in retrain_metrics
            and "complexity" in retrain_metrics
        ):
            fig, ax = self._create_figure(
                figsize=(10, 6),
                xlabel="Epoch",
                ylabel="Complexity",
                title="Complexity over Time",
            )
            ax.plot(
                retrain_metrics["epoch"],
                retrain_metrics["complexity"],
                label="complexity",
            )
            ax.legend()
            out2 = os.path.join(plot_dir, "plot_complexity.png")
            fig.savefig(out2)
            plt.close(fig)

        if (
            retrain_metrics
            and "epoch" in retrain_metrics
            and "test_acc" in retrain_metrics
        ):
            fig, ax = self._create_figure(
                figsize=(10, 6),
                xlabel="Epoch",
                ylabel="Accuracy",
                title="Test Accuracy over Time",
            )
            ax.plot(
                retrain_metrics["epoch"], retrain_metrics["test_acc"], label="test acc"
            )
            ax.legend()
            out3 = os.path.join(plot_dir, "plot_test_acc.png")
            fig.savefig(out3)
            plt.close(fig)

    def _generate_mixture_dynamics_plot(self, plot_dir: str):
        """Generate the mixture dynamics plot."""
        if "mixture_dynamics_plot" not in self.plot_data:
            return

        plot_data = self.plot_data["mixture_dynamics_plot"]
        mixture_history = plot_data["mixture_history"]

        if not mixture_history:
            return

        # Extract values for plotting
        epochs = range(len(mixture_history))
        mu_history = []
        sigma_history = []
        pi_history = []

        for epoch_data in mixture_history:
            mu_history.append(epoch_data.get("mu", []))
            sigma_history.append(
                epoch_data.get("sigma2", [])
            )  # Assuming sigma2 is stored
            pi_history.append(epoch_data.get("pi", []))

        # Plot for each mixture component over time
        if mu_history:
            num_components = len(mu_history[0]) if mu_history[0] else 0
            if num_components > 0:
                fig, ax = self._create_figure(
                    figsize=(10, 6),
                    xlabel="Epoch",
                    ylabel="Mean (μ)",
                    title="Mixture Component Means Over Time",
                )
                for j in range(num_components):
                    component_mu = [
                        epoch[j] if j < len(epoch) else np.nan for epoch in mu_history
                    ]
                    ax.plot(
                        epochs,
                        component_mu,
                        label=f"Component {j} μ",
                        linewidth=1.0,
                        alpha=0.9,
                    )
                ax.legend()
                out = os.path.join(plot_dir, "plot_mixture_dynamics.png")
                fig.savefig(out)
                plt.close(fig)

    def _generate_filters_plot(self, plot_dir: str):
        """Generate the filters visualization plot."""
        if "filters_plot" not in self.plot_data:
            return

        plot_data = self.plot_data["filters_plot"]
        model = plot_data["model"]
        layer_idx = plot_data["layer_idx"]
        conv_layers = plot_data["conv_layers"]

        if not conv_layers or len(conv_layers) <= layer_idx:
            return

        layer_idx_actual, conv_layer = conv_layers[layer_idx]
        kernel = conv_layer.kernel.numpy()

        # Handle different kernel shapes and visualize filters
        if (
            len(kernel.shape) == 4
        ):  # Conv2D kernel: (height, width, in_channels, out_channels)
            out_channels = kernel.shape[3]
            in_channels = kernel.shape[2]

            # For visualization, take the first few filters and channels
            rows = min(4, out_channels)  # Limit number of rows for visualization
            cols = min(8, in_channels)  # Limit number of columns for visualization

            fig, axes = plt.subplots(rows, cols, figsize=(1.2 * cols, 1.2 * rows))

            if rows == 1 and cols == 1:
                axes = [[axes]]
            elif rows == 1:
                axes = [axes]
            elif cols == 1:
                axes = [[ax] for ax in axes]

            for i in range(rows):
                for j in range(cols):
                    if i < out_channels and j < in_channels:
                        # Take the first filter from the output channel, visualize the j-th input channel
                        filter_weights = kernel[:, :, j, i]
                        if len(axes) > i and len(axes[i]) > j:
                            axes[i][j].imshow(filter_weights, cmap="viridis")
                            axes[i][j].axis("off")
                            axes[i][j].set_title(f"F{i},C{j}", fontsize=8)

            fig.suptitle(f"Filters from {conv_layer.name}")
            fig.tight_layout()
            out = os.path.join(plot_dir, "plot_filters.png")
            fig.savefig(out)
            plt.close(fig)

    def _generate_pareto_plot(self, plot_dir: str):
        """Generate the Pareto plot."""
        if "pareto_plot" not in self.plot_data:
            return

        plot_data = self.plot_data["pareto_plot"]
        experiments_data = plot_data["experiments_data"]

        if not experiments_data:
            return

        # Extract accuracy and compression ratio values
        accuracies = []
        compression_ratios = []
        labels = []

        for exp in experiments_data:
            if "accuracy" in exp and "compression_ratio" in exp:
                accuracies.append(exp["accuracy"])
                compression_ratios.append(exp["compression_ratio"])
                labels.append(exp.get("name", f"Exp {len(accuracies)}"))

        if not accuracies:
            return

        # Plot Pareto front
        fig, ax = self._create_figure(
            figsize=(10, 6),
            xlabel="Compression Ratio",
            ylabel="Accuracy",
            title="Pareto Front: Accuracy vs Compression",
        )
        ax.scatter(compression_ratios, accuracies)
        ax.grid(True, alpha=0.3)

        # Add labels to points if few points
        if len(accuracies) <= 20:
            for i, label in enumerate(labels):
                ax.annotate(
                    label,
                    (compression_ratios[i], accuracies[i]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                )

        out = os.path.join(plot_dir, "plot_pareto.png")
        fig.savefig(out)
        plt.close(fig)

    def store(self, dir_name: str) -> str:
        """
        Store all collected metrics and data to the specified directory.

        Args:
            dir_name: Directory name to save all metrics

        Returns:
            Path to the directory where metrics were saved
        """
        metrics_dir = os.path.join(dir_name, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)

        # Save retraining arguments
        if self.retraining_args:
            with open(os.path.join(dir_name, "retraining_args.json"), "w") as f:
                json.dump(self.retraining_args, f, indent=2)

        # Save environment information
        if self.environment_info:
            with open(os.path.join(dir_name, "environment.json"), "w") as f:
                json.dump(self.environment_info, f, indent=2)

        # Save layer pruning data
        if self.layer_pruning_data:
            with open(os.path.join(dir_name, "layer_pruning.json"), "w") as f:
                json.dump(self.layer_pruning_data, f, indent=2)

        # Save summary metrics
        if self.summary_metrics_data:
            with open(os.path.join(dir_name, "summary_metrics.json"), "w") as f:
                json.dump(self.summary_metrics_data, f, indent=2)

        # Save plot data
        if self.plot_data:
            with open(os.path.join(dir_name, "plot_data.json"), "w") as f:
                json.dump(
                    {
                        k: v
                        for k, v in self.plot_data.items()
                        if not k.endswith("_plot")
                    },
                    f,
                    indent=2,
                )

        # Create actual plot files
        plot_dir = os.path.join(dir_name, "figures")
        os.makedirs(plot_dir, exist_ok=True)

        self._generate_mixture_components_plot(plot_dir)
        self._generate_weights_scatter_plot(plot_dir)
        self._generate_curves_plot(plot_dir)
        self._generate_mixture_dynamics_plot(plot_dir)
        self._generate_filters_plot(plot_dir)
        self._generate_pareto_plot(plot_dir)
