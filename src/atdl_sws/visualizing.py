"""
Keras implementation of training visualization for soft weight sharing.

This module provides functionality for creating animated visualizations of the soft weight sharing process.
"""

import os
import numpy as np
import imageio
import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf


def flatten_weights(model) -> tf.Tensor:
    """Return every trainable weight of `model` as one 1-D tensor."""
    flats = [
        tf.reshape(v, [-1])  # flatten each matrix / vector
        for v in model.trainable_variables
    ]  # every kernel, bias, …
    return tf.concat(flats, axis=0)


class TrainingGifVisualizer:
    """
    Keras equivalent of the PyTorch TrainingGifVisualizer.
    - Captures pretrained weights (epoch 0) vs. current weights each epoch.
    - Overlays mixture means and +/- 2 std bands from the learned prior.
    - Writes per-epoch PNGs and composes them into a GIF at the end.

    Usage:
      viz = TrainingGifVisualizer(out_dir=..., tag="retraining")
      viz.on_train_begin(model, prior, total_epochs=epochs)
      viz.on_epoch_end(epoch, model, prior, test_acc=acc)   # call each epoch (epoch >= 1)
      path = viz.on_train_end()  # returns GIF path
    """

    def __init__(
        self,
        out_dir: str,
        tag: str = "retraining",
        framerate: int = 2,
        notebook_display: bool = False,
        cleanup_frames: bool = True,
    ):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.tag = tag
        self.frames_dir = os.path.join(out_dir, f"{tag}_frames")
        os.makedirs(self.frames_dir, exist_ok=True)
        self.gif_path = os.path.join(out_dir, f"{tag}.gif")
        self.framerate = framerate
        self.notebook_display = notebook_display
        self.cleanup_frames = cleanup_frames
        self._w0 = None
        self.total_epochs = None

        # Import IPython display if needed
        if self.notebook_display:
            try:
                from IPython import display as ipython_display

                self.ipython_display = ipython_display
            except ImportError:
                print("Warning: IPython not available, notebook_display disabled")
                self.notebook_display = False

    def on_train_begin(self, model, prior, total_epochs: int):
        self.total_epochs = total_epochs
        self._w0 = flatten_weights(model)
        self._make_frame(epoch=0, model=model, prior=prior, test_acc=None)

    def on_epoch_end(
        self, epoch: int, model, prior, test_acc=None, title_extra: str = ""
    ):
        self._make_frame(
            epoch=epoch,
            model=model,
            prior=prior,
            test_acc=test_acc,
            title_extra=title_extra,
        )

    def on_train_end(self) -> str:
        frames = []
        frame_paths = []
        for e in range(0, (self.total_epochs or 0) + 1):
            fp = os.path.join(self.frames_dir, f"frame_{e:03d}.png")
            if os.path.exists(fp):
                frames.append(imageio.imread(fp))
                frame_paths.append(fp)
        if frames:
            imageio.mimsave(
                self.gif_path, frames, duration=1.0 / max(1, self.framerate)
            )

            if self.cleanup_frames:
                for fp in frame_paths:
                    os.remove(fp)
                try:
                    os.rmdir(self.frames_dir)
                except OSError:
                    pass  # Directory not empty or doesn't exist - ignore
        return self.gif_path

    def _make_frame(
        self, epoch: int, model, prior, test_acc=None, title_extra: str = ""
    ):
        if self._w0 is None:
            return

        if self.notebook_display:
            self.ipython_display.clear_output(wait=True)

        wT = flatten_weights(model)
        w0 = self._w0

        # Make sure both arrays have the same length (in case model changed)
        min_len = min(len(w0), len(wT))
        w0 = w0[:min_len]
        wT = wT[:min_len]

        # Full random permutation per epoch (like Keras, not a fixed subset)
        # Use epoch-based seed for reproducibility
        rng = np.random.RandomState(seed=epoch + 12345)
        I = rng.permutation(len(w0))

        # Get mixture params
        mu, sigma2, _ = prior.mixture_params()
        # Convert to numpy
        mu = mu.numpy()  # type: ignore
        sigma2 = sigma2.numpy()  # type: ignore
        sigma2 = np.squeeze(sigma2)
        std = np.sqrt(sigma2)

        x0 = -1.2
        x1 = 1.2

        sns.set(style="whitegrid", rc={"figure.figsize": (8, 8)})
        g = sns.jointplot(
            x=tf.gather(w0, I),
            y=tf.gather(wT, I),
            height=8,
            kind="scatter",
            color="g",
            marker="o",
            joint_kws={"s": 8, "edgecolor": "w"},
            marginal_kws=dict(bins=1000),
            ratio=4,
        )
        ax = g.ax_joint

        for k, (muk, stdk) in enumerate(zip(mu, std)):
            ax.hlines(muk, x0, x1, lw=0.5)
            if k == 0:
                ax.fill_between(
                    [x0, x1],
                    [muk - 2 * stdk, muk - 2 * stdk],
                    [muk + 2 * stdk, muk + 2 * stdk],
                    color="blue",
                    alpha=0.1,
                )
            else:
                ax.fill_between(
                    [x0, x1],
                    [muk - 2 * stdk, muk - 2 * stdk],
                    [muk + 2 * stdk, muk + 2 * stdk],
                    color="red",
                    alpha=0.1,
                )

        g.set_axis_labels("Pretrained", "Retrained")

        g.ax_marg_x.set_xlim(-1, 1)
        g.ax_marg_y.set_ylim(-1, 1)

        try:
            g.ax_marg_y.set_xscale("log")
        except Exception:
            pass

        title = f"Epoch: {epoch} /{self.total_epochs or 0}"
        if test_acc is not None:
            title += f"\nTest accuracy: {test_acc:.4f} "
        if title_extra:
            title += f"\n{title_extra}"

        g.fig.text(
            0.98,
            0.98,
            title,
            transform=g.fig.transFigure,
            fontsize=12,
            ha="right",
            va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="gray"),
        )

        if self.notebook_display:
            self.ipython_display.display(g.fig)

        # Save frame
        fn = os.path.join(self.frames_dir, f"frame_{epoch:03d}.png")
        g.savefig(fn, bbox_inches="tight", dpi=100)  # DPI 100 like Keras
        plt.close(g.fig)
