"""
Methods to compliment the keras engine

Karen Ullrich, Jan 2017
"""
import os
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import imageio

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .helpers import special_flatten


# ---------------------------------------------------------
# helpers
# ---------------------------------------------------------

def collect_trainable_weights(layer):
    """
    Collects all `trainable_weights` attributes,
    excluding any sublayers where `trainable` is set the `False`.
    """
    trainable = getattr(layer, 'trainable', True)
    if not trainable:
        return []
    weights = []
    if layer.__class__.__name__ == 'Sequential':
        for sublayer in layer.layers:  # Updated for TF 2.x
            weights += collect_trainable_weights(sublayer)
    elif hasattr(layer, 'layers'):  # Model/Functional API
        for sublayer in layer.layers:
            weights += collect_trainable_weights(sublayer)
    else:
        weights += layer.trainable_weights
    # dedupe weights
    weights = list(set(weights))
    # Sort by name for consistency
    if weights:
        weights.sort(key=lambda x: x.name)
    return weights


def extract_weights(model):
    """
    Extract symbolic, trainable weights from a Model.
    """
    trainable_weights = []
    for layer in model.layers:
        trainable_weights += collect_trainable_weights(layer)
    return trainable_weights


# ---------------------------------------------------------
# objectives
# ---------------------------------------------------------

def identity_objective(y_true, y_pred):
    """
    Hack to turn Keras' Layer engine into an empirical prior on the weights
    """
    return y_pred


# ---------------------------------------------------------
# logsumexp
# ---------------------------------------------------------

def logsumexp(t, w=None, axis=1):
    """
    t... tensor
    w... weight tensor
    """
    t_max = tf.reduce_max(t, axis=axis, keepdims=True)

    if w is not None:
        tmp = w * tf.exp(t - t_max)
    else:
        tmp = tf.exp(t - t_max)

    out = tf.reduce_sum(tmp, axis=axis)
    out = tf.math.log(out)

    t_max = tf.reduce_max(t, axis=axis)

    return out + t_max


# ---------------------------------------------------------
# Callbacks
# ---------------------------------------------------------

class VisualisationCallback(keras.callbacks.Callback):
    """A callback for visualizing the progress in training."""

    def __init__(self, model, X_test, Y_test, epochs):

        self.model = model
        self.X_test = X_test
        self.Y_test = Y_test
        self.epochs = epochs

        super(VisualisationCallback, self).__init__()

    def on_train_begin(self, logs=None):
        if logs is None:
            logs = {}
        self.W_0 = self.model.get_weights()

    def on_epoch_begin(self, epoch, logs=None):
        if logs is None:
            logs = {}
        self.plot_histogram(epoch)

    def on_train_end(self, logs=None):
        if logs is None:
            logs = {}
        self.plot_histogram(epoch=self.epochs)
        images = []
        filenames = ["./.tmp%d.png" % epoch for epoch in range(self.epochs + 1)]
        for filename in filenames:
            if os.path.exists(filename):
                images.append(imageio.imread(filename))
                os.remove(filename)
        if images:
            imageio.mimsave('./figures/retraining.gif', images, duration=.5)

    def plot_histogram(self, epoch):
        # get network weights
        W_T = self.model.get_weights()
        W_0 = self.W_0
        weights_0 = np.squeeze(special_flatten([w for w in W_0 if len(w.shape) > 0]))  # Filter out shape issues
        weights_T = np.squeeze(special_flatten([w for w in W_T if len(w.shape) > 0]))
        
        # Limit the number of points for performance
        if len(weights_0) > 5000:
            indices = np.random.choice(len(weights_0), 5000, replace=False)
            weights_0 = weights_0[indices]
            weights_T = weights_T[indices]
        
        # get means, variances and mixing proportions
        # Assuming the last 3 arrays in W_T are the prior parameters
        pi_T = W_T[-1] if len(W_T) > 0 else np.array([0.99])
        mu_T = np.concatenate([np.zeros(1), W_T[-3] if len(W_T) >= 3 else np.array([0.1])]).flatten()
        
        # plot histograms and GMM
        x0 = -1.2
        x1 = 1.2
        I = np.random.permutation(len(weights_0))
        
        plt.figure(figsize=(12, 8))
        plt.scatter(weights_0[I], weights_T[I], alpha=0.5, s=1, color="g")
        
        # Add horizontal lines for means
        for k in range(len(mu_T)):
            if k == 0:
                plt.axhline(y=mu_T[k], color='blue', alpha=0.3, linewidth=0.5)
                plt.axhspan(mu_T[k] - 0.1, mu_T[k] + 0.1, alpha=0.1, color='blue')
            else:
                plt.axhline(y=mu_T[k], color='red', alpha=0.3, linewidth=0.5)
                plt.axhspan(mu_T[k] - 0.1, mu_T[k] + 0.1, alpha=0.1, color='red')
        
        # Evaluate model accuracy
        score = self.model.evaluate({'input': self.X_test}, {"error_loss": self.Y_test}, verbose=0)
        if isinstance(score, (list, tuple)) and len(score) > 1:
            accuracy = score[1]  # Assume accuracy is at index 1
        else:
            accuracy = score[0] if isinstance(score, (list, tuple)) else score

        plt.title("Epoch: %d /%d\nTest accuracy: %.4f " % (epoch, self.epochs, accuracy))
        plt.xlabel("Pretrained")
        plt.ylabel("Retrained")
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.tight_layout()
        
        plt.savefig("./.tmp%d.png" % epoch, bbox_inches='tight')
        plt.close()