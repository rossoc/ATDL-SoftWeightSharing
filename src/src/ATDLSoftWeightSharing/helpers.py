"""
Methods for general purpose

Karen Ullrich, Sep 2016
"""

import numpy as np
from scipy.special import logsumexp as scipy_logsumexp

import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------
# RESHAPING LISTS FILLED WITH ARRAYS
# ---------------------------------------------------------

def special_flatten(arraylist):
    """
    Flattens the output of model.get_weights()
    """
    # Filter out arrays with no elements (empty arrays)
    valid_arrays = [array for array in arraylist if array.size > 0]
    if not valid_arrays:
        return np.array([]).reshape((0, 1))
    out = np.concatenate([array.flatten() for array in valid_arrays])
    return out.reshape((len(out), 1))


def reshape_like(in_array, shaped_array):
    """
    Inverts special_flatten
    """
    if isinstance(in_array, np.ndarray) and in_array.ndim == 1:
        in_array = in_array.reshape(-1, 1)
    flattened_array = list(in_array.flatten())
    out = []
    for array in shaped_array:
        num_samples = array.size
        if len(flattened_array) >= num_samples:
            dummy = flattened_array[:num_samples]
            del flattened_array[:num_samples]
            out.append(np.asarray(dummy).reshape(array.shape))
        else:
            # If not enough elements, pad with zeros
            dummy = list(dummy) + [0] * (num_samples - len(dummy))
            out.append(np.asarray(dummy).reshape(array.shape))
    return out


# ---------------------------------------------------------
# DISCRETESIZE
# ---------------------------------------------------------

def merger(inputs):
    """
    Comparing and merging components.
    """
    for _ in range(3):
        lists = []
        for inpud in inputs:
            for i in inpud:
                tmp = 1
                for l in lists:
                    if i in l:
                        for j in inpud:
                            if j not in l:  # Add element only if not already in list
                                l.append(j)
                        tmp = 0
                if tmp == 1:
                    lists.append(list(inpud))
        lists = [np.unique(l) for l in lists]
        inputs = lists
    return lists


def KL(means, logprecisions):
    """
    Compute the KL-divergence between 2 Gaussian Components.
    """
    precisions = np.exp(logprecisions)
    return 0.5 * (logprecisions[0] - logprecisions[1]) + precisions[1] / 2. * (
        1. / precisions[0] + (means[0] - means[1]) ** 2) - 0.5


def compute_responsibilies(xs, mus, logprecisions, pis):
    """
    Computing the unnormalized responsibilities.
    """
    xs = xs.flatten()
    K = len(pis)
    responsibilies = np.zeros((K, len(xs)))
    for k in range(K):
        # Not normalized!!!
        responsibilies[k] = pis[k] * np.exp(0.5 * logprecisions[k]) * np.exp(
            - np.exp(logprecisions[k]) / 2 * (xs - mus[k]) ** 2)
    return np.argmax(responsibilies, axis=0)


def discretesize(W, pi_zero=0.999):
    """
    Discrete quantization of weights
    """
    # Get the weight parameters (excluding the prior parameters)
    weight_params = W[:-3] if len(W) >= 3 else W
    weights = special_flatten(weight_params).flatten()

    # Get the prior parameters
    means = np.concatenate([np.zeros(1), W[-3]]) if len(W) >= 3 else np.array([0.0])
    logprecisions = W[-2] if len(W) >= 2 else np.array([1.0])
    logpis = np.concatenate([np.log([pi_zero]), W[-1]]) if len(W) >= 1 else np.array([np.log(pi_zero)])

    # Classes K
    J = len(logprecisions)
    
    # Compute KL-divergence
    K_matrix = np.zeros((J, J))
    L = np.zeros((J, J))

    for i, (m1, pr1, pi1) in enumerate(zip(means, logprecisions, logpis)):
        for j, (m2, pr2, pi2) in enumerate(zip(means, logprecisions, logpis)):
            K_matrix[i, j] = KL([m1, m2], [pr1, pr2])
            L[i, j] = np.exp(pi1) * (pi1 - pi2 + K_matrix[i, j])

    # merge
    idx, idy = np.where(K_matrix < 1e-10)
    lists = merger(list(zip(idx, idy)))
    
    # compute merged components
    new_means, new_logprecisions, new_logpis = [], [], []

    for l in lists:
        l = np.array(l)  # Convert to numpy array
        if len(l) > 0:
            # Use scipy's logsumexp for numerical stability
            new_logpis.append(scipy_logsumexp(logpis[l]))
            # Calculate weighted average for means
            weights_exp = np.exp(logpis[l] - np.min(logpis[l]))
            new_means.append(
                np.sum(means[l] * weights_exp) / np.sum(weights_exp)
            )
            # Calculate weighted average for logprecisions
            new_logprecisions.append(
                np.log(
                    np.sum(np.exp(logprecisions[l]) * weights_exp) / np.sum(weights_exp)
                )
            )

    if new_means:
        new_means = np.array(new_means)
        # set the closest mean to 0.0
        closest_idx = np.argmin(np.abs(new_means))
        new_means[closest_idx] = 0.0

    # compute responsibilities
    if len(new_means) > 0 and len(new_logprecisions) > 0 and len(new_logpis) > 0:
        argmax_responsibilities = compute_responsibilies(
            weights, new_means, np.array(new_logprecisions), np.exp(new_logpis)
        )
        out = [new_means[i] for i in argmax_responsibilities]
    else:
        # If no valid components, return original weights
        return W

    out = reshape_like(out, shaped_array=weight_params)
    
    # Append back the original prior parameters
    result = out + W[-3:] if len(W) >= 3 else out
    return result


def save_histogram(W_T, save, upper_bound=200):
    """
    Save histogram of weights
    """
    # Extract only the weight parameters, not the prior parameters
    weight_params = W_T[:-3] if len(W_T) >= 3 else W_T
    w = np.squeeze(special_flatten(weight_params))
    
    # Regular histogram
    plt.figure(figsize=(10, 7))
    sns.set_style("whitegrid")
    plt.xlim(-1, 1)
    plt.ylim(0, upper_bound)
    plt.hist(w, bins=200, density=True, color="g", alpha=0.7)
    plt.title("Weight Distribution")
    plt.savefig(f"./{save}.png", bbox_inches='tight')
    plt.close()

    # Log-scaled histogram
    plt.figure(figsize=(10, 7))
    plt.yscale("log")
    sns.set_style("whitegrid")
    plt.xlim(-1, 1)
    plt.ylim(0.001, upper_bound * 5)
    plt.hist(w, bins=200, density=True, color="g", alpha=0.7)
    plt.title("Weight Distribution (Log Scale)")
    plt.savefig(f"./{save}_log.png", bbox_inches='tight')
    plt.close()