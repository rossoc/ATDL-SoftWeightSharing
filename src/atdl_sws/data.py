"""
Keras implementation of data loading utilities for soft weight sharing.

This module provides Keras equivalents of PyTorch data utilities from the ATDL2/sws/data.py file.
"""

import numpy as np
import tensorflow as tf
from typing import Tuple


def _normalize_mnist_data(train_images, test_images):
    normalized_train_images = (
        train_images.reshape(train_images.shape[0], 28, 28, 1).astype("float32") / 255.0
    )
    normalized_test_images = (
        test_images.reshape(test_images.shape[0], 28, 28, 1).astype("float32") / 255.0
    )
    return normalized_train_images, normalized_test_images


def _normalize_cifar10_data(train_images, test_images):
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.247, 0.243, 0.261])

    train_images = train_images.astype("float32")
    test_images = test_images.astype("float32")

    normalized_train_images = (train_images - mean.reshape(1, 1, 1, -1)) / std.reshape(
        1, 1, 1, -1
    )
    normalized_test_images = (test_images - mean.reshape(1, 1, 1, -1)) / std.reshape(
        1, 1, 1, -1
    )

    return normalized_train_images, normalized_test_images


def _normalize_cifar100_data(train_images, test_images):
    mean = np.array([0.5071, 0.4867, 0.4408])
    std = np.array([0.2675, 0.2565, 0.2761])

    train_images = train_images.astype("float32")
    test_images = test_images.astype("float32")

    normalized_train_images = (train_images - mean.reshape(1, 1, 1, -1)) / std.reshape(
        1, 1, 1, -1
    )
    normalized_test_images = (test_images - mean.reshape(1, 1, 1, -1)) / std.reshape(
        1, 1, 1, -1
    )

    return normalized_train_images, normalized_test_images


def _convert_labels_to_categorical(train_labels, test_labels, num_classes: int):
    categorical_train_labels = tf.keras.utils.to_categorical(train_labels, num_classes)
    categorical_test_labels = tf.keras.utils.to_categorical(test_labels, num_classes)
    return categorical_train_labels, categorical_test_labels


def _create_tf_dataset_with_batching(
    images, labels, batch_size: int, is_training: bool = True, seed: int = 42
):
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))

    if is_training:
        # Shuffle and batch the training dataset
        dataset = dataset.shuffle(buffer_size=10000, seed=seed)
        dataset = dataset.batch(batch_size)
    else:
        # Batch the test dataset (no shuffling)
        dataset = dataset.batch(2 * batch_size)

    # Prefetch for performance
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def make_loaders(
    dataset: str, batch_size: int, num_workers: int = 0, seed: int = 42
) -> Tuple[tf.data.Dataset, tf.data.Dataset, int]:
    """
    Create data loaders for the specified dataset using tf.data API.

    Args:
        dataset: Name of the dataset ('mnist', 'cifar10', 'cifar100')
        batch_size: Batch size for the data loaders
        num_workers: Number of workers for data loading (not used in tf.data)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_loader, test_loader, num_classes)
    """
    # Set random seed for reproducibility
    tf.random.set_seed(seed)
    np.random.seed(seed)

    if dataset == "mnist":
        (
            (normalized_train_images, categorical_train_labels),
            (
                normalized_test_images,
                categorical_test_labels,
            ),
        ) = get_mnist_data()
        categories = 10
    elif dataset == "cifar10":
        (
            (normalized_train_images, categorical_train_labels),
            (
                normalized_test_images,
                categorical_test_labels,
            ),
        ) = get_cifar10_data()
        categories = 10
    elif dataset == "cifar100":
        (
            (normalized_train_images, categorical_train_labels),
            (
                normalized_test_images,
                categorical_test_labels,
            ),
        ) = get_cifar100_data()
        categories = 100
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    train_dataset = _create_tf_dataset_with_batching(
        normalized_train_images,
        categorical_train_labels,
        batch_size,
        is_training=True,
        seed=seed,
    )
    test_dataset = _create_tf_dataset_with_batching(
        normalized_test_images,
        categorical_test_labels,
        batch_size,
        is_training=False,
        seed=seed,
    )

    return train_dataset, test_dataset, categories


def get_mnist_data():
    (raw_train_images, raw_train_labels), (raw_test_images, raw_test_labels) = (
        tf.keras.datasets.mnist.load_data()
    )

    normalized_train_images, normalized_test_images = _normalize_mnist_data(
        raw_train_images, raw_test_images
    )

    categorical_train_labels, categorical_test_labels = _convert_labels_to_categorical(
        raw_train_labels, raw_test_labels, 10
    )

    return (normalized_train_images, categorical_train_labels), (
        normalized_test_images,
        categorical_test_labels,
    )


def get_cifar10_data():
    (raw_train_images, raw_train_labels), (raw_test_images, raw_test_labels) = (
        tf.keras.datasets.cifar10.load_data()
    )

    normalized_train_images, normalized_test_images = _normalize_cifar10_data(
        raw_train_images, raw_test_images
    )

    categorical_train_labels, categorical_test_labels = _convert_labels_to_categorical(
        raw_train_labels, raw_test_labels, 10
    )

    return (normalized_train_images, categorical_train_labels), (
        normalized_test_images,
        categorical_test_labels,
    )


def get_cifar100_data():
    (raw_train_images, raw_train_labels), (raw_test_images, raw_test_labels) = (
        tf.keras.datasets.cifar100.load_data()
    )

    normalized_train_images, normalized_test_images = _normalize_cifar100_data(
        raw_train_images, raw_test_images
    )

    categorical_train_labels, categorical_test_labels = _convert_labels_to_categorical(
        raw_train_labels, raw_test_labels, 100
    )

    return (normalized_train_images, categorical_train_labels), (
        normalized_test_images,
        categorical_test_labels,
    )
