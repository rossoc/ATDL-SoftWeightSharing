"""
Keras implementation of data loading utilities for soft weight sharing.

This module provides Keras equivalents of PyTorch data utilities from the ATDL2/sws/data.py file.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import Tuple


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
        # Load MNIST dataset
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        
        # Reshape and normalize
        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255.0
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255.0
        
        # Convert labels to categorical
        y_train = keras.utils.to_categorical(y_train, 10)
        y_test = keras.utils.to_categorical(y_test, 10)
        
        num_classes = 10
        
    elif dataset in ("cifar10", "cifar100"):
        if dataset == "cifar10":
            (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
            mean = np.array([0.4914, 0.4822, 0.4465])
            std = np.array([0.247, 0.243, 0.261])
            num_classes = 10
        else:
            (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
            mean = np.array([0.5071, 0.4867, 0.4408])
            std = np.array([0.2675, 0.2565, 0.2761])
            num_classes = 100
        
        # Convert to float32
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        
        # Normalize
        x_train = (x_train - mean.reshape(1, 1, 1, -1)) / std.reshape(1, 1, 1, -1)
        x_test = (x_test - mean.reshape(1, 1, 1, -1)) / std.reshape(1, 1, 1, -1)
        
        # Convert labels to categorical
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)
        
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    # Create tf.data.Dataset objects
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    
    # Shuffle and batch the training dataset
    train_dataset = train_dataset.shuffle(buffer_size=10000, seed=seed)
    train_dataset = train_dataset.batch(batch_size)
    
    # Batch the test dataset (no shuffling)
    test_dataset = test_dataset.batch(2 * batch_size)
    
    # Prefetch for performance
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)
    
    return train_dataset, test_dataset, num_classes


def get_mnist_data():
    """
    Get MNIST data in the format expected by the original Keras tutorial.
    """
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Reshape and normalize the data
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255.0
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255.0

    # Convert labels to categorical
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    print(f"Successfully loaded {x_train.shape[0]} train samples and {x_test.shape[0]} test samples.")

    return (x_train, y_train), (x_test, y_test)


def get_cifar_data(dataset_name="cifar10"):
    """
    Get CIFAR data in a format suitable for training.
    """
    if dataset_name == "cifar10":
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
        num_classes = 10
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.247, 0.243, 0.261])
    elif dataset_name == "cifar100":
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
        num_classes = 100
        mean = np.array([0.5071, 0.4867, 0.4408])
        std = np.array([0.2675, 0.2565, 0.2761])
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Convert to float32
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    
    # Apply data augmentation for training data
    x_train_aug = []
    for img in x_train:
        # Random crops with padding
        img_padded = tf.image.resize_with_crop_or_pad(img, 40, 40)
        img_cropped = tf.image.random_crop(img_padded, [32, 32, 3])
        
        # Random horizontal flip
        img_flipped = tf.image.random_flip_left_right(img_cropped)
        
        x_train_aug.append(img_flipped.numpy())
    
    x_train = np.array(x_train_aug)
    
    # Normalize
    x_train = (x_train - mean.reshape(1, 1, 1, -1)) / std.reshape(1, 1, 1, -1)
    x_test = (x_test - mean.reshape(1, 1, 1, -1)) / std.reshape(1, 1, 1, -1)
    
    # Convert labels to categorical
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    
    print(f"Successfully loaded {x_train.shape[0]} train samples and {x_test.shape[0]} test samples.")
    
    return (x_train, y_train), (x_test, y_test)


def prepare_dataset(dataset_name, batch_size=32, augment=True):
    """
    Prepare a dataset for training with optional augmentation.
    """
    if dataset_name == "mnist":
        (x_train, y_train), (x_test, y_test) = get_mnist_data()
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        
        # For MNIST, just shuffle and batch
        train_dataset = train_dataset.shuffle(buffer_size=10000).batch(batch_size)
        test_dataset = test_dataset.batch(batch_size)
        
    elif dataset_name in ["cifar10", "cifar100"]:
        (x_train, y_train), (x_test, y_test) = get_cifar_data(dataset_name)
        
        # Create tf.data datasets
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        
        if augment:
            # Define data augmentation
            def augment_image(image, label):
                # Random crops with padding
                image = tf.image.resize_with_crop_or_pad(image, 40, 40)
                image = tf.image.random_crop(image, [32, 32, 3])
                # Random horizontal flip
                image = tf.image.random_flip_left_right(image)
                return image, label
            
            train_dataset = train_dataset.map(augment_image)
        
        # Shuffle and batch datasets
        train_dataset = train_dataset.shuffle(buffer_size=50000).batch(batch_size)
        test_dataset = test_dataset.batch(batch_size)
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Prefetch for performance
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)
    
    return train_dataset, test_dataset