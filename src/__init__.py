"""
ATDL-SoftWeightSharing package.

This package implements soft weight sharing techniques for neural networks.
"""

# Import main functions and classes to make them accessible at the package level
from .base_model import pretraining
from .data import (
    get_mnist_data,
    get_cifar10_data,
    get_cifar100_data,
    make_loaders,
    _normalize_mnist_data,
    _normalize_cifar10_data,
    _normalize_cifar100_data,
    _convert_labels_to_categorical,
    _create_tf_dataset_with_batching,
)
from .prior import *
from .retraining import retraining

__version__ = "0.2.0"
__author__ = "ATDL"