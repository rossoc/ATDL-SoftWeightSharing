import sys
import numpy as np
import pytest
from unittest.mock import patch


from data import make_loaders, get_mnist_data, get_cifar_data


def test_get_mnist_data():
    """Test get_mnist_data function returns correct data shapes."""
    (x_train, y_train), (x_test, y_test) = get_mnist_data()

    # Check shapes
    assert x_train.shape[0] > 0  # Should have training samples
    assert x_test.shape[0] > 0  # Should have test samples
    assert x_train.shape[1:] == (28, 28, 1)  # MNIST image dimensions
    assert x_test.shape[1:] == (28, 28, 1)  # MNIST image dimensions
    assert y_train.shape[0] == x_train.shape[0]  # Same number of labels as samples
    assert y_test.shape[0] == x_test.shape[0]  # Same number of labels as samples
    assert y_train.shape[1] == 10  # 10 classes for MNIST
    assert y_test.shape[1] == 10  # 10 classes for MNIST

    # Check data type and range
    assert x_train.dtype == np.float32
    assert x_test.dtype == np.float32
    assert np.all(x_train >= 0) and np.all(x_train <= 1)  # Normalized
    assert np.all(x_test >= 0) and np.all(x_test <= 1)  # Normalized


def test_get_cifar10_data():
    """Test get_cifar_data function returns correct data shapes for CIFAR-10."""
    (x_train, y_train), (x_test, y_test) = get_cifar_data("cifar10")

    # Check shapes
    assert x_train.shape[0] > 0  # Should have training samples
    assert x_test.shape[0] > 0  # Should have test samples
    assert x_train.shape[1:] == (32, 32, 3)  # CIFAR-10 image dimensions
    assert x_test.shape[1:] == (32, 32, 3)  # CIFAR-10 image dimensions
    assert y_train.shape[0] == x_train.shape[0]  # Same number of labels as samples
    assert y_test.shape[0] == x_test.shape[0]  # Same number of labels as samples
    assert y_train.shape[1] == 10  # 10 classes for CIFAR-10
    assert y_test.shape[1] == 10  # 10 classes for CIFAR-10

    # Check data type
    assert x_train.dtype == np.float64
    assert x_test.dtype == np.float64


# def test_get_cifar100_data():
#     """Test get_cifar_data function returns correct data shapes for CIFAR-100."""
#     (x_train, y_train), (x_test, y_test) = get_cifar_data("cifar100")
#
#     # Check shapes
#     assert x_train.shape[0] > 0  # Should have training samples
#     assert x_test.shape[0] > 0  # Should have test samples
#     assert x_train.shape[1:] == (32, 32, 3)  # CIFAR-100 image dimensions
#     assert x_test.shape[1:] == (32, 32, 3)  # CIFAR-100 image dimensions
#     assert y_train.shape[0] == x_train.shape[0]  # Same number of labels as samples
#     assert y_test.shape[0] == x_test.shape[0]  # Same number of labels as samples
#     assert y_train.shape[1] == 100  # 100 classes for CIFAR-100
#     assert y_test.shape[1] == 100  # 100 classes for CIFAR-100
#
#     # Check data type
#     assert x_train.dtype == np.float64
#     assert x_test.dtype == np.float64


def test_make_loaders_mnist():
    """Test make_loaders function with MNIST dataset."""
    train_loader, test_loader, num_classes = make_loaders("mnist", batch_size=32)

    assert num_classes == 10  # MNIST has 10 classes

    # Check that we can get a batch from the loaders
    train_batch = next(iter(train_loader))
    test_batch = next(iter(test_loader))

    assert len(train_batch) == 2  # (images, labels)
    assert len(test_batch) == 2  # (images, labels)

    x_batch, y_batch = train_batch
    assert x_batch.shape[1:] == (28, 28, 1)  # Image dimensions
    assert y_batch.shape[1] == 10  # Number of classes


def test_make_loaders_cifar10():
    """Test make_loaders function with CIFAR-10 dataset."""
    train_loader, test_loader, num_classes = make_loaders("cifar10", batch_size=32)

    assert num_classes == 10  # CIFAR-10 has 10 classes

    # Check that we can get a batch from the loaders
    train_batch = next(iter(train_loader))
    test_batch = next(iter(test_loader))

    assert len(train_batch) == 2  # (images, labels)
    assert len(test_batch) == 2  # (images, labels)

    x_batch, y_batch = train_batch
    assert x_batch.shape[1:] == (32, 32, 3)  # CIFAR-10 image dimensions
    assert y_batch.shape[1] == 10  # Number of classes


# def test_make_loaders_cifar100():
#     """Test make_loaders function with CIFAR-100 dataset."""
#     train_loader, test_loader, num_classes = make_loaders("cifar100", batch_size=32)
#
#     assert num_classes == 100  # CIFAR-100 has 100 classes
#
#     # Check that we can get a batch from the loaders
#     train_batch = next(iter(train_loader))
#     test_batch = next(iter(test_loader))
#
#     assert len(train_batch) == 2  # (images, labels)
#     assert len(test_batch) == 2  # (images, labels)
#
#     x_batch, y_batch = train_batch
#     assert x_batch.shape[1:] == (32, 32, 3)  # CIFAR-100 image dimensions
#     assert y_batch.shape[1] == 100  # Number of classes


def test_make_loaders_invalid_dataset():
    """Test make_loaders function with invalid dataset raises ValueError."""
    with pytest.raises(ValueError, match="Unknown dataset: invalid_dataset"):
        make_loaders("invalid_dataset", batch_size=32)


def test_get_cifar_data_invalid_dataset():
    """Test get_cifar_data function with invalid dataset raises ValueError."""
    with pytest.raises(ValueError, match="Unknown dataset: invalid_dataset"):
        get_cifar_data("invalid_dataset")


if __name__ == "__main__":
    pytest.main([__file__])
