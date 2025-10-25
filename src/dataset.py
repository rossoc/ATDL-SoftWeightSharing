import numpy as np
from datasets import load_dataset


def split(X, y, n):
    """
    Split dataset into n parts

    Args:
        X: Input data
        y: Labels
        n: Number of splits

    Returns:
        tuple: Split X and y arrays
    """
    idx = np.random.permutation(len(X))
    split_idx = np.array_split(idx, n)

    X = [X[i] for i in split_idx]
    y = [y[i] for i in split_idx]
    return X, y


def sample_results(X, y, results):
    """
    Filter dataset to include only samples with specific labels

    Args:
        X: Input data
        y: Labels
        results: List of labels to include

    Returns:
        tuple: Filtered X and y arrays
    """
    mask = np.isin(y, results)
    X_res = X[mask]
    y_res = y[mask]
    return X_res, y_res


def get_mnist(digits=None):
    """
    Load MNIST dataset with option to filter specific digits

    Args:
        digits: List of digits to include (default: None, includes all digits 0-9)

    Returns:
        tuple: Training and test data (X_train, X_test, y_train, y_test)
    """
    train = load_dataset("ylecun/mnist", split="train")
    test = load_dataset("ylecun/mnist", split="test")

    X_tr = np.array(train["image"])
    y_tr = np.array(train["label"])
    X_te = np.array(test["image"])
    y_te = np.array(test["label"])

    # Convert PIL images to numpy arrays and flatten
    X_tr = np.array([np.array(img).flatten() for img in X_tr])
    X_te = np.array([np.array(img).flatten() for img in X_te])

    if digits:
        X_tr, y_tr = sample_results(X_tr, y_tr, digits)
        X_te, y_te = sample_results(X_te, y_te, digits)

    return X_tr, X_te, y_tr, y_te


if __name__ == "__main__":
    # Test loading all digits
    print("Loading MNIST dataset for all digits...")
    X_train, X_test, y_train, y_test = get_mnist()

    print(f"Training data shape: {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Test labels shape: {y_test.shape}")
    print(f"Training data range: [{X_train.min()}, {X_train.max()}]")
    print(f"Unique training labels: {np.unique(y_train)}")
    print(f"Unique test labels: {np.unique(y_test)}")

    # Test loading specific digits (e.g., only digits 3 and 8)
    print("\nLoading MNIST dataset for digits 3 and 8...")
    X_train_38, X_test_38, y_train_38, y_test_38 = get_mnist(digits=[3, 8])

    print(f"Training data shape (digits 3,8): {X_train_38.shape}")
    print(f"Training labels shape (digits 3,8): {y_train_38.shape}")
    print(f"Unique training labels (digits 3,8): {np.unique(y_train_38)}")
    print(f"Unique test labels (digits 3,8): {np.unique(y_test_38)}")
