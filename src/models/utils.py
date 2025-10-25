import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_X_y, check_array
from sklearn.utils.multiclass import unique_labels
from tqdm import tqdm


class PyTorchNeuralNetworkWrapper(BaseEstimator, ClassifierMixin):
    """
    Scikit-learn compatible wrapper for PyTorch neural networks.

    This class allows PyTorch models to be used with scikit-learn's API,
    enabling easy pretraining and compatibility with other scikit-learn tools.
    """

    def __init__(
        self,
        model,
        loss_fn=None,
        optimizer=None,
        learning_rate=5e-4,
        batch_size=64,
        epochs=10,
        device=None,
        verbose=False,
        loss_every_num_epochs=1,
    ):
        """
        Initialize the wrapper.

        Parameters:
        -----------
        model : torch.nn.Module
            The PyTorch neural network model
        loss_fn : torch.nn.Module, optional
            Loss function (default: CrossEntropyLoss)
        optimizer : torch.optim.Optimizer, optional
            Optimizer (default: Adam)
        learning_rate : float, default=0.001
            Learning rate for the optimizer
        batch_size : int, default=64
            Batch size for training
        epochs : int, default=10
            Number of training epochs
        device : str, optional
            Device to run training on (default: 'cuda' if available, else 'cpu')
        verbose : bool, default=False
            Whether to print training progress
        loss_every_num_epochs : int, default=1
            Display loss every this many epochs
        """
        self.model = model
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()
        self.optimizer = optimizer or optim.Adam(model.parameters(), lr=learning_rate)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose
        self.loss_every_num_epochs = loss_every_num_epochs

        # Move model to device
        self.model.to(self.device)

    def fit(self, X, y):
        """
        Fit the neural network to the training data.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values (class labels)

        Returns:
        --------
        self : object
            Returns self for method chaining
        """
        # Validate inputs
        X, y = check_X_y(X, y, accept_sparse=False)

        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        self.n_classes_ = len(self.classes_)

        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)

        # If X is 2D but the model expects 4D (for CNNs), reshape appropriately
        # This is a simple heuristic - for images like MNIST we assume (N, 784) -> (N, 1, 28, 28)
        if len(X.shape) == 2 and X.shape[1] == 784:  # Likely flattened MNIST
            batch_size = X_tensor.size(0)
            X_tensor = X_tensor.view(batch_size, 1, 28, 28)

        # Create dataset and dataloader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Training loop
        self.model.train()

        # Use tqdm for progress bar
        epoch_range = tqdm(range(self.epochs), desc="Training")

        for epoch in epoch_range:
            total_loss = 0.0
            num_batches = len(dataloader)

            for batch_X, batch_y in dataloader:
                # Zero gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(batch_X)
                loss = self.loss_fn(outputs, batch_y)

                # Backward pass
                loss.backward()

                # Update weights
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / num_batches

            # Display loss based on loss_every_num_epochs parameter
            if (epoch + 1) % self.loss_every_num_epochs == 0:
                epoch_range.set_postfix({"loss": f"{avg_loss:.4f}"})

        # Mark as fitted
        self.fitted_ = True
        return self

    def save_model(self, filepath):
        """
        Save the trained model to a file.

        Parameters:
        -----------
        filepath : str
            Path to save the model
        """
        if not hasattr(self, "fitted_"):
            raise ValueError("Model must be fitted before saving. Call fit() first.")

        # Create a dictionary with model state and other necessary attributes
        model_data = {
            "model_state_dict": self.model.state_dict(),
            "model_class": self.model.__class__.__name__,
            "model_config": str(self.model.__class__),
            "classes_": self.classes_,
            "n_classes_": self.n_classes_,
            "fitted_": self.fitted_,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "device": self.device,
            "verbose": self.verbose,
            "loss_every_num_epochs": self.loss_every_num_epochs,
        }

        torch.save(model_data, filepath)

    @classmethod
    def load_model(cls, filepath, model_instance):
        """
        Load a trained model from a file.

        Parameters:
        -----------
        filepath : str
            Path to load the model from
        model_instance: torch.nn.Module
            An instance of the model class to load the weights into

        Returns:
        --------
        PyTorchNeuralNetworkWrapper
            A loaded instance of the wrapper with the trained model
        """
        # Load the saved data
        model_data = torch.load(filepath, map_location="cpu", weights_only=False)

        # Create a new instance of the wrapper with the provided model
        wrapper = cls(
            model=model_instance,
            learning_rate=model_data["learning_rate"],
            batch_size=model_data["batch_size"],
            epochs=model_data["epochs"],
            device=model_data["device"],
            verbose=model_data["verbose"],
            loss_every_num_epochs=model_data["loss_every_num_epochs"],
        )

        # Load the model state dict
        wrapper.model.load_state_dict(model_data["model_state_dict"])

        # Restore other attributes
        wrapper.classes_ = model_data["classes_"]
        wrapper.n_classes_ = model_data["n_classes_"]
        wrapper.fitted_ = model_data["fitted_"]

        return wrapper

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples to predict

        Returns:
        --------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels
        """
        if not hasattr(self, "fitted_"):
            raise NotFittedError(
                "This PyTorchNeuralNetworkWrapper instance is not fitted yet."
            )

        # Validate input
        X = check_array(X, accept_sparse=False)

        # Convert to tensor
        X_tensor = torch.FloatTensor(X).to(self.device)

        # Reshape if needed (for image data)
        if len(X.shape) == 2 and X.shape[1] == 784:  # Likely flattened MNIST
            batch_size = X_tensor.size(0)
            X_tensor = X_tensor.view(batch_size, 1, 28, 28)

        # Set model to evaluation mode
        self.model.eval()

        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs.data, 1)

        # Convert back to numpy and return
        return predicted.cpu().numpy()

    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples to predict

        Returns:
        --------
        y_prob : ndarray of shape (n_samples, n_classes)
            Predicted class probabilities
        """
        if not hasattr(self, "fitted_"):
            raise NotFittedError(
                "This PyTorchNeuralNetworkWrapper instance is not fitted yet."
            )

        # Validate input
        X = check_array(X, accept_sparse=False)

        # Convert to tensor
        X_tensor = torch.FloatTensor(X).to(self.device)

        # Reshape if needed (for image data)
        if len(X.shape) == 2 and X.shape[1] == 784:  # Likely flattened MNIST
            batch_size = X_tensor.size(0)
            X_tensor = X_tensor.view(batch_size, 1, 28, 28)

        # Set model to evaluation mode
        self.model.eval()

        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)

        # Convert back to numpy and return
        return probabilities.cpu().numpy()

    def score(self, X, y, sample_weight=None):
        """
        Return the mean accuracy on the given test data and labels.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test samples
        y : array-like, shape (n_samples,)
            True labels for X

        Returns:
        --------
        score : float
            Mean accuracy of self.predict(X) wrt. y
        """
        sample_weight = sample_weight or 1
        predictions = self.predict(X)
        return np.mean((predictions == y) * sample_weight)


# Example usage and test
if __name__ == "__main__":
    # Import the model
    import sys
    import os

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    try:
        from models.lenet import LeNet
        from dataset import get_mnist

        print("Testing the scikit-learn wrapper with LeNet...")

        # Load a small subset of MNIST for quick testing
        X_train, X_test, y_train, y_test = get_mnist()

        X_train = X_train[:200]
        y_train = y_train[:200]

        print(f"Training on {len(X_train)} samples...")
        print(f"Testing on {len(X_test)} samples...")

        # Create and train the model
        clf = PyTorchNeuralNetworkWrapper(
            model=LeNet(),
            epochs=20,
            batch_size=64,
            verbose=True,
        )

        # Fit the model
        clf.fit(X_train, y_train)
        print("Model fitted successfully!")

        # Test prediction
        predictions = clf.predict(X_test[:5])  # Predict on 5 samples
        print(f"Sample predictions: {predictions}")
        print(f"Actual labels: {y_test[:5]}")

        # Test score
        score = clf.score(X_test, y_test)
        print(f"Model accuracy on test data: {score:.4f}")

        # Test save and load functionality
        print("\nTesting save and load functionality...")
        model_path = "test_lenet_model.pth"

        # Save the original trained model
        clf.save_model(model_path)
        print(f"Model saved to {model_path}")

        # Create a new instance of the model architecture
        new_model = LeNet()

        # Load the model back
        loaded_clf = PyTorchNeuralNetworkWrapper.load_model(model_path, new_model)
        print("Model loaded successfully!")

        # Verify that the loaded model produces the same predictions
        original_predictions = clf.predict(X_test[:5])
        loaded_predictions = loaded_clf.predict(X_test[:5])

        print(f"Original predictions: {original_predictions}")
        print(f"Loaded model predictions: {loaded_predictions}")
        print(
            f"Predictions match: {np.array_equal(original_predictions, loaded_predictions)}"
        )

        # Test the score of the loaded model
        loaded_score = loaded_clf.score(X_test, y_test)
        print(f"Score of loaded model: {loaded_score:.4f}")

        # Clean up
        import os

        if os.path.exists(model_path):
            os.remove(model_path)
            print(f"Test model file {model_path} removed.")

    except ImportError as e:
        print(f"Could not import required modules: {e}")
        print("Make sure all dependencies are installed.")
    except Exception as e:
        print(f"An error occurred during testing: {e}")
