import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    """
    LeNet model based on the Caffe implementation.
    
    Architecture:
    - Conv1: 20 filters of size 5x5
    - MaxPool1: 2x2 kernel, stride 2
    - Conv2: 50 filters of size 5x5
    - MaxPool2: 2x2 kernel, stride 2
    - FC1: 500 units
    - ReLU: activation
    - FC2: 10 units (for MNIST classification)
    """
    
    def __init__(self):
        super(LeNet, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5, stride=1)
        
        # Max pooling layers
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(50 * 4 * 4, 500)  # Input size after convolutions and pooling
        self.fc2 = nn.Linear(500, 10)  # Output layer for 10 classes (MNIST digits)
        
        # Initialize weights using Xavier initialization
        self._initialize_weights()
    
    def forward(self, x):
        # First convolutional block
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Second convolutional block
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers with ReLU activation
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        
        return x
    
    def _initialize_weights(self):
        """Initialize weights using Xavier initialization and constant bias."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


if __name__ == "__main__":
    # Test the model with a dummy input
    model = LeNet()
    dummy_input = torch.randn(1, 1, 28, 28)  # MNIST input size
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model: {model}")
