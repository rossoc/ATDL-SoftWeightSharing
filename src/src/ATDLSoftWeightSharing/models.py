"""
Keras implementation of neural network models for soft weight sharing.

This module provides Keras equivalents of PyTorch models from the ATDL2/sws/models.py file.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


class LeNet300100(keras.Model):
    """Keras implementation of LeNet300100 model."""
    
    def __init__(self, num_classes=10):
        super().__init__()
        self.fc1 = layers.Dense(300, activation='relu', name='fc1')
        self.fc2 = layers.Dense(100, activation='relu', name='fc2')
        self.fc3 = layers.Dense(num_classes, name='fc3')
        
    def call(self, x):
        x = tf.reshape(x, [tf.shape(x)[0], -1])  # Flatten
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class LeNet5Caffe(keras.Model):
    """Keras implementation of LeNet5Caffe model."""
    
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = layers.Conv2D(20, 5, activation='relu')
        self.conv2 = layers.Conv2D(50, 5, activation='relu')
        self.maxpool = layers.MaxPooling2D(2)
        self.fc1 = layers.Dense(500, activation='relu')
        self.fc2 = layers.Dense(num_classes)
        
    def call(self, x):
        x = self.maxpool(self.conv1(x))
        x = self.maxpool(self.conv2(x))
        x = tf.reshape(x, [tf.shape(x)[0], -1])  # Flatten
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class TutorialNet(keras.Model):
    """
    Keras implementation of TutorialNet matching the original Keras tutorial (642K params).
    Architecture:
    - Conv2d(1->25, 5x5, stride=2) + ReLU  [28x28 -> 12x12]
    - Conv2d(25->50, 3x3, stride=2) + ReLU [12x12 -> 5x5]
    - Flatten -> Linear(1250->500) + ReLU
    - Linear(500->10)
    """
    
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = layers.Conv2D(25, 5, strides=2, activation='relu')
        self.conv2 = layers.Conv2D(50, 3, strides=2, activation='relu')
        self.fc1 = layers.Dense(500, activation='relu')
        self.fc2 = layers.Dense(num_classes)
        
    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = tf.reshape(x, [tf.shape(x)[0], -1])  # Flatten
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class WideResNet16x4(keras.Model):
    """
    Keras implementation of WideResNet-16-4 (light, no dropout).
    """
    
    def __init__(self, num_classes=10, k=4):
        super().__init__()
        n = (16 - 4) // 6  # Number of layers per block
        self.widths = [16, 16 * k, 32 * k, 64 * k]
        
        # Input convolution
        self.conv1 = layers.Conv2D(self.widths[0], 3, strides=1, padding='same', use_bias=False)
        
        # Residual blocks
        self.block1 = NetworkBlock(n, self.widths[0], self.widths[1], 1)
        self.block2 = NetworkBlock(n, self.widths[1], self.widths[2], 2)
        self.block3 = NetworkBlock(n, self.widths[2], self.widths[3], 2)
        
        # Final batch normalization and classifier
        self.bn = layers.BatchNormalization()
        self.global_avg_pool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Initialize weights similar to PyTorch implementation.
        Note: This is called after model is built.
        """
        pass  # Weights will be initialized automatically by Keras layers
    
    def call(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = tf.nn.relu(self.bn(out))
        out = self.global_avg_pool(out)
        out = self.fc(out)
        return out


class BasicBlock(keras.Model):
    """Keras implementation of a basic ResNet block."""
    
    def __init__(self, in_planes, out_planes, stride):
        super().__init__()
        self.bn1 = layers.BatchNormalization()
        self.conv1 = layers.Conv2D(
            out_planes, 3, strides=stride, padding='same', use_bias=False
        )
        self.bn2 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(
            out_planes, 3, strides=1, padding='same', use_bias=False
        )
        
        self.shortcut = None
        if stride != 1 or in_planes != out_planes:
            self.shortcut = layers.Conv2D(
                out_planes, 1, strides=stride, use_bias=False
            )

    def call(self, x):
        out = tf.nn.relu(self.bn1(x))
        residual = self.shortcut(out) if self.shortcut is not None else x
        out = self.conv1(out)
        out = self.conv2(tf.nn.relu(self.bn2(out)))
        return out + residual


class NetworkBlock(keras.Model):
    """Keras implementation of a network block containing multiple basic blocks."""
    
    def __init__(self, n_layers, in_planes, out_planes, stride):
        super().__init__()
        layers_list = []
        for i in range(n_layers):
            stride_for_layer = stride if i == 0 else 1
            in_planes_for_layer = in_planes if i == 0 else out_planes
            layers_list.append(
                BasicBlock(in_planes_for_layer, out_planes, stride_for_layer)
            )
        self.net = keras.Sequential(layers_list)

    def call(self, x):
        return self.net(x)


def make_model(name: str, dataset: str, num_classes: int):
    """Factory function to create models by name."""
    if name == "lenet_300_100":
        assert dataset == "mnist"
        return LeNet300100(num_classes)
    if name == "lenet5":
        assert dataset == "mnist"
        return LeNet5Caffe(num_classes)
    if name == "tutorial":
        assert dataset == "mnist"
        return TutorialNet(num_classes)
    if name == "wrn_16_4":
        assert dataset in ("cifar10", "cifar100")
        return WideResNet16x4(num_classes=num_classes, k=4)
    raise ValueError(f"Unknown model: {name}")


def create_tutorial_model(input_shape=(28, 28, 1), num_classes=10):
    """
    Create the tutorial model architecture that matches the original keras tutorial
    for compatibility with the compression pipeline.
    """
    model = keras.Sequential([
        layers.Conv2D(25, (5, 5), strides=(2, 2), activation="relu", input_shape=input_shape),
        layers.Conv2D(50, (3, 3), strides=(2, 2), activation="relu"),
        layers.Flatten(),
        layers.Dense(500, activation="relu"),
        layers.Dense(num_classes, activation="softmax")
    ])
    return model