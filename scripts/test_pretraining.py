#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script 1: Pretraining a model with Keras implementation
This script tests the model creation and basic training functionality
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Import our new modules
from src.ATDLSoftWeightSharing.models import create_tutorial_model
from src.ATDLSoftWeightSharing.data import get_mnist_data


def main():
    print("=== Test Script 1: Pretraining a Neural Network ===")
    print("Testing Keras model creation and basic training...")
    
    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
    
    # Load data
    print("Loading MNIST data...")
    (x_train, y_train), (x_test, y_test) = get_mnist_data()
    
    # Create model using our implementation
    print("Creating model...")
    model = create_tutorial_model(input_shape=(28, 28, 1), num_classes=10)
    model.summary()
    
    # Compile model
    print("Compiling model...")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train for just 5 epochs as requested
    print("Training model for 5 epochs...")
    epochs = 5
    batch_size = 256

    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        validation_data=(x_test, y_test)
    )
    
    # Evaluate
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy after pretraining: {test_accuracy:.4f}")
    
    # Save the pretrained model
    model.save("./scripts/pretrained_model.keras")
    print("Pretrained model saved as pretrained_model.keras")
    
    print("Pretraining test completed successfully!")
    return model


if __name__ == "__main__":
    main()