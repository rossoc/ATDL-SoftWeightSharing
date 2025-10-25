#!/usr/bin/env python3
"""
Script to pretrain the LeNet model and then compress it using Soft Weight Sharing (SWS).
"""

import sys
import os
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.lenet import LeNet
from models.utils import PyTorchNeuralNetworkWrapper
from sws_compressor import SoftWeightSharingCompressor
from dataset import get_mnist
from compression_utils import compression_report, Prior


def main():
    print("Starting pretraining and compression process...")
    
    # Load MNIST dataset
    print("Loading MNIST dataset...")
    X_train, X_test, y_train, y_test = get_mnist()
    
    # Use a smaller subset for faster training
    X_train = X_train[:5000]  # Use first 5000 samples
    y_train = y_train[:5000]
    X_test = X_test[:1000]    # Use first 1000 samples
    y_test = y_test[:1000]
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Initialize the LeNet model
    print("Initializing LeNet model...")
    model = LeNet()
    
    # Wrap the model with the sklearn-compatible wrapper
    print("Creating sklearn-compatible wrapper...")
    clf = PyTorchNeuralNetworkWrapper(
        model=model,
        epochs=5,  # Fewer epochs for testing
        batch_size=64,
        learning_rate=1e-3,  # Standard learning rate
        verbose=True
    )
    
    print("Starting pretraining...")
    # Pretrain the model
    clf.fit(X_train, y_train)
    
    # Evaluate the pretrained model
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    print(f"Pretrained model - Train accuracy: {train_score:.4f}")
    print(f"Pretrained model - Test accuracy: {test_score:.4f}")
    
    # Save the pretrained model
    pretrained_model_path = "pretrained_lenet_model.pth"
    clf.save_model(pretrained_model_path)
    print(f"Pretrained model saved to {pretrained_model_path}")
    
    # Now, let's compress the model using SWS
    print("Starting SWS compression...")
    
    # Create a new instance of the model to load the pretrained weights
    compressed_model = LeNet()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compressed_model = compressed_model.to(device)
    
    # Load the pretrained weights
    model_data = torch.load(pretrained_model_path, map_location=device, weights_only=False)
    compressed_model.load_state_dict(model_data["model_state_dict"])
    compressed_model.eval()
    
    # Prepare test data for compression (we need a DataLoader)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.LongTensor(y_test).to(device)
    
    # Reshape if needed (for image data)
    if len(X_test_tensor.shape) == 2 and X_test_tensor.shape[1] == 784:  # Likely flattened MNIST
        batch_size = X_test_tensor.size(0)
        X_test_tensor = X_test_tensor.view(batch_size, 1, 28, 28)
    
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Initialize the SWS compressor
    # Using random values for the gamma inverse distribution and Beta priors
    sws_compressor = SoftWeightSharingCompressor(
        tau=5e-3,
        n_components=17,
        zero_component_prior_alpha=2.0,  # Beta(2.0, 2.0) promotes reasonable zero component weight
        zero_component_prior_beta=2.0,   # Beta(2.0, 2.0) promotes reasonable zero component weight
        other_var_prior_alpha=np.random.uniform(0.5, 2.0),  # Random gamma inv dist values
        other_var_prior_beta=np.random.uniform(0.5, 2.0),   # Random gamma inv dist values
        lr_weights=1e-4,
        lr_mixture=5e-4,
        prune_threshold=1e-4,
        device=device
    )
    
    print("Compressing model with SWS...")
    # Apply compression
    compressed_state_dict, code_book, mixture_params = sws_compressor(compressed_model, test_loader, epochs=10)
    
    # Create a Prior object using the final mixture parameters
    prior = Prior(mixture_params['mu'], mixture_params['sigma'], mixture_params['pi'])
    
    # Load the compressed weights back into the model
    compressed_model.load_state_dict(compressed_state_dict)
    
    # Evaluate the compressed model
    compressed_model.eval()
    with torch.no_grad():
        X_test_final = torch.FloatTensor(X_test).to(device)
        if len(X_test_final.shape) == 2 and X_test_final.shape[1] == 784:
            batch_size = X_test_final.size(0)
            X_test_final = X_test_final.view(batch_size, 1, 28, 28)
        
        outputs = compressed_model(X_test_final)
        _, predicted = torch.max(outputs.data, 1)
        compressed_accuracy = (predicted.cpu().numpy() == y_test).mean()
    
    print(f"Compressed model - Test accuracy: {compressed_accuracy:.4f}")
    
    # Calculate compression statistics
    original_params = sum(p.numel() for p in model.parameters())
    unique_weights = len(torch.unique(torch.cat([p.flatten() for p in model.parameters()])))
    compressed_params = len(code_book['centres'])
    
    print(f"\nCompression Results:")
    print(f"Original number of parameters: {original_params}")
    print(f"Number of unique weights after compression: {compressed_params}")
    print(f"Compression ratio: {original_params / len(code_book['centres']):.2f}x")
    print(f"Accuracy drop: {test_score - compressed_accuracy:.4f}")
    
    # Generate detailed compression report using the new function
    print("\nGenerating detailed compression report...")
    compression_details = compression_report(
        model=compressed_model,  # Use the compressed model
        prior=prior,
        dataset="MNIST",  # Dataset name
        use_huffman=True,
        pbits_fc=5,
        pbits_conv=8,
        skip_last_matrix=False,
        assign_mode="ml"  # Should match the quantization method used in compression
    )
    
    print(f"Detailed Compression Report:")
    print(f"  Original bits: {compression_details['orig_bits']:,}")
    print(f"  Compressed bits: {compression_details['compressed_bits']:,}")
    print(f"  Compression Ratio: {compression_details['CR']:.2f}x")
    print(f"  Non-zero elements: {compression_details['nnz']:,}")
    print(f"  Layer-by-layer breakdown:")
    for layer_info in compression_details['layers']:
        print(f"    {layer_info['layer']} {layer_info['shape']}: {layer_info['orig_bits']} -> {layer_info['bits_IR'] + layer_info['bits_IC'] + layer_info['bits_A'] + layer_info['bits_codebook']} bits")
    
    # Save the compressed model
    compressed_model_path = "compressed_lenet_model.pth"
    torch.save({
        'state_dict': compressed_state_dict,
        'code_book': code_book,
        'original_accuracy': test_score,
        'compressed_accuracy': compressed_accuracy
    }, compressed_model_path)
    print(f"Compressed model saved to {compressed_model_path}")
    
    print("Process completed successfully!")
    

if __name__ == "__main__":
    main()