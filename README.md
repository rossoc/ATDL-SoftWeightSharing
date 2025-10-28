# ATDL-SoftWeightSharing

This package implements soft weight sharing techniques for neural networks using Keras/TensorFlow.

## Installation

To install the package in development mode:

```bash
pip install -e .
```

## Dependencies

- matplotlib>=3.10.7
- tensorflow>=2.20.0
- tqdm>=4.66.0
- pillow>=10.0.0
- imageio>=2.9.0

## Usage

The package provides utilities for:
- Data loading and preprocessing
- Model pretraining with various datasets (MNIST, CIFAR-10, CIFAR-100)
- Soft weight sharing with mixture priors
- Model retraining with prior regularization

Example usage:

```python
from atdl_swss import pretraining, retraining
from atdl_swss.data import make_loaders
```

## Features

- Keras implementations of soft weight sharing techniques
- Support for multiple datasets
- Mixture prior regularization
- Model quantization capabilities