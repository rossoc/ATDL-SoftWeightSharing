from .data import (
    get_mnist_data,
    get_cifar10_data,
    get_cifar100_data,
)

from .models import (
    lenet_caffe,
    lenet_300_100,
    ResNet,
)

from .visualizing import TrainingGifVisualizer
from .csv_logger import CSVLogger
from .metrics import Metrics

import logging
import sys
import os
import absl.logging


def setup_log(save_dir=None):
    print(save_dir)
    if save_dir:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            filename=save_dir + "/train.log",
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            stream=sys.stdout,
        )

    return absl.logging.use_python_logging()


def evaluate_model(model, dataset):
    """Evaluate the model on the test set of the specified dataset."""
    # Load test data
    x_test, y_test = dataset

    # Compile the model if not already compiled
    if not hasattr(model, "optimizer"):
        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

    _, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    return test_accuracy


def get_data_model_mappings(model_name):
    """Define the model, dataset, and class mappings."""
    dataset_fn = {
        "lenet300": get_mnist_data,
        "lenet5": get_mnist_data,
        "cifar10": get_cifar10_data,
        "cifar100": get_cifar100_data,
    }

    model_fn = {
        "lenet300": lenet_300_100,
        "lenet5": lenet_caffe,
        "cifar10": ResNet,
        "cifar100": ResNet,
    }

    n_classes = {
        "lenet300": 10,
        "lenet5": 10,
        "cifar10": 10,
        "cifar100": 100,
    }

    return dataset_fn[model_name](), model_fn[model_name](
        num_classes=n_classes[model_name]
    )


def setup_visualization_and_logging(args, model_name, mode="retraining"):
    """Set up visualization and logging based on arguments."""
    viz = None
    logger = None

    os.makedirs(args.save_dir, exist_ok=True)

    if args.visualize:
        viz = TrainingGifVisualizer(
            out_dir=args.save_dir, tag=f"{model_name}_{mode}", framerate=args.framerate
        )

    if args.log:
        logger = CSVLogger(
            os.path.join(args.save_dir, f"{model_name}_{mode}_log.csv"),
            [
                "phase",
                "epoch",
                "train_ce",
                "complexity",
                "total_loss",
                "tau",
                "test_acc",
                "elapsed",
            ],
        )

    metrics = Metrics(vars(args))

    return viz, logger, metrics


def sparsity(model):
    """Calculate sparsity (|W=0|/|W|)"""
    total_params = 0
    zero_params = 0

    for var in model.trainable_variables:
        var_numpy = var.numpy()
        var_size = var_numpy.size
        total_params += var_size
        zero_params += (var_numpy == 0).sum()

    return zero_params / total_params if total_params > 0 else 0.0
