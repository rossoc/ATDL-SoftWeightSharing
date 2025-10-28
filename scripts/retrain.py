from argparse import ArgumentParser
from atdl_sws import lenet_caffe, lenet_300_100, ResNet, retraining, TrainingGifVisualizer, CSVLogger
import os


def create_model_and_dataset_mapping():
    """Define the model, dataset, and class mappings."""
    dataset = {
        "lenet300": "mnist",
        "lenet5": "mnist",
        "cifar10": "cifar10",
        "cifar100": "cifar100",
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
    
    return dataset, model_fn, n_classes


def parse_args():
    """Parse command line arguments."""
    ap = ArgumentParser()
    ap.add_argument(
        "--model", choices=["lenet300", "lenet5", "cifar10", "cifar100"], required=True
    )
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--save-dir", type=str, required=False, default=None)
    ap.add_argument("--run-dir", type=str, required=False, default="./runs", help="Directory to save visualizations and logs")
    ap.add_argument("--visualize", action="store_true", help="Enable GIF visualization")
    ap.add_argument("--framerate", type=float, default=2, help="Framerate for the visualization GIF")
    ap.add_argument("--log", action="store_true", help="Enable CSV logging")

    # Prior-specific arguments
    ap.add_argument(
        "--prior-J", type=int, default=16, help="Number of mixture components"
    )
    ap.add_argument(
        "--prior-pi0", type=float, default=0.99, help="Probability of zero component"
    )
    ap.add_argument(
        "--prior-init-sigma",
        type=float,
        default=0.25,
        help="Initial sigma for mixture components",
    )
    ap.add_argument(
        "--tau", type=float, default=0.005, help="Regularization coefficient"
    )

    # Learning rates for different parameters
    ap.add_argument(
        "--lr-w", type=float, help="Learning rate for model weights (default: lr)"
    )
    ap.add_argument(
        "--lr-mu", type=float, help="Learning rate for means (default: lr*10)"
    )
    ap.add_argument(
        "--lr-sigma", type=float, help="Learning rate for log_sigma2 (default: lr*100)"
    )
    ap.add_argument(
        "--lr-pi", type=float, help="Learning rate for pi_logits (default: lr*100)"
    )

    return ap.parse_args()


def setup_visualization_and_logging(args, model_name):
    """Set up visualization and logging based on arguments."""
    viz = None
    logger = None
    
    # Create run directory
    os.makedirs(args.run_dir, exist_ok=True)
    
    # Setup visualization if requested
    if args.visualize:
        viz = TrainingGifVisualizer(
            out_dir=args.run_dir,
            tag=f"{model_name}_retrain",
            framerate=args.framerate
        )
    
    # Setup logging if requested
    if args.log:
        logger = CSVLogger(
            os.path.join(args.run_dir, f"{model_name}_retrain_log.csv"),
            ["phase", "epoch", "train_ce", "complexity", "total_loss", "tau", "test_acc", "elapsed"]
        )
    
    return viz, logger


if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    
    # Get model and dataset mappings
    dataset_map, model_fn, n_classes = create_model_and_dataset_mapping()
    
    # Create the model
    model = model_fn[args.model](num_classes=n_classes[args.model])
    dataset = dataset_map[args.model]

    # Setup visualization and logging
    viz, logger = setup_visualization_and_logging(args, args.model)
    
    # Call the retraining function with visualization and logging support
    retraining(
        model,
        args.epochs,
        args.lr,
        dataset,
        prior_J=args.prior_J,
        prior_pi0=args.prior_pi0,
        prior_init_sigma=args.prior_init_sigma,
        tau=args.tau,
        lr_w=args.lr_w,
        lr_mu=args.lr_mu,
        lr_sigma=args.lr_sigma,
        lr_pi=args.lr_pi,
        batch_size=args.batch_size,
        save_dir=args.save_dir,
        viz=viz,
        logger=logger,
        run_dir=args.run_dir,
    )