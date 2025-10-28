from argparse import ArgumentParser
from models import lenet_caffe, lenet_300_100, ResNet
from retraining import retraining


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument(
        "--model", choices=["lenet300", "lenet5", "cifar10", "cifar100"], required=True
    )
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--save-dir", type=str, required=False, default=None)

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

    args = ap.parse_args()

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

    # Create the model
    model = model_fn[args.model](num_classes=n_classes[args.model])
    dataset = dataset[args.model]

    # Call the retraining function
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
    )

