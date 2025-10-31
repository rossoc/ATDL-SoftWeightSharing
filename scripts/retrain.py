from argparse import ArgumentParser
import os
from atdl_sws import (
    MixturePrior,
    Metrics,
    retraining,
    get_data_model_mappings,
    setup_visualization_and_logging,
    evaluate_model,
    setup_log,
)


def parse_args():
    """Parse command line arguments."""
    ap = ArgumentParser()
    ap.add_argument(
        "--model", choices=["lenet300", "lenet5", "cifar10", "cifar100"], required=True
    )
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--save-dir", type=str, required=False, default=None)
    ap.add_argument("--visualize", action="store_true", help="Enable GIF visualization")
    ap.add_argument(
        "--framerate", type=float, default=2, help="Framerate for the visualization GIF"
    )
    ap.add_argument(
        "--log", default="True", action="store_true", help="Enable CSV logging"
    )

    ap.add_argument(
        "--prior-J", type=int, default=17, help="Number of mixture components"
    )
    ap.add_argument(
        "--prior-pi0",
        type=float,
        default=0.99,
        help="Probability of zero component (default 0.99)",
    )
    ap.add_argument(
        "--prior-init-sigma",
        type=float,
        default=0.25,
        help="Initial sigma for mixture components",
    )
    ap.add_argument(
        "--tau",
        type=float,
        default=0.005,
        help="Regularization coefficient default 0.005",
    )

    ap.add_argument(
        "--lr-w",
        type=float,
        default=1e-4,
        help="Learning rate for model weights (default: 0.0001)",
    )
    ap.add_argument(
        "--lr-mu",
        type=float,
        default=1e-4,
        help="Learning rate for means (default: 0.0001)",
    )
    ap.add_argument(
        "--lr-sigma",
        type=float,
        default=3e-3,
        help="Learning rate for log_sigma2 (default: 0.003)",
    )
    ap.add_argument(
        "--lr-pi",
        type=float,
        default=3e-3,
        help="Learning rate for pi_logits (default: 0.003)",
    )
    ap.add_argument(
        "--prior-gamma-alpha",
        type=float,
        default=250,
        help="Alpha of Inverse Gamma of j!=0 (default: 250)",
    )
    ap.add_argument(
        "--prior-gamma-beta",
        type=float,
        default=0.1,
        help="Beta of Inverse Gamma of j!=0 components (default: 0.1)",
    )
    ap.add_argument(
        "--prior-gamma-alpha0",
        type=float,
        default=5000,
        help="Alpha of Inverse Gamma of j=0 components (default: 5000)",
    )
    ap.add_argument(
        "--prior-gamma-beta0",
        type=float,
        default=2.0,
        help="Beta of Inverse Gamma of j=0 components (default: 2.0)",
    )
    ap.add_argument(
        "--weight-decay",
        type=float,
        default=0.1,
        help="Model's weight decay (default: 0.00001)",
    )
    ap.add_argument(
        "--regularizer",
        type=float,
        default=0.1,
        help="Model's regularizer (default: 0.1)",
    )

    return ap.parse_args()


if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    setup_log(save_dir=args.save_dir)

    # Get model and dataset mappings
    (train_dataset, test_dataset), model = get_data_model_mappings(
        args.model, args.regularizer
    )

    pre_acc = evaluate_model(model, test_dataset)

    # Setup visualization and logging
    viz, logger, metrics = setup_visualization_and_logging(args, args.model)

    prior = MixturePrior(
        model,
        J=args.prior_J,
        pi0=args.prior_pi0,
        init_log_sigma2=args.prior_init_sigma,
        gamma_alpha=args.prior_gamma_alpha,
        gamma_beta=args.prior_gamma_beta,
        gamma_alpha0=args.prior_gamma_alpha0,
        gamma_beta0=args.prior_gamma_beta0,
    )

    retraining(
        model,
        prior,
        args.epochs,
        train_dataset,
        eval=test_dataset,
        tau=args.tau,
        lr_w=args.lr_w,
        lr_mu=args.lr_mu,
        lr_sigma=args.lr_sigma,
        lr_pi=args.lr_pi,
        batch_size=args.batch_size,
        save_dir=args.save_dir,
        viz=viz,
        logger=logger,
        weight_decay=args.weight_decay,
    )

    # Get post-retraining accuracy
    print("Evaluating model after retraining...")
    post_acc = evaluate_model(model, test_dataset)
    print(f"Post-retraining accuracy: {post_acc:.4f}")

    prior.merge_components()

    prior.quantize_model(model, skip_last_matrix=True)
    quant_acc = evaluate_model(model, test_dataset)

    metrics.store_layer_pruning(model)
    metrics.store_summary_metrics(
        model=model,
        pre_acc=pre_acc,
        post_acc=post_acc,
        quantized_acc=quant_acc,
    )

    metrics.store(args.save_dir)

    # Save the quantized model
    if args.save_dir:
        model.save(os.path.join(args.save_dir, "quantized.keras"))
        print(
            f"Quantized model saved to {os.path.join(args.save_dir, 'quantized.keras')}"
        )
