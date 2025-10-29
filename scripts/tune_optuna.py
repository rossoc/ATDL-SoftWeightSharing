import argparse
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import optuna
import tensorflow as tf

from atdl_sws import (
    MixturePrior,
    retraining,
    get_data_model_mappings,
    setup_visualization_and_logging,
    evaluate_model,
    setup_log,
    Metrics,
    sparsity,
    lenet_caffe,
    lenet_300_100,
    ResNet,
    get_mnist_data,
    get_cifar10_data,
    get_cifar100_data,
)


class DataModelProvider:
    """
    A class to provide dataset and model once, reducing downloads and repeated initializations.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._train_data = None
        self._test_data = None
        self._model_fn = None
        self._n_classes = None

        # Set up model and dataset mappings
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

        self._model_fn = model_fn[model_name]
        self._n_classes = n_classes[model_name]

        # Load dataset once
        self._train_data, self._test_data = dataset_fn[model_name]()

    def get_data(self):
        """Returns (x_train, y_train), (x_test, y_test)"""
        return self._train_data, self._test_data

    def get_model(self):
        """Returns a brand new model with the needed features"""
        return self._model_fn(num_classes=self._n_classes)


def _now_tag() -> str:
    import time

    return time.strftime("%Y%m%d_%H%M%S")


def objective(
    trial: optuna.trial.Trial,
    fixed_args: argparse.Namespace,
    data_provider: DataModelProvider,
) -> Tuple[float, float]:
    """
    Objective function for Optuna optimization with dual objectives:
    1. Minimize accuracy loss (acc_pre - acc_post)
    2. Maximize sparsity (|W=0|/|W|)
    """
    # Sample hyperparameters to optimize
    tau = trial.suggest_float("tau", 1e-6, 6e-3, log=True)
    lr_w = trial.suggest_float("lr_w", 1e-5, 1e-3, log=True)
    lr_mu = trial.suggest_float("lr_mu", 1e-5, 5e-3, log=True)
    lr_sigma = trial.suggest_float("lr_sigma", 1e-5, 1e-2, log=True)
    lr_pi = trial.suggest_float("lr_pi", 1e-5, 5e-3, log=True)

    # Prior parameters
    prior_pi0 = trial.suggest_float("prior_pi0", 0.85, 0.999)
    prior_init_sigma = trial.suggest_float("prior_init_sigma", -1 * 1e-2, 0.5)
    prior_gamma_alpha = trial.suggest_float("prior_gamma_alpha", 1e4, 1e5)
    prior_gamma_beta = trial.suggest_float("prior_gamma_beta", 1, 100)
    prior_gamma_alpha0 = trial.suggest_float("prior_gamma_alpha0", 1e4, 1e5)
    prior_gamma_beta0 = trial.suggest_float("prior_gamma_beta0", 5.0, 20.0)

    # Fixed parameters that don't change
    prior_J = fixed_args.prior_J or 17
    epochs = fixed_args.epochs or 40
    batch_size = fixed_args.batch_size or 64

    # Get dataset and model from the provider (not downloaded again)
    train_dataset, test_dataset = data_provider.get_data()
    model = data_provider.get_model()

    # Initialize prior with sampled parameters
    prior = MixturePrior(
        model,
        J=prior_J,
        pi0=prior_pi0,
        init_log_sigma2=prior_init_sigma,
        gamma_alpha=prior_gamma_alpha,
        gamma_beta=prior_gamma_beta,
        gamma_alpha0=prior_gamma_alpha0,
        gamma_beta0=prior_gamma_beta0,
    )

    prior.initialize_weights_from_mixture(model)

    # Setup visualization and logging
    args_for_viz = argparse.Namespace(
        **vars(fixed_args),
        tau=tau,
        lr_w=lr_w,
        lr_mu=lr_mu,
        lr_sigma=lr_sigma,
        lr_pi=lr_pi,
        prior_pi0=prior_pi0,
        prior_init_sigma=prior_init_sigma,
        prior_gamma_alpha=prior_gamma_alpha,
        prior_gamma_beta=prior_gamma_beta,
        prior_gamma_alpha0=prior_gamma_alpha0,
        prior_gamma_beta0=prior_gamma_beta0,
    )

    viz, logger, metrics = setup_visualization_and_logging(
        args_for_viz, fixed_args.model
    )

    # Perform retraining
    try:
        retraining(
            model,
            prior,
            epochs,
            train_dataset,
            tau=tau,
            lr_w=lr_w,
            lr_mu=lr_mu,
            lr_sigma=lr_sigma,
            lr_pi=lr_pi,
            batch_size=batch_size,
            save_dir=fixed_args.save_dir + "/t" + str(trial._trial_id),
            viz=viz,
            logger=logger,
            verbose=0,
        )
    except Exception as e:
        print(f"Error during retraining: {e}")
        # Return worst-case values to penalize this trial
        return float("inf"), 0.0  # High accuracy loss, no sparsity

    pre_acc = evaluate_model(model, test_dataset)
    # Evaluate post-retraining accuracy
    model = prior.quantize_model(model, skip_last_matrix=True)
    acc_loss, post_acc = model.evaluate(test_dataset[0], test_dataset[1], verbose=0)

    # Calculate sparsity (|W=0|/|W|)
    sparsity_ratio = sparsity(model)

    metrics.store_summary_metrics(model, pre_acc=pre_acc, post_acc=post_acc)

    metrics.store(fixed_args.save_dir + "/t" + str(trial._trial_id))

    return acc_loss, sparsity_ratio


def parse_args():
    """Parse command line arguments."""
    ap = argparse.ArgumentParser(
        description="Hyperparameter optimization for SWS using Optuna"
    )
    ap.add_argument(
        "--model", choices=["lenet300", "lenet5", "cifar10", "cifar100"], required=True
    )
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--save-dir", type=str, required=False, default="runs/optuna")
    ap.add_argument("--n-trials", type=int, default=30, help="Number of Optuna trials")
    ap.add_argument("--study-name", type=str, default=None)
    ap.add_argument(
        "--storage", type=str, default=None, help="e.g., sqlite:///sws_optuna.db"
    )
    # Fixed parameters (not optimized)
    ap.add_argument(
        "--prior-J", type=int, default=17, help="Number of mixture components (fixed)"
    )
    ap.add_argument("--visualize", action="store_true", help="Enable GIF visualization")
    ap.add_argument(
        "--framerate", type=float, default=2, help="Framerate for the visualization GIF"
    )
    ap.add_argument(
        "--log", default="True", action="store_true", help="Enable CSV logging"
    )
    ap.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    return ap.parse_args()


def main():
    args = parse_args()

    # Create data provider once to avoid repeated downloads
    data_provider = DataModelProvider(args.model)

    # Create study for multi-objective optimization (Pareto front)
    study_name = args.study_name or f"sws_optuna_{args.model}_{_now_tag()}_pareto"
    sampler = optuna.samplers.NSGAIISampler(seed=args.seed)

    # Create multi-objective study: minimize accuracy loss, maximize sparsity
    if args.storage:
        study = optuna.create_study(
            study_name=study_name,
            directions=[
                "minimize",
                "maximize",
            ],  # [acc_loss, sparsity] -> min acc_loss and max sparsity
            sampler=sampler,
            storage=args.storage,
            load_if_exists=True,
        )
    else:
        study = optuna.create_study(
            study_name=study_name,
            directions=[
                "minimize",
                "maximize",
            ],  # [acc_loss, sparsity] -> min acc_loss and max sparsity
            sampler=sampler,
        )

    print(f"Starting optimization with {args.n_trials} trials...")
    print(f"Study name: {study_name}")
    print(
        "Optimizing for Pareto front with objectives: minimize accuracy loss, maximize sparsity"
    )

    # Run optimization
    study.optimize(
        lambda trial: objective(trial, args, data_provider),
        n_trials=args.n_trials,
        show_progress_bar=True,
        timeout=None,
    )

    # Print and save results
    print("\n=== Pareto Front Results ===")
    pareto_trials = study.best_trials
    print(f"Found {len(pareto_trials)} Pareto-optimal solutions:")

    base_runs = Path(args.save_dir)
    base_runs.mkdir(parents=True, exist_ok=True)

    pareto_results = []
    for i, trial in enumerate(pareto_trials):
        if len(trial.values) == 2:
            acc_loss, sparsity = trial.values

            print(f"Solution #{i + 1}:")
            print(f"  Accuracy Loss: {acc_loss:.6f}")
            print(f"  Sparsity: {sparsity:.6f}")
            print("  Parameters:")
            for k, v in trial.params.items():
                print(f"    {k}: {v}")

            pareto_results.append(
                {
                    "solution_number": i + 1,
                    "acc_loss": acc_loss,
                    "sparsity": sparsity,
                    "params": trial.params,
                    "trial_number": trial.number,
                }
            )

    # Save Pareto results to JSON
    pareto_file = base_runs / f"{study_name}_pareto_results.json"
    with open(pareto_file, "w") as f:
        json.dump(
            {
                "study_name": study_name,
                "n_trials": len(study.trials),
                "n_pareto": len(pareto_results),
                "model": args.model,
                "dataset": {
                    "model": args.model,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                },
                "fixed_params": {
                    "prior_J": args.prior_J,
                    "visualize": args.visualize,
                    "framerate": args.framerate,
                    "log": args.log,
                },
                "pareto_front": pareto_results,
            },
            f,
            indent=2,
        )
    print(f"\nSaved Pareto front results to: {pareto_file}")

    print(f"\nStudy directory: {args.save_dir}")
    print("Optimization complete!")


if __name__ == "__main__":
    main()
