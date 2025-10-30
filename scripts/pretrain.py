from atdl_sws import lenet_caffe, lenet_300_100, resnet, pretraining
from argparse import ArgumentParser


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument(
        "--model", choices=["lenet300", "lenet5", "cifar10", "cifar100"], required=True
    )
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=float, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--save-dir", type=str, required=False, default=None)
    args = ap.parse_args()

    dataset = {
        "lenet300": "mnist",
        "lenet5": "mnist",
        "cifar10": "cifar10",
        "cifar100": "cifar100",
    }

    model = {
        "lenet300": lenet_300_100,
        "lenet5": lenet_caffe,
        "cifar10": resnet,
        "cifar100": resnet,
    }

    n_classes = {
        "lenet300": 10,
        "lenet5": 10,
        "cifar10": 10,
        "cifar100": 100,
    }

    model = model[args.model](num_classes=n_classes[args.model])
    dataset = dataset[args.model]

    pretraining(model, args.epochs, args.lr, dataset, args.batch_size, args.save_dir)
