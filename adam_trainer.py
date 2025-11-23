import argparse
import itertools
import math
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple
import csv

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
##


def build_transforms(dataset: str) -> Tuple[transforms.Compose, transforms.Compose]:
    """Return train/test transforms tailored to each dataset."""
    normalize_stats = {
        "cifar10": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        "svhn": ((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
    }
    mean, std = normalize_stats[dataset]

    train_tfms = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    test_tfms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return train_tfms, test_tfms


def get_dataloaders(
    dataset: str, batch_size: int, num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, int]:
    """Create train/test dataloaders and return number of classes."""
    train_tfms, test_tfms = build_transforms(dataset)
    root = os.path.join("data", dataset)

    if dataset == "cifar10":
        train_set = datasets.CIFAR10(root, train=True, download=True, transform=train_tfms)
        test_set = datasets.CIFAR10(root, train=False, download=True, transform=test_tfms)
        num_classes = 10
    elif dataset == "cifar100":
        train_set = datasets.CIFAR100(root, train=True, download=True, transform=train_tfms)
        test_set = datasets.CIFAR100(root, train=False, download=True, transform=test_tfms)
        num_classes = 100
    elif dataset == "svhn":
        train_set = datasets.SVHN(root, split="train", download=True, transform=train_tfms)
        test_set = datasets.SVHN(root, split="test", download=True, transform=test_tfms)
        num_classes = 10
    else:
        raise ValueError(f"Unsupported dataset {dataset}")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader, num_classes



def build_model(model_name: str, num_classes: int) -> nn.Module:
    """Load model (ResNet-18 or VGG16) and adapt the classifier."""
    if model_name == "resnet":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "vgg16":
        model = models.vgg16(weights=None)

        model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model.classifier[0] = nn.Linear(512, 4096)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
     
        for m in model.features:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        # Re-initialize the newly created classifier layers with proper initialization
        nn.init.kaiming_normal_(model.classifier[0].weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(model.classifier[0].bias, 0)
        nn.init.kaiming_normal_(model.classifier[6].weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(model.classifier[6].bias, 0)
        # Also re-initialize existing classifier layers for consistency
        nn.init.kaiming_normal_(model.classifier[3].weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(model.classifier[3].bias, 0)
    else:
        raise ValueError(f"Unsupported model {model_name}")
    return model


class NoisySGD(Optimizer):
    """SGD with additive Gaussian noise after the gradient step."""

    def __init__(self, params, lr: float, momentum: float = 0.0, weight_decay: float = 0.0, noise_std: float = 1e-3):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, noise_std=noise_std)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            noise_std = group["noise_std"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                state = self.state.setdefault(p, {})
                if momentum != 0:
                    buf = state.setdefault("momentum_buffer", torch.zeros_like(p))
                    buf.mul_(momentum).add_(grad)
                    update = buf
                else:
                    update = grad

                p.add_(update, alpha=-lr)
                # Add Gaussian noise to parameters after the gradient step.
                if noise_std > 0:
                    noise = torch.randn_like(p) * noise_std * math.sqrt(lr)
                    p.add_(noise)

        return loss


def get_optimizer(name: str, model: nn.Module, lr: float) -> Optimizer:
    """Instantiate optimizer by name."""
    if name == "sgd":
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    if name == "adam":
     
        return optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4, eps=1e-4)
    if name == "adamw":
 
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-3, eps=1e-8)
    if name == "noisy_sgd":
        return NoisySGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, noise_std=1e-3)
    raise ValueError(f"Unknown optimizer {name}")




@dataclass
class EpochStats:
    losses: List[float]
    accuracies: List[float]


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer = None,
    device: torch.device = torch.device("cpu"),
) -> Tuple[float, float]:
    """Run one pass over loader; training if optimizer provided, else eval."""
    is_train = optimizer is not None
    model.train(mode=is_train)
    context = torch.enable_grad() if is_train else torch.no_grad()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with context:
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == targets).sum().item()
            total_samples += inputs.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


def run_train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Train for one epoch and return average loss, accuracy, and grad-norm."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    total_grad_norm = 0.0
    num_batches = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        # Compute gradient norm (L2) over all parameters for this batch
        total_norm_sq = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2).item()
                total_norm_sq += param_norm * param_norm
        batch_grad_norm = math.sqrt(total_norm_sq) if total_norm_sq > 0.0 else 0.0
        total_grad_norm += batch_grad_norm
        num_batches += 1

        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(dim=1)
        total_correct += (preds == targets).sum().item()
        total_samples += inputs.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    avg_grad_norm = total_grad_norm / max(num_batches, 1)
    return avg_loss, avg_acc, avg_grad_norm


def train_model(
    dataset: str,
    model_name: str,
    optimizer_name: str,
    lr: float,
    epochs: int,
    batch_size: int,
    device: torch.device,
    output_dir: str,
    num_workers: int,
):
    """Train/evaluate model for a given dataset/model/optimizer combo."""
    train_loader, test_loader, num_classes = get_dataloaders(dataset, batch_size, num_workers=num_workers)
    model = build_model(model_name, num_classes).to(device)
    optimizer = get_optimizer(optimizer_name, model, lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=[100,150],gamma=0.1) 
    criterion = nn.CrossEntropyLoss()

    eval_epochs: List[int] = []
    eval_train_stats = EpochStats([], [])
    eval_test_stats = EpochStats([], [])
    loss_gaps: List[float] = []
    acc_gaps: List[float] = []
    grad_norms: List[float] = []
    # epoch, train_loss, train_acc, test_loss, test_acc, loss_gap, acc_gap, grad_norm
    metrics_records: List[Tuple[int, float, float, float, float, float, float, float]] = []

    def evaluate(epoch_idx: int, train_grad_norm: float = 0.0):
        train_loss_eval, train_acc_eval = run_epoch(
            model, train_loader, criterion, optimizer=None, device=device
        )
        test_loss_eval, test_acc_eval = run_epoch(
            model, test_loader, criterion, optimizer=None, device=device
        )
        loss_gap = test_loss_eval - train_loss_eval
        acc_gap = train_acc_eval - test_acc_eval

        eval_epochs.append(epoch_idx)
        eval_train_stats.losses.append(train_loss_eval)
        eval_train_stats.accuracies.append(train_acc_eval)
        eval_test_stats.losses.append(test_loss_eval)
        eval_test_stats.accuracies.append(test_acc_eval)
        loss_gaps.append(loss_gap)
        acc_gaps.append(acc_gap)
        grad_norms.append(train_grad_norm)
        metrics_records.append(
            (
                epoch_idx,
                train_loss_eval,
                train_acc_eval,
                test_loss_eval,
                test_acc_eval,
                loss_gap,
                acc_gap,
                train_grad_norm,
            )
        )
        print(
            f"[{dataset.upper()}][{model_name.upper()}][{optimizer_name.upper()}] "
            f"Eval @ Epoch {epoch_idx:02d} "
            f"Train Loss {train_loss_eval:.4f} Acc {train_acc_eval:.3f} "
            f"Test Loss {test_loss_eval:.4f} Acc {test_acc_eval:.3f} "
            f"Loss Gap {loss_gap:.4f} Acc Gap {acc_gap:.4f} "
            f"Train GradNorm {train_grad_norm:.4f}"
        )

    # Initial evaluation before any training (gradient norm set to 0)
    evaluate(epoch_idx=0, train_grad_norm=0.0)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc, train_grad_norm = run_train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        scheduler.step()

        print(
            f"[{dataset.upper()}][{model_name.upper()}][{optimizer_name.upper()}] "
            f"Finished Epoch {epoch:02d}/{epochs} "
            f"Train Loss {train_loss:.4f} Acc {train_acc:.3f} "
            f"Train GradNorm {train_grad_norm:.4f}"
        )

        evaluate(epoch_idx=epoch, train_grad_norm=train_grad_norm)

    save_curves(
        dataset,
        model_name,
        optimizer_name,
        eval_epochs,
        eval_train_stats,
        eval_test_stats,
        loss_gaps,
        acc_gaps,
        grad_norms,
        output_dir,
    )

    metrics_csv_path = os.path.join(output_dir, f"{dataset}_{model_name}_{optimizer_name}_metrics.csv")
    with open(metrics_csv_path, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,train_acc,test_loss,test_acc,loss_gap,acc_gap,grad_norm\n")
        for (
            epoch_idx,
            train_loss_eval,
            train_acc_eval,
            test_loss_eval,
            test_acc_eval,
            loss_gap,
            acc_gap,
            grad_norm,
        ) in metrics_records:
            f.write(
                f"{epoch_idx},{train_loss_eval:.6f},{train_acc_eval:.6f},"
                f"{test_loss_eval:.6f},{test_acc_eval:.6f},"
                f"{loss_gap:.6f},{acc_gap:.6f},{grad_norm:.6f}\n"
            )


def save_curves(
    dataset: str,
    model_name: str,
    optimizer_name: str,
    eval_epochs: List[int],
    train_stats: EpochStats,
    test_stats: EpochStats,
    loss_gaps: List[float],
    acc_gaps: List[float],
    grad_norms: List[float],
    output_dir: str,
):
    """Plot loss/accuracy curves (and gaps, grad-norm) for a single optimizer."""
    os.makedirs(output_dir, exist_ok=True)
    epoch_axis = eval_epochs

    # train_loss
    plt.figure(figsize=(8, 5))
    plt.plot(epoch_axis, train_stats.losses, label="train_loss")
    plt.title(f"{dataset.upper()}")
    plt.xlabel("Epoch")
    plt.ylabel("train_loss")
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5)
    plot_path = os.path.join(output_dir, f"{dataset}_{model_name}_{optimizer_name}_train_loss.png")
    plt.savefig(plot_path, dpi=400, bbox_inches="tight")
    plt.close()

    # train_acc
    plt.figure(figsize=(8, 5))
    plt.plot(epoch_axis, train_stats.accuracies, label="train_acc")
    plt.title(f"{dataset.upper()}")
    plt.xlabel("Epoch")
    plt.ylabel("train_acc")
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5)
    acc_plot_path = os.path.join(output_dir, f"{dataset}_{model_name}_{optimizer_name}_train_acc.png")
    plt.savefig(acc_plot_path, dpi=400, bbox_inches="tight")
    plt.close()

    # test_loss
    plt.figure(figsize=(8, 5))
    plt.plot(epoch_axis, test_stats.losses, label="test_loss")
    plt.title(f"{dataset.upper()}")
    plt.xlabel("Epoch")
    plt.ylabel("test_loss")
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5)
    test_loss_path = os.path.join(output_dir, f"{dataset}_{model_name}_{optimizer_name}_test_loss.png")
    plt.savefig(test_loss_path, dpi=400, bbox_inches="tight")
    plt.close()

    # test_acc
    plt.figure(figsize=(8, 5))
    plt.plot(epoch_axis, test_stats.accuracies, label="test_acc")
    plt.title(f"{dataset.upper()}")
    plt.xlabel("Epoch")
    plt.ylabel("test_acc")
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5)
    test_acc_path = os.path.join(output_dir, f"{dataset}_{model_name}_{optimizer_name}_test_acc.png")
    plt.savefig(test_acc_path, dpi=400, bbox_inches="tight")
    plt.close()

    # loss_gap
    plt.figure(figsize=(8, 5))
    plt.plot(epoch_axis, loss_gaps, label="loss_gap")
    plt.title(f"{dataset.upper()}")
    plt.xlabel("Epoch")
    plt.ylabel("loss_gap")
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5)
    gap_plot_path = os.path.join(output_dir, f"{dataset}_{model_name}_{optimizer_name}_loss_gap.png")
    plt.savefig(gap_plot_path, dpi=400, bbox_inches="tight")
    plt.close()

    # acc_gap
    plt.figure(figsize=(8, 5))
    plt.plot(epoch_axis, acc_gaps, label="acc_gap")
    plt.title(f"{dataset.upper()}")
    plt.xlabel("Epoch")
    plt.ylabel("acc_gap")
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5)
    acc_gap_path = os.path.join(output_dir, f"{dataset}_{model_name}_{optimizer_name}_acc_gap.png")
    plt.savefig(acc_gap_path, dpi=400, bbox_inches="tight")
    plt.close()

    # gradient_norm
    plt.figure(figsize=(8, 5))
    plt.plot(epoch_axis, grad_norms, label="gradient_norm")
    plt.title(f"{dataset.upper()}")
    plt.xlabel("Epoch")
    plt.ylabel("gradient_norm")
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5)
    grad_plot_path = os.path.join(output_dir, f"{dataset}_{model_name}_{optimizer_name}_gradient_norm.png")
    plt.savefig(grad_plot_path, dpi=400, bbox_inches="tight")
    plt.close()

    torch.save(
        {
            "train_loss": train_stats.losses,
            "train_acc": train_stats.accuracies,
            "test_loss": test_stats.losses,
            "test_acc": test_stats.accuracies,
            "loss_gap": loss_gaps,
            "acc_gap": acc_gaps,
            "grad_norm": grad_norms,
            "epochs": eval_epochs,
        },
        os.path.join(output_dir, f"{dataset}_{model_name}_{optimizer_name}_metrics.pt"),
    )


def plot_adam_vs_adamw(
    dataset: str,
    model_name: str,
    output_dir: str,
    optimizer_names: Tuple[str, str] = ("adam", "adamw"),
) -> None:
    """Load CSV metrics for Adam and AdamW and plot them on shared figures."""
    os.makedirs(output_dir, exist_ok=True)

    metrics: Dict[str, Dict[str, List[float]]] = {}
    for opt_name in optimizer_names:
        csv_path = os.path.join(output_dir, f"{dataset}_{model_name}_{opt_name}_metrics.csv")
        if not os.path.exists(csv_path):
            continue
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            epochs: List[float] = []
            train_loss: List[float] = []
            test_loss: List[float] = []
            train_acc: List[float] = []
            test_acc: List[float] = []
            loss_gap: List[float] = []
            acc_gap: List[float] = []
            grad_norm: List[float] = []
            for row in reader:
                epochs.append(float(row["epoch"]))
                train_loss.append(float(row["train_loss"]))
                train_acc.append(float(row["train_acc"]))
                test_loss.append(float(row["test_loss"]))
                test_acc.append(float(row["test_acc"]))
                loss_gap.append(float(row["loss_gap"]))
                acc_gap.append(float(row["acc_gap"]))
                grad_norm.append(float(row["grad_norm"]))
        metrics[opt_name] = {
            "epochs": epochs,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "loss_gap": loss_gap,
            "acc_gap": acc_gap,
            "grad_norm": grad_norm,
        }

    if not metrics:
        return

    def _plot_metric(metric_key: str, ylabel: str, filename_suffix: str):
        plt.figure(figsize=(8, 5))
        for opt_name, m in metrics.items():
            plt.plot(m["epochs"], m[metric_key], label=opt_name.upper())
        plt.title(f"{dataset.upper()}")
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True, linestyle="--", linewidth=0.5)
        plot_path = os.path.join(output_dir, f"{dataset}_{model_name}_adam_vs_adamw_{filename_suffix}.png")
        plt.savefig(plot_path, dpi=400, bbox_inches="tight")
        plt.close()

    # Train / test loss & accuracy, gap, and grad norm
    _plot_metric("train_loss", "train_loss", "train_loss")
    _plot_metric("test_loss", "test_loss", "test_loss")
    _plot_metric("train_acc", "train_acc", "train_acc")
    _plot_metric("test_acc", "test_acc", "test_acc")
    _plot_metric("loss_gap", "loss_gap", "loss_gap")
    _plot_metric("acc_gap", "acc_gap", "acc_gap")
    _plot_metric("grad_norm", "gradient_norm", "grad_norm")




def parse_args():
    parser = argparse.ArgumentParser(description="ResNet18/VGG16 optimizer comparison")
    parser.add_argument("--datasets", nargs="+", default=["cifar10", "cifar100", "svhn"])
    parser.add_argument("--models", nargs="+", default=["resnet", "vgg16"])
    parser.add_argument("--optimizers", nargs="+", default=["adam", "adamw"])#, "sgd", "noisy_sgd"
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    print(f"Running on device: {device}")

    for dataset, model_name, optimizer_name in itertools.product(args.datasets, args.models, args.optimizers):
        train_model(
            dataset=dataset,
            model_name=model_name,
            optimizer_name=optimizer_name,
            lr=args.lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=device,
            output_dir=args.output_dir,
            num_workers=args.num_workers,
        )


    for dataset, model_name in itertools.product(args.datasets, args.models):
        plot_adam_vs_adamw(dataset, model_name, args.output_dir, optimizer_names=("adam", "adamw"))


if __name__ == "__main__":
    main()
    