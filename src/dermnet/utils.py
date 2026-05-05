"""
Utility functions: seeding, logging setup, training curve plotting.
"""
from __future__ import annotations

import logging
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(log_dir: str = "outputs/logs") -> None:
    """Configure root logger to write to console and a file."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(log_dir) / "train.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file),
        ],
    )


def plot_training_curves(history, save_dir: str = "outputs/figures") -> None:
    """
    Plot loss and accuracy curves from a TrainingHistory object.
    Saves two PNGs: loss_curves.png and accuracy_curves.png.

    Args:
        history: TrainingHistory dataclass with train_loss, val_loss,
                 train_acc, val_acc, lr lists.
        save_dir: directory to save figures.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history.train_loss) + 1)

    # ── Loss ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, history.train_loss, label="Train loss", linewidth=2)
    ax.plot(epochs, history.val_loss, label="Val loss", linewidth=2, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path / "loss_curves.png", dpi=150)
    plt.close(fig)

    # ── Accuracy ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, [a * 100 for a in history.train_acc],
            label="Train acc", linewidth=2)
    ax.plot(epochs, [a * 100 for a in history.val_acc],
            label="Val acc", linewidth=2, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Training & Validation Accuracy")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path / "accuracy_curves.png", dpi=150)
    plt.close(fig)

    # ── LR schedule ───────────────────────────────────────────────────────────
    if history.lr:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(epochs, history.lr, linewidth=2, color="green")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning Rate")
        ax.set_title("Learning Rate Schedule")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(save_path / "lr_schedule.png", dpi=150)
        plt.close(fig)

    print(f"Training curves saved to {save_path}/")
