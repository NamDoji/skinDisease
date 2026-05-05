"""
Evaluation: accuracy, top-5 accuracy, per-class F1, confusion matrix.
All heavy computation runs under @torch.no_grad() to avoid OOM on test set.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    top_k_accuracy_score,
)
from torch.utils.data import DataLoader


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Batch-level top-1 accuracy (float in [0, 1])."""
    return (logits.argmax(1) == labels).float().mean().item()


@torch.no_grad()
def run_evaluation(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: list[str],
    figure_dir: str | Path | None = None,
) -> dict:
    """
    Full evaluation pass over a DataLoader.

    Returns:
        dict with keys:
          accuracy, top5_accuracy, macro_f1, weighted_f1,
          per_class_f1 (dict), classification_report (str)
    """
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []
    all_probs: list[np.ndarray] = []

    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)

        all_preds.extend(logits.argmax(1).cpu().numpy().tolist())
        all_labels.extend(labels.numpy().tolist())
        all_probs.extend(probs.cpu().numpy())

    preds = np.array(all_preds)
    labels_np = np.array(all_labels)
    probs_np = np.array(all_probs)         # (N, num_classes)

    accuracy = float((preds == labels_np).mean())
    top5_acc = float(top_k_accuracy_score(labels_np, probs_np, k=min(5, len(class_names))))
    macro_f1 = float(f1_score(labels_np, preds, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(labels_np, preds, average="weighted", zero_division=0))
    per_class = f1_score(labels_np, preds, average=None, zero_division=0)
    report = classification_report(
        labels_np, preds, target_names=class_names, zero_division=0
    )

    metrics = {
        "accuracy": accuracy,
        "top5_accuracy": top5_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class_f1": {class_names[i]: float(v) for i, v in enumerate(per_class)},
        "classification_report": report,
    }

    if figure_dir is not None:
        fig_dir = Path(figure_dir)
        fig_dir.mkdir(parents=True, exist_ok=True)
        _plot_confusion_matrix(labels_np, preds, class_names, fig_dir)
        _plot_per_class_f1(metrics["per_class_f1"], fig_dir)

    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Private plotting helpers
# ──────────────────────────────────────────────────────────────────────────────

def _plot_confusion_matrix(
    labels: np.ndarray,
    preds: np.ndarray,
    class_names: list[str],
    save_dir: Path,
) -> None:
    """Row-normalised confusion matrix (recall per class on diagonal)."""
    cm = confusion_matrix(labels, preds, normalize="true")
    n = len(class_names)
    fig_size = max(12, n)
    fig, ax = plt.subplots(figsize=(fig_size, int(fig_size * 0.85)))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.4,
        ax=ax,
        annot_kws={"size": 7},
    )
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual", fontsize=11)
    ax.set_title("Normalised Confusion Matrix", fontsize=13)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    fig.tight_layout()
    fig.savefig(save_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {save_dir / 'confusion_matrix.png'}")


def _plot_per_class_f1(per_class_f1: dict[str, float], save_dir: Path) -> None:
    """Horizontal bar chart of per-class F1, colour-coded by score."""
    items = sorted(per_class_f1.items(), key=lambda x: x[1])
    names, scores = zip(*items)
    colors = [
        "#d62728" if s < 0.5 else "#ff7f0e" if s < 0.75 else "#2ca02c"
        for s in scores
    ]

    fig, ax = plt.subplots(figsize=(10, max(8, len(names) * 0.4)))
    ax.barh(names, scores, color=colors)
    ax.axvline(x=0.5, color="gray", linestyle="--", linewidth=1, label="F1=0.50")
    ax.axvline(x=0.75, color="lightgray", linestyle=":", linewidth=1, label="F1=0.75")
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1 Score")
    ax.set_title("Per-Class F1 Score  (red < 0.5 | orange 0.5–0.75 | green > 0.75)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(save_dir / "per_class_f1.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {save_dir / 'per_class_f1.png'}")
