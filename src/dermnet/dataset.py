"""
Dataset class and DataLoader factory for DermNet skin disease classification.

Key design decisions:
  - DermNetDataset reads images via OpenCV (faster than PIL for large batches)
    and applies Albumentations transforms.
  - Class imbalance is handled with TWO complementary strategies:
      1. WeightedRandomSampler: controls batch *composition* so rare classes
         appear every epoch.
      2. compute_class_weights: used in CrossEntropyLoss to penalise mistakes
         on rare classes more heavily.
  - build_dataloaders() is the single entry-point called from train.py.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import albumentations as A


class DermNetDataset(Dataset):
    """
    PyTorch Dataset for DermNet skin disease images.

    Args:
        df: DataFrame with columns [path, label, class_name].
        transform: Albumentations Compose pipeline. If None, returns raw PIL-like numpy.
    """

    def __init__(self, df: pd.DataFrame, transform: Optional[A.Compose] = None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]

        # Read as BGR, convert to RGB — Albumentations expects HWC uint8 numpy
        image = cv2.imread(str(row["path"]))
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {row['path']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image=image)["image"]  # → CHW float32 tensor

        return image, int(row["label"])


def compute_class_weights(labels: list[int], num_classes: int) -> torch.Tensor:
    """
    Inverse-frequency class weights for CrossEntropyLoss.
    Weight of class c = (1 / count_c) * normalisation_factor.

    Returns:
        Float tensor of shape (num_classes,).
    """
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    counts = np.where(counts == 0, 1.0, counts)       # guard division-by-zero
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes    # normalise: sum = num_classes
    return torch.tensor(weights, dtype=torch.float32)


def make_weighted_sampler(labels: list[int], num_classes: int) -> WeightedRandomSampler:
    """
    Per-sample weights so every class is sampled ~equally per epoch.
    Uses replacement=True so rare classes can be over-sampled.

    Returns:
        WeightedRandomSampler over the full dataset.
    """
    class_weights = compute_class_weights(labels, num_classes)
    sample_weights = [float(class_weights[lbl]) for lbl in labels]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def build_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_transform: A.Compose,
    eval_transform: A.Compose,
    batch_size: int,
    num_workers: int,
    num_classes: int,
    use_weighted_sampler: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train, val, and test DataLoaders.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_ds = DermNetDataset(train_df, train_transform)
    val_ds = DermNetDataset(val_df, eval_transform)
    test_ds = DermNetDataset(test_df, eval_transform)

    sampler = None
    shuffle = True
    if use_weighted_sampler:
        sampler = make_weighted_sampler(train_df["label"].tolist(), num_classes)
        shuffle = False  # mutually exclusive with sampler

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader
