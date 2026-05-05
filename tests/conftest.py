"""
Shared pytest fixtures for all test modules.
"""
import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image


@pytest.fixture
def num_classes() -> int:
    return 23


@pytest.fixture
def image_size() -> int:
    return 224


@pytest.fixture
def batch_size() -> int:
    return 4


@pytest.fixture
def dummy_batch(batch_size, num_classes, image_size):
    """Random (images, labels) batch — no disk I/O."""
    images = torch.randn(batch_size, 3, image_size, image_size)
    labels = torch.randint(0, num_classes, (batch_size,))
    return images, labels


@pytest.fixture
def dummy_df(tmp_path, num_classes):
    """
    Build a minimal fake dataset: 3 images per class, 64×64 PNGs on disk.
    Returns a DataFrame with columns [path, label, class_name].
    """
    rows = []
    for class_idx in range(num_classes):
        class_dir = tmp_path / f"class_{class_idx:02d}"
        class_dir.mkdir()
        for i in range(3):
            img_path = class_dir / f"img_{i}.png"
            arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            Image.fromarray(arr).save(img_path)
            rows.append({
                "path": str(img_path),
                "label": class_idx,
                "class_name": f"class_{class_idx:02d}",
            })
    return pd.DataFrame(rows)
